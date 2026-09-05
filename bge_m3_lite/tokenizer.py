"""XLM-RoBERTa tokenizer implemented from scratch on top of a SentencePiece model.

Reproduces exactly what ``transformers``' fast tokenizer does for BAAI/bge-m3,
without ``sentencepiece`` / ``tokenizers`` / ``transformers``. The reference
pipeline (dumped from the runtime ``tokenizers`` object) is::

    AddedVocabulary  : split out literal special tokens (<s> </s> <pad> <unk> <mask>)
    Precompiled      : SentencePiece nmt_nfkc charsmap, applied per grapheme cluster
    WhitespaceSplit  : drop Unicode whitespace, keep the words
    Metaspace        : prefix every word with U+2581
    Unigram          : Viterbi best segmentation, consecutive <unk> fused
    ids              : <s>=0 <pad>=1 </s>=2 <unk>=3 <mask>=250001, other = spm_id + 1

Implementation notes were taken from ``sentencepiece/src/normalizer.cc``, the
``spm_precompiled`` crate and ``tokenizers/src/models/unigram/model.rs``.
"""

from __future__ import annotations

import os
import re
import struct
from dataclasses import dataclass

from bge_m3_lite import _proto
from bge_m3_lite._grapheme import graphemes

_UNK_PENALTY = 10.0
SPACE_SYMBOL = "▁"

# Piece types from sentencepiece_model.proto
_PIECE_NORMAL = 1
_PIECE_UNKNOWN = 2
_PIECE_CONTROL = 3
_PIECE_USER_DEFINED = 4

# Rust's char::is_whitespace == Unicode White_Space property.
_WHITESPACE = frozenset(
    "\t\n\x0b\x0c\r \x85\xa0\u1680\u2000\u2001\u2002\u2003\u2004\u2005\u2006"
    "\u2007\u2008\u2009\u200a\u2028\u2029\u202f\u205f\u3000"
)
_WS_CLASS = "[" + "".join(re.escape(c) for c in sorted(_WHITESPACE)) + "]"


class _DoubleArrayTrie:
    """Read-only Darts double array (the layout used by SentencePiece)."""

    __slots__ = ("_units",)

    def __init__(self, blob: bytes) -> None:
        if len(blob) % 4:
            raise ValueError("double array blob size must be a multiple of 4")
        self._units = struct.unpack(f"<{len(blob) // 4}I", blob)

    def common_prefix_search(self, key: bytes) -> list[int]:
        """Values of every trie key that is a prefix of ``key`` (shortest first)."""
        units = self._units
        node = 0
        unit = units[node]
        node ^= (unit >> 10) << ((unit & 0x200) >> 6)
        results: list[int] = []
        for byte in key:
            node ^= byte
            unit = units[node]
            if unit & 0x800000FF != byte:
                break
            node ^= (unit >> 10) << ((unit & 0x200) >> 6)
            if (unit >> 8) & 1:
                results.append(units[node] & 0x7FFFFFFF)
        return results


class PrecompiledCharsmap:
    """SentencePiece ``precompiled_charsmap`` with Hugging Face's application rule."""

    def __init__(self, blob: bytes) -> None:
        (trie_size,) = struct.unpack("<I", blob[:4])
        self._trie = _DoubleArrayTrie(blob[4 : 4 + trie_size])
        self._normalized = blob[4 + trie_size :]
        if not self._normalized.endswith(b"\0"):
            raise ValueError("charsmap normalized block must be NUL terminated")
        self._cache: dict[str, str | None] = {}

    def transform(self, chunk: str) -> str | None:
        """Replacement for ``chunk`` if any prefix of it is a charsmap key."""
        if chunk in self._cache:
            return self._cache[chunk]
        results = self._trie.common_prefix_search(chunk.encode("utf-8"))
        if not results:
            out = None
        else:
            start = results[0]  # HF takes the *first* (shortest) match
            end = self._normalized.index(b"\0", start)
            out = self._normalized[start:end].decode("utf-8")
        if len(self._cache) < 65536:
            self._cache[chunk] = out
        return out

    def normalize(self, text: str) -> str:
        """Port of ``Precompiled::normalize`` from Hugging Face ``tokenizers``."""
        if text.isascii() and self._ascii_clean(text):
            return text
        out: list[str] = []
        for g in graphemes(text):
            if len(g) == 1 or len(g.encode("utf-8")) < 6:
                norm = self.transform(g)
                if norm is not None:
                    out.append(norm)
                    continue
                if len(g) == 1:
                    out.append(g)
                    continue
            for c in g:
                norm = self.transform(c)
                out.append(c if norm is None else norm)
        return "".join(out)

    def _ascii_clean(self, text: str) -> bool:
        # Fast path: printable ASCII other than control chars is never remapped.
        return all(0x20 <= ord(c) < 0x7F for c in text)


@dataclass(frozen=True)
class SentencePieceModel:
    pieces: list[str]
    scores: list[float]
    types: list[int]
    unk_id: int
    charsmap: PrecompiledCharsmap | None

    @classmethod
    def from_bytes(cls, data: bytes) -> SentencePieceModel:
        pieces: list[str] = []
        scores: list[float] = []
        types: list[int] = []
        trainer: _proto.Message = {}
        norm: _proto.Message = {}
        for field, _wire, value in _proto.iter_fields(data):
            if not isinstance(value, bytes):
                continue
            if field == 1:  # repeated SentencePiece pieces
                msg = _proto.as_message(value)
                pieces.append(_proto.get_bytes(msg, 1).decode("utf-8"))
                score = _proto.get_bytes(msg, 2)
                scores.append(_proto.as_float32(score) if score else 0.0)
                types.append(_proto.get_int(msg, 3, _PIECE_NORMAL))
            elif field == 2:
                trainer = _proto.as_message(value)
            elif field == 3:
                norm = _proto.as_message(value)
        model_type = _proto.get_int(trainer, 3, 1)
        if model_type != 1:
            raise ValueError(f"only unigram models are supported (got {model_type})")
        unk_ids = [i for i, t in enumerate(types) if t == _PIECE_UNKNOWN]
        if len(unk_ids) != 1:
            raise ValueError("model must define exactly one <unk> piece")
        blob = _proto.get_bytes(norm, 2)
        charsmap = PrecompiledCharsmap(blob) if blob else None
        return cls(pieces, scores, types, unk_ids[0], charsmap)

    @classmethod
    def from_file(cls, path: str | os.PathLike[str]) -> SentencePieceModel:
        with open(path, "rb") as fh:
            return cls.from_bytes(fh.read())


class UnigramModel:
    """Unigram Viterbi encoder over a SentencePiece vocabulary (HF semantics)."""

    def __init__(self, model: SentencePieceModel, *, fuse_unk: bool = True) -> None:
        self.model = model
        self.fuse_unk = fuse_unk
        self.piece_to_id: dict[str, int] = {}
        for i, (piece, ptype) in enumerate(zip(model.pieces, model.types, strict=True)):
            if ptype in (_PIECE_NORMAL, _PIECE_USER_DEFINED):
                self.piece_to_id[piece] = i
        self.max_piece_len = max((len(p) for p in self.piece_to_id), default=1)
        self.min_score = min(model.scores, default=0.0)
        self.unk_score = self.min_score - _UNK_PENALTY

    def encode(self, word: str) -> list[int]:
        """Encode one pre-tokenized word into spm piece ids."""
        n = len(word)
        if n == 0:
            return []
        piece_to_id = self.piece_to_id
        scores = self.model.scores
        max_len = self.max_piece_len
        unk_id = self.model.unk_id
        best_score = [0.0] * (n + 1)
        best_start = [-1] * (n + 1)
        best_id = [-1] * (n + 1)
        for start in range(n):
            base = best_score[start]
            has_single = False
            limit = min(n, start + max_len)
            for end in range(start + 1, limit + 1):
                pid = piece_to_id.get(word[start:end])
                if pid is None:
                    continue
                cand = base + scores[pid]
                if best_start[end] == -1 or cand > best_score[end]:
                    best_score[end] = cand
                    best_start[end] = start
                    best_id[end] = pid
                if end == start + 1:
                    has_single = True
            if not has_single:
                end = start + 1
                cand = base + self.unk_score
                if best_start[end] == -1 or cand > best_score[end]:
                    best_score[end] = cand
                    best_start[end] = start
                    best_id[end] = unk_id
        ids: list[int] = []
        end = n
        while end > 0:
            pid = best_id[end]
            if not (self.fuse_unk and pid == unk_id and ids and ids[-1] == unk_id):
                ids.append(pid)
            end = best_start[end]
        ids.reverse()
        return ids


class XLMRobertaTokenizer:
    """Tokenizer producing the exact ids ``transformers`` yields for BAAI/bge-m3."""

    BOS_ID = 0  # <s>
    PAD_ID = 1  # <pad>
    EOS_ID = 2  # </s>
    UNK_ID = 3  # <unk>
    _FAIRSEQ_OFFSET = 1

    def __init__(self, spm_model: SentencePieceModel) -> None:
        self.unigram = UnigramModel(spm_model)
        self.charsmap = spm_model.charsmap
        self.vocab_size = len(spm_model.pieces) + self._FAIRSEQ_OFFSET + 1
        self.mask_id = self.vocab_size - 1  # <mask> = 250001
        self.special_ids = frozenset(
            {self.BOS_ID, self.PAD_ID, self.EOS_ID, self.UNK_ID}
        )
        self._specials = {
            "<s>": self.BOS_ID,
            "<pad>": self.PAD_ID,
            "</s>": self.EOS_ID,
            "<unk>": self.UNK_ID,
            "<mask>": self.mask_id,
        }
        # <mask> has lstrip=True: it swallows the whitespace before it.
        self._special_re = re.compile(
            "|".join(
                [f"(?:{_WS_CLASS}*<mask>)"]
                + [re.escape(s) for s in self._specials if s != "<mask>"]
            )
        )

    @classmethod
    def from_file(cls, path: str | os.PathLike[str]) -> XLMRobertaTokenizer:
        return cls(SentencePieceModel.from_file(path))

    # -- pipeline -----------------------------------------------------------

    def _split_specials(self, text: str) -> list[tuple[str, int | None]]:
        out: list[tuple[str, int | None]] = []
        pos = 0
        for m in self._special_re.finditer(text):
            if m.start() > pos:
                out.append((text[pos : m.start()], None))
            out.append((m.group(0), self._specials[m.group(0).lstrip()]))
            pos = m.end()
        if pos < len(text):
            out.append((text[pos:], None))
        return out

    def normalize(self, text: str) -> str:
        return self.charsmap.normalize(text) if self.charsmap else text

    @staticmethod
    def pre_tokenize(normalized: str) -> list[str]:
        """WhitespaceSplit + Metaspace: words, each prefixed with U+2581."""
        words: list[str] = []
        current: list[str] = []
        for c in normalized:
            if c in _WHITESPACE:
                if current:
                    words.append("".join(current))
                    current = []
            else:
                current.append(c)
        if current:
            words.append("".join(current))
        return [w if w.startswith(SPACE_SYMBOL) else SPACE_SYMBOL + w for w in words]

    def _remap(self, spm_id: int) -> int:
        if spm_id == self.unigram.model.unk_id:
            return self.UNK_ID
        return spm_id + self._FAIRSEQ_OFFSET

    def tokenize_ids(self, text: str) -> list[int]:
        """Token ids without the surrounding ``<s>`` / ``</s>``."""
        ids: list[int] = []
        encode = self.unigram.encode
        remap = self._remap
        for segment, special_id in self._split_specials(text):
            if special_id is not None:
                ids.append(special_id)
                continue
            for word in self.pre_tokenize(self.normalize(segment)):
                ids.extend(remap(i) for i in encode(word))
        return ids

    def encode(self, text: str, max_length: int = 8192) -> list[int]:
        """``<s> ... </s>`` with truncation, matching ``tokenizer(text)`` in HF."""
        if max_length < 2:
            raise ValueError("max_length must be at least 2 (for <s> and </s>)")
        body = self.tokenize_ids(text)
        if len(body) > max_length - 2:
            body = body[: max_length - 2]
        return [self.BOS_ID, *body, self.EOS_ID]

    # -- helpers ------------------------------------------------------------

    def id_to_token(self, token_id: int) -> str:
        for token, sid in self._specials.items():
            if token_id == sid:
                return token
        return self.unigram.model.pieces[token_id - self._FAIRSEQ_OFFSET]

    def tokenize(self, text: str) -> list[str]:
        return [self.id_to_token(i) for i in self.tokenize_ids(text)]

    @staticmethod
    def convert_tokens_to_string(tokens: list[str]) -> str:
        return "".join(tokens).replace(SPACE_SYMBOL, " ").strip()
