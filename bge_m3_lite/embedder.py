"""BGE-M3 embedder: dense, sparse (lexical weights) and ColBERT multi-vectors."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np

from bge_m3_lite import hub
from bge_m3_lite._torch_pickle import load_state_dict
from bge_m3_lite.model import OnnxBackbone
from bge_m3_lite.tokenizer import XLMRobertaTokenizer

HIDDEN_SIZE = 1024
MAX_LENGTH = 8192
QUERY_MAX_LENGTH = 512
BATCH_SIZE = 12
MAX_BATCH_TOKENS = 16384  # padded tokens per forward pass (texts x width)


class BGEM3Embedder:
    """CPU inference for BAAI/bge-m3 producing the same outputs as FlagEmbedding.

    >>> embedder = BGEM3Embedder()            # downloads ~2.3 GB on first use
    >>> out = embedder.encode(["hello"], return_sparse=True, return_colbert_vecs=True)
    >>> out["dense_vecs"].shape
    (1, 1024)
    """

    def __init__(
        self,
        cache_dir: str | Path | None = None,
        *,
        num_threads: int | None = None,
        max_length: int = MAX_LENGTH,
        quiet: bool = False,
        precision: Literal["fp32", "int8"] = "fp32",
        model_path: str | Path | None = None,
        query_max_length: int = QUERY_MAX_LENGTH,
        passage_max_length: int | None = None,
    ) -> None:
        """``precision="int8"`` uses the quantised backbone (see docs/quantization.md);
        ``model_path`` points at any backbone ONNX file and skips the download.

        ``query_max_length`` / ``passage_max_length`` are the defaults for
        :meth:`encode_queries`, :meth:`encode_corpus` and :meth:`compute_score`
        (passages default to ``max_length``; FlagEmbedding truncates both at 512).
        """
        files = hub.ensure_files(
            hub.TOKENIZER_FILES + hub.HEAD_FILES, cache_dir, quiet=quiet
        )
        if model_path is not None:
            backbone_path = Path(model_path)
        elif precision == "int8":
            backbone_path = hub.ensure_files((hub.INT8_FILE,), cache_dir, quiet=quiet)[
                hub.INT8_FILE.name
            ]
        elif precision == "fp32":
            backbone_path = hub.ensure_files(hub.MODEL_FILES, cache_dir, quiet=quiet)[
                "model.onnx"
            ]
        else:
            raise ValueError(f"unknown precision {precision!r}")
        self.precision = precision
        self.tokenizer = XLMRobertaTokenizer.from_file(files["sentencepiece.bpe.model"])
        self.backbone = OnnxBackbone(backbone_path, num_threads=num_threads)
        self.max_length = max_length
        self.query_max_length = query_max_length
        self.passage_max_length = (
            max_length if passage_max_length is None else passage_max_length
        )

        sparse = load_state_dict(files["sparse_linear.pt"])
        colbert = load_state_dict(files["colbert_linear.pt"])
        # Both heads are stored in fp16 upstream and used as fp32 on CPU.
        self.sparse_w = np.ascontiguousarray(sparse["weight"].T, dtype=np.float32)
        self.sparse_b = sparse["bias"].astype(np.float32)
        self.colbert_w = np.ascontiguousarray(colbert["weight"].T, dtype=np.float32)
        self.colbert_b = colbert["bias"].astype(np.float32)
        if self.sparse_w.shape != (HIDDEN_SIZE, 1):
            raise ValueError(f"unexpected sparse head shape {self.sparse_w.shape}")
        if self.colbert_w.shape != (HIDDEN_SIZE, HIDDEN_SIZE):
            raise ValueError(f"unexpected colbert head shape {self.colbert_w.shape}")

    def close(self) -> None:
        """Release the onnxruntime session (also happens on garbage collection)."""
        self.backbone.close()

    def __enter__(self) -> BGEM3Embedder:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- tokenization -------------------------------------------------------

    def tokenize(
        self, texts: Sequence[str], max_length: int | None = None
    ) -> list[list[int]]:
        if max_length is None:
            max_length = self.max_length
        max_length = min(max_length, MAX_LENGTH)
        return [self.tokenizer.encode(t, max_length=max_length) for t in texts]

    @staticmethod
    def _batches(
        token_ids: list[list[int]], batch_size: int, max_batch_tokens: int
    ) -> list[list[int]]:
        """Indices grouped longest-first so that every batch holds at most
        ``batch_size`` texts and ``batch_size * width <= max_batch_tokens``
        padded tokens (a single text longer than the budget gets its own batch).
        Peak memory then depends on the token budget, not on the longest text."""
        if batch_size < 1 or max_batch_tokens < 1:
            raise ValueError("batch_size and max_batch_tokens must be >= 1")
        order = sorted(range(len(token_ids)), key=lambda i: -len(token_ids[i]))
        batches: list[list[int]] = []
        current: list[int] = []
        width = 0
        for i in order:
            if current and (
                len(current) >= batch_size
                or (len(current) + 1) * width > max_batch_tokens
            ):
                batches.append(current)
                current = []
            if not current:
                width = len(token_ids[i])  # longest first: the width of this batch
            current.append(i)
        if current:
            batches.append(current)
        return batches

    @staticmethod
    def _pad(batch: list[list[int]], pad_id: int) -> tuple[np.ndarray, np.ndarray]:
        width = max(len(ids) for ids in batch)
        input_ids = np.full((len(batch), width), pad_id, dtype=np.int64)
        mask = np.zeros((len(batch), width), dtype=np.int64)
        for i, ids in enumerate(batch):
            input_ids[i, : len(ids)] = ids
            mask[i, : len(ids)] = 1
        return input_ids, mask

    # -- encoding -----------------------------------------------------------

    def encode(
        self,
        texts: str | Sequence[str],
        *,
        batch_size: int = BATCH_SIZE,
        max_batch_tokens: int = MAX_BATCH_TOKENS,
        max_length: int | None = None,
        return_dense: bool = True,
        return_sparse: bool = False,
        return_colbert_vecs: bool = False,
    ) -> dict[str, Any]:
        """Return ``dense_vecs`` / ``lexical_weights`` / ``colbert_vecs``.

        Keys that were not requested are ``None``. For a single string input the
        per-text containers are unwrapped, mirroring FlagEmbedding.

        Batches are formed longest-first and are bounded both by ``batch_size``
        texts and by ``max_batch_tokens`` padded tokens, so mixed short/long
        inputs never blow up memory: the hidden state of a batch is
        ``max_batch_tokens * 4 KiB`` at most (64 MiB by default).
        """
        single = isinstance(texts, str)
        if single:
            texts = [texts]
        texts = list(texts)
        n = len(texts)
        token_ids = self.tokenize(texts, max_length)

        dense = np.empty((n, HIDDEN_SIZE), dtype=np.float32) if return_dense else None
        lexical: list[dict[str, float] | None] = [None] * n
        colbert: list[np.ndarray | None] = [None] * n

        # Longest first: peak memory shows up in the first batch, less padding overall.
        for idx in self._batches(token_ids, batch_size, max_batch_tokens):
            batch = [token_ids[i] for i in idx]
            input_ids, mask = self._pad(batch, self.tokenizer.PAD_ID)
            hidden = self.backbone(input_ids, mask)  # (b, s, 1024)
            if dense is not None:
                dense[idx] = _normalize(hidden[:, 0, :])
            if return_sparse:
                weights = np.maximum(hidden @ self.sparse_w + self.sparse_b, 0.0)
                for row, i in enumerate(idx):
                    lexical[i] = self._lexical_weights(weights[row, :, 0], batch[row])
            if return_colbert_vecs:
                for row, i in enumerate(idx):
                    # Drop <s>, keep </s> and skip padding, like FlagEmbedding.
                    tokens = hidden[row, 1 : len(batch[row]), :]
                    colbert[i] = _normalize(tokens @ self.colbert_w + self.colbert_b)

        out: dict[str, Any] = {
            "dense_vecs": dense,
            "lexical_weights": lexical if return_sparse else None,
            "colbert_vecs": colbert if return_colbert_vecs else None,
        }
        if single:
            out = {k: (v[0] if v is not None else None) for k, v in out.items()}
        return out

    def encode_queries(
        self, queries: str | Sequence[str], *, max_length: int | None = None, **kw: Any
    ) -> dict[str, Any]:
        """:meth:`encode` with ``max_length`` defaulting to ``query_max_length``."""
        if max_length is None:
            max_length = self.query_max_length
        return self.encode(queries, max_length=max_length, **kw)

    def encode_corpus(
        self, corpus: str | Sequence[str], *, max_length: int | None = None, **kw: Any
    ) -> dict[str, Any]:
        """:meth:`encode` with ``max_length`` defaulting to ``passage_max_length``."""
        if max_length is None:
            max_length = self.passage_max_length
        return self.encode(corpus, max_length=max_length, **kw)

    def compute_score(
        self,
        sentence_pairs: tuple[str, str] | Sequence[tuple[str, str]],
        *,
        batch_size: int = BATCH_SIZE,
        max_batch_tokens: int = MAX_BATCH_TOKENS,
        max_query_length: int | None = None,
        max_passage_length: int | None = None,
        weights_for_different_modes: Sequence[float] | None = None,
    ) -> dict[str, Any]:
        """Relevance of each (query, passage) pair under every BGE-M3 mode.

        Returns ``dense`` (cosine), ``sparse`` (lexical matching), ``colbert``
        (late interaction) and the weighted combinations ``sparse+dense`` and
        ``colbert+sparse+dense`` as lists of floats, like FlagEmbedding's
        ``compute_score``. ``weights_for_different_modes`` orders the weights as
        ``[dense, sparse, colbert]`` (default ``[1, 1, 1]``). A single pair
        returns plain floats.
        """
        single = len(sentence_pairs) == 2 and isinstance(sentence_pairs[0], str)
        pairs: list[tuple[str, str]] = (
            [sentence_pairs]  # type: ignore[list-item]
            if single
            else list(sentence_pairs)  # type: ignore[arg-type]
        )
        w = (
            [1.0, 1.0, 1.0]
            if weights_for_different_modes is None
            else list(map(float, weights_for_different_modes))
        )
        if len(w) != 3:
            raise ValueError(
                "weights_for_different_modes needs [dense, sparse, colbert]"
            )
        kw: dict[str, Any] = dict(
            batch_size=batch_size,
            max_batch_tokens=max_batch_tokens,
            return_sparse=True,
            return_colbert_vecs=True,
        )
        queries: list[str] = [pair[0] for pair in pairs]
        passages: list[str] = [pair[1] for pair in pairs]
        q = self.encode_queries(queries, max_length=max_query_length, **kw)
        p = self.encode_corpus(passages, max_length=max_passage_length, **kw)
        dense = (q["dense_vecs"] * p["dense_vecs"]).sum(axis=1).tolist()
        sparse = [
            self.compute_lexical_matching_score(a, b)
            for a, b in zip(q["lexical_weights"], p["lexical_weights"], strict=True)
        ]
        colbert = [
            self.colbert_score(a, b)
            for a, b in zip(q["colbert_vecs"], p["colbert_vecs"], strict=True)
        ]
        scores: dict[str, list[float]] = {
            "colbert": colbert,
            "sparse": sparse,
            "dense": dense,
            "sparse+dense": [
                (w[1] * s + w[0] * d) / (w[0] + w[1])
                for s, d in zip(sparse, dense, strict=True)
            ],
            "colbert+sparse+dense": [
                (w[2] * c + w[1] * s + w[0] * d) / sum(w)
                for c, s, d in zip(colbert, sparse, dense, strict=True)
            ],
        }
        if single:
            return {k: v[0] for k, v in scores.items()}
        return scores

    def _lexical_weights(self, weights: np.ndarray, ids: list[int]) -> dict[str, float]:
        special = self.tokenizer.special_ids
        result: dict[str, float] = {}
        for w, tid in zip(weights[: len(ids)].tolist(), ids, strict=True):
            if tid in special or w <= 0:
                continue
            key = str(tid)
            if w > result.get(key, 0.0):
                result[key] = w
        return result

    # -- scoring helpers (same semantics as FlagEmbedding) -------------------

    def convert_id_to_token(
        self, lexical_weights: dict[str, float]
    ) -> dict[str, float]:
        return {
            self.tokenizer.id_to_token(int(k)): v for k, v in lexical_weights.items()
        }

    @staticmethod
    def compute_lexical_matching_score(
        lw1: dict[str, float], lw2: dict[str, float]
    ) -> float:
        return float(sum(w * lw2[t] for t, w in lw1.items() if t in lw2))

    @staticmethod
    def colbert_score(q_reps: np.ndarray, p_reps: np.ndarray) -> float:
        token_scores = q_reps @ p_reps.T
        return float(token_scores.max(axis=1).sum() / q_reps.shape[0])


def _normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(norm, eps)
