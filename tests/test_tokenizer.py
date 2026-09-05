import json

import pytest

from bge_m3_lite.tokenizer import SentencePieceModel, XLMRobertaTokenizer
from tests.conftest import FIXTURES

CASES = json.loads((FIXTURES / "tokenizer_cases.json").read_text(encoding="utf-8"))


def test_model_metadata(tokenizer):
    assert tokenizer.vocab_size == 250002
    assert tokenizer.mask_id == 250001
    assert tokenizer.unigram.model.unk_id == 0
    assert tokenizer.unigram.max_piece_len == 16
    assert tokenizer.charsmap is not None


@pytest.mark.parametrize("case", CASES, ids=[c["text"][:24] for c in CASES])
def test_matches_transformers(tokenizer, case):
    assert tokenizer.encode(case["text"]) == case["ids"]


def test_special_tokens_and_lstrip_mask(tokenizer):
    assert tokenizer.encode("") == [0, 2]
    assert tokenizer.tokenize_ids("a <mask>") == [10, 250001]
    assert tokenizer.tokenize_ids("<s></s><pad><unk>") == [0, 2, 1, 3]


def test_truncation(tokenizer):
    ids = tokenizer.encode("word " * 100, max_length=16)
    assert len(ids) == 16 and ids[0] == 0 and ids[-1] == 2


def test_roundtrip_tokens(tokenizer):
    tokens = tokenizer.tokenize("Hello world, 你好世界")
    assert tokens[0] == "▁Hello"
    assert tokenizer.convert_tokens_to_string(tokens) == "Hello world, 你好世界"


def test_hf_precompiled_quirks(tokenizer):
    # whole-grapheme lookup < 6 bytes keeps only the shortest key match ("Ａ" -> "A")
    assert tokenizer.tokenize("Ａ́") == ["▁A"]
    # >= 6 bytes: per character, so halfwidth katakana + voiced mark stay separate
    assert tokenizer.tokenize("ﾊﾟ") == ["▁", "ハ", "゚"]
    # consecutive unknown characters are fused into one <unk>
    assert tokenizer.tokenize("\U00020000\U00020001") == ["▁", "<unk>"]


def test_rejects_non_unigram():
    with pytest.raises(ValueError):
        SentencePieceModel.from_bytes(b"")


def test_from_file(tokenizer_path):
    tok = XLMRobertaTokenizer.from_file(str(tokenizer_path))
    assert tok.encode("Hello world!") == [0, 35378, 8999, 38, 2]


def test_max_length_validation(tokenizer):
    with pytest.raises(ValueError):
        tokenizer.encode("x", max_length=1)
    assert tokenizer.encode("hello there", max_length=2) == [0, 2]


def test_vocab_cache_roundtrip(tokenizer_path, tmp_path):
    import shutil

    model = tmp_path / "spm.model"
    shutil.copy(tokenizer_path, model)
    cache = tmp_path / "spm.model.cache"
    parsed = SentencePieceModel.from_file(model, cache=False)
    assert not cache.exists()
    first = SentencePieceModel.from_file(model)
    assert cache.is_file()
    second = SentencePieceModel.from_file(model)
    for m in (first, second):
        assert m.pieces == parsed.pieces
        assert m.scores == parsed.scores
        assert m.types == parsed.types
        assert m.unk_id == parsed.unk_id
        assert m.charsmap is not None and parsed.charsmap is not None
        assert m.charsmap.blob == parsed.charsmap.blob
    # a stale or corrupt cache is ignored and rewritten
    cache.write_bytes(b"garbage")
    assert SentencePieceModel.from_file(model).pieces == parsed.pieces
    assert cache.read_bytes() != b"garbage"
    # a cache written for another model file (different digest) is not used
    head, body = cache.read_bytes().split(b"\n\n", 1)
    lines = head.split(b"\n")
    lines[1] = b"0" * 64
    lines[2] = b"1"  # would change unk_id if the stale cache were trusted
    cache.write_bytes(b"\n".join(lines) + b"\n\n" + body)
    assert SentencePieceModel.from_file(model).unk_id == parsed.unk_id
