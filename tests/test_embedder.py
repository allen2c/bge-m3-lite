import json

import numpy as np
import pytest

from bge_m3_lite.embedder import BGEM3Embedder, _normalize
from tests.conftest import FIXTURES


def test_scoring_helpers():
    q = _normalize(np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
    p = _normalize(np.array([[1.0, 1.0], [1.0, 0.0]], dtype=np.float32))
    assert BGEM3Embedder.colbert_score(q, p) == pytest.approx(
        (1.0 + 0.7071) / 2, abs=1e-3
    )
    assert BGEM3Embedder.compute_lexical_matching_score(
        {"1": 0.5, "2": 0.25}, {"2": 2.0, "3": 9.0}
    ) == pytest.approx(0.5)


def test_lexical_weights_max_pool_and_special_tokens(tokenizer):
    emb = BGEM3Embedder.__new__(BGEM3Embedder)
    emb.tokenizer = tokenizer
    weights = np.array([9.0, 0.5, 0.7, 0.0, 9.0], dtype=np.float32)
    ids = [0, 42, 42, 43, 2]  # <s> x x y </s>
    assert emb._lexical_weights(weights, ids) == {"42": pytest.approx(0.7)}


def test_pad():
    ids, mask = BGEM3Embedder._pad([[0, 5, 2], [0, 2]], pad_id=1)
    assert ids.tolist() == [[0, 5, 2], [0, 2, 1]]
    assert mask.tolist() == [[1, 1, 1], [1, 1, 0]]


@pytest.mark.slow
def test_matches_flagembedding_reference():
    ref = json.loads((FIXTURES / "embeddings_ref.json").read_text(encoding="utf-8"))
    npz = np.load(FIXTURES / "embeddings_ref.npz")
    emb = BGEM3Embedder(quiet=True)
    out = emb.encode(
        ref["sentences"], batch_size=4, return_sparse=True, return_colbert_vecs=True
    )
    dense = out["dense_vecs"]
    assert dense.shape == npz["dense"].shape
    np.testing.assert_allclose(dense, npz["dense"], atol=1e-4)
    for lw, ref_lw in zip(out["lexical_weights"], ref["lexical_weights"], strict=True):
        assert set(lw) == set(ref_lw)
        for k, v in ref_lw.items():
            assert lw[k] == pytest.approx(v, abs=1e-4)
    for i, cv in enumerate(out["colbert_vecs"]):
        np.testing.assert_allclose(cv, npz[f"colbert_{i}"], atol=1e-3)


@pytest.mark.slow
def test_single_string_and_batching_order():
    emb = BGEM3Embedder(quiet=True)
    texts = ["short", "a much longer sentence about retrieval " * 5, "中文"]
    batched = emb.encode(texts, batch_size=2)["dense_vecs"]
    for i, t in enumerate(texts):
        single = emb.encode(t)["dense_vecs"]
        assert single.shape == (1024,)
        np.testing.assert_allclose(single, batched[i], atol=1e-4)


def test_batches_respect_text_and_token_budgets():
    ids = [[0] * n for n in (5, 100, 3, 60, 60, 8)]
    # longest first; 100 alone (budget 128 < 2 * 100), then 60+60, then 8+5+3
    assert BGEM3Embedder._batches(ids, batch_size=12, max_batch_tokens=128) == [
        [1],
        [3, 4],
        [5, 0, 2],
    ]
    # batch_size still caps the number of texts
    assert BGEM3Embedder._batches(ids, batch_size=2, max_batch_tokens=10**6) == [
        [1, 3],
        [4, 5],
        [0, 2],
    ]
    # a text longer than the budget gets a batch of its own
    assert BGEM3Embedder._batches(ids, batch_size=12, max_batch_tokens=1) == [
        [1],
        [3],
        [4],
        [5],
        [0],
        [2],
    ]
    assert BGEM3Embedder._batches([], 12, 128) == []
    with pytest.raises(ValueError):
        BGEM3Embedder._batches(ids, 0, 128)


class _FakeBackbone:
    """Deterministic hidden states: token id and position encoded in the vector."""

    calls: list[tuple[int, int]]

    def __init__(self) -> None:
        self.calls = []

    def __call__(self, input_ids: np.ndarray, mask: np.ndarray) -> np.ndarray:
        self.calls.append(input_ids.shape)
        b, s = input_ids.shape
        h = np.zeros((b, s, 1024), dtype=np.float32)
        h[:, :, 0] = 1.0
        h[:, :, 1] = input_ids % 7
        h[:, :, 2] = np.arange(s)[None, :] % 5
        return h * mask[:, :, None]


@pytest.fixture
def fake_embedder(tokenizer):
    emb = BGEM3Embedder.__new__(BGEM3Embedder)
    emb.tokenizer = tokenizer
    emb.backbone = _FakeBackbone()  # pyright: ignore[reportAttributeAccessIssue]
    emb.max_length = 8192
    emb.query_max_length = 6
    emb.passage_max_length = 32
    rng = np.random.default_rng(0)
    emb.sparse_w = rng.standard_normal((1024, 1)).astype(np.float32)
    emb.sparse_b = np.zeros(1, dtype=np.float32)
    emb.colbert_w = rng.standard_normal((1024, 1024)).astype(np.float32)
    emb.colbert_b = np.zeros(1024, dtype=np.float32)
    return emb


def test_encode_is_independent_of_batching(fake_embedder):
    texts = ["short", "a much longer sentence about retrieval " * 5, "中文", "x y z"]
    a = fake_embedder.encode(
        texts, batch_size=1, return_sparse=True, return_colbert_vecs=True
    )
    budget = max(len(ids) for ids in fake_embedder.tokenize(texts)) + 4
    b = fake_embedder.encode(
        texts,
        batch_size=12,
        max_batch_tokens=budget,
        return_sparse=True,
        return_colbert_vecs=True,
    )
    np.testing.assert_allclose(a["dense_vecs"], b["dense_vecs"])
    assert a["lexical_weights"] == b["lexical_weights"]
    for x, y in zip(a["colbert_vecs"], b["colbert_vecs"], strict=True):
        np.testing.assert_allclose(x, y, atol=1e-6)
    later = fake_embedder.backbone.calls[len(texts) :]
    assert 1 < len(later) < len(texts) and max(s[0] * s[1] for s in later) <= budget


def test_encode_queries_and_corpus_defaults(fake_embedder):
    text = "one two three four five six seven eight nine ten"
    fake_embedder.encode_queries(text)
    fake_embedder.encode_corpus(text)
    fake_embedder.encode_corpus(text, max_length=4)
    widths = [s[1] for s in fake_embedder.backbone.calls]
    assert widths == [6, len(fake_embedder.tokenize([text])[0]), 4]


def test_compute_score_matches_manual_combination(fake_embedder):
    pairs = [("what is bge", "BGE M3 is an embedding model"), ("hi", "unrelated")]
    scores = fake_embedder.compute_score(pairs, weights_for_different_modes=[2, 1, 1])
    assert set(scores) == {
        "colbert",
        "sparse",
        "dense",
        "sparse+dense",
        "colbert+sparse+dense",
    }
    q = fake_embedder.encode_queries(
        [p[0] for p in pairs], return_sparse=True, return_colbert_vecs=True
    )
    p = fake_embedder.encode_corpus(
        [p[1] for p in pairs], return_sparse=True, return_colbert_vecs=True
    )
    for i in range(2):
        d = float(q["dense_vecs"][i] @ p["dense_vecs"][i])
        s = fake_embedder.compute_lexical_matching_score(
            q["lexical_weights"][i], p["lexical_weights"][i]
        )
        c = fake_embedder.colbert_score(q["colbert_vecs"][i], p["colbert_vecs"][i])
        assert scores["dense"][i] == pytest.approx(d)
        assert scores["sparse"][i] == pytest.approx(s)
        assert scores["colbert"][i] == pytest.approx(c)
        assert scores["sparse+dense"][i] == pytest.approx((2 * d + s) / 3)
        assert scores["colbert+sparse+dense"][i] == pytest.approx((2 * d + s + c) / 4)
    single = fake_embedder.compute_score(
        pairs[0], weights_for_different_modes=[2, 1, 1]
    )
    assert single["dense"] == pytest.approx(scores["dense"][0])
    with pytest.raises(ValueError):
        fake_embedder.compute_score(pairs, weights_for_different_modes=[1, 1])


@pytest.mark.slow
def test_int8_backbone_close_to_reference():
    ref = json.loads((FIXTURES / "embeddings_ref.json").read_text(encoding="utf-8"))
    npz = np.load(FIXTURES / "embeddings_ref.npz")
    emb = BGEM3Embedder(quiet=True, precision="int8")
    out = emb.encode(ref["sentences"], batch_size=4, return_sparse=True)
    cos = (out["dense_vecs"] * npz["dense"]).sum(axis=1)
    assert cos.min() > 0.995 and cos.mean() > 0.997
    for lw, ref_lw in zip(out["lexical_weights"], ref["lexical_weights"], strict=True):
        top = sorted(ref_lw, key=ref_lw.get, reverse=True)[:3]
        assert set(lw) >= set(top)  # the strongest reference tokens survive
