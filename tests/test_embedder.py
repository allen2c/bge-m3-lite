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
