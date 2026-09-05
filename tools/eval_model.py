"""Compare any backbone ONNX file against the FlagEmbedding fixtures and time it.

Usage:  uv run tools/eval_model.py [MODEL.onnx ...]   (default: cached fp32 model)
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

from bge_m3_lite import hub
from bge_m3_lite.embedder import BGEM3Embedder

FIXTURES = Path(__file__).resolve().parent.parent / "tests" / "fixtures"


def evaluate(model_path: Path) -> None:
    ref = json.loads((FIXTURES / "embeddings_ref.json").read_text(encoding="utf-8"))
    npz = np.load(FIXTURES / "embeddings_ref.npz")
    t0 = time.perf_counter()
    emb = BGEM3Embedder(quiet=True, model_path=model_path)
    print(
        f"\n== {model_path} ({model_path.stat().st_size / 2**20:.0f} MiB) "
        f"session {time.perf_counter() - t0:.1f}s"
    )
    out = emb.encode(
        ref["sentences"], batch_size=4, return_sparse=True, return_colbert_vecs=True
    )
    dense_cos = (out["dense_vecs"] * npz["dense"]).sum(1)
    print(f"dense cos: min {dense_cos.min():.5f} mean {dense_cos.mean():.5f}")
    same_keys = top5 = spearman = 0.0
    for lw, rlw in zip(out["lexical_weights"], ref["lexical_weights"], strict=True):
        same_keys += set(lw) == set(rlw)
        a = sorted(lw, key=lw.get, reverse=True)[:5]
        b = sorted(rlw, key=rlw.get, reverse=True)[:5]
        top5 += a == b
        keys = sorted(rlw)
        if len(keys) > 2:
            ra = np.argsort(np.argsort([-lw.get(k, 0.0) for k in keys]))
            rb = np.argsort(np.argsort([-rlw[k] for k in keys]))
            spearman += np.corrcoef(ra, rb)[0, 1]
        else:
            spearman += 1.0
    n = len(ref["sentences"])
    print(
        f"sparse: same key set {same_keys:.0f}/{n}, top-5 identical {top5:.0f}/{n}, "
        f"mean Spearman {spearman / n:.4f}"
    )
    cols = [
        (cv * npz[f"colbert_{i}"]).sum(1) for i, cv in enumerate(out["colbert_vecs"])
    ]
    allc = np.concatenate(cols)
    p1, p5 = np.percentile(allc, [1, 5])
    print(
        f"colbert token cos: min {allc.min():.5f} p1 {p1:.5f} p5 {p5:.5f} mean {allc.mean():.5f}"
    )
    # late-interaction ranking agreement: query 0 against all others
    q, rq = out["colbert_vecs"][0], npz["colbert_0"]
    s = [emb.colbert_score(q, c) for c in out["colbert_vecs"][1:]]
    rs = [emb.colbert_score(rq, npz[f"colbert_{i}"]) for i in range(1, n)]
    print(
        f"colbert ranking (q0 vs rest) identical: {np.argsort(s).tolist() == np.argsort(rs).tolist()}"
    )
    for name, text, bs in [
        ("16tok x32", "What is the capital of France? " * 2, 32),
        ("128tok x16", "The quick brown fox jumps over the lazy dog. " * 12, 16),
        (
            "512tok x4",
            "Retrieval augmented generation combines search and language models. " * 40,
            4,
        ),
    ]:
        texts = [text] * bs
        ntok = sum(len(x) for x in emb.tokenize(texts))
        emb.encode(texts[:1])
        t = time.perf_counter()
        emb.encode(texts, batch_size=bs)
        dt = time.perf_counter() - t
        print(f"{name:12s} {ntok / dt:6.0f} tok/s")
    emb.close()


if __name__ == "__main__":
    paths = [Path(p) for p in sys.argv[1:]] or [hub.default_cache_dir() / "model.onnx"]
    for p in paths:
        evaluate(p)
