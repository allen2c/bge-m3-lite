"""Regenerate tests/fixtures/embeddings_ref.* with FlagEmbedding (PyTorch) as ground truth.

Run with the ``ref`` dependency group:  uv run --group ref tools/make_embedding_fixtures.py
"""

# pyright: reportMissingImports=false

import json
import sys
from pathlib import Path

import numpy as np
from FlagEmbedding import BGEM3FlagModel

SENTENCES = [
    "What is BGE M3?",
    "BGE M3 is an embedding model supporting dense retrieval, lexical matching and multi-vector interaction.",
    "BGE-M3 是一個支援稠密檢索、詞彙匹配與多向量互動的嵌入模型。",
    "機器學習與深度學習有什麼差別？",
    "日本語のテキストをトークン化する。",
    "한국어 문장을 토큰화합니다.",
    "Русский текст для проверки токенизатора.",
    "The quick brown fox jumps over the lazy dog. " * 3,
    "emoji 😀🚀 café ﬁｒｅ ① mixed 中文 English",
    "",
    " ".join(
        f"sentence number {i} talks about retrieval augmented generation and vector databases."
        for i in range(40)
    ),
]

model = BGEM3FlagModel(
    "BAAI/bge-m3", use_fp16=False, devices="cpu", normalize_embeddings=True
)
out = model.encode(
    SENTENCES,
    batch_size=4,
    max_length=8192,
    return_dense=True,
    return_sparse=True,
    return_colbert_vecs=True,
)
fixtures = Path(__file__).resolve().parent.parent / "tests" / "fixtures"
dense = np.asarray(out["dense_vecs"], dtype=np.float32)
arrays: dict[str, np.ndarray] = {"dense": dense}
for i, v in enumerate(out["colbert_vecs"]):
    arrays[f"colbert_{i}"] = np.asarray(v, dtype=np.float32)
np.savez_compressed(fixtures / "embeddings_ref.npz", **arrays)  # pyright: ignore[reportArgumentType]
lexical = [{str(k): float(v) for k, v in lw.items()} for lw in out["lexical_weights"]]
(fixtures / "embeddings_ref.json").write_text(
    json.dumps(
        {"sentences": SENTENCES, "lexical_weights": lexical},
        ensure_ascii=False,
        indent=0,
    )
    + "\n",
    encoding="utf-8",
)
print(
    "dense",
    dense.shape,
    "colbert shapes",
    [v.shape for k, v in arrays.items() if k != "dense"],
    file=sys.stderr,
)
