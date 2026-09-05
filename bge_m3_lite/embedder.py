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
    ) -> None:
        """``precision="int8"`` uses the quantised backbone (see docs/quantization.md);
        ``model_path`` points at any backbone ONNX file and skips the download."""
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
        batch_size: int = 12,
        max_length: int | None = None,
        return_dense: bool = True,
        return_sparse: bool = False,
        return_colbert_vecs: bool = False,
    ) -> dict[str, Any]:
        """Return ``dense_vecs`` / ``lexical_weights`` / ``colbert_vecs``.

        Keys that were not requested are ``None``. For a single string input the
        per-text containers are unwrapped, mirroring FlagEmbedding.

        Memory: the hidden state of one batch is ``batch_size * seq_len * 4 KiB``
        (a batch of 12 texts at 8192 tokens is ~400 MiB), so lower ``batch_size``
        when encoding long documents.
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
        order = sorted(range(n), key=lambda i: -len(token_ids[i]))
        for start in range(0, n, batch_size):
            idx = order[start : start + batch_size]
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
