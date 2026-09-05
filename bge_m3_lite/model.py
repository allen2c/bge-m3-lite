"""onnxruntime wrapper around the BAAI/bge-m3 XLM-RoBERTa backbone."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np


class OnnxBackbone:
    """Runs ``model.onnx`` and returns the last hidden state (token embeddings)."""

    INPUT_IDS = "input_ids"
    ATTENTION_MASK = "attention_mask"
    OUTPUT = "token_embeddings"

    def __init__(
        self,
        model_path: str | Path,
        *,
        num_threads: int | None = None,
        inter_op_threads: int | None = None,
    ) -> None:
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        if num_threads is None:
            env = os.environ.get("BGE_M3_LITE_THREADS")
            num_threads = int(env) if env else 0
        opts.intra_op_num_threads = num_threads  # 0 = let ORT pick physical cores
        if inter_op_threads is not None:
            opts.inter_op_num_threads = inter_op_threads
        self.session = ort.InferenceSession(
            str(model_path), opts, providers=["CPUExecutionProvider"]
        )
        names = {i.name for i in self.session.get_inputs()}
        if not {self.INPUT_IDS, self.ATTENTION_MASK} <= names:
            raise ValueError(f"unexpected model inputs: {sorted(names)}")
        outputs = {o.name for o in self.session.get_outputs()}
        if self.OUTPUT not in outputs:
            raise ValueError(f"model has no '{self.OUTPUT}' output: {sorted(outputs)}")

    def close(self) -> None:
        self.session = None

    def __call__(self, input_ids: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
        """``(batch, seq)`` int64 arrays -> ``(batch, seq, 1024)`` float32."""
        if self.session is None:
            raise RuntimeError("backbone session is closed")
        outputs = self.session.run(
            [self.OUTPUT],
            {
                self.INPUT_IDS: np.ascontiguousarray(input_ids, dtype=np.int64),
                self.ATTENTION_MASK: np.ascontiguousarray(
                    attention_mask, dtype=np.int64
                ),
            },
        )
        return np.asarray(outputs[0], dtype=np.float32)
