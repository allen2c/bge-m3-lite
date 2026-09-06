"""onnxruntime wrapper around the BAAI/bge-m3 XLM-RoBERTa backbone."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import numpy as np


def performance_cores() -> int:
    """Performance cores of an Apple Silicon Mac (``hw.perflevel0.logicalcpu``),
    0 when unknown. The efficiency cores add nothing to throughput but cost
    CPU time (docs/resources.md)."""
    if sys.platform != "darwin":
        return 0
    import ctypes
    import ctypes.util

    try:
        libc = ctypes.CDLL(ctypes.util.find_library("c"))
        value = ctypes.c_int(0)
        size = ctypes.c_size_t(ctypes.sizeof(value))
        ok = libc.sysctlbyname(
            b"hw.perflevel0.logicalcpu",
            ctypes.byref(value),
            ctypes.byref(size),
            None,
            0,
        )
        return value.value if ok == 0 else 0
    except (OSError, AttributeError):
        return 0


def default_threads() -> int:
    """Intra-op threads: ``BGE_M3_LITE_THREADS``, else the performance cores on
    Apple Silicon, else 0 (onnxruntime picks the physical cores)."""
    env = os.environ.get("BGE_M3_LITE_THREADS")
    if env:
        return int(env)
    return performance_cores()


def session_options(
    num_threads: int | None = None,
    *,
    inter_op_threads: int | None = None,
    spin: bool | None = None,
) -> Any:
    """``ort.SessionOptions`` for the backbone.

    ``spin=None`` reads ``BGE_M3_LITE_SPIN`` (default off): onnxruntime's
    worker threads otherwise keep spinning after every run, which costs
    ~60 ms of CPU per idle second and 30–40 % of a short query's CPU time
    for 0–5 % throughput (docs/resources.md).
    """
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.intra_op_num_threads = (
        default_threads() if num_threads is None else num_threads
    )
    if inter_op_threads is not None:
        opts.inter_op_num_threads = inter_op_threads
    if spin is None:
        spin = os.environ.get("BGE_M3_LITE_SPIN") == "1"
    opts.add_session_config_entry(
        "session.intra_op.allow_spinning", "1" if spin else "0"
    )
    return opts


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
        spin: bool | None = None,
    ) -> None:
        import onnxruntime as ort

        opts = session_options(
            num_threads, inter_op_threads=inter_op_threads, spin=spin
        )
        # resolve(): ORT validates that external data files (model.onnx_data)
        # stay inside the model's directory, comparing *real* paths — a
        # symlinked cache directory would otherwise be rejected.
        self.session = ort.InferenceSession(
            str(Path(model_path).resolve()), opts, providers=["CPUExecutionProvider"]
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
