"""Build the quantised backbone from the cached fp32 model.

Requires the optional ``quant`` extra (``pip install "bge-m3-lite[quant]"``):
``onnx`` and ``onnx-ir`` are only imported inside :func:`quantize`, never at
runtime. Two methods are supported:

* ``dynamic``: int8 weights + uint8 activations (``quantize_dynamic``,
  per-channel). Smallest compute cost on x86 with VNNI.
* ``nbits``: weight-only ``MatMulNBits`` (4 or 8 bit, block-wise scales,
  ``accuracy_level=4`` computes in int8 on ARM/x86). Better accuracy.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class QuantConfig:
    method: Literal["dynamic", "nbits"] = "dynamic"
    bits: int = 8
    block_size: int = 128
    accuracy_level: int = 4  # 0 = fp32 compute, 4 = int8 compute (nbits only)
    quantize_embeddings: bool = True  # also quantise the word-embedding Gather


def quantize(
    model_in: str | Path,
    model_out: str | Path,
    config: QuantConfig | None = None,
) -> tuple[int, str]:
    """Write the quantised model and return ``(size_bytes, sha256)``."""
    config = config or QuantConfig()
    try:
        import onnx
        from onnxruntime.quantization import QuantType, quantize_dynamic
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            'quantisation needs the "quant" extra: pip install "bge-m3-lite[quant]"'
        ) from exc

    model_in, model_out = Path(model_in), Path(model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    ops = ("MatMul", "Gather") if config.quantize_embeddings else ("MatMul",)
    if config.method == "dynamic":
        quantize_dynamic(
            str(model_in),
            str(model_out),
            op_types_to_quantize=list(ops),
            per_channel=True,
            weight_type=QuantType.QInt8,
            use_external_data_format=False,
        )
    else:
        from onnxruntime.quantization.matmul_nbits_quantizer import (
            MatMulNBitsQuantizer,
        )

        model = onnx.load(str(model_in), load_external_data=True)
        quantizer = MatMulNBitsQuantizer(
            model,
            bits=config.bits,
            block_size=config.block_size,
            is_symmetric=True,
            accuracy_level=config.accuracy_level or None,
            op_types_to_quantize=ops,
            quant_axes=(("MatMul", 0), ("Gather", 1)),
        )
        quantizer.process()
        # ``quantizer.model`` is an ONNXModel wrapper at runtime (typed as ModelProto).
        result = getattr(quantizer.model, "model", quantizer.model)
        if not isinstance(result, onnx.ModelProto):
            raise TypeError(f"unexpected quantizer output {type(result)!r}")
        onnx.save_model(result, str(model_out))
    digest = hashlib.sha256()
    with open(model_out, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return model_out.stat().st_size, digest.hexdigest()
