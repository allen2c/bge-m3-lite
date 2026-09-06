"""Build the quantised backbone from the cached fused fp32 graph.

Requires the optional ``quant`` extra (``pip install "bge-m3-lite[quant]"``);
``onnx`` is only imported inside :func:`quantize`, never at runtime. Methods:

* ``rowwise`` (default, shipped): SmoothQuant, then per-token uint8
  activations with a zero point and per-column int8 weights, written out with
  standard ops (``MatMulInteger``) so every CPU computes the same thing.
* ``dynamic``: ORT ``quantize_dynamic`` (per-tensor activations). Faster
  kernels, but its accuracy depends on the platform (see docs/quantization.md).
* ``nbits``: weight-only ``MatMulNBits`` (4 or 8 bit, block-wise scales).

SmoothQuant (Xiao et al. 2022) moves the activation outliers of the LayerNorm
outputs into the following weights: for ``Y = X W`` and a per-input-channel
scale ``s``, ``Y = (X / s) (diag(s) W)``; ``s_k = max|X_k|^alpha /
max|W_k|^(1 - alpha)`` with statistics from ``calibration.txt``. One ``Mul``
per projection, weights rescaled in place, fp32 outputs unchanged.
"""

from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

CALIBRATION_FILE = Path(__file__).with_name("calibration.txt")
SMOOTH_TARGETS = (  # node-name patterns of the projections to smooth (fused graph)
    r"^Attention_\d+$",  # merged QKV projection
    r"attention/output/dense/MatMul$",
    r"/intermediate/dense/MatMul$",  # FFN in
    r"layer\.\d+/output/dense/MatMul$",  # FFN out
)


@dataclass(frozen=True)
class QuantConfig:
    method: Literal["rowwise", "dynamic", "nbits"] = "rowwise"
    bits: int = 8
    block_size: int = 128
    accuracy_level: int = 4  # 0 = fp32 compute, 4 = int8 compute (nbits only)
    quantize_embeddings: bool = True  # also quantise the word-embedding Gather
    smooth_alpha: float | None = 0.5  # SmoothQuant strength, None = off
    calibration_max_length: int = 512


def load_calibration_texts(path: str | Path = CALIBRATION_FILE) -> list[str]:
    with open(path, encoding="utf-8") as fh:
        return [line.rstrip("\n") for line in fh if line.strip()]


def quantize(
    model_in: str | Path,
    model_out: str | Path,
    config: QuantConfig | None = None,
    *,
    tokenizer_path: str | Path | None = None,
    calibration_texts: Sequence[str] | None = None,
) -> tuple[int, str]:
    """Write the quantised model and return ``(size_bytes, sha256)``.

    ``model_in`` is normally the fused graph (``model_fused.onnx``); the raw
    Hub export works too but is not smoothed. ``tokenizer_path`` defaults to
    ``sentencepiece.bpe.model`` next to the model and is only needed for
    SmoothQuant calibration.
    """
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
    ops = ["MatMul", "Gather"] if config.quantize_embeddings else ["MatMul"]
    # The fused graph (docs/fusion.md) carries com.microsoft contrib ops: the
    # quantizer needs the domain declared and a default tensor type, and it
    # turns ``Attention`` into ``QAttention``.
    model = onnx.load(str(model_in))
    fused = any(n.domain == "com.microsoft" for n in model.graph.node)
    if fused:
        ops.append("Attention")
        if not any(o.domain == "com.microsoft" for o in model.opset_import):
            model.opset_import.add(domain="com.microsoft", version=1)
    if config.method == "rowwise" and not fused:
        raise ValueError(
            "rowwise quantisation needs the fused graph (bge-m3-lite fuse)"
        )
    if config.smooth_alpha is not None and fused and config.method != "nbits":
        texts = (
            list(calibration_texts)
            if calibration_texts is not None
            else load_calibration_texts()
        )
        tok = tokenizer_path or model_in.with_name("sentencepiece.bpe.model")
        _smooth(
            model,
            _activation_stats(model_in, model, texts, tok, config),
            config.smooth_alpha,
        )
    if config.method == "rowwise":
        _quantize_rowwise(model, quantize_embeddings=config.quantize_embeddings)
        onnx.save_model(model, str(model_out))
    elif config.method == "dynamic":
        quantize_dynamic(
            model,
            str(model_out),
            op_types_to_quantize=ops,
            per_channel=True,
            weight_type=QuantType.QInt8,
            use_external_data_format=False,
            extra_options={"DefaultTensorType": onnx.TensorProto.FLOAT},
        )
    else:
        from onnxruntime.quantization.matmul_nbits_quantizer import (
            MatMulNBitsQuantizer,
        )

        quantizer = MatMulNBitsQuantizer(
            model,
            bits=config.bits,
            block_size=config.block_size,
            is_symmetric=True,
            accuracy_level=config.accuracy_level or None,
            op_types_to_quantize=tuple(o for o in ops if o != "Attention"),
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


def smooth_targets(model: Any) -> list[Any]:
    """Nodes of ``model`` whose first input is smoothed (fused graph only)."""
    return [
        n for n in model.graph.node if any(re.search(p, n.name) for p in SMOOTH_TARGETS)
    ]


def _activation_stats(
    model_path: Path,
    model: Any,
    texts: Sequence[str],
    tokenizer_path: str | Path,
    config: QuantConfig,
) -> dict[str, np.ndarray]:
    """Per-channel ``max|x|`` of every smoothed input over the calibration texts."""
    import onnx
    import onnxruntime as ort

    from bge_m3_lite.tokenizer import XLMRobertaTokenizer

    names = sorted({n.input[0] for n in smooth_targets(model)})
    # A graph-only copy with the activations as extra outputs; it must live next
    # to the model so the external weight files resolve.
    probe = onnx.load(str(model_path), load_external_data=False)
    if not any(o.domain == "com.microsoft" for o in probe.opset_import):
        probe.opset_import.add(domain="com.microsoft", version=1)
    for name in names:
        probe.graph.output.add().CopyFrom(
            onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, None)
        )
    probe_path = model_path.with_name(f".probe-{os.getpid()}.onnx")
    onnx.save_model(probe, str(probe_path))
    try:
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        session = ort.InferenceSession(
            str(probe_path.resolve()), opts, providers=["CPUExecutionProvider"]
        )
    finally:
        probe_path.unlink()
    tokenizer = XLMRobertaTokenizer.from_file(tokenizer_path)
    ids = sorted(
        (tokenizer.encode(t, max_length=config.calibration_max_length) for t in texts),
        key=len,
        reverse=True,
    )
    stats: dict[str, np.ndarray] = {}
    for start in range(0, len(ids), 8):
        batch = ids[start : start + 8]
        width = max(len(s) for s in batch)
        input_ids = np.full((len(batch), width), tokenizer.PAD_ID, dtype=np.int64)
        mask = np.zeros((len(batch), width), dtype=np.int64)
        for row, seq in enumerate(batch):
            input_ids[row, : len(seq)] = seq
            mask[row, : len(seq)] = 1
        outputs = session.run(names, {"input_ids": input_ids, "attention_mask": mask})
        keep = mask.reshape(-1) == 1
        for name, out in zip(names, outputs, strict=True):
            act = np.asarray(out, dtype=np.float32)
            cur = np.abs(act.reshape(-1, act.shape[-1])[keep]).max(axis=0)
            stats[name] = np.maximum(stats[name], cur) if name in stats else cur
    return stats


def _smooth(model: Any, stats: dict[str, np.ndarray], alpha: float) -> None:
    """Insert ``X / s`` and rescale ``W`` to ``diag(s) W`` for every target."""
    import onnx
    from onnx import numpy_helper

    inits = {t.name: t for t in model.graph.initializer}
    nodes = list(model.graph.node)
    for node in smooth_targets(model):
        weight = inits[node.input[1]]
        w = numpy_helper.to_array(weight)  # (K, N) MatMul, (K, 3N) Attention
        x_max = np.maximum(stats[node.input[0]].astype(np.float64), 1e-5)
        w_max = np.maximum(np.abs(w).max(axis=1).astype(np.float64), 1e-5)
        s = np.clip(x_max**alpha / w_max ** (1.0 - alpha), 1e-2, 1e2)
        weight.CopyFrom(
            numpy_helper.from_array((w * s[:, None]).astype(np.float32), weight.name)
        )
        inv_name = f"{node.name}_smooth_scale"
        model.graph.initializer.append(
            numpy_helper.from_array((1.0 / s).astype(np.float32), inv_name)
        )
        smoothed = f"{node.input[0]}_smoothed_{node.name}"
        mul = onnx.helper.make_node(
            "Mul", [node.input[0], inv_name], [smoothed], name=f"{node.name}_smooth"
        )
        model.graph.node.insert(nodes.index(node), mul)
        nodes.insert(nodes.index(node), mul)
        node.input[0] = smoothed


# -- row-wise dynamic int8 (platform independent) ------------------------------
#
# ORT's ``quantize_dynamic`` quantises activations per *tensor*. Its
# ``DynamicQuantizeMatMul`` kernel on Apple Silicon (KleidiAI) silently switches
# to per-row scales and is far more accurate (dense cosine 0.998 vs 0.983 on
# every other CPU). The graph below spells that per-row scheme out with
# standard ops so every platform computes the same thing:
#
#   Xr = Reshape(X, [-1, K]);  s = (max Xr - min Xr) / 255 per row;  z = -min / s
#   Q  = uint8(round(Xr / s) + z)
#   Y  = s * (MatMulInteger(Q, Wq) - z * colsum(Wq)) * w_scale
#
# ``MatMulInteger`` has no per-row zero point, hence the ``colsum`` correction.
# Weights are per-column symmetric int8 stored as uint8 with zero point 128
# (removed by ``MatMulInteger`` itself): the u8·s8 AVX2 kernel saturates its
# int16 intermediates, u8·u8 does not, and the integer results are identical.
# Signed int8 activations (``zero_point=False``) lose a bit and are 5x slower
# on x86 (no s8·s8 kernel); both switches stay only for the tests.
# The merged QKV ``Attention`` becomes the same projection + ``MultiHeadAttention``.

_QI8_MAX = 127.0


def _quantize_rowwise(
    model: Any,
    *,
    quantize_embeddings: bool,
    zero_point: bool = True,
    weight_uint8: bool = True,
) -> None:
    import onnx
    from onnx import helper, numpy_helper

    graph = model.graph
    inits = {t.name: t for t in graph.initializer}
    consts: list[Any] = []
    new_nodes: list[Any] = []

    def const(name: str, value: np.ndarray) -> str:
        consts.append(numpy_helper.from_array(value, name))
        return name

    c_qmax = const(
        "rowwise/qmax", np.array(255.0 if zero_point else _QI8_MAX, dtype=np.float32)
    )
    c_eps = const("rowwise/eps", np.array(1e-10, dtype=np.float32))
    c_zero = const("rowwise/zero", np.array(0.0, dtype=np.float32))
    c_slice0 = const("rowwise/s0", np.array([0], dtype=np.int64))
    c_slice2 = const("rowwise/s2", np.array([2], dtype=np.int64))

    def rowwise_matmul(prefix: str, x: str, w_name: str, out: str) -> list[Any]:
        """Nodes computing ``out = x @ W`` with per-row int8 activations."""
        w = numpy_helper.to_array(inits[w_name]).astype(np.float32)  # (K, N)
        k, n = w.shape
        w_scale = np.maximum(np.abs(w).max(axis=0) / _QI8_MAX, 1e-12)
        w_q = np.clip(np.round(w / w_scale), -_QI8_MAX, _QI8_MAX).astype(np.int8)
        mm_inputs = [f"{prefix}/xq", w_name]
        if weight_uint8:
            inits[w_name].CopyFrom(
                numpy_helper.from_array(
                    (w_q.astype(np.int16) + 128).astype(np.uint8), w_name
                )
            )
            mm_inputs += [
                const(
                    f"{prefix}/a_zp",
                    np.array(0, dtype=np.uint8 if zero_point else np.int8),
                ),
                const(f"{prefix}/b_zp", np.full((n,), 128, dtype=np.uint8)),
            ]
        else:
            inits[w_name].CopyFrom(numpy_helper.from_array(w_q, w_name))
        ws = const(f"{prefix}/w_scale", w_scale.astype(np.float32).reshape(1, n))
        shp = const(f"{prefix}/shape2d", np.array([-1, k], dtype=np.int64))
        n_const = const(f"{prefix}/n", np.array([n], dtype=np.int64))
        p = prefix
        mk = helper.make_node
        nodes = [mk("Reshape", [x, shp], [f"{p}/x2d"])]
        if zero_point:
            colsum = const(
                f"{p}/colsum",
                w_q.astype(np.int32)
                .sum(axis=0, dtype=np.int32)
                .astype(np.float32)
                .reshape(1, n),
            )
            nodes += [
                mk("ReduceMax", [f"{p}/x2d"], [f"{p}/mx"], axes=[1], keepdims=1),
                mk("ReduceMin", [f"{p}/x2d"], [f"{p}/mn0"], axes=[1], keepdims=1),
                mk("Min", [f"{p}/mn0", c_zero], [f"{p}/mn"]),  # range must hold 0
                mk("Max", [f"{p}/mx", c_zero], [f"{p}/mx0"]),
                mk("Sub", [f"{p}/mx0", f"{p}/mn"], [f"{p}/range"]),
                mk("Div", [f"{p}/range", c_qmax], [f"{p}/scale0"]),
                mk("Max", [f"{p}/scale0", c_eps], [f"{p}/scale"]),
                mk("Div", [f"{p}/mn", f"{p}/scale"], [f"{p}/negzp"]),
                mk("Neg", [f"{p}/negzp"], [f"{p}/zp0"]),
                mk("Round", [f"{p}/zp0"], [f"{p}/zp"]),  # (M, 1) in [0, 255]
                mk("Div", [f"{p}/x2d", f"{p}/scale"], [f"{p}/xs"]),
                mk("Round", [f"{p}/xs"], [f"{p}/xr"]),
                mk("Add", [f"{p}/xr", f"{p}/zp"], [f"{p}/xz"]),
                mk("Clip", [f"{p}/xz", c_zero, c_qmax], [f"{p}/xc"]),
                mk("Cast", [f"{p}/xc"], [f"{p}/xq"], to=onnx.TensorProto.UINT8),
                mk("MatMulInteger", mm_inputs, [f"{p}/y32"]),
                mk("Cast", [f"{p}/y32"], [f"{p}/yf"], to=onnx.TensorProto.FLOAT),
                mk("Mul", [f"{p}/zp", colsum], [f"{p}/corr"]),
                mk("Sub", [f"{p}/yf", f"{p}/corr"], [f"{p}/yc"]),
                mk("Mul", [f"{p}/yc", f"{p}/scale"], [f"{p}/ys"]),
            ]
        else:
            nodes += [
                mk("Abs", [f"{p}/x2d"], [f"{p}/abs"]),
                mk("ReduceMax", [f"{p}/abs"], [f"{p}/amax"], axes=[1], keepdims=1),
                mk("Div", [f"{p}/amax", c_qmax], [f"{p}/scale0"]),
                mk("Max", [f"{p}/scale0", c_eps], [f"{p}/scale"]),
                mk("Div", [f"{p}/x2d", f"{p}/scale"], [f"{p}/xs"]),
                mk("Round", [f"{p}/xs"], [f"{p}/xr"]),  # |xr| <= 127 by construction
                mk("Cast", [f"{p}/xr"], [f"{p}/xq"], to=onnx.TensorProto.INT8),
                mk("MatMulInteger", mm_inputs, [f"{p}/y32"]),
                mk("Cast", [f"{p}/y32"], [f"{p}/yf"], to=onnx.TensorProto.FLOAT),
                mk("Mul", [f"{p}/yf", f"{p}/scale"], [f"{p}/ys"]),
            ]
        nodes += [
            mk("Mul", [f"{p}/ys", ws], [f"{p}/y2d"]),
            mk("Shape", [x], [f"{p}/xshape"]),
            mk("Slice", [f"{p}/xshape", c_slice0, c_slice2], [f"{p}/bs"]),
            mk("Concat", [f"{p}/bs", n_const], [f"{p}/yshape"], axis=0),
            mk("Reshape", [f"{p}/y2d", f"{p}/yshape"], [out]),
        ]
        return nodes

    for node in graph.node:
        if node.op_type == "MatMul" and node.input[1] in inits:
            prefix = f"rowwise/{node.name or node.output[0]}"
            new_nodes.extend(
                rowwise_matmul(prefix, node.input[0], node.input[1], node.output[0])
            )
        elif node.op_type == "Attention" and node.domain == "com.microsoft":
            x, w_name, bias, mask = node.input[:4]
            heads = next(a.i for a in node.attribute if a.name == "num_heads")
            prefix = f"rowwise/{node.name}"
            new_nodes.extend(rowwise_matmul(prefix, x, w_name, f"{prefix}/qkv"))
            new_nodes.append(
                helper.make_node(
                    "Split",
                    [f"{prefix}/qkv"],
                    [f"{prefix}/q", f"{prefix}/k", f"{prefix}/v"],
                    axis=2,
                )
            )
            new_nodes.append(
                helper.make_node(
                    "MultiHeadAttention",
                    [f"{prefix}/q", f"{prefix}/k", f"{prefix}/v", bias, mask],
                    [node.output[0]],
                    domain="com.microsoft",
                    num_heads=heads,
                )
            )
        elif (
            node.op_type == "Gather"
            and quantize_embeddings
            and node.input[0] in inits
            and len(inits[node.input[0]].dims) == 2
            and inits[node.input[0]].dims[0] > 10000
        ):
            # word embeddings: int8 rows with one scale per row
            w_name = node.input[0]
            w = numpy_helper.to_array(inits[w_name]).astype(np.float32)
            scale = np.maximum(np.abs(w).max(axis=1, keepdims=True) / _QI8_MAX, 1e-12)
            w_q = np.clip(np.round(w / scale), -_QI8_MAX, _QI8_MAX).astype(np.int8)
            inits[w_name].CopyFrom(numpy_helper.from_array(w_q, w_name))
            sc = const(f"{w_name}/row_scale", scale.astype(np.float32))
            p = f"rowwise/{node.name or node.output[0]}"
            new_nodes.extend(
                [
                    helper.make_node("Gather", [w_name, node.input[1]], [f"{p}/eq"]),
                    helper.make_node("Gather", [sc, node.input[1]], [f"{p}/es"]),
                    helper.make_node(
                        "Cast", [f"{p}/eq"], [f"{p}/ef"], to=onnx.TensorProto.FLOAT
                    ),
                    helper.make_node("Mul", [f"{p}/ef", f"{p}/es"], [node.output[0]]),
                ]
            )
        else:
            new_nodes.append(node)
    del graph.node[:]
    graph.node.extend(new_nodes)
    graph.initializer.extend(consts)
    # value_info entries of the raw export may now carry stale types
    del graph.value_info[:]
