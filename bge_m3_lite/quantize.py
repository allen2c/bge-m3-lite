"""Build the quantised backbone from the cached fused fp32 graph.

Requires the optional ``quant`` extra (``pip install "bge-m3-lite[quant]"``);
``onnx`` is only imported inside :func:`quantize`, never at runtime. Methods:

* ``rowwise`` (default, shipped): SmoothQuant, then per-token uint8
  activations with a zero point and per-column int8 weights (per-axis
  ``QuantizeLinear`` + ``MatMulIntegerToFloat``), and attention in query
  chunks, so every CPU computes the same thing within a bounded buffer.
* ``dynamic``: ORT ``quantize_dynamic`` (per-tensor activations). Faster
  kernels, but its accuracy depends on the platform (see docs/quantization/).
* ``nbits``: weight-only ``MatMulNBits`` (4 or 8 bit, block-wise scales).

SmoothQuant (Xiao et al. 2022) moves the activation outliers of the LayerNorm
outputs into the following weights: for ``Y = X W`` and a per-input-channel
scale ``s``, ``Y = (X / s) (diag(s) W)``; ``s_k = max|X_k|^alpha /
max|W_k|^(1 - alpha)`` with statistics from the calibration texts
(docs/calibration.md). One ``Mul`` per projection, weights rescaled in place,
fp32 outputs unchanged.
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
CALIBRATION_EXTRA = Path(__file__).with_name(
    "calibration_miracl.txt"
)  # docs/calibration.md
ATTENTION_CHUNK = 256  # see docs/memory.md (512 before v0.5.2)
# Where the per-token tail of a layer (output projection, FFN) runs: a second
# Loop over TAIL_ROWS rows of the flattened batch (v0.6.1), inside the
# attention Loop (v0.5.2), or on the whole padded batch (docs/memory.md).
Tail = Literal["rows", "loop", "none"]
TAIL_ROWS = {"fp32": 256, "int8": 256}  # measured optimum on every runner
SMOOTH_TARGETS = (  # node-name patterns of the projections to smooth (fused graph)
    r"^Attention_\d+(/MatMul)?$",  # merged QKV projection (chunked: its MatMul)
    r"attention/output/dense/MatMul$",
    r"/intermediate/dense/MatMul$",  # FFN in
    r"layer\.\d+/output/dense/MatMul$",  # FFN out
)


INLINE_LIMIT = 1024  # tensors up to this many bytes stay inside the graph file
ALIGN = 64  # byte alignment of every tensor in the external data file


@dataclass(frozen=True)
class QuantResult:
    """Sizes and SHA-256 digests of ``model_int8.onnx`` + ``model_int8.onnx_data``."""

    graph_size: int
    graph_sha256: str
    data_size: int
    data_sha256: str


@dataclass(frozen=True)
class QuantConfig:
    method: Literal["rowwise", "dynamic", "nbits"] = "rowwise"
    bits: int = 8
    block_size: int = 128
    accuracy_level: int = 4  # 0 = fp32 compute, 4 = int8 compute (nbits only)
    quantize_embeddings: bool = True  # also quantise the word-embedding Gather
    smooth_alpha: float | None = 0.5  # SmoothQuant strength, None = off
    calibration_max_length: int = 512
    symmetric: bool = False  # rowwise: fixed zero point 128 instead of per-row
    attention_chunk: int = ATTENTION_CHUNK  # query rows per attention pass, 0 = off
    tail: Tail = "rows"  # where the layer tail runs (docs/memory.md)
    tail_rows: int = TAIL_ROWS["int8"]  # rows per window of the tail Loop
    # node-name regexes whose MatMul / Attention stay fp32 (experiments)
    keep_fp32: tuple[str, ...] = ()


def load_calibration_texts(path: str | Path | None = None) -> list[str]:
    """Texts of ``path``, or the bundled set (hand-written + MIRACL sample)."""
    paths = [Path(path)] if path is not None else [CALIBRATION_FILE, CALIBRATION_EXTRA]
    texts: list[str] = []
    for p in paths:
        with open(p, encoding="utf-8") as fh:
            texts += [line.rstrip("\n") for line in fh if line.strip()]
    return texts


def quantize(
    model_in: str | Path,
    model_out: str | Path,
    config: QuantConfig | None = None,
    *,
    tokenizer_path: str | Path | None = None,
    calibration_texts: Sequence[str] | None = None,
) -> QuantResult:
    """Write the quantised model pair and return sizes and digests.

    ``model_in`` is normally the fused graph (``model_fused.onnx``); the raw
    Hub export works too but is not smoothed. ``tokenizer_path`` defaults to
    ``sentencepiece.bpe.model`` next to the model and is only needed for
    SmoothQuant calibration.

    A shipped fused graph already carries the attention ``Loop`` (with the
    layer tail inside it since v0.5.2), which would hide the projections from
    the calibration probe and freeze the chunk: the rowwise recipe therefore
    rebuilds an unchunked fused graph from ``model.onnx`` next to ``model_in``
    (temporary files in the same directory) and quantises that.

    The weights go to ``<model_out>_data`` next to the graph (external data,
    like the fused graph): onnxruntime then maps the file instead of keeping
    a parsed protobuf copy of every tensor, see docs/resources.md.
    """
    config = config or QuantConfig()
    try:
        import onnx
        import onnxruntime.quantization  # noqa: F401  (needed by _quantize)
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
    fused = has_contrib_ops(model.graph)
    temp: list[Path] = []
    if fused and any(n.op_type == "Loop" for n in model.graph.node):  # chunked
        from bge_m3_lite.fuse import fuse

        base = f".fused-{os.getpid()}"
        fuse(model_in.with_name("model.onnx"), attention_chunk=0, basename=base)
        model_in = model_in.with_name(f"{base}.onnx")
        temp = [model_in, model_in.with_name(f"{base}.onnx_data")]
        model = onnx.load(str(model_in))
    try:
        return _quantize(
            model,
            model_in,
            model_out,
            config,
            ops,
            fused,
            tokenizer_path,
            calibration_texts,
        )
    finally:
        for path in temp:
            path.unlink(missing_ok=True)


def _quantize(
    model: Any,
    model_in: Path,
    model_out: Path,
    config: QuantConfig,
    ops: list[str],
    fused: bool,
    tokenizer_path: str | Path | None,
    calibration_texts: Sequence[str] | None,
) -> QuantResult:
    import onnx
    from onnxruntime.quantization import QuantType, quantize_dynamic

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
        _quantize_rowwise(
            model,
            quantize_embeddings=config.quantize_embeddings,
            zero_point=not config.symmetric,
            keep_fp32=config.keep_fp32,
            attention_chunk=config.attention_chunk,
            tail=config.tail,
            tail_rows=config.tail_rows,
        )
    elif config.method == "dynamic":
        tmp = model_out.with_name(model_out.name + ".tmp")
        quantize_dynamic(
            model,
            str(tmp),
            op_types_to_quantize=ops,
            per_channel=True,
            weight_type=QuantType.QInt8,
            use_external_data_format=False,
            extra_options={"DefaultTensorType": onnx.TensorProto.FLOAT},
        )
        model = onnx.load(str(tmp))
        tmp.unlink()
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
        model = getattr(quantizer.model, "model", quantizer.model)
        if not isinstance(model, onnx.ModelProto):
            raise TypeError(f"unexpected quantizer output {type(model)!r}")
    data_path = model_out.with_name(model_out.name + "_data")
    write_external_data(model, data_path)
    onnx.save_model(model, str(model_out))
    return QuantResult(
        model_out.stat().st_size,
        sha256(model_out),
        data_path.stat().st_size,
        sha256(data_path),
    )


def has_contrib_ops(graph: Any) -> bool:
    """True if ``graph`` (or a ``Loop`` body / ``If`` branch in it) uses
    ``com.microsoft`` ops: the fused graph, whose contrib ops all sit inside
    the loops since v0.5.2."""
    for node in graph.node:
        if node.domain == "com.microsoft":
            return True
        if any(
            a.name in ("body", "then_branch", "else_branch") and has_contrib_ops(a.g)
            for a in node.attribute
        ):
            return True
    return False


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_external_data(
    model: Any,
    data_path: Path,
    *,
    shared: dict[bytes, tuple[str, int, int]] | None = None,
) -> tuple[int, int]:
    """Move the initializers of ``model`` out of the graph into ``data_path``.

    Tensors whose SHA-1 is in ``shared`` are referenced at their existing
    ``(location, offset, length)`` instead of being written; tensors up to
    ``INLINE_LIMIT`` bytes stay inline. Deterministic (sorted by name, 64-byte
    aligned). Returns ``(shared, written)`` counts; the caller saves the graph.
    """
    import onnx
    from onnx.external_data_helper import set_external_data

    n_shared = n_written = 0
    with open(data_path, "wb") as out:
        for tensor in sorted(model.graph.initializer, key=lambda t: t.name):
            if tensor.data_location == onnx.TensorProto.EXTERNAL:
                raise RuntimeError(f"{tensor.name}: unexpected external tensor")
            blob = tensor.raw_data
            if not blob:
                continue  # typed fields (small constants), leave as they are
            hit = shared.get(hashlib.sha1(blob).digest()) if shared else None
            if hit is not None:
                set_external_data(tensor, hit[0], offset=hit[1], length=hit[2])
                n_shared += 1
            elif len(blob) > INLINE_LIMIT:
                out.write(b"\0" * ((-out.tell()) % ALIGN))
                set_external_data(
                    tensor, data_path.name, offset=out.tell(), length=len(blob)
                )
                out.write(blob)
                n_written += 1
            else:
                continue
            tensor.data_location = onnx.TensorProto.EXTERNAL
            # ClearField, not ``= b""``: onnx.save_model re-writes every tensor
            # that still *has* raw_data into its external location.
            tensor.ClearField("raw_data")
    return n_shared, n_written


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
        # Named as the projection's MatMul so that quantising from the raw
        # export or from a chunked fused graph writes identical data files.
        stem = node.name if node.op_type == "MatMul" else f"{node.name}/MatMul"
        inv_name = f"{stem}_smooth_scale"
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
# every other CPU). The graph below spells that per-row scheme out so every
# platform computes the same thing, using the two ops ORT can run per row:
#
#   Xr = Reshape(X, [-1, K]);  s = (max Xr - min Xr) / 255 per row;  z = -min / s
#   Q  = QuantizeLinear(Xr, s, z, axis=0)                      (uint8, per row)
#   Y  = s * (MatMulIntegerToFloat(Q, Wq, 1, w_scale, 0, 128) - z * colsum(Wq))
#
# ``MatMulIntegerToFloat`` (u8 x u8 with a per-column ``b_zero_point`` of 128:
# MLAS's AVX2 u8·s8 kernel saturates its int16 intermediates, u8·u8 does not)
# folds the int32 -> float cast and the weight scale; it takes a per-row
# ``a_scale`` but only a scalar ``a_zero_point``, hence the ``colsum``
# correction for the per-row zero point. ``zero_point=False`` is the symmetric
# variant: scale = max|x| / 127, a fixed zero point of 128, no correction (three
# fewer passes over the (M, N) output, half a bit less precision).
# The merged QKV ``Attention`` becomes the same projection + ``MultiHeadAttention``.

_QI8_MAX = 127.0
_ROWWISE_OPSET = 13  # per-axis QuantizeLinear


def _bump_opset(model: Any, version: int) -> None:
    """Raise the default-domain opset of the fused graph (opset 11) to 13.

    Only ``Unsqueeze`` and ``ReduceSum`` in that graph changed signature (their
    ``axes`` attribute became an input).
    """
    from onnx import numpy_helper

    cur = next((o for o in model.opset_import if o.domain == ""), None)
    if cur is None or cur.version >= version:
        return
    cur.version = version
    for node in model.graph.node:
        if node.domain == "" and node.op_type in ("Unsqueeze", "ReduceSum"):
            axes = next((a for a in node.attribute if a.name == "axes"), None)
            if axes is None:
                continue
            name = f"{node.name or node.output[0]}/axes"
            model.graph.initializer.append(
                numpy_helper.from_array(np.array(axes.ints, dtype=np.int64), name)
            )
            node.attribute.remove(axes)
            node.input.append(name)
    if model.ir_version < 7:
        model.ir_version = 7


# -- attention in query chunks -------------------------------------------------
#
# ORT's CPU ``Attention`` / ``MultiHeadAttention`` allocate the full score
# matrix ``batch x heads x S x S`` (4 GiB for one 8192-token text). Softmax is
# per query row, so running the same op on ``chunk`` query rows at a time
# against the full K/V is exact and caps that buffer at
# ``batch x heads x chunk x S``. A ``Loop`` keeps the graph static: one
# iteration for texts up to ``chunk`` tokens (no measurable overhead), ceil(S /
# chunk) iterations otherwise; the outputs are concatenated on the way.


def attention_nodes(
    prefix: str,
    inputs: list[str],
    num_heads: int,
    out: str,
    *,
    chunk: int = ATTENTION_CHUNK,
) -> tuple[list[Any], list[Any]]:
    """``(nodes, initializers)`` computing ``MultiHeadAttention(inputs)``.

    ``inputs`` are ``[q, k, v, bias, key_padding_mask]``; the graph must be
    opset 13 (see :func:`_bump_opset`).
    """
    import onnx
    from onnx import helper, numpy_helper

    mk = helper.make_node
    if chunk <= 0:
        node = mk(
            "MultiHeadAttention",
            inputs,
            [out],
            domain="com.microsoft",
            num_heads=num_heads,
            name=f"{prefix}/mha",
        )
        return [node], []
    q = inputs[0]
    p = prefix
    inits: list[Any] = []

    def c64(name: str, values: list[int]) -> str:
        inits.append(numpy_helper.from_array(np.array(values, dtype=np.int64), name))
        return name

    c_chunk = c64(f"{p}/chunk", [chunk])
    c_chunk1 = c64(f"{p}/chunk-1", [chunk - 1])
    c_0, c_1, c_2, c_3 = (c64(f"{p}/{i}", [i]) for i in range(4))
    inits.append(numpy_helper.from_array(np.array(True), f"{p}/true"))
    nodes = [
        mk("Shape", [q], [f"{p}/qshape"]),
        mk("Slice", [f"{p}/qshape", c_1, c_2], [f"{p}/S"]),
        mk("Add", [f"{p}/S", c_chunk1], [f"{p}/S+"]),
        mk("Div", [f"{p}/S+", c_chunk], [f"{p}/trips"]),
        mk("Squeeze", [f"{p}/trips", c_0], [f"{p}/trip"]),
        mk("Slice", [f"{p}/qshape", c_0, c_1], [f"{p}/B"]),
        mk("Slice", [f"{p}/qshape", c_2, c_3], [f"{p}/H"]),
        mk("Concat", [f"{p}/B", c_0, f"{p}/H"], [f"{p}/acc_shape"], axis=0),
        mk(
            "ConstantOfShape",
            [f"{p}/acc_shape"],
            [f"{p}/acc0"],
            value=helper.make_tensor("v", onnx.TensorProto.FLOAT, [1], [0.0]),
        ),
    ]
    # body(i, cond, acc) -> (cond, acc ++ MHA(q[:, i*chunk : (i+1)*chunk]))
    body = helper.make_graph(
        [
            mk("Unsqueeze", [f"{p}/i", c_0], [f"{p}/i1"]),
            mk("Mul", [f"{p}/i1", c_chunk], [f"{p}/start"]),
            mk("Add", [f"{p}/start", c_chunk], [f"{p}/end"]),  # Slice clamps to S
            mk("Slice", [q, f"{p}/start", f"{p}/end", c_1], [f"{p}/qc"]),
            mk(
                "MultiHeadAttention",
                [f"{p}/qc", *inputs[1:]],
                [f"{p}/oc"],
                domain="com.microsoft",
                num_heads=num_heads,
                name=f"{p}/mha",
            ),
            mk("Concat", [f"{p}/acc", f"{p}/oc"], [f"{p}/acc_next"], axis=1),
            mk("Identity", [f"{p}/cond"], [f"{p}/cond_next"]),
        ],
        f"{p}/body",
        [
            helper.make_tensor_value_info(f"{p}/i", onnx.TensorProto.INT64, []),
            helper.make_tensor_value_info(f"{p}/cond", onnx.TensorProto.BOOL, []),
            helper.make_tensor_value_info(f"{p}/acc", onnx.TensorProto.FLOAT, None),
        ],
        [
            helper.make_tensor_value_info(f"{p}/cond_next", onnx.TensorProto.BOOL, []),
            helper.make_tensor_value_info(
                f"{p}/acc_next", onnx.TensorProto.FLOAT, None
            ),
        ],
    )
    nodes.append(
        mk(
            "Loop",
            [f"{p}/trip", f"{p}/true", f"{p}/acc0"],
            [out],
            body=body,
            name=f"{p}/loop",
        )
    )
    return nodes, inits


def _layer_tail(
    nodes: list[Any], li: int, consts: set[str], graph: Any
) -> tuple[list[int], str, str] | None:
    """``(tail node indices, residual, final)`` of the per-token chain rooted
    at the attention ``Loop`` ``nodes[li]``: output projection ->
    SkipLayerNormalization -> FFN in -> BiasGelu -> FFN out ->
    SkipLayerNormalization, every node reading only constants, the chain and
    one outer tensor (the residual, i.e. the layer input). ``None`` when the
    graph does not have that shape (int8 ``keep_fp32`` ``Attention`` layers,
    an intermediate used elsewhere)."""
    loop = nodes[li]
    out = loop.output[0]
    tail: list[int] = []
    avail, residual, lns = {out}, None, 0
    for j in range(li + 1, len(nodes)):
        node = nodes[j]
        for name in node.input:
            if not name or name in consts or name in avail:
                continue
            if residual in (None, name):
                residual = name
            else:  # not a per-token chain rooted at this loop
                return None
        tail.append(j)
        avail.update(node.output)
        if node.op_type == "SkipLayerNormalization":
            lns += 1
            if lns == 2:
                break
    first_ln = next(
        (nodes[j] for j in tail if nodes[j].op_type == "SkipLayerNormalization"),
        None,
    )
    if lns != 2 or first_ln is None or residual != first_ln.input[1]:
        return None
    final = nodes[tail[-1]].output[0]
    inner = avail - {final}
    used_outside = {o.name for o in graph.output}
    for k, n in enumerate(nodes):
        if k not in tail:
            used_outside.update(n.input)
    if used_outside & inner:
        return None  # an intermediate is used elsewhere (extra graph outputs)
    return tail, str(residual), final


def _copy_renamed(node: Any, rename: dict[str, str]) -> Any:
    copy = type(node)()
    copy.CopyFrom(node)
    for k, name in enumerate(copy.input):
        copy.input[k] = rename.get(name, name)
    for k, name in enumerate(copy.output):
        copy.output[k] = rename.get(name, name)
    return copy


def layer_tail_into_loop(model: Any) -> int:
    """Move the per-token tail of every layer into its attention ``Loop``
    (v0.5.2 layout, ``fuse --tail loop``; see :func:`layer_tail_row_loop`).

    The residual input of the first SkipLayerNormalization (the layer input)
    is sliced like the query rows; the nodes are otherwise moved verbatim.
    Returns the number of loops rewritten.
    """
    from onnx import helper

    graph = model.graph
    nodes = list(graph.node)
    consts = {t.name for t in graph.initializer}
    moved: set[int] = set()
    rewritten = 0
    for li, loop in enumerate(nodes):
        if loop.op_type != "Loop" or not loop.name.endswith("/loop"):
            continue
        found = _layer_tail(nodes, li, consts, graph)
        if found is None:
            continue
        tail, residual, final = found
        p = loop.name[: -len("/loop")]
        out = loop.output[0]
        body = next(a.g for a in loop.attribute if a.name == "body")
        body_nodes = [n for n in body.node if n.op_type not in ("Concat", "Identity")]
        rename = {out: f"{p}/oc", residual: f"{p}/xc", final: f"{p}/layer_out"}
        body_nodes.append(
            helper.make_node(
                "Slice",
                [residual, f"{p}/start", f"{p}/end", f"{p}/1"],
                [f"{p}/xc"],
            )
        )
        body_nodes += [_copy_renamed(nodes[j], rename) for j in tail]
        body_nodes += [
            helper.make_node(
                "Concat", [f"{p}/acc", f"{p}/layer_out"], [f"{p}/acc_next"], axis=1
            ),
            helper.make_node("Identity", [f"{p}/cond"], [f"{p}/cond_next"]),
        ]
        del body.node[:]
        body.node.extend(body_nodes)
        loop.output[0] = final
        moved.update(tail)
        rewritten += 1
    if moved:
        del graph.node[:]
        graph.node.extend(n for k, n in enumerate(nodes) if k not in moved)
        del graph.value_info[:]
    return rewritten


def layer_tail_row_loop(model: Any, rows: int = TAIL_ROWS["fp32"]) -> int:
    """Run the per-token tail of every layer in a second ``Loop`` over
    ``rows`` rows of the flattened ``(1, batch x seq, hidden)`` activations.

    The tail (see :func:`_layer_tail`) is per token, so its intermediates
    (the FFN's ``rows x 4096`` above all) are bounded by ``rows`` whatever the
    batch shape, unlike inside the attention ``Loop`` where a batch of short
    texts still runs the tail on ``batch x seq`` rows (docs/memory.md). The
    body's result is a scan output (one buffer, no accumulator copies), which
    needs every iteration to produce the same shape: the last window is
    shifted back to ``rows`` full rows (the tail recomputes up to ``rows - 1``
    rows), and the outer graph reassembles the ``batch x seq`` rows from the
    windows. Returns the number of layers rewritten.
    """
    import onnx
    from onnx import helper, numpy_helper

    graph = model.graph
    nodes = list(graph.node)
    consts = {t.name for t in graph.initializer}
    hidden = next(  # the LayerNorm gamma of any layer tail
        (
            int(t.dims[0])
            for n in nodes
            if n.op_type == "SkipLayerNormalization"
            for t in graph.initializer
            if t.name == n.input[2]
        ),
        None,
    )
    if hidden is None:
        return 0
    inits: list[Any] = []

    def c64(name: str, values: list[int]) -> str:
        if name not in consts:
            inits.append(
                numpy_helper.from_array(np.array(values, dtype=np.int64), name)
            )
            consts.add(name)
        return name

    mk = helper.make_node
    c_rows = c64("tail/rows", [rows])
    c_rows1 = c64("tail/rows-1", [rows - 1])
    c_flat = c64("tail/flat", [1, -1, hidden])
    c_0, c_1, c_2 = c64("tail/0", [0]), c64("tail/1", [1]), c64("tail/2", [2])
    if "tail/true" not in consts:
        inits.append(numpy_helper.from_array(np.array(True), "tail/true"))
        consts.add("tail/true")
    replaced: dict[int, list[Any]] = {}
    moved: set[int] = set()
    for li, loop in enumerate(nodes):
        if loop.op_type != "Loop" or not loop.name.endswith("/loop"):
            continue
        found = _layer_tail(nodes, li, consts, graph)
        if found is None:
            continue
        tail, residual, final = found
        p = loop.name[: -len("/loop")] + "/tail"
        out = loop.output[0]
        rename = {out: f"{p}/oc", residual: f"{p}/xc", final: f"{p}/layer_out"}
        body = helper.make_graph(
            [
                mk("Unsqueeze", [f"{p}/i", c_0], [f"{p}/i1"]),
                mk("Mul", [f"{p}/i1", c_rows], [f"{p}/start0"]),
                mk("Min", [f"{p}/start0", f"{p}/last"], [f"{p}/start"]),
                mk("Add", [f"{p}/start", f"{p}/W"], [f"{p}/end"]),
                mk("Slice", [f"{p}/out2d", f"{p}/start", f"{p}/end", c_1], [f"{p}/oc"]),
                mk("Slice", [f"{p}/x2d", f"{p}/start", f"{p}/end", c_1], [f"{p}/xc"]),
                *(_copy_renamed(nodes[j], rename) for j in tail),
                mk("Identity", [f"{p}/cond"], [f"{p}/cond_next"]),
            ],
            f"{p}/body",
            [
                helper.make_tensor_value_info(f"{p}/i", onnx.TensorProto.INT64, []),
                helper.make_tensor_value_info(f"{p}/cond", onnx.TensorProto.BOOL, []),
            ],
            [
                helper.make_tensor_value_info(
                    f"{p}/cond_next", onnx.TensorProto.BOOL, []
                ),
                helper.make_tensor_value_info(
                    f"{p}/layer_out", onnx.TensorProto.FLOAT, None
                ),
            ],
        )
        replaced[li] = [
            loop,
            mk("Shape", [out], [f"{p}/shape"]),
            mk("Reshape", [out, c_flat], [f"{p}/out2d"]),
            mk("Reshape", [residual, c_flat], [f"{p}/x2d"]),
            mk("Shape", [f"{p}/out2d"], [f"{p}/shape2d"]),
            mk("Slice", [f"{p}/shape2d", c_1, c_2], [f"{p}/M"]),
            mk("Min", [f"{p}/M", c_rows], [f"{p}/W"]),  # rows per window
            mk("Sub", [f"{p}/M", f"{p}/W"], [f"{p}/last"]),  # start of the last one
            mk("Add", [f"{p}/M", c_rows1], [f"{p}/M+"]),
            mk("Div", [f"{p}/M+", c_rows], [f"{p}/trips"]),
            mk("Squeeze", [f"{p}/trips", c_0], [f"{p}/trip"]),
            mk(
                "Loop",
                [f"{p}/trip", "tail/true"],
                [f"{p}/scan"],
                body=body,
                name=f"{p}/loop",
            ),
            # windows (trips, 1, W, H) -> rows [0, (trips-1) rows) of the full
            # windows, then the last window's rows that are not covered yet
            mk("Reshape", [f"{p}/scan", c_flat], [f"{p}/flat"]),
            mk("Sub", [f"{p}/trips", c_1], [f"{p}/trips-1"]),
            mk("Mul", [f"{p}/trips-1", c_rows], [f"{p}/head"]),
            mk("Slice", [f"{p}/flat", c_0, f"{p}/head", c_1], [f"{p}/head_rows"]),
            mk("Mul", [f"{p}/trips", f"{p}/W"], [f"{p}/flat_rows"]),
            mk("Sub", [f"{p}/M", f"{p}/head"], [f"{p}/rest"]),
            mk("Sub", [f"{p}/flat_rows", f"{p}/rest"], [f"{p}/rest_start"]),
            mk(
                "Slice",
                [f"{p}/flat", f"{p}/rest_start", f"{p}/flat_rows", c_1],
                [f"{p}/rest_rows"],
            ),
            mk("Concat", [f"{p}/head_rows", f"{p}/rest_rows"], [f"{p}/rows"], axis=1),
            mk("Reshape", [f"{p}/rows", f"{p}/shape"], [final]),
        ]
        moved.update(tail)
    if replaced:
        new_nodes: list[Any] = []
        for k, n in enumerate(nodes):
            if k in replaced:
                new_nodes.extend(replaced[k])
            elif k not in moved:
                new_nodes.append(n)
        del graph.node[:]
        graph.node.extend(new_nodes)
        graph.initializer.extend(inits)
        del graph.value_info[:]
    return len(replaced)


def apply_tail(model: Any, tail: Tail, rows: int) -> int:
    """Run the layer-tail pass named by ``tail`` (``"none"`` leaves the graph)."""
    if tail == "rows":
        return layer_tail_row_loop(model, rows)
    if tail == "loop":
        return layer_tail_into_loop(model)
    if tail != "none":
        raise ValueError(f"unknown tail {tail!r}")
    return 0


def _quantize_rowwise(
    model: Any,
    *,
    quantize_embeddings: bool,
    zero_point: bool = True,
    keep_fp32: tuple[str, ...] = (),
    attention_chunk: int = ATTENTION_CHUNK,
    tail: Tail = "none",
    tail_rows: int = TAIL_ROWS["int8"],
) -> None:
    import onnx
    from onnx import helper, numpy_helper

    _bump_opset(model, _ROWWISE_OPSET)
    graph = model.graph
    inits = {t.name: t for t in graph.initializer}
    consts: list[Any] = []
    new_nodes: list[Any] = []

    def keep(node: Any) -> bool:
        # smoothing (if any) already ran and is exact in fp32, so a kept node
        # simply stays a plain fp32 MatMul/Attention.
        return any(re.search(p, node.name) for p in keep_fp32)

    def const(name: str, value: np.ndarray) -> str:
        consts.append(numpy_helper.from_array(value, name))
        return name

    c_qmax = const(
        "rowwise/qmax", np.array(255.0 if zero_point else _QI8_MAX, dtype=np.float32)
    )
    c_eps = const("rowwise/eps", np.array(1e-10, dtype=np.float32))
    c_zero = const("rowwise/zero", np.array(0.0, dtype=np.float32))
    c_one = const("rowwise/one", np.array(1.0, dtype=np.float32))
    c_flat = const("rowwise/flat", np.array([-1], dtype=np.int64))
    c_slice0 = const("rowwise/s0", np.array([0], dtype=np.int64))
    c_slice2 = const("rowwise/s2", np.array([2], dtype=np.int64))
    c_a_zp = const("rowwise/a_zp", np.array(0 if zero_point else 128, dtype=np.uint8))
    c_zp128 = helper.make_tensor("v", onnx.TensorProto.UINT8, [1], [128])

    def rowwise_matmul(prefix: str, x: str, w_name: str, out: str) -> list[Any]:
        """Nodes computing ``out = x @ W`` with per-row uint8 activations."""
        w = numpy_helper.to_array(inits[w_name]).astype(np.float32)  # (K, N)
        k, n = w.shape
        w_scale = np.maximum(np.abs(w).max(axis=0) / _QI8_MAX, 1e-12)
        w_q = np.clip(np.round(w / w_scale), -_QI8_MAX, _QI8_MAX).astype(np.int8)
        inits[w_name].CopyFrom(
            numpy_helper.from_array(
                (w_q.astype(np.int16) + 128).astype(np.uint8), w_name
            )
        )
        ws = const(f"{prefix}/w_scale", w_scale.astype(np.float32))
        b_zp = const(f"{prefix}/b_zp", np.full((n,), 128, dtype=np.uint8))
        shp = const(f"{prefix}/shape2d", np.array([-1, k], dtype=np.int64))
        n_const = const(f"{prefix}/n", np.array([n], dtype=np.int64))
        p = prefix
        mk = helper.make_node
        nodes = [
            mk("Reshape", [x, shp], [f"{p}/x2d"]),
            mk("ReduceMax", [f"{p}/x2d"], [f"{p}/mx0"], axes=[1], keepdims=1),
            mk("ReduceMin", [f"{p}/x2d"], [f"{p}/mn0"], axes=[1], keepdims=1),
        ]
        mm = [f"{p}/xq", w_name, c_one, ws, c_a_zp, b_zp]
        if zero_point:
            colsum = const(
                f"{p}/colsum",
                (w_q.astype(np.int32).sum(axis=0) * w_scale)
                .astype(np.float32)
                .reshape(1, n),
            )
            nodes += [
                mk("Min", [f"{p}/mn0", c_zero], [f"{p}/mn"]),  # range must hold 0
                mk("Max", [f"{p}/mx0", c_zero], [f"{p}/mx"]),
                mk("Sub", [f"{p}/mx", f"{p}/mn"], [f"{p}/range"]),
                mk("Div", [f"{p}/range", c_qmax], [f"{p}/scale0"]),
                mk("Max", [f"{p}/scale0", c_eps], [f"{p}/scale"]),  # (M, 1)
                mk("Div", [f"{p}/mn", f"{p}/scale"], [f"{p}/negzp"]),
                mk("Neg", [f"{p}/negzp"], [f"{p}/zp0"]),
                mk("Round", [f"{p}/zp0"], [f"{p}/zp"]),  # (M, 1) in [0, 255]
                mk("Reshape", [f"{p}/zp", c_flat], [f"{p}/zp1d"]),
                mk("Cast", [f"{p}/zp1d"], [f"{p}/zpq"], to=onnx.TensorProto.UINT8),
                mk("Reshape", [f"{p}/scale", c_flat], [f"{p}/scale1d"]),
                mk(
                    "QuantizeLinear",
                    [f"{p}/x2d", f"{p}/scale1d", f"{p}/zpq"],
                    [f"{p}/xq"],
                    axis=0,
                ),
                mk("MatMulIntegerToFloat", mm, [f"{p}/y0"], domain="com.microsoft"),
                mk("Mul", [f"{p}/zp", colsum], [f"{p}/corr"]),
                mk("Sub", [f"{p}/y0", f"{p}/corr"], [f"{p}/yc"]),
                mk("Mul", [f"{p}/yc", f"{p}/scale"], [f"{p}/y2d"]),
            ]
        else:
            nodes += [
                mk("Neg", [f"{p}/mn0"], [f"{p}/nmn"]),
                mk("Max", [f"{p}/mx0", f"{p}/nmn"], [f"{p}/amax"]),
                mk("Div", [f"{p}/amax", c_qmax], [f"{p}/scale0"]),
                mk("Max", [f"{p}/scale0", c_eps], [f"{p}/scale"]),  # (M, 1)
                mk("Reshape", [f"{p}/scale", c_flat], [f"{p}/scale1d"]),
                mk("Shape", [f"{p}/scale1d"], [f"{p}/rows"]),
                mk("ConstantOfShape", [f"{p}/rows"], [f"{p}/zpq"], value=c_zp128),
                mk(
                    "QuantizeLinear",
                    [f"{p}/x2d", f"{p}/scale1d", f"{p}/zpq"],
                    [f"{p}/xq"],
                    axis=0,
                ),
                mk("MatMulIntegerToFloat", mm, [f"{p}/y0"], domain="com.microsoft"),
                mk("Mul", [f"{p}/y0", f"{p}/scale"], [f"{p}/y2d"]),
            ]
        nodes += [
            mk("Shape", [x], [f"{p}/xshape"]),
            mk("Slice", [f"{p}/xshape", c_slice0, c_slice2], [f"{p}/bs"]),
            mk("Concat", [f"{p}/bs", n_const], [f"{p}/yshape"], axis=0),
            mk("Reshape", [f"{p}/y2d", f"{p}/yshape"], [out]),
        ]
        return nodes

    for node in graph.node:
        if keep(node) and node.op_type in ("MatMul", "Attention"):
            new_nodes.append(node)
        elif node.op_type == "MatMul" and node.input[1] in inits:
            prefix = f"rowwise/{node.name or node.output[0]}"
            new_nodes.extend(
                rowwise_matmul(prefix, node.input[0], node.input[1], node.output[0])
            )
        elif node.op_type == "Attention" and node.domain == "com.microsoft":
            x, w_name, bias, mask = node.input[:4]
            heads = next(a.i for a in node.attribute if a.name == "num_heads")
            prefix = f"rowwise/{node.name}"  # the Loop's prefix
            qkv = rowwise_matmul(f"{prefix}/MatMul", x, w_name, f"{prefix}/qkv")
            new_nodes.extend(qkv)
            new_nodes.append(
                helper.make_node(
                    "Split",
                    [f"{prefix}/qkv"],
                    [f"{prefix}/q", f"{prefix}/k", f"{prefix}/v"],
                    axis=2,
                )
            )
            nodes, extra = attention_nodes(
                prefix,
                [f"{prefix}/q", f"{prefix}/k", f"{prefix}/v", bias, mask],
                heads,
                node.output[0],
                chunk=attention_chunk,
            )
            new_nodes.extend(nodes)
            consts.extend(extra)
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
    if attention_chunk > 0:
        apply_tail(model, tail, tail_rows)
