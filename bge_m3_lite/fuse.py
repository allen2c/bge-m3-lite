"""Build the fused fp32 backbone from the cached Hub export.

Requires the optional ``quant`` extra (``onnx``); nothing here is imported at
runtime. ``onnxruntime.transformers`` rewrites the opset-11 export into 24
``Attention``, 48 ``SkipLayerNormalization`` and 24 ``BiasGelu`` contrib ops.
The outputs are unchanged (see docs/verification.md) and every weight except
the merged QKV projections is byte-identical to the Hub weights, so the result
is two small files next to the original ``model.onnx_data``:

* ``model_fused.onnx``       – the graph (~160 KB); shared tensors point into
                               ``model.onnx_data`` by offset
* ``model_fused.onnx_data``  – the 48 merged QKV weights and biases (288 MiB)

With ``attention_chunk > 0`` (default) every ``Attention`` is then rewritten as
``MatMul`` + ``Split`` + ``MultiHeadAttention`` over query chunks in a ``Loop``
(:func:`bge_m3_lite.quantize.attention_nodes`), which bounds the attention
score buffer for long inputs (docs/memory.md); the outputs are unchanged.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bge_m3_lite.quantize import ATTENTION_CHUNK, _bump_opset, attention_nodes

NUM_HEADS = 16
HIDDEN_SIZE = 1024
SHARED_DATA = "model.onnx_data"
INLINE_LIMIT = 1024  # tensors up to this size stay inside the graph file
ALIGN = 64


@dataclass(frozen=True)
class FuseResult:
    graph_size: int
    graph_sha256: str
    data_size: int
    data_sha256: str
    shared: int  # tensors served from model.onnx_data
    fused: int  # tensors written to model_fused.onnx_data


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _chunk_attention(model: Any, chunk: int) -> None:
    """``Attention`` -> ``MatMul`` + ``Split`` + chunked ``MultiHeadAttention``."""
    from onnx import helper

    _bump_opset(model, 13)
    new_nodes: list[Any] = []
    inits: list[Any] = []
    for node in model.graph.node:
        if node.op_type != "Attention" or node.domain != "com.microsoft":
            new_nodes.append(node)
            continue
        x, weight, bias, mask = node.input[:4]
        heads = next(a.i for a in node.attribute if a.name == "num_heads")
        p = node.name
        new_nodes += [
            helper.make_node("MatMul", [x, weight], [f"{p}/qkv"], name=f"{p}/MatMul"),
            helper.make_node(
                "Split", [f"{p}/qkv"], [f"{p}/q", f"{p}/k", f"{p}/v"], axis=2
            ),
        ]
        nodes, extra = attention_nodes(
            p,
            [f"{p}/q", f"{p}/k", f"{p}/v", bias, mask],
            heads,
            node.output[0],
            chunk=chunk,
        )
        new_nodes += nodes
        inits += extra
    del model.graph.node[:]
    model.graph.node.extend(new_nodes)
    model.graph.initializer.extend(inits)


def fuse(
    model_in: str | Path,
    out_dir: str | Path | None = None,
    *,
    attention_chunk: int = ATTENTION_CHUNK,
) -> FuseResult:
    """Write ``model_fused.onnx`` + ``model_fused.onnx_data`` next to ``model_in``
    (or into ``out_dir``) and return sizes and digests. Deterministic."""
    try:
        import onnx
        from onnx.external_data_helper import ExternalDataInfo, set_external_data
        from onnxruntime.transformers import optimizer
        from onnxruntime.transformers.fusion_options import FusionOptions
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            'fusion needs the "quant" extra: pip install "bge-m3-lite[quant]"'
        ) from exc

    model_in = Path(model_in)
    out_dir = model_in.parent if out_dir is None else Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    graph_path = out_dir / "model_fused.onnx"
    data_path = out_dir / "model_fused.onnx_data"

    # Content -> (offset, length) of every tensor stored in the shared data file.
    original = onnx.load(str(model_in), load_external_data=False)
    shared: dict[bytes, tuple[int, int]] = {}
    with open(model_in.parent / SHARED_DATA, "rb") as fh:
        for tensor in original.graph.initializer:
            if tensor.data_location != onnx.TensorProto.EXTERNAL:
                continue
            info = ExternalDataInfo(tensor)
            if info.location != SHARED_DATA:
                continue
            offset, length = int(info.offset or 0), int(info.length or 0)
            fh.seek(offset)
            blob = fh.read(length)
            shared[hashlib.sha1(blob).digest()] = (offset, length)

    options = FusionOptions("bert")
    fused = optimizer.optimize_model(
        str(model_in),
        model_type="bert",
        num_heads=NUM_HEADS,
        hidden_size=HIDDEN_SIZE,
        optimization_options=options,
        opt_level=0,
        use_gpu=False,
    )
    stats = fused.get_fused_operator_statistics()
    if stats.get("Attention", 0) != 24:
        raise RuntimeError(f"attention fusion failed: {stats}")
    model = fused.model
    model.producer_name = "bge-m3-lite"
    # The contrib ops need their domain declared for onnx tooling (ORT itself
    # tolerates the omission); the optimizer only adds it in save_model_to_file.
    if not any(o.domain == "com.microsoft" for o in model.opset_import):
        model.opset_import.add(domain="com.microsoft", version=1)
    if attention_chunk > 0:
        _chunk_attention(model, attention_chunk)
    del model.metadata_props[:]
    entry = model.metadata_props.add()
    entry.key, entry.value = "bge_m3_lite.source_sha256", _sha256(model_in)

    n_shared = n_fused = 0
    with open(data_path, "wb") as out:
        for tensor in sorted(model.graph.initializer, key=lambda t: t.name):
            if tensor.data_location == onnx.TensorProto.EXTERNAL:
                raise RuntimeError(f"{tensor.name}: unexpected external tensor")
            blob = tensor.raw_data
            if not blob:
                continue  # typed fields (small constants), leave as they are
            hit = shared.get(hashlib.sha1(blob).digest())
            if hit is not None:
                set_external_data(tensor, SHARED_DATA, offset=hit[0], length=hit[1])
                n_shared += 1
            elif len(blob) > INLINE_LIMIT:
                pad = (-out.tell()) % ALIGN
                out.write(b"\0" * pad)
                set_external_data(
                    tensor, data_path.name, offset=out.tell(), length=len(blob)
                )
                out.write(blob)
                n_fused += 1
            else:
                continue
            tensor.data_location = onnx.TensorProto.EXTERNAL
            # ClearField, not ``= b""``: onnx.save_model re-writes every tensor
            # that still *has* raw_data into its external location.
            tensor.ClearField("raw_data")
    onnx.save_model(model, str(graph_path))
    return FuseResult(
        graph_path.stat().st_size,
        _sha256(graph_path),
        data_path.stat().st_size,
        _sha256(data_path),
        n_shared,
        n_fused,
    )
