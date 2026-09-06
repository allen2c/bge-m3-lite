import pytest

from bge_m3_lite.quantize import QuantConfig


def test_default_config_is_shipped_profile():
    cfg = QuantConfig()
    assert (
        cfg.method == "rowwise" and cfg.quantize_embeddings and cfg.smooth_alpha == 0.5
    )


def test_embedder_rejects_unknown_precision(tokenizer_path, head_paths):
    from bge_m3_lite.embedder import BGEM3Embedder

    with pytest.raises(ValueError):
        BGEM3Embedder(precision="int4")  # type: ignore[arg-type]


def test_cli_quantize_help():
    from bge_m3_lite.cli import build_parser

    args = build_parser().parse_args(["quantize", "--method", "nbits", "--bits", "8"])
    assert args.method == "nbits" and args.bits == 8 and args.accuracy_level == 4


def test_smoothquant_is_exact_in_fp32():
    """X @ W == (X / s) @ (diag(s) W): smoothing must not change fp32 outputs."""
    onnx = pytest.importorskip("onnx")
    import numpy as np
    import onnxruntime as ort
    from onnx import helper, numpy_helper

    from bge_m3_lite.quantize import _smooth, smooth_targets

    rng = np.random.default_rng(0)
    w = rng.standard_normal((8, 4)).astype(np.float32)
    graph = helper.make_graph(
        [
            helper.make_node(
                "MatMul", ["x", "w"], ["y"], name="layer.0/output/dense/MatMul"
            ),
            helper.make_node("Relu", ["y"], ["z"], name="relu"),
        ],
        "g",
        [helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [None, 8])],
        [helper.make_tensor_value_info("z", onnx.TensorProto.FLOAT, [None, 4])],
        [numpy_helper.from_array(w, "w")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    x = rng.standard_normal((5, 8)).astype(np.float32)
    x[:, 3] *= 50.0  # an outlier channel
    before = np.asarray(
        ort.InferenceSession(model.SerializeToString()).run(None, {"x": x})[0]
    )
    assert [n.name for n in smooth_targets(model)] == ["layer.0/output/dense/MatMul"]
    _smooth(model, {"x": np.abs(x).max(axis=0)}, alpha=0.5)
    ops = [n.op_type for n in model.graph.node]
    assert ops == ["Mul", "MatMul", "Relu"]
    after = np.asarray(
        ort.InferenceSession(model.SerializeToString()).run(None, {"x": x})[0]
    )
    np.testing.assert_allclose(after, before, rtol=1e-5, atol=1e-5)
    scale = numpy_helper.to_array(model.graph.initializer[1])
    assert scale[3] < scale[0]  # the outlier channel is scaled down the most


def test_calibration_texts_bundled():
    from bge_m3_lite.quantize import load_calibration_texts

    texts = load_calibration_texts()
    assert len(texts) > 150 and all(t.strip() for t in texts)
    assert any("中" <= ch <= "鿿" for t in texts for ch in t)  # CJK present


def test_cli_quantize_smooth_flags():
    from bge_m3_lite.cli import build_parser

    args = build_parser().parse_args(["quantize"])
    assert args.method == "rowwise" and args.alpha == 0.5 and not args.no_smooth
    args = build_parser().parse_args(["quantize", "--method", "dynamic", "--no-smooth"])
    assert args.method == "dynamic" and args.no_smooth


def _ops(model) -> set[str]:
    """Op types of the graph, including the bodies of ``Loop`` nodes."""
    from onnx import helper

    ops: set[str] = set()
    for n in model.graph.node:
        ops.add(n.op_type)
        for a in n.attribute:
            if a.name == "body":
                ops |= {b.op_type for b in helper.get_attribute_value(a).node}
    return ops


@pytest.mark.parametrize(("zero_point", "chunk"), [(True, 4), (True, 0), (False, 4)])
def test_rowwise_matmul_and_attention(zero_point, chunk):
    onnx = pytest.importorskip("onnx")
    import numpy as np
    import onnxruntime as ort
    from onnx import helper, numpy_helper

    from bge_m3_lite.quantize import _quantize_rowwise

    rng = np.random.default_rng(1)
    b, s, h, heads = 2, 6, 16, 2
    w_qkv = (rng.standard_normal((h, 3 * h)) * 0.2).astype(np.float32)
    bias = (rng.standard_normal((3 * h,)) * 0.1).astype(np.float32)
    w_out = (rng.standard_normal((h, h)) * 0.2).astype(np.float32)
    emb = rng.standard_normal((20000, h)).astype(np.float32)
    graph = helper.make_graph(
        [
            helper.make_node("Gather", ["emb", "ids"], ["x"], name="embed/Gather"),
            helper.make_node(
                "Attention",
                ["x", "w_qkv", "bias", "mask"],
                ["a"],
                name="Attention_0",
                domain="com.microsoft",
                num_heads=heads,
            ),
            helper.make_node(
                "MatMul", ["a", "w_out"], ["y"], name="layer.0/output/dense/MatMul"
            ),
        ],
        "g",
        [
            helper.make_tensor_value_info("ids", onnx.TensorProto.INT64, [b, s]),
            helper.make_tensor_value_info("mask", onnx.TensorProto.INT32, [b, s]),
        ],
        [helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [b, s, h])],
        [
            numpy_helper.from_array(emb, "emb"),
            numpy_helper.from_array(w_qkv, "w_qkv"),
            numpy_helper.from_array(bias, "bias"),
            numpy_helper.from_array(w_out, "w_out"),
        ],
    )
    model = helper.make_model(
        graph,
        opset_imports=[
            helper.make_opsetid("", 13),
            helper.make_opsetid("com.microsoft", 1),
        ],
    )
    ids = rng.integers(0, 20000, (b, s)).astype(np.int64)
    mask = np.array([[1] * s, [1] * 4 + [0] * 2], dtype=np.int32)
    feeds = {"ids": ids, "mask": mask}
    ref = np.asarray(
        ort.InferenceSession(model.SerializeToString()).run(None, feeds)[0]
    )
    _quantize_rowwise(
        model,
        quantize_embeddings=True,
        zero_point=zero_point,
        attention_chunk=chunk,
    )
    ops = _ops(model)
    assert {"MatMulIntegerToFloat", "QuantizeLinear", "MultiHeadAttention"} <= ops
    assert ("Loop" in ops) == (chunk > 0)  # seq 6 with chunk 4: two iterations
    assert "Attention" not in ops and "MatMul" not in ops
    assert all(
        t.data_type != onnx.TensorProto.FLOAT
        for t in model.graph.initializer
        if t.name in ("emb", "w_qkv", "w_out")
    )
    out = np.asarray(
        ort.InferenceSession(model.SerializeToString()).run(None, feeds)[0]
    )
    valid = mask.astype(bool)
    rel = np.abs(out - ref)[valid].max() / np.abs(ref)[valid].max()
    assert rel < 0.03  # int8 noise, not a structural error


def test_rowwise_keep_fp32_leaves_matched_nodes_exact():
    """A ``keep_fp32`` pattern leaves that MatMul/Attention bit-exact in fp32."""
    onnx = pytest.importorskip("onnx")
    import numpy as np
    import onnxruntime as ort
    from onnx import helper, numpy_helper

    from bge_m3_lite.quantize import _quantize_rowwise

    rng = np.random.default_rng(2)
    b, s, h, heads = 2, 6, 16, 2
    w_qkv = (rng.standard_normal((h, 3 * h)) * 0.2).astype(np.float32)
    bias = (rng.standard_normal((3 * h,)) * 0.1).astype(np.float32)
    w_out = (rng.standard_normal((h, h)) * 0.2).astype(np.float32)
    emb = rng.standard_normal((20000, h)).astype(np.float32)
    graph = helper.make_graph(
        [
            helper.make_node("Gather", ["emb", "ids"], ["x"], name="embed/Gather"),
            helper.make_node(
                "Attention",
                ["x", "w_qkv", "bias", "mask"],
                ["a"],
                name="Attention_0",
                domain="com.microsoft",
                num_heads=heads,
            ),
            helper.make_node(
                "MatMul", ["a", "w_out"], ["y"], name="layer.0/output/dense/MatMul"
            ),
        ],
        "g",
        [
            helper.make_tensor_value_info("ids", onnx.TensorProto.INT64, [b, s]),
            helper.make_tensor_value_info("mask", onnx.TensorProto.INT32, [b, s]),
        ],
        [helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [b, s, h])],
        [
            numpy_helper.from_array(emb, "emb"),
            numpy_helper.from_array(w_qkv, "w_qkv"),
            numpy_helper.from_array(bias, "bias"),
            numpy_helper.from_array(w_out, "w_out"),
        ],
    )
    model = helper.make_model(
        graph,
        opset_imports=[
            helper.make_opsetid("", 13),
            helper.make_opsetid("com.microsoft", 1),
        ],
    )
    ids = rng.integers(0, 20000, (b, s)).astype(np.int64)
    mask = np.array([[1] * s, [1] * 4 + [0] * 2], dtype=np.int32)
    feeds = {"ids": ids, "mask": mask}
    ref = np.asarray(
        ort.InferenceSession(model.SerializeToString()).run(None, feeds)[0]
    )
    _quantize_rowwise(
        model,
        quantize_embeddings=True,
        keep_fp32=(r"layer\.0/output/dense/MatMul$",),
    )
    ops = _ops(model)
    # the kept MatMul stays a plain fp32 MatMul; the Attention is still quantised
    assert {"MatMul", "MatMulIntegerToFloat", "MultiHeadAttention"} <= ops
    kept = [n for n in model.graph.node if n.name == "layer.0/output/dense/MatMul"]
    assert len(kept) == 1 and kept[0].op_type == "MatMul"
    out = np.asarray(
        ort.InferenceSession(model.SerializeToString()).run(None, feeds)[0]
    )
    valid = mask.astype(bool)
    # only the attention projection is quantised, so the error comes solely
    # from the int8 noise in the attention output feeding the fp32 MatMul.
    rel = np.abs(out - ref)[valid].max() / np.abs(ref)[valid].max()
    assert rel < 0.03


def test_rowwise_bumps_opset_11_graph_to_13():
    """The fused graph is opset 11; per-axis QuantizeLinear needs 13."""
    onnx = pytest.importorskip("onnx")
    import numpy as np
    import onnxruntime as ort
    from onnx import helper, numpy_helper

    from bge_m3_lite.quantize import _quantize_rowwise

    w = np.eye(4, dtype=np.float32)
    graph = helper.make_graph(
        [
            helper.make_node("Unsqueeze", ["x"], ["x3"], axes=[0]),
            helper.make_node("ReduceSum", ["x3"], ["s"], axes=[1], keepdims=1),
            helper.make_node("MatMul", ["s", "w"], ["y"], name="m"),
        ],
        "g",
        [helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [3, 4])],
        [helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [1, 1, 4])],
        [numpy_helper.from_array(w, "w")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])
    model.opset_import.add(domain="com.microsoft", version=1)
    x = np.arange(12, dtype=np.float32).reshape(3, 4)
    ref = np.asarray(
        ort.InferenceSession(model.SerializeToString()).run(None, {"x": x})[0]
    )
    _quantize_rowwise(model, quantize_embeddings=False)
    assert next(o.version for o in model.opset_import if o.domain == "") == 13
    onnx.checker.check_model(model)
    out = np.asarray(
        ort.InferenceSession(model.SerializeToString()).run(None, {"x": x})[0]
    )
    np.testing.assert_allclose(out, ref, rtol=0.02, atol=0.05)


@pytest.mark.parametrize("chunk", [1, 4, 64])
def test_fused_attention_chunking_is_exact(chunk):
    """``fuse._chunk_attention`` keeps the fp32 ``Attention`` output bit-exact."""
    onnx = pytest.importorskip("onnx")
    import numpy as np
    import onnxruntime as ort
    from onnx import helper, numpy_helper

    from bge_m3_lite.fuse import _chunk_attention

    rng = np.random.default_rng(3)
    b, s, h, heads = 2, 7, 16, 2
    graph = helper.make_graph(
        [
            helper.make_node(
                "Attention",
                ["x", "w_qkv", "bias", "mask"],
                ["y"],
                name="Attention_0",
                domain="com.microsoft",
                num_heads=heads,
            )
        ],
        "g",
        [
            helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [b, s, h]),
            helper.make_tensor_value_info("mask", onnx.TensorProto.INT32, [b, s]),
        ],
        [helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [b, s, h])],
        [
            numpy_helper.from_array(
                (rng.standard_normal((h, 3 * h)) * 0.2).astype(np.float32), "w_qkv"
            ),
            numpy_helper.from_array(
                (rng.standard_normal((3 * h,)) * 0.1).astype(np.float32), "bias"
            ),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])
    model.opset_import.add(domain="com.microsoft", version=1)
    feeds = {
        "x": rng.standard_normal((b, s, h)).astype(np.float32),
        "mask": np.array([[1] * s, [1] * 5 + [0] * 2], dtype=np.int32),
    }
    ref = np.asarray(
        ort.InferenceSession(model.SerializeToString()).run(None, feeds)[0]
    )
    _chunk_attention(model, chunk)
    assert "Attention" not in _ops(model) and "Loop" in _ops(model)
    onnx.checker.check_model(model)
    out = np.asarray(
        ort.InferenceSession(model.SerializeToString()).run(None, feeds)[0]
    )
    valid = feeds["mask"].astype(bool)
    np.testing.assert_allclose(out[valid], ref[valid], rtol=1e-5, atol=1e-6)


def _two_layer_model(rng, b, s, h, heads):
    """Two fused-style encoder layers: Attention, output projection,
    SkipLayerNormalization, FFN (MatMul, BiasGelu, MatMul), SkipLayerNormalization."""
    import numpy as np
    import onnx
    from onnx import helper, numpy_helper

    nodes, inits = [], []
    x = "x"
    for layer in range(2):
        p = f"layer.{layer}"
        w = {
            f"{p}/w_qkv": (h, 3 * h),
            f"{p}/w_out": (h, h),
            f"{p}/w_in": (h, 4 * h),
            f"{p}/w_ffn_out": (4 * h, h),
        }
        for name, shape in w.items():
            inits.append(
                numpy_helper.from_array(
                    (rng.standard_normal(shape) * 0.2).astype(np.float32), name
                )
            )
        for name, size in (
            (f"{p}/b_qkv", 3 * h),
            (f"{p}/b_out", h),
            (f"{p}/b_in", 4 * h),
            (f"{p}/b_ffn_out", h),
            (f"{p}/beta1", h),
            (f"{p}/beta2", h),
        ):
            inits.append(
                numpy_helper.from_array(
                    (rng.standard_normal(size) * 0.1).astype(np.float32), name
                )
            )
        for name in (f"{p}/gamma1", f"{p}/gamma2"):
            inits.append(numpy_helper.from_array(np.ones(h, dtype=np.float32), name))
        mk = helper.make_node
        ms = "com.microsoft"
        nodes += [
            mk(
                "Attention",
                [x, f"{p}/w_qkv", f"{p}/b_qkv", "mask"],
                [f"{p}/a"],
                name=f"Attention_{layer}",
                domain=ms,
                num_heads=heads,
            ),
            mk("MatMul", [f"{p}/a", f"{p}/w_out"], [f"{p}/o"], name=f"{p}/out/MatMul"),
            mk(
                "SkipLayerNormalization",
                [f"{p}/o", x, f"{p}/gamma1", f"{p}/beta1", f"{p}/b_out"],
                [f"{p}/ln1"],
                name=f"SkipLayerNorm_{2 * layer}",
                domain=ms,
                epsilon=1e-5,
            ),
            mk("MatMul", [f"{p}/ln1", f"{p}/w_in"], [f"{p}/f"], name=f"{p}/in/MatMul"),
            mk("BiasGelu", [f"{p}/f", f"{p}/b_in"], [f"{p}/g"], domain=ms),
            mk(
                "MatMul",
                [f"{p}/g", f"{p}/w_ffn_out"],
                [f"{p}/f2"],
                name=f"{p}/ffn_out/MatMul",
            ),
            mk(
                "SkipLayerNormalization",
                [f"{p}/f2", f"{p}/ln1", f"{p}/gamma2", f"{p}/beta2", f"{p}/b_ffn_out"],
                [f"{p}/y"],
                name=f"SkipLayerNorm_{2 * layer + 1}",
                domain=ms,
                epsilon=1e-5,
            ),
        ]
        x = f"{p}/y"
    graph = helper.make_graph(
        nodes,
        "g",
        [
            helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [b, s, h]),
            helper.make_tensor_value_info("mask", onnx.TensorProto.INT32, [b, s]),
        ],
        [helper.make_tensor_value_info(x, onnx.TensorProto.FLOAT, [b, s, h])],
        inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])
    model.opset_import.add(domain="com.microsoft", version=1)
    return model


def _body_ops(model, loop_name: str) -> list[str]:
    loop = next(n for n in model.graph.node if n.name == loop_name)
    body = next(a.g for a in loop.attribute if a.name == "body")
    return [n.op_type for n in body.node]


@pytest.mark.parametrize(
    ("b", "s", "chunk"), [(2, 7, 3), (2, 7, 4), (1, 5, 64), (3, 4, 4)]
)
def test_layer_tail_row_loop_is_exact(b, s, chunk):
    """The per-token tail of a layer runs in a scan-output ``Loop`` over
    ``chunk`` rows of the flattened batch (14 rows with chunk 3/4: 5/4
    windows, the last one shifted back; 12 rows with chunk 4: exact fit; 5
    rows with chunk 64: one window). Bit-identical on Apple Silicon; x86 MLAS
    picks its GEMM kernel by row count, so the rows recomputed in the shifted
    window can differ in the last bit."""
    onnx = pytest.importorskip("onnx")
    import numpy as np
    import onnxruntime as ort

    from bge_m3_lite.fuse import _chunk_attention

    rng = np.random.default_rng(5)
    h, heads = 16, 2
    model = _two_layer_model(rng, b, s, h, heads)
    feeds = {
        "x": rng.standard_normal((b, s, h)).astype(np.float32),
        "mask": np.array(
            [[1] * s] * (b - 1) + [[1] * (s - 2) + [0] * 2], dtype=np.int32
        ),
    }
    ref = np.asarray(
        ort.InferenceSession(model.SerializeToString()).run(None, feeds)[0]
    )
    _chunk_attention(model, chunk)  # tail="rows" is the default
    onnx.checker.check_model(model)
    outer = [n.op_type for n in model.graph.node]
    assert outer.count("MatMul") == 2 and outer.count("Loop") == 4  # QKV only
    assert "SkipLayerNormalization" not in outer and "BiasGelu" not in outer
    body = _body_ops(model, "Attention_1/tail/loop")
    assert body.count("MatMul") == 3 and body.count("SkipLayerNormalization") == 2
    assert body.count("Slice") == 2 and body[-1] == "Identity"
    loop = next(n for n in model.graph.node if n.name == "Attention_1/tail/loop")
    assert len(loop.input) == 2 and len(loop.output) == 1  # scan output only
    assert model.graph.output[0].name == "layer.1/y"
    out = np.asarray(
        ort.InferenceSession(model.SerializeToString()).run(None, feeds)[0]
    )
    np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("chunk", [3, 4, 64])
def test_layer_tail_moves_into_loop_and_stays_exact(chunk):
    """``tail="loop"`` (v0.5.2): the whole per-token tail of a layer runs inside
    the attention ``Loop`` and the fp32 output is unchanged (seq 7 with chunk
    3/4: 3/2 iterations)."""
    onnx = pytest.importorskip("onnx")
    import numpy as np
    import onnxruntime as ort

    from bge_m3_lite.fuse import _chunk_attention

    rng = np.random.default_rng(5)
    b, s, h, heads = 2, 7, 16, 2
    model = _two_layer_model(rng, b, s, h, heads)
    feeds = {
        "x": rng.standard_normal((b, s, h)).astype(np.float32),
        "mask": np.array([[1] * s, [1] * 5 + [0] * 2], dtype=np.int32),
    }
    ref = np.asarray(
        ort.InferenceSession(model.SerializeToString()).run(None, feeds)[0]
    )
    _chunk_attention(model, chunk, tail="loop")
    onnx.checker.check_model(model)
    outer = [n.op_type for n in model.graph.node]
    assert outer.count("MatMul") == 2 and outer.count("Loop") == 2  # QKV only
    assert "SkipLayerNormalization" not in outer and "BiasGelu" not in outer
    body = _body_ops(model, "Attention_1/loop")
    assert body.count("MatMul") == 3 and body.count("SkipLayerNormalization") == 2
    assert body.count("Slice") == 2 and body[-2:] == ["Concat", "Identity"]
    assert model.graph.output[0].name == "layer.1/y"
    out = np.asarray(
        ort.InferenceSession(model.SerializeToString()).run(None, feeds)[0]
    )
    valid = feeds["mask"].astype(bool)
    np.testing.assert_allclose(out[valid], ref[valid], rtol=1e-5, atol=1e-6)


def test_layer_tail_pass_is_optional_and_skips_broken_chains():
    onnx = pytest.importorskip("onnx")
    import numpy as np

    from bge_m3_lite.fuse import _chunk_attention
    from bge_m3_lite.quantize import layer_tail_into_loop

    rng = np.random.default_rng(6)
    model = _two_layer_model(rng, 1, 5, 8, 2)
    _chunk_attention(model, 4, tail="none")
    assert [n.op_type for n in model.graph.node].count("SkipLayerNormalization") == 4
    # an FFN intermediate that is also a graph output breaks layer 0's chain
    model.graph.output.append(
        onnx.helper.make_tensor_value_info("layer.0/g", onnx.TensorProto.FLOAT, None)
    )
    assert layer_tail_into_loop(model) == 1
    assert "BiasGelu" in [n.op_type for n in model.graph.node]
    assert "BiasGelu" in _body_ops(model, "Attention_1/loop")


@pytest.mark.parametrize("tail", ["loop", "rows"])
def test_rowwise_layer_tail_in_loop(tail):
    """int8: the quantised projections of the tail move into the loop too."""
    pytest.importorskip("onnx")
    import numpy as np
    import onnxruntime as ort

    from bge_m3_lite.quantize import _quantize_rowwise

    rng = np.random.default_rng(7)
    b, s, h, heads = 2, 6, 16, 2
    model = _two_layer_model(rng, b, s, h, heads)
    feeds = {
        "x": rng.standard_normal((b, s, h)).astype(np.float32),
        "mask": np.array([[1] * s, [1] * 4 + [0] * 2], dtype=np.int32),
    }
    ref = np.asarray(
        ort.InferenceSession(model.SerializeToString()).run(None, feeds)[0]
    )
    _quantize_rowwise(model, quantize_embeddings=False, attention_chunk=4, tail=tail)
    outer = [n.op_type for n in model.graph.node]
    assert "SkipLayerNormalization" not in outer
    assert outer.count("MatMulIntegerToFloat") == 2  # the two QKV projections
    body = _body_ops(
        model,
        "rowwise/Attention_0/loop"
        if tail == "loop"
        else "rowwise/Attention_0/tail/loop",
    )
    assert body.count("MatMulIntegerToFloat") == 3
    assert body.count("SkipLayerNormalization") == 2
    out = np.asarray(
        ort.InferenceSession(model.SerializeToString()).run(None, feeds)[0]
    )
    valid = feeds["mask"].astype(bool)
    rel = np.abs(out - ref)[valid].max() / np.abs(ref)[valid].max()
    assert rel < 0.05


def test_has_contrib_ops_looks_inside_loop_bodies():
    pytest.importorskip("onnx")
    import numpy as np

    from bge_m3_lite.fuse import _chunk_attention
    from bge_m3_lite.quantize import has_contrib_ops

    model = _two_layer_model(np.random.default_rng(8), 1, 5, 8, 2)
    assert has_contrib_ops(model.graph)
    _chunk_attention(model, 4)  # every contrib op is now inside a Loop body
    assert all(n.domain != "com.microsoft" for n in model.graph.node)
    assert has_contrib_ops(model.graph)
