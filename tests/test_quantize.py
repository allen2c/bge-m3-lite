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
