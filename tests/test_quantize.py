import pytest

from bge_m3_lite.quantize import QuantConfig


def test_default_config_is_shipped_profile():
    cfg = QuantConfig()
    assert cfg.method == "dynamic" and cfg.quantize_embeddings


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
    assert args.alpha == 0.5 and not args.no_smooth and args.calibration is None
    args = build_parser().parse_args(["quantize", "--no-smooth", "--alpha", "0.7"])
    assert args.no_smooth and args.alpha == 0.7
