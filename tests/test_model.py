"""Session options and external-data loading of the onnxruntime wrapper."""

from pathlib import Path

import numpy as np
import pytest

from bge_m3_lite import model as m


def test_session_options_defaults(monkeypatch):
    monkeypatch.delenv("BGE_M3_LITE_THREADS", raising=False)
    monkeypatch.delenv("BGE_M3_LITE_SPIN", raising=False)
    opts = m.session_options()
    assert opts.get_session_config_entry("session.intra_op.allow_spinning") == "0"
    assert opts.intra_op_num_threads == m.performance_cores()  # 0 off Apple Silicon
    assert (
        m.session_options(spin=True).get_session_config_entry(
            "session.intra_op.allow_spinning"
        )
        == "1"
    )
    monkeypatch.setenv("BGE_M3_LITE_THREADS", "3")
    monkeypatch.setenv("BGE_M3_LITE_SPIN", "1")
    opts = m.session_options()
    assert opts.intra_op_num_threads == 3
    assert opts.get_session_config_entry("session.intra_op.allow_spinning") == "1"
    assert m.session_options(2).intra_op_num_threads == 2
    with pytest.raises(RuntimeError):  # not set unless low_memory
        opts.get_session_config_entry("session.disable_prepacking")
    low = m.session_options(low_memory=True)
    assert low.get_session_config_entry("session.disable_prepacking") == "1"


def _tiny_backbone(onnx, rng):
    """Gather(emb, input_ids) * mask -> token_embeddings, like the real graph."""
    from onnx import helper, numpy_helper

    emb = rng.standard_normal((64, 32)).astype(np.float32)  # 8 KiB: external
    bias = rng.standard_normal((32,)).astype(np.float32)  # 128 B: stays inline
    graph = helper.make_graph(
        [
            helper.make_node("Gather", ["emb", "input_ids"], ["x"]),
            helper.make_node("Add", ["x", "bias"], ["xb"]),
            helper.make_node("Cast", ["attention_mask"], ["mf"], to=1),
            helper.make_node("Unsqueeze", ["mf", "axes"], ["m3"]),
            helper.make_node("Mul", ["xb", "m3"], ["token_embeddings"]),
        ],
        "tiny",
        [
            helper.make_tensor_value_info("input_ids", onnx.TensorProto.INT64, [1, 4]),
            helper.make_tensor_value_info(
                "attention_mask", onnx.TensorProto.INT64, [1, 4]
            ),
        ],
        [
            helper.make_tensor_value_info(
                "token_embeddings", onnx.TensorProto.FLOAT, [1, 4, 32]
            )
        ],
        [
            numpy_helper.from_array(emb, "emb"),
            numpy_helper.from_array(bias, "bias"),
            numpy_helper.from_array(np.array([2], dtype=np.int64), "axes"),
        ],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])


def test_backbone_loads_external_data(tmp_path: Path):
    """The int8 asset ships as graph + ``_data`` file; ORT must resolve it on
    every platform (Windows included) and compute the same as the embedded graph."""
    onnx = pytest.importorskip("onnx")
    from bge_m3_lite.quantize import INLINE_LIMIT, write_external_data

    rng = np.random.default_rng(0)
    model = _tiny_backbone(onnx, rng)
    embedded = tmp_path / "embedded.onnx"
    onnx.save_model(model, str(embedded))

    graph_path = tmp_path / "model_int8.onnx"
    data_path = tmp_path / "model_int8.onnx_data"
    assert write_external_data(model, data_path) == (0, 1)
    onnx.save_model(model, str(graph_path))
    loaded = onnx.load(str(graph_path), load_external_data=False)
    external = {
        t.name for t in loaded.graph.initializer if t.data_location == 1
    }  # EXTERNAL
    assert external == {"emb"} and data_path.stat().st_size == 64 * 32 * 4
    assert graph_path.stat().st_size < INLINE_LIMIT * 2

    ids = np.array([[3, 1, 4, 1]], dtype=np.int64)
    mask = np.array([[1, 1, 1, 0]], dtype=np.int64)
    ref = m.OnnxBackbone(embedded, num_threads=1)(ids, mask)
    out = m.OnnxBackbone(graph_path, num_threads=1)(ids, mask)
    assert out.shape == (1, 4, 32) and out.dtype == np.float32
    np.testing.assert_array_equal(out, ref)
    assert np.all(out[0, 3] == 0.0)


def test_write_external_data_is_aligned_and_shares(tmp_path: Path):
    onnx = pytest.importorskip("onnx")
    import hashlib

    from onnx import numpy_helper
    from onnx.external_data_helper import ExternalDataInfo

    from bge_m3_lite.quantize import ALIGN, write_external_data

    rng = np.random.default_rng(1)
    model = _tiny_backbone(onnx, rng)
    a = rng.standard_normal((300,)).astype(np.float32)  # 1200 B, odd size
    b = rng.standard_normal((400,)).astype(np.float32)
    model.graph.initializer.extend(
        [numpy_helper.from_array(a, "z_a"), numpy_helper.from_array(b, "z_b")]
    )
    shared = {hashlib.sha1(b.tobytes()).digest(): ("model.onnx_data", 4096, 1600)}
    data_path = tmp_path / "g.onnx_data"
    assert write_external_data(model, data_path, shared=shared) == (1, 2)
    infos = {
        t.name: ExternalDataInfo(t)
        for t in model.graph.initializer
        if t.data_location == onnx.TensorProto.EXTERNAL
    }
    assert set(infos) == {"emb", "z_a", "z_b"}
    assert (infos["z_b"].location, infos["z_b"].offset) == ("model.onnx_data", 4096)
    assert (infos["emb"].offset, infos["z_a"].offset) == (0, 64 * 32 * 4)
    assert 64 * 32 * 4 % ALIGN == 0  # sorted by name, emb is already aligned
    assert data_path.stat().st_size == 64 * 32 * 4 + 1200
    assert not any(
        t.HasField("raw_data") for t in model.graph.initializer if t.name in infos
    )
