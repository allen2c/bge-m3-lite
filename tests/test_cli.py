from bge_m3_lite import __version__
from bge_m3_lite.cli import build_parser, main


def test_info(capsys, monkeypatch, tmp_path):
    monkeypatch.setenv("BGE_M3_LITE_CACHE", str(tmp_path))
    assert main(["info"]) == 0
    out = capsys.readouterr().out
    assert __version__ in out and "model.onnx_data" in out and "missing" in out
    assert "model_int8.onnx" in out


def test_encode_flags():
    args = build_parser().parse_args(["encode", "--int8", "--sparse", "x"])
    assert args.int8 and args.sparse and args.text == ["x"] and args.model is None
    assert args.batch_size == 12 and args.max_batch_tokens == 16384
    args = build_parser().parse_args(["encode", "--max-batch-tokens", "4096", "x"])
    assert args.max_batch_tokens == 4096
