from bge_m3_lite import __version__
from bge_m3_lite.cli import main


def test_info(capsys, monkeypatch, tmp_path):
    monkeypatch.setenv("BGE_M3_LITE_CACHE", str(tmp_path))
    assert main(["info"]) == 0
    out = capsys.readouterr().out
    assert __version__ in out and "model.onnx_data" in out and "missing" in out
