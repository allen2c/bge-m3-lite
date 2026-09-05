import hashlib
from pathlib import Path

from bge_m3_lite import hub


def test_urls_and_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("BGE_M3_LITE_CACHE", str(tmp_path))
    assert hub.default_cache_dir() == tmp_path
    monkeypatch.setenv("HF_ENDPOINT", "https://mirror.example/")
    url = hub.file_url(hub.MODEL_FILES[0])
    assert (
        url
        == f"https://mirror.example/BAAI/bge-m3/resolve/{hub.REVISION}/onnx/model.onnx"
    )


def test_is_complete(tmp_path: Path):
    data = b"abc" * 10
    remote = hub.RemoteFile(
        "x.bin", "x.bin", len(data), hashlib.sha256(data).hexdigest()
    )
    path = tmp_path / "x.bin"
    assert not hub.is_complete(path, remote)
    path.write_bytes(data)
    assert hub.is_complete(path, remote, verify=True)
    path.write_bytes(b"x" * len(data))
    assert hub.is_complete(path, remote)  # size only
    assert not hub.is_complete(path, remote, verify=True)


def test_offline_mode_raises(monkeypatch, tmp_path):
    monkeypatch.setenv("BGE_M3_LITE_OFFLINE", "1")
    import pytest

    with pytest.raises(FileNotFoundError):
        hub.ensure_files(hub.HEAD_FILES, tmp_path)


def test_file_table_consistent():
    names = [f.name for f in hub.ALL_FILES]
    assert len(names) == len(set(names))
    assert sum(f.size for f in hub.MODEL_FILES) > 2_000_000_000


def test_finished_part_is_reused_not_redownloaded(tmp_path, monkeypatch):
    data = b"payload" * 100
    remote = hub.RemoteFile(
        "f.bin", "f.bin", len(data), hashlib.sha256(data).hexdigest()
    )
    dest = tmp_path / "f.bin"
    dest.with_name("f.bin.part").write_bytes(data)

    def boom(*a, **k):
        raise AssertionError("network must not be used")

    monkeypatch.setattr(hub.urllib.request, "urlopen", boom)
    assert hub.download(remote, dest, quiet=True) == dest
    assert dest.read_bytes() == data and not dest.with_name("f.bin.lock").exists()


def test_stale_lock_is_removed(tmp_path):
    lock = tmp_path / "x.lock"
    lock.write_text("999999999")  # no such pid
    hub._acquire_lock(lock, timeout=5)
    assert lock.read_text() == str(hub.os.getpid())
    lock.unlink()


def test_int8_url_override(monkeypatch):
    assert hub.file_url(hub.INT8_FILE) == hub.INT8_FILE.remote_path
    assert hub.file_url(hub.INT8_FILE).startswith("https://github.com/")
    monkeypatch.setenv("BGE_M3_LITE_INT8_URL", "https://mirror.example/int8.onnx")
    assert hub.file_url(hub.INT8_FILE) == "https://mirror.example/int8.onnx"
    monkeypatch.setenv("HF_ENDPOINT", "https://hf.example")
    assert hub.file_url(hub.TOKENIZER_FILES[0]).startswith("https://hf.example/BAAI")


def test_pid_alive():
    assert hub._pid_alive(hub.os.getpid())
    assert not hub._pid_alive(999999999)
