"""Model file resolution: local cache + download from the Hugging Face Hub.

Only ``urllib`` is used. Files are verified against pinned SHA-256 digests so
a corrupted or tampered download is never loaded.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import ssl
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

REPO_ID = "BAAI/bge-m3"
REVISION = "5617a9f61b028005a4858fdac845db406aefb181"


@dataclass(frozen=True)
class RemoteFile:
    name: str  # local file name
    remote_path: str  # path inside the HF repo (or a full URL for release assets)
    size: int
    sha256: str | None


MODEL_FILES: tuple[RemoteFile, ...] = (
    RemoteFile(
        "model.onnx",
        "onnx/model.onnx",
        724923,
        "f84251230831afb359ab26d9fd37d5936d4d9bb5d1d5410e66442f630f24435b",
    ),
    RemoteFile(
        "model.onnx_data",
        "onnx/model.onnx_data",
        2266820608,
        "1eebfb28493f67bba03ce0ef64bfdc7fc5a3bd9d7493f818bb1d78cd798416b4",
    ),
    RemoteFile(
        "Constant_7_attr__value",
        "onnx/Constant_7_attr__value",
        65552,
        "cdf16f72c5d07b36484056e601ed9687f78477e5d85cee85a34f2406b7fb5906",
    ),
)
TOKENIZER_FILES: tuple[RemoteFile, ...] = (
    RemoteFile(
        "sentencepiece.bpe.model",
        "sentencepiece.bpe.model",
        5069051,
        "cfc8146abe2a0488e9e2a0c56de7952f7c11ab059eca145a0a727afce0db2865",
    ),
)
HEAD_FILES: tuple[RemoteFile, ...] = (
    RemoteFile(
        "colbert_linear.pt",
        "colbert_linear.pt",
        2100674,
        "19bfbae397c2b7524158c919d0e9b19393c5639d098f0a66932c91ed8f5f9abb",
    ),
    RemoteFile(
        "sparse_linear.pt",
        "sparse_linear.pt",
        3516,
        "45c93804d2142b8f6d7ec6914ae23a1eee9c6a1d27d83d908a20d2afb3595ad9",
    ),
)
ALL_FILES = TOKENIZER_FILES + HEAD_FILES + MODEL_FILES

# int8 backbone (SmoothQuant + row-wise dynamic int8 of the fused graph,
# built by ``bge-m3-lite quantize``, see docs/quantization.md). Hosted
# as a GitHub release asset; override the URL with BGE_M3_LITE_INT8_URL or
# build it locally into the cache.
INT8_RELEASE = "https://github.com/allen2c/bge-m3-lite/releases/download/v0.3.1"
INT8_FILE = RemoteFile(
    "model_int8.onnx",
    f"{INT8_RELEASE}/model_int8.onnx",
    597151197,
    "d87b5cc0b0953c5336eebe3bce99cf3c55683edb9fe321fa6fdec73425d864e4",
)

# Fused fp32 backbone (Attention / SkipLayerNorm / BiasGelu contrib ops, built by
# ``bge-m3-lite fuse``, see docs/fusion.md). The graph references the Hub
# weights in ``model.onnx_data`` by offset; only the merged QKV projections
# live in ``model_fused.onnx_data``. Override the base URL with
# BGE_M3_LITE_FUSED_URL.
FUSED_RELEASE = "https://github.com/allen2c/bge-m3-lite/releases/download/v0.3.0"
FUSED_FILES: tuple[RemoteFile, ...] = (
    RemoteFile(
        "model_fused.onnx",
        f"{FUSED_RELEASE}/model_fused.onnx",
        158733,
        "113d3c707e0578387a6da8f33621f78d4d17fd8bca7327674a3cb41bb215d417",
    ),
    RemoteFile(
        "model_fused.onnx_data",
        f"{FUSED_RELEASE}/model_fused.onnx_data",
        302284800,
        "d90723fed1af6a11089cbed0e0ae148366c502a8475a6cf22a5aa69d466c84e5",
    ),
)
VERIFY_LIMIT = 64 << 20  # files smaller than this are digest-checked on every load


def default_cache_dir() -> Path:
    env = os.environ.get("BGE_M3_LITE_CACHE")
    if env:
        return Path(env).expanduser()
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".cache"
    return base / "bge-m3-lite" / REPO_ID.replace("/", "--")


def hf_endpoint() -> str:
    return os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")


def file_url(remote: RemoteFile) -> str:
    if remote is INT8_FILE and os.environ.get("BGE_M3_LITE_INT8_URL"):
        return os.environ["BGE_M3_LITE_INT8_URL"]
    if remote in FUSED_FILES and os.environ.get("BGE_M3_LITE_FUSED_URL"):
        base = os.environ["BGE_M3_LITE_FUSED_URL"].rstrip("/")
        return f"{base}/{remote.name}"
    if remote.remote_path.startswith(("https://", "http://")):
        return remote.remote_path
    return f"{hf_endpoint()}/{REPO_ID}/resolve/{REVISION}/{remote.remote_path}"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def is_complete(path: Path, remote: RemoteFile, *, verify: bool = False) -> bool:
    if not path.is_file() or path.stat().st_size != remote.size:
        return False
    return not (verify and remote.sha256) or _sha256(path) == remote.sha256


def _progress(name: str, done: int, total: int, start: float) -> None:
    if not sys.stderr.isatty():
        return
    pct = 100.0 * done / total if total else 0.0
    speed = done / max(time.monotonic() - start, 1e-6) / (1 << 20)
    sys.stderr.write(
        f"\r{name}: {done / (1 << 20):8.1f}/{total / (1 << 20):.1f} MiB "
        f"({pct:5.1f}%) {speed:6.1f} MiB/s"
    )
    if done >= total:
        sys.stderr.write("\n")
    sys.stderr.flush()


def download(remote: RemoteFile, dest: Path, *, quiet: bool = False) -> Path:
    """Download ``remote`` to ``dest`` (resumable), verify size + digest."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    lock = dest.with_name(dest.name + ".lock")
    _acquire_lock(lock)
    try:
        if is_complete(dest, remote):
            return dest  # another process finished it while we waited
        return _download_locked(remote, dest, part, quiet=quiet)
    finally:
        lock.unlink(missing_ok=True)


def _acquire_lock(lock: Path, timeout: float = 6 * 3600) -> None:
    deadline = time.monotonic() + timeout
    while True:
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            return
        except FileExistsError:
            # Stale lock (crashed process): older than a day, or holder gone.
            try:
                age = time.time() - lock.stat().st_mtime
                holder = int(lock.read_text() or 0)
            except (OSError, ValueError):
                continue
            if age > 86400 or (holder and not _pid_alive(holder)):
                lock.unlink(missing_ok=True)
                continue
            if time.monotonic() > deadline:
                raise TimeoutError(f"could not acquire {lock}") from None
            time.sleep(1.0)


def _pid_alive(pid: int) -> bool:
    if sys.platform == "win32":
        # os.kill(pid, 0) would *terminate* the process on Windows.
        import ctypes

        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        handle = kernel32.OpenProcess(0x1000, False, pid)  # QUERY_LIMITED_INFORMATION
        if not handle:
            return kernel32.GetLastError() == 5  # ERROR_ACCESS_DENIED: exists
        kernel32.CloseHandle(handle)
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _finish(part: Path, dest: Path, remote: RemoteFile) -> Path:
    if part.stat().st_size != remote.size:
        raise RuntimeError(
            f"{remote.name}: size mismatch ({part.stat().st_size} != {remote.size})"
        )
    if remote.sha256 and _sha256(part) != remote.sha256:
        part.unlink()
        raise RuntimeError(f"{remote.name}: SHA-256 mismatch, file removed")
    shutil.move(part, dest)
    return dest


RETRIES = 3  # attempts per file; HF answers 429 after many downloads in a day
BACKOFF = 2.0  # seconds, doubled after every failed attempt


def _download_locked(
    remote: RemoteFile, dest: Path, part: Path, *, quiet: bool
) -> Path:
    url = file_url(remote)
    for attempt in range(1, RETRIES + 1):
        try:
            _fetch(url, remote, part, quiet=quiet)
            break
        except (urllib.error.URLError, OSError) as exc:
            if attempt == RETRIES or not _retryable(exc):
                hint = ""
                if "CERTIFICATE_VERIFY_FAILED" in str(exc):
                    hint = (
                        " (no CA certificates for this Python: run the bundled "
                        '"Install Certificates.command" on macOS python.org builds, '
                        "or pip install certifi)"
                    )
                raise RuntimeError(f"failed to download {url}: {exc}{hint}") from exc
            delay = _retry_delay(exc, attempt)
            if not quiet:
                print(
                    f"\n{remote.name}: {exc}; retry {attempt}/{RETRIES - 1} "
                    f"in {delay:.0f}s",
                    file=sys.stderr,
                )
            time.sleep(delay)
    return _finish(part, dest, remote)


def _retryable(exc: BaseException) -> bool:
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code == 429 or exc.code >= 500
    return "CERTIFICATE_VERIFY_FAILED" not in str(exc)  # network / timeout


def _retry_delay(exc: BaseException, attempt: int) -> float:
    if isinstance(exc, urllib.error.HTTPError):
        try:
            return min(float(exc.headers.get("Retry-After", "")), 120.0)
        except ValueError:
            pass
    return BACKOFF * 2 ** (attempt - 1)


def _fetch(url: str, remote: RemoteFile, part: Path, *, quiet: bool) -> None:
    """One (resumable) attempt: append to ``part`` until it holds ``remote.size``."""
    offset = part.stat().st_size if part.exists() else 0
    if offset == remote.size:
        return  # fully downloaded, interrupted before verify
    if offset > remote.size:
        offset = 0
        part.unlink()
    headers = {"User-Agent": "bge-m3-lite"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    req = urllib.request.Request(url, headers=headers)
    start = time.monotonic()
    with urllib.request.urlopen(req, timeout=60, context=_ssl_context()) as resp:
        if offset and resp.status != 206:
            offset = 0  # server ignored the range; start over
        with open(part, "ab" if offset else "wb") as fh:
            done = offset
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                fh.write(chunk)
                done += len(chunk)
                if not quiet:
                    _progress(remote.name, done, remote.size, start)


def _ssl_context() -> ssl.SSLContext:
    """Default context, with certifi's CA bundle when the system has none."""
    ctx = ssl.create_default_context()
    if not ctx.get_ca_certs() and not ssl.get_default_verify_paths().cafile:
        try:
            import certifi  # type: ignore[import-not-found]  # optional

            ctx.load_verify_locations(cafile=certifi.where())
        except ImportError:
            pass
    return ctx


def ensure_files(
    files: tuple[RemoteFile, ...] = ALL_FILES,
    cache_dir: Path | str | None = None,
    *,
    quiet: bool = False,
) -> dict[str, Path]:
    """Return ``{name: local_path}`` for ``files``, downloading what is missing."""
    cache = Path(cache_dir).expanduser() if cache_dir else default_cache_dir()
    out: dict[str, Path] = {}
    for remote in files:
        path = cache / remote.name
        # Small files are re-hashed on every run (cheap); the 2.3 GB model data
        # is hashed once at download time and size-checked afterwards.
        if not is_complete(path, remote, verify=remote.size < VERIFY_LIMIT):
            if path.exists():
                path.unlink()
            if os.environ.get("BGE_M3_LITE_OFFLINE") == "1":
                raise FileNotFoundError(f"{path} missing and offline mode is on")
            download(remote, path, quiet=quiet)
        out[remote.name] = path
    return out
