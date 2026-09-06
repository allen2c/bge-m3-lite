"""Compare any backbone ONNX file against the FlagEmbedding fixtures and time it.

Usage:  uv run tools/eval_model.py [--low-memory] [MODEL.onnx ...]   (default: cached fp32 model)

Besides accuracy and tok/s it reports what docs/resources.md tracks: resident
memory after loading and at the peak, CPU-seconds per 1 000 tokens, and the
CPU an idle session burns per second (thread spinning).
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from bge_m3_lite import hub
from bge_m3_lite.embedder import BGEM3Embedder

FIXTURES = Path(__file__).resolve().parent.parent / "tests" / "fixtures"


def rss_mib() -> float:
    """Resident set size of this process in MiB."""
    if sys.platform == "win32":
        return _win_memory()[0]
    if sys.platform == "darwin":
        import subprocess

        out = subprocess.check_output(["ps", "-o", "rss=", "-p", str(os.getpid())])
        return int(out) / 1024
    with open("/proc/self/statm") as fh:
        return int(fh.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2**20


def peak_rss_mib() -> float:
    if sys.platform == "win32":
        return _win_memory()[1]
    import resource

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 2**20 if sys.platform == "darwin" else peak / 1024


def _win_memory() -> tuple[float, float]:  # pragma: no cover - Windows only
    import ctypes
    from ctypes import wintypes

    class Counters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    ctypes.windll.psapi.GetProcessMemoryInfo(  # type: ignore[attr-defined]
        ctypes.windll.kernel32.GetCurrentProcess(),  # type: ignore[attr-defined]
        ctypes.byref(counters),
        counters.cb,
    )
    return counters.WorkingSetSize / 2**20, counters.PeakWorkingSetSize / 2**20


def os_threads() -> int:
    """Threads of this process (shows how many workers onnxruntime started)."""
    if sys.platform == "linux":
        with open("/proc/self/status") as fh:
            return next(int(x.split()[1]) for x in fh if x.startswith("Threads:"))
    if sys.platform == "darwin":
        import subprocess

        out = subprocess.check_output(["ps", "-M", "-p", str(os.getpid())])
        return out.count(b"\n") - 1
    return 0


def idle_cpu_ms_per_s(seconds: float = 1.0) -> float:
    """CPU time the process burns while nothing runs (onnxruntime spinning)."""
    time.sleep(0.5)  # let the run's tail (thread hand-off, page reclaim) settle
    cpu0 = time.process_time()
    time.sleep(seconds)
    return (time.process_time() - cpu0) / seconds * 1000


def _report_sparse_colbert(
    emb: BGEM3Embedder, out: dict, ref: dict, npz: np.lib.npyio.NpzFile, label: str
) -> None:
    dense_cos = (out["dense_vecs"] * npz["dense"]).sum(1)
    print(f"[{label}] dense cos: min {dense_cos.min():.5f} mean {dense_cos.mean():.5f}")
    same_keys = top5 = spearman = 0.0
    for lw, rlw in zip(out["lexical_weights"], ref["lexical_weights"], strict=True):
        same_keys += set(lw) == set(rlw)
        a = sorted(lw, key=lw.get, reverse=True)[:5]
        b = sorted(rlw, key=rlw.get, reverse=True)[:5]
        top5 += a == b
        keys = sorted(rlw)
        if len(keys) > 2:
            ra = np.argsort(np.argsort([-lw.get(k, 0.0) for k in keys]))
            rb = np.argsort(np.argsort([-rlw[k] for k in keys]))
            spearman += np.corrcoef(ra, rb)[0, 1]
        else:
            spearman += 1.0
    n = len(ref["sentences"])
    print(
        f"[{label}] sparse: same key set {same_keys:.0f}/{n}, top-5 identical {top5:.0f}/{n}, "
        f"mean Spearman {spearman / n:.4f}"
    )
    cols = [
        (cv * npz[f"colbert_{i}"]).sum(1) for i, cv in enumerate(out["colbert_vecs"])
    ]
    allc = np.concatenate(cols)
    p1, p5 = np.percentile(allc, [1, 5])
    print(
        f"[{label}] colbert token cos: min {allc.min():.5f} p1 {p1:.5f} p5 {p5:.5f} "
        f"mean {allc.mean():.5f}"
    )
    # late-interaction ranking agreement: query 0 against all others
    q, rq = out["colbert_vecs"][0], npz["colbert_0"]
    s = [emb.colbert_score(q, c) for c in out["colbert_vecs"][1:]]
    rs = [emb.colbert_score(rq, npz[f"colbert_{i}"]) for i in range(1, n)]
    print(
        f"[{label}] colbert ranking (q0 vs rest) identical: "
        f"{np.argsort(s).tolist() == np.argsort(rs).tolist()}"
    )


def evaluate(
    model_path: Path, *, heldout: bool = True, low_memory: bool = False
) -> None:
    ref = json.loads((FIXTURES / "embeddings_ref.json").read_text(encoding="utf-8"))
    npz = np.load(FIXTURES / "embeddings_ref.npz")
    t0 = time.perf_counter()
    emb = BGEM3Embedder(quiet=True, model_path=model_path, low_memory=low_memory)
    assert emb.backbone.session is not None
    size = model_path.stat().st_size
    data = model_path.with_name(model_path.name + "_data")
    if data.exists():
        size += data.stat().st_size
    print(
        f"\n== {model_path} ({size / 2**20:.0f} MiB){' low-memory' if low_memory else ''} "
        f"session {time.perf_counter() - t0:.2f}s rss {rss_mib():.0f} MiB "
        f"threads {emb.backbone.session.get_session_options().intra_op_num_threads} "
        f"os-threads {os_threads()}"
    )
    out = emb.encode(
        ref["sentences"], batch_size=4, return_sparse=True, return_colbert_vecs=True
    )
    _report_sparse_colbert(emb, out, ref, npz, "11-set")

    if heldout and (FIXTURES / "heldout_ref.npz").exists():
        href = json.loads((FIXTURES / "heldout_ref.json").read_text(encoding="utf-8"))
        hnpz = np.load(FIXTURES / "heldout_ref.npz")
        hout = emb.encode(
            href["sentences"],
            batch_size=4,
            return_sparse=True,
            return_colbert_vecs=True,
        )
        _report_sparse_colbert(emb, hout, href, hnpz, "held-out")

    for name, text, bs in [
        ("16tok x32", "What is the capital of France? " * 2, 32),
        ("128tok x16", "The quick brown fox jumps over the lazy dog. " * 12, 16),
        (
            "512tok x4",
            "Retrieval augmented generation combines search and language models. " * 40,
            4,
        ),
    ]:
        texts = [text] * bs
        ntok = sum(len(x) for x in emb.tokenize(texts))
        emb.encode(texts[:1])
        t, cpu = time.perf_counter(), time.process_time()
        emb.encode(texts, batch_size=bs)
        dt, dcpu = time.perf_counter() - t, time.process_time() - cpu
        print(
            f"{name:12s} {ntok / dt:6.0f} tok/s  {dcpu / ntok * 1000:5.2f} cpu-s/ktok"
        )
    # One short query at a time is the serving pattern: wall and CPU per call.
    query = ["What is the capital of France?"] * 20
    emb.encode(query[:1])
    t, cpu = time.perf_counter(), time.process_time()
    for q in query:
        emb.encode(q)
    n = len(query)
    print(
        f"short query   {(time.perf_counter() - t) / n * 1000:5.1f} ms wall "
        f"{(time.process_time() - cpu) / n * 1000:5.1f} ms cpu"
    )
    print(
        f"idle cpu {idle_cpu_ms_per_s(2.0):.0f} ms/s  rss {rss_mib():.0f} MiB  "
        f"peak rss {peak_rss_mib():.0f} MiB"
    )
    emb.close()


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="evaluate backbone ONNX files")
    ap.add_argument("models", nargs="*", type=Path)
    ap.add_argument("--low-memory", action="store_true", help="no weight prepacking")
    args = ap.parse_args()
    paths = args.models or [hub.default_cache_dir() / "model.onnx"]
    if len(paths) == 1:
        evaluate(paths[0], low_memory=args.low_memory)
    else:  # one process per model, so RSS and thread counts are not inherited
        import subprocess

        flags = ["--low-memory"] if args.low_memory else []
        for p in paths:
            subprocess.run([sys.executable, __file__, str(p), *flags], check=True)
