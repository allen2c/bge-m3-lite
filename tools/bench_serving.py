"""Serving benchmark: one shared embedder under asyncio, closed-loop clients.

Usage:  uv run tools/bench_serving.py [--precision fp32|int8] [--model PATH]
                                      [--concurrency 1,2,4,8] [--sessions 4]
                                      [--queries 40] [--passages 16] [--repeat 2]

Prints one Markdown table per precision (the CI `bench` job appends it to the
step summary) with, per mode: requests per second, p50 / p95 latency seen by a
client, CPU-seconds per request, the worst event-loop stall while requests run
(a coroutine ticks every 10 ms) and peak RSS. Modes: sequential calls in the
main thread; `asyncio.to_thread(emb.encode, q)` at each concurrency, with a
closed loop of that many clients (a client sends its next request as soon as
the previous one returns, like `wrk`); `encode(list)` in batches of several
sizes (what a micro-batcher would submit: every request waits for the whole
batch); the same for 128-token passages; and, with `--sessions N`, N sessions
of ``threads / N`` intra-op threads each, one client per session.
"""

from __future__ import annotations

import argparse
import asyncio
import inspect
import os
import sys
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from statistics import quantiles

sys.path.insert(0, str(Path(__file__).resolve().parent))  # for eval_model helpers

from eval_model import peak_rss_mib, rss_mib  # noqa: E402

from bge_m3_lite.embedder import BGEM3Embedder  # noqa: E402
from bge_m3_lite.serving import AsyncEmbedder  # noqa: E402

QUERY = "What is the capital of France?"  # 9 tokens
SENTENCE = "The quick brown fox jumps over the lazy dog. "  # 10 tokens
TICK_S = 0.01
Result = tuple[list[float], float, float, float | None]  # latencies, wall, cpu, lag


class Loop:
    """A coroutine that sleeps ``TICK_S`` in a loop and records how late every
    wake-up was: the event-loop stall an unrelated request would see."""

    def __init__(self) -> None:
        self.max_lag_ms = 0.0
        self.ticks = 0

    async def run(self) -> None:
        try:
            while True:
                t = time.perf_counter()
                await asyncio.sleep(TICK_S)
                self.max_lag_ms = max(
                    self.max_lag_ms, (time.perf_counter() - t - TICK_S) * 1000
                )
                self.ticks += 1
        except asyncio.CancelledError:
            pass


Client = Callable[[str], object] | Callable[[str], Awaitable[object]]


async def closed_loop(texts: list[str], clients: list[Client]) -> Result:
    """Run ``texts`` through the ``clients`` (sync ``encode`` goes through
    ``asyncio.to_thread``), each one taking the next text as soon as its
    previous call returned. Returns per-call latencies, wall time, CPU time
    and the worst event-loop stall."""
    queue = list(texts)
    latencies: list[float] = []

    async def client(encode: Client) -> None:
        while queue:
            text = queue.pop()
            t = time.perf_counter()
            if inspect.iscoroutinefunction(encode):
                await encode(text)
            else:
                await asyncio.to_thread(encode, text)
            latencies.append(time.perf_counter() - t)

    ticker = Loop()
    tick_task = asyncio.create_task(ticker.run())
    t0, cpu0 = time.perf_counter(), time.process_time()
    await asyncio.gather(*(client(c) for c in clients))
    wall, cpu = time.perf_counter() - t0, time.process_time() - cpu0
    tick_task.cancel()
    await tick_task
    return latencies, wall, cpu, ticker.max_lag_ms


def stats(latencies: list[float]) -> tuple[float, float]:
    if len(latencies) < 2:
        return latencies[0] * 1000, latencies[0] * 1000
    q = quantiles(latencies, n=20)  # 5 % steps
    return q[9] * 1000, q[18] * 1000


def timed(fn: Callable[[list[float]], None]) -> Result:
    """Call ``fn`` with a list it appends per-request latencies to; return
    them with wall time and CPU time."""
    latencies: list[float] = []
    t0, cpu0 = time.perf_counter(), time.process_time()
    fn(latencies)
    return latencies, time.perf_counter() - t0, time.process_time() - cpu0, None


class Bench:
    def __init__(self, embedders: list[BGEM3Embedder], repeat: int) -> None:
        self.embedders = embedders
        self.repeat = repeat

    def report(self, mode: str, n: int, run: Callable[[], Result]) -> None:
        """Run ``run`` ``repeat`` times and print the last result as a row."""
        result = run()
        for _ in range(self.repeat - 1):
            result = run()
        latencies, wall, cpu, lag = result
        p50, p95 = stats(latencies)
        lag_s = f"{lag:.0f}" if lag is not None else ""
        print(
            f"| {mode} | {n / wall:.1f} | {p50:.0f} | {p95:.0f} | "
            f"{cpu / n * 1000:.0f} | {lag_s} | {peak_rss_mib():.0f} |",
            flush=True,
        )

    def sequential(self, name: str, texts: list[str]) -> None:
        emb = self.embedders[0]

        def run(latencies: list[float]) -> None:
            for text in texts:
                t = time.perf_counter()
                emb.encode(text)
                latencies.append(time.perf_counter() - t)

        self.report(f"{name} sequential", len(texts), lambda: timed(run))

    def concurrent(self, name: str, texts: list[str], concurrency: int) -> None:
        # Client k uses session k % len(embedders): one shared session is the
        # `to_thread` pattern, N sessions give every client its own queue.
        embs = self.embedders
        clients: list[Client] = [embs[k % len(embs)].encode for k in range(concurrency)]
        label = f"{name} to_thread ×{concurrency}"
        if len(embs) > 1:
            label += f", {len(embs)} sessions"
        self.report(label, len(texts), lambda: asyncio.run(closed_loop(texts, clients)))

    def served(
        self, name: str, texts: list[str], concurrency: int, window_ms: float
    ) -> None:
        """Closed-loop clients through ``AsyncEmbedder`` (its default
        ``max_concurrency``), with or without the micro-batch window."""

        async def run() -> Result:
            async with AsyncEmbedder(
                self.embedders[0], batch_window_ms=window_ms
            ) as served:
                # every client awaits the same coroutine function: the batcher
                # sees ``concurrency`` requests per round
                return await closed_loop(texts, [served.encode] * concurrency)

        label = f"{name} AsyncEmbedder ×{concurrency}"
        if window_ms:
            label += f", window {window_ms:g} ms"
        self.report(label, len(texts), lambda: asyncio.run(run()))

    def batched(self, name: str, texts: list[str], batch: int) -> None:
        """``encode`` on ``batch`` texts at a time: the micro-batcher's view,
        every request in a batch waits for the batch."""
        emb = self.embedders[0]

        def run(latencies: list[float]) -> None:
            for i in range(0, len(texts), batch):
                chunk = texts[i : i + batch]
                t = time.perf_counter()
                emb.encode(chunk, batch_size=len(chunk))
                latencies.extend([time.perf_counter() - t] * len(chunk))

        self.report(
            f"{name} encode(list) batch {batch}", len(texts), lambda: timed(run)
        )


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("--precision", default="fp32", choices=["fp32", "int8"])
    ap.add_argument("--model", type=Path, help="backbone ONNX file (skips --precision)")
    ap.add_argument("--concurrency", default="1,2,4,8")
    ap.add_argument(
        "--sessions", type=int, default=1, help="split the threads over N sessions"
    )
    ap.add_argument(
        "--threads",
        type=int,
        default=None,
        help="intra-op threads (default: the package default)",
    )
    ap.add_argument("--queries", type=int, default=40)
    ap.add_argument("--passages", type=int, default=16)
    ap.add_argument(
        "--passage-tokens",
        type=int,
        default=128,
        help="length of a passage; 512+ makes peak RSS versus concurrency visible",
    )
    ap.add_argument(
        "--repeat", type=int, default=2, help="runs per mode, the last one is reported"
    )
    ap.add_argument("--low-memory", action="store_true")
    ap.add_argument(
        "--batch-window-ms",
        type=float,
        default=10.0,
        help="micro-batch window for the AsyncEmbedder rows",
    )
    args = ap.parse_args()
    concurrency = [int(c) for c in args.concurrency.split(",") if c]

    kw: dict = dict(quiet=True, low_memory=args.low_memory)
    if args.model:  # the file name tells AsyncEmbedder which defaults to use
        kw["model_path"] = args.model
        kw["precision"] = "int8" if "int8" in args.model.name else "fp32"
    else:
        kw["precision"] = args.precision
    t0 = time.perf_counter()
    first = BGEM3Embedder(num_threads=args.threads, **kw)
    assert first.backbone.session is not None
    threads = first.backbone.session.get_session_options().intra_op_num_threads
    embedders = [first]
    if args.sessions > 1:  # the same thread budget, split over N sessions
        first.close()
        threads = max(1, threads // args.sessions)
        embedders = [
            BGEM3Embedder(num_threads=threads, **kw) for _ in range(args.sessions)
        ]
    load_s = time.perf_counter() - t0
    name = args.model.name if args.model else args.precision
    print(
        f"\n#### serving {name}: {len(embedders)} session(s) × {threads} threads, "
        f"load {load_s:.2f} s, rss {rss_mib():.0f} MiB, python {sys.version.split()[0]}, "
        f"os {sys.platform} cpus {os.cpu_count()}\n"
    )
    queries = [QUERY] * args.queries
    passage = SENTENCE * (args.passage_tokens * 12 // 128)  # same text as eval_model
    passages = [passage] * args.passages
    ntok = len(first.tokenize([passage])[0])
    for emb in embedders:  # warm up: memory patterns for both shapes
        emb.encode(QUERY)
        emb.encode(passage)

    bench = Bench(embedders, args.repeat)
    print(
        "| mode | req/s | p50 ms | p95 ms | cpu ms/req | loop lag max ms | peak rss MiB |"
    )
    print("|---|---|---|---|---|---|---|")
    if len(embedders) == 1:
        bench.sequential("query", queries)
    for c in concurrency:
        bench.concurrent("query", queries, c)
    if len(embedders) == 1:
        for b in sorted({4, 8, 16, args.queries}):
            if b <= args.queries:
                bench.batched("query", queries, b)
        for c in concurrency:
            bench.served("query", queries, c, 0)
            bench.served("query", queries, c, args.batch_window_ms)
        bench.sequential("passage", passages)
    for c in concurrency:
        bench.concurrent("passage", passages, c)
    if len(embedders) == 1:
        bench.batched("passage", passages, args.passages)
        bench.served("passage", passages, 4, 0)
        bench.served("passage", passages, 4, args.batch_window_ms)
    print(
        f"\n(query = {args.queries} × 9-token texts, passage = {args.passages} × "
        f"{ntok}-token texts; latency as seen by a closed-loop client; loop lag = "
        f"worst delay of a 10 ms ticking coroutine; peak rss = ru_maxrss so far)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
