import asyncio
import threading
import time
from collections.abc import Sequence
from typing import Any

import numpy as np
import pytest

from bge_m3_lite.embedder import BGEM3Embedder
from bge_m3_lite.serving import (
    DEFAULT_BATCH_WINDOW_MS,
    DEFAULT_CONCURRENCY,
    LENGTH_BUCKETS,
    AsyncEmbedder,
    length_bucket,
)


class FakeEmbedder(BGEM3Embedder):
    """Stands in for the model: ``encode`` sleeps (releasing the GIL like
    ``session.run``) and returns per-text outputs derived from the text."""

    def __init__(self, delay: float = 0.02, precision: str = "int8") -> None:
        self.precision = precision  # type: ignore[assignment]
        self.query_max_length = 512
        self.passage_max_length = 8192
        self.delay = delay
        self.calls: list[tuple[list[str], dict[str, Any]]] = []
        self.peak_in_flight = 0
        self.closed = False
        self._in_flight = 0
        self._lock = threading.Lock()

    def close(self) -> None:
        self.closed = True

    def encode(self, texts: str | Sequence[str], **kw: Any) -> dict[str, Any]:  # type: ignore[override]
        single = isinstance(texts, str)
        items = [texts] if single else list(texts)
        with self._lock:
            self._in_flight += 1
            self.peak_in_flight = max(self.peak_in_flight, self._in_flight)
            self.calls.append((items, kw))
        try:
            time.sleep(self.delay)
            if any(t == "boom" for t in items):
                raise ValueError("boom")
        finally:
            with self._lock:
                self._in_flight -= 1
        dense = np.array([[len(t), hash(t) % 97, 1.0] for t in items], dtype=np.float32)
        out: dict[str, Any] = {
            "dense_vecs": dense if kw.get("return_dense", True) else None,
            "lexical_weights": (
                [{"1": float(len(t))} for t in items]
                if kw.get("return_sparse")
                else None
            ),
            "colbert_vecs": (
                [np.full((len(t), 2), len(t), dtype=np.float32) for t in items]
                if kw.get("return_colbert_vecs")
                else None
            ),
        }
        if single:
            out = {k: (v[0] if v is not None else None) for k, v in out.items()}
        return out


class Ticker:
    """Wakes up every ``period`` seconds and records the worst delay."""

    def __init__(self, period: float = 0.005) -> None:
        self.period = period
        self.max_lag = 0.0

    async def run(self) -> None:
        try:
            while True:
                t = time.perf_counter()
                await asyncio.sleep(self.period)
                self.max_lag = max(self.max_lag, time.perf_counter() - t - self.period)
        except asyncio.CancelledError:
            pass


def run(coro):
    return asyncio.run(coro)


def test_constructor_validation():
    fake = FakeEmbedder()
    with pytest.raises(TypeError):
        AsyncEmbedder(fake, precision="int8")
    with pytest.raises(ValueError):
        AsyncEmbedder(fake, max_concurrency=0)
    with pytest.raises(ValueError):
        AsyncEmbedder(fake, batch_window_ms=-1)
    for precision in ("fp32", "int8"):
        emb = AsyncEmbedder(FakeEmbedder(precision=precision))
        assert emb.max_concurrency == DEFAULT_CONCURRENCY[precision]
        assert emb.batch_window_ms == DEFAULT_BATCH_WINDOW_MS[precision]
    assert DEFAULT_BATCH_WINDOW_MS["int8"] == 0  # batching loses on int8


def test_same_outputs_as_sync_api():
    fake = FakeEmbedder(delay=0.0)

    async def main():
        async with AsyncEmbedder(fake, max_concurrency=2) as emb:
            single = await emb.encode("hello", return_sparse=True)
            many = await emb.encode_queries(["hello", "hi"], return_colbert_vecs=True)
            corpus = await emb.encode_corpus(["a passage"], return_dense=False)
        return single, many, corpus

    single, many, corpus = run(main())
    kws = [kw for _, kw in fake.calls]  # the options and defaults that reached encode
    assert kws[1]["max_length"] == 512 and kws[2]["max_length"] == 8192
    assert all(kw["batch_size"] == 12 and kw["max_batch_tokens"] == 16384 for kw in kws)
    ref_single = fake.encode("hello", return_sparse=True)
    assert single["dense_vecs"].shape == (3,)
    np.testing.assert_array_equal(single["dense_vecs"], ref_single["dense_vecs"])
    assert single["lexical_weights"] == {"1": 5.0}
    assert many["dense_vecs"].shape == (2, 3)
    assert len(many["colbert_vecs"]) == 2 and many["lexical_weights"] is None
    assert corpus["dense_vecs"] is None


def test_event_loop_keeps_ticking_during_encode():
    fake = FakeEmbedder(delay=0.05)

    async def main():
        ticker = Ticker()
        task = asyncio.create_task(ticker.run())
        async with AsyncEmbedder(fake, max_concurrency=4) as emb:
            await asyncio.gather(*(emb.encode(f"q{i}") for i in range(8)))
        task.cancel()
        await task
        return ticker.max_lag

    assert run(main()) < 0.02


def test_back_pressure_bounds_in_flight_and_counts_queue():
    fake = FakeEmbedder(delay=0.03)

    async def main():
        emb = AsyncEmbedder(fake, max_concurrency=3)
        tasks = [asyncio.create_task(emb.encode(f"q{i}")) for i in range(10)]
        await asyncio.sleep(0.01)
        snapshot = (emb.in_flight, emb.queue_depth)
        await asyncio.gather(*tasks)
        after = (emb.in_flight, emb.queue_depth)
        await emb.close()
        return snapshot, after

    snapshot, after = run(main())
    assert snapshot == (3, 7)
    assert after == (0, 0)
    assert fake.peak_in_flight == 3
    assert len(fake.calls) == 10


async def occupy(emb: AsyncEmbedder) -> asyncio.Task:
    """Take the only slot so that the next requests are held and batched."""
    task = asyncio.create_task(emb.encode("blocker"))
    for _ in range(100):  # Windows timers tick every 15 ms: poll instead
        await asyncio.sleep(0.001)
        if emb.in_flight == 1:
            return task
    raise AssertionError("blocker never started")


def test_micro_batcher_starts_at_once_when_a_slot_is_free():
    fake = FakeEmbedder(delay=0.0)

    async def main():
        async with AsyncEmbedder(fake, max_concurrency=2, batch_window_ms=500) as emb:
            t0 = time.perf_counter()
            await emb.encode("a")
            return time.perf_counter() - t0

    assert run(main()) < 0.1
    assert [texts for texts, _ in fake.calls] == [["a"]]


def test_micro_batcher_takes_a_burst_as_one_call():
    fake = FakeEmbedder(delay=0.02)

    async def main():
        async with AsyncEmbedder(fake, max_concurrency=2, batch_window_ms=500) as emb:
            await asyncio.gather(*(emb.encode(f"q{i}") for i in range(8)))
            await emb.encode("alone")  # idle again: no wait, own call

    run(main())
    # requests arriving in one loop iteration leave together, on one slot
    assert [len(texts) for texts, _ in fake.calls] == [8, 1]
    assert fake.peak_in_flight == 1


def test_micro_batcher_merges_requests_and_splits_results():
    fake = FakeEmbedder(delay=0.02)

    async def main():
        emb = AsyncEmbedder(fake, max_concurrency=1, batch_window_ms=500, batch_size=16)
        blocker = await occupy(emb)
        results = await asyncio.gather(
            emb.encode("a"),
            emb.encode(["bb", "ccc"]),
            emb.encode("dddd", return_sparse=True),  # different options: own batch
            emb.encode("eeeee"),
            blocker,
        )
        await emb.close()
        return results

    a, bc, d, e, _ = run(main())
    calls = [texts for texts, _ in fake.calls]
    assert calls[0] == ["blocker"]  # the rest were flushed when its slot freed
    assert sorted(calls[1:]) == [["a", "bb", "ccc", "eeeee"], ["dddd"]]
    assert a["dense_vecs"].shape == (3,) and a["dense_vecs"][0] == 1
    assert bc["dense_vecs"].shape == (2, 3)
    assert bc["dense_vecs"][:, 0].tolist() == [2, 3]
    assert d["lexical_weights"] == {"1": 4.0} and d["dense_vecs"][0] == 4
    assert e["dense_vecs"][0] == 5 and e["lexical_weights"] is None


def test_micro_batcher_keeps_long_texts_out_of_short_batches():
    """A batch is padded to its longest text, so requests only merge within a
    character-length bucket; a burst with one long passage leaves as two
    calls and the short ones do not wait for it."""
    fake = FakeEmbedder(delay=0.02)
    assert LENGTH_BUCKETS == (128, 512, 2048)
    assert [length_bucket([t]) for t in ("", "x" * 128, "x" * 129, "x" * 3000)] == [
        0,
        0,
        1,
        3,
    ]
    assert length_bucket(["short", "x" * 600]) == 2  # a request: its longest text

    async def main():
        emb = AsyncEmbedder(fake, max_concurrency=1, batch_window_ms=500)
        blocker = await occupy(emb)
        await asyncio.gather(
            emb.encode("a"), emb.encode("x" * 600), emb.encode(["bb", "ccc"]), blocker
        )
        await emb.close()

    run(main())
    calls = [texts for texts, _ in fake.calls]
    assert calls[0] == ["blocker"]
    assert sorted(calls[1:]) == [["a", "bb", "ccc"], ["x" * 600]]


def test_micro_batcher_window_and_full_batch_release_early():
    fake = FakeEmbedder(delay=0.1)

    async def main():
        emb = AsyncEmbedder(fake, max_concurrency=1, batch_window_ms=20, batch_size=3)
        blocker = await occupy(emb)
        t0 = time.perf_counter()
        first = [asyncio.create_task(emb.encode(f"q{i}")) for i in range(2)]
        await asyncio.sleep(0.05)  # the window passed: a batch of two is queued
        assert emb.queue_depth == 2 and len(fake.calls) == 1
        more = [asyncio.create_task(emb.encode(f"m{i}")) for i in range(3)]
        await asyncio.sleep(0)  # batch_size reached: flushed without waiting
        assert emb.queue_depth == 5
        await asyncio.gather(blocker, *first, *more)
        await emb.close()
        return time.perf_counter() - t0

    assert run(main()) < 2.0  # three 0.1 s calls; the 20 ms window never stalls
    assert [texts for texts, _ in fake.calls] == [
        ["blocker"],
        ["q0", "q1"],
        ["m0", "m1", "m2"],
    ]


def test_micro_batcher_propagates_errors_to_every_waiter():
    fake = FakeEmbedder(delay=0.02)

    async def main():
        async with AsyncEmbedder(fake, max_concurrency=1, batch_window_ms=50) as emb:
            blocker = await occupy(emb)
            results = await asyncio.gather(
                emb.encode("fine"), emb.encode("boom"), return_exceptions=True
            )
            await blocker
            return results

    results = run(main())
    assert all(isinstance(r, ValueError) for r in results)
    assert [texts for texts, _ in fake.calls] == [["blocker"], ["fine", "boom"]]


def test_close_drains_queue_then_refuses_requests():
    fake = FakeEmbedder(delay=0.02)

    async def main():
        emb = AsyncEmbedder(fake, max_concurrency=1, batch_window_ms=20)
        tasks = [asyncio.create_task(emb.encode(f"q{i}")) for i in range(3)]
        await asyncio.sleep(0)
        await emb.close()
        results = await asyncio.gather(*tasks)
        with pytest.raises(RuntimeError):
            await emb.encode("late")
        await emb.close()  # idempotent
        return results

    results = run(main())
    assert [r["dense_vecs"][0] for r in results] == [2, 2, 2]
    assert fake.closed is False  # a wrapped embedder stays open


def test_owned_embedder_is_closed(monkeypatch):
    fake = FakeEmbedder()
    monkeypatch.setattr("bge_m3_lite.serving.BGEM3Embedder", lambda **kw: fake)

    async def main():
        async with AsyncEmbedder(precision="int8") as emb:
            assert emb.embedder is fake

    run(main())
    assert fake.closed is True


@pytest.mark.slow
def test_real_model_matches_sync_api_bit_exact():
    sync = BGEM3Embedder(quiet=True)
    texts = ["short", "a much longer sentence about retrieval " * 5, "中文", "x" * 300]
    kw = dict(return_sparse=True, return_colbert_vecs=True)

    async def main():
        ticker = Ticker()
        task = asyncio.create_task(ticker.run())
        async with AsyncEmbedder(sync, max_concurrency=2, batch_window_ms=0) as emb:
            direct = await asyncio.gather(*(emb.encode(t, **kw) for t in texts))
        async with AsyncEmbedder(sync, max_concurrency=1, batch_window_ms=50) as emb:
            blocker = await occupy(emb)
            batched = await asyncio.gather(*(emb.encode(t, **kw) for t in texts))
            await blocker
        task.cancel()
        await task
        return direct, batched, ticker.max_lag

    direct, batched, lag = run(main())
    assert lag < 0.02
    # the burst leaves as one padded batch per length bucket (short, 中文 |
    # the sentence, x * 300): each is exactly what encode(list) returns
    groups = {
        b: [i for i, t in enumerate(texts) if length_bucket([t]) == b] for b in (0, 1)
    }
    assert [len(g) for g in groups.values()] == [2, 2]
    for group in groups.values():
        ref_list = sync.encode([texts[i] for i in group], **kw)
        for j, i in enumerate(group):
            assert np.array_equal(batched[i]["dense_vecs"], ref_list["dense_vecs"][j])
            assert batched[i]["lexical_weights"] == ref_list["lexical_weights"][j]
            assert np.array_equal(
                batched[i]["colbert_vecs"], ref_list["colbert_vecs"][j]
            )
    for i, t in enumerate(texts):
        ref = sync.encode(t, **kw)
        assert np.array_equal(direct[i]["dense_vecs"], ref["dense_vecs"])
        assert direct[i]["lexical_weights"] == ref["lexical_weights"]
        assert np.array_equal(direct[i]["colbert_vecs"], ref["colbert_vecs"])
        np.testing.assert_allclose(
            batched[i]["dense_vecs"], ref["dense_vecs"], atol=1e-4
        )
