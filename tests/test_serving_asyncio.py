"""The asyncio scheduling facts ``serving.py`` relies on, one test each
(docs/serving/asyncio.md; ``tools/asyncio_probe.py`` reproduces them without
the embedder). Every test is green on Python 3.11 and 3.13; ``pytest-timeout``
turns a hang into a failure."""

import asyncio
import sys

import pytest

from bge_m3_lite.serving import AsyncEmbedder
from tests.test_serving import FakeEmbedder, Ticker, occupy, run


async def settle(emb: AsyncEmbedder, **state: int) -> None:
    """Poll until the counters reach ``state`` (Windows timers tick every 15 ms)."""
    for _ in range(200):
        if all(getattr(emb, k) == v for k, v in state.items()):
            return
        await asyncio.sleep(0.001)
    raise AssertionError(f"never reached {state}")


def test_close_returns_with_a_done_task_awaiting_its_callback():
    """v0.6 regression: a finished batch task stays in ``_running`` until its
    done-callback runs one iteration later; ``await gather(*running)`` does
    not yield on 3.12+ for done tasks (gh-104144), so close() spun forever."""
    fake = FakeEmbedder(delay=0.0)

    async def main():
        emb = AsyncEmbedder(fake, max_concurrency=1, batch_window_ms=20)
        request = asyncio.create_task(emb.encode("a"))
        while not emb._running:  # the burst flush created the batch task
            await asyncio.sleep(0)
        batch_task = next(iter(emb._running))
        while not batch_task.done():
            await asyncio.sleep(0)
        assert batch_task in emb._running  # done, its discard is still queued
        await emb.close()  # must not hang
        assert not emb._running
        return await request

    assert run(main())["dense_vecs"][0] == 1


def test_slot_waiters_run_in_arrival_order():
    """``Semaphore`` hands released slots to waiters FIFO; a newcomer queues
    behind them (both versions; the 3.13 rewrite keeps the order)."""
    fake = FakeEmbedder(delay=0.01)

    async def main():
        async with AsyncEmbedder(fake, max_concurrency=1) as emb:
            first = [asyncio.create_task(emb.encode(f"q{i}")) for i in range(4)]
            await settle(emb, in_flight=1, queue_depth=3)
            late = asyncio.create_task(emb.encode("late"))
            await asyncio.gather(*first, late)

    run(main())
    assert [t[0] for t, _ in fake.calls] == ["q0", "q1", "q2", "q3", "late"]


def test_cancelled_slot_waiter_leaves_the_queue():
    """A request cancelled while waiting for a slot drops out of
    ``queue_depth`` and never runs; the slot goes to the next waiter."""
    fake = FakeEmbedder(delay=0.05)

    async def main():
        emb = AsyncEmbedder(fake, max_concurrency=1)
        blocker = await occupy(emb)
        waiting = asyncio.create_task(emb.encode("cancelled"))
        await settle(emb, queue_depth=1)
        waiting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiting
        assert (emb.in_flight, emb.queue_depth) == (1, 0)
        await emb.encode("after")
        await blocker
        await emb.close()

    run(main())
    assert [t[0] for t, _ in fake.calls] == ["blocker", "after"]


def test_timed_out_request_keeps_its_slot_until_the_call_returns():
    """``asyncio.timeout`` cancels the awaiting task at once, but the thread
    keeps running ``encode``: the slot stays taken, ``in_flight`` counts it,
    the next request waits for it, and close() waits in the loop instead of
    blocking it in ``executor.shutdown``."""
    fake = FakeEmbedder(delay=0.2)

    async def main():
        emb = AsyncEmbedder(fake, max_concurrency=1)
        ticker = Ticker()
        tick = asyncio.create_task(ticker.run())
        with pytest.raises(TimeoutError):
            async with asyncio.timeout(0.01):
                await emb.encode("slow")
        assert (emb.in_flight, emb.queue_depth) == (1, 0)
        after = asyncio.create_task(emb.encode("after"))
        await settle(emb, queue_depth=1)
        assert emb.in_flight == 1  # the abandoned call still holds the slot
        await emb.close()  # drains "after" too
        assert (emb.in_flight, emb.queue_depth) == (0, 0)
        tick.cancel()
        await tick
        return (await after)["dense_vecs"][0], ticker.max_lag

    value, lag = run(main())
    assert value == 5
    assert lag < 0.1  # close() never blocked the loop for the 0.2 s call
    assert [t[0] for t, _ in fake.calls] == ["slow", "after"]
    assert fake.peak_in_flight == 1


def test_cancelled_batch_waiter_does_not_break_the_batch():
    """A waiter cancelled while its batch runs is skipped when the results
    are split (``set_result`` on a cancelled future raises); the others get
    theirs."""
    fake = FakeEmbedder(delay=0.05)

    async def main():
        async with AsyncEmbedder(fake, max_concurrency=1, batch_window_ms=500) as emb:
            blocker = await occupy(emb)
            results = await asyncio.gather(
                asyncio.wait_for(emb.encode("a"), 0.01),
                emb.encode("bb"),
                blocker,
                return_exceptions=True,
            )
            return results

    a, bb, _ = run(main())
    assert isinstance(a, TimeoutError)
    assert isinstance(bb, dict) and bb["dense_vecs"][0] == 2
    # 3.11's wait_for wraps its coroutine in a task one iteration later than
    # gather does (3.12 awaits it directly, gh-96764): the batch order differs
    assert [sorted(texts) for texts, _ in fake.calls] == [["blocker"], ["a", "bb"]]


@pytest.mark.skipif(sys.version_info < (3, 12), reason="eager_task_factory is 3.12+")
def test_burst_leaves_together_under_eager_tasks():
    """With ``asyncio.eager_task_factory`` a task runs inside ``create_task``;
    the burst still leaves as one call because the flush is a ``call_soon``
    callback, which only the loop runs."""
    fake = FakeEmbedder(delay=0.0)

    async def main():
        factory = getattr(asyncio, "eager_task_factory")  # noqa: B009 (pyright on 3.11)
        asyncio.get_running_loop().set_task_factory(factory)
        async with AsyncEmbedder(fake, max_concurrency=2, batch_window_ms=500) as emb:
            await asyncio.gather(*(emb.encode(f"q{i}") for i in range(8)))

    run(main())
    assert [len(texts) for texts, _ in fake.calls] == [8]
