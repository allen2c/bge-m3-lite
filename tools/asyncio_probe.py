"""Minimal reproductions of the asyncio scheduling facts ``serving.py`` relies on.

Usage:  uv run --no-project -p 3.11 tools/asyncio_probe.py [PROBE ...]
        uv run --no-project -p 3.13 tools/asyncio_probe.py

Each probe prints one line ``name: result``; docs/serving/asyncio.md records
the lines per interpreter next to the assumption they test and the CPython
change behind any difference. Needs nothing but the standard library, so it
runs on any interpreter ``uv`` can find.
"""

from __future__ import annotations

import asyncio
import inspect
import sys
import threading
import time
from collections.abc import Awaitable, Callable
from concurrent.futures import ThreadPoolExecutor

PROBES: dict[str, Callable[[], Awaitable[str]]] = {}


def probe(fn: Callable[[], Awaitable[str]]) -> Callable[[], Awaitable[str]]:
    PROBES[fn.__name__] = fn
    return fn


async def _iterations(aw: Awaitable[object]) -> int:
    """How many loop iterations pass while awaiting ``aw`` (0 = no yield)."""
    loop = asyncio.get_running_loop()
    count = 0
    stop = False

    def tick() -> None:
        nonlocal count
        if not stop:
            count += 1
            loop.call_soon(tick)

    loop.call_soon(tick)
    await aw
    stop = True
    return count


async def _done_task() -> asyncio.Task[None]:
    async def nothing() -> None:
        pass

    task = asyncio.create_task(nothing())
    while not task.done():  # step it without a Future callback of our own
        await asyncio.sleep(0)
    return task


@probe
async def gather_done() -> str:
    """close() in v0.6 looped on ``await gather(*running)`` with every task
    done but not yet discarded by its done-callback: on 3.12+ gather of done
    futures completes without yielding, so the loop never ran the callback."""
    tasks = [await _done_task() for _ in range(2)]
    yields = await _iterations(asyncio.gather(*tasks))
    wait_yields = await _iterations(asyncio.wait(tasks))
    return f"gather yields {yields} iteration(s), wait yields {wait_yields}"


@probe
async def close_drain_shape() -> str:
    """The v0.6 drain loop (``gather``) versus the v0.6.1 one (``wait`` on
    unfinished futures, else ``sleep(0)``): iterations until ``running``
    empties, with a task whose discard callback is still queued."""

    async def drain(use_gather: bool) -> str:
        running: set[asyncio.Future[None]] = set()
        task = asyncio.create_task(asyncio.sleep(0))
        running.add(task)
        task.add_done_callback(running.discard)
        while not task.done():  # done, its discard callback still queued
            await asyncio.sleep(0)
        rounds = 0
        while running:
            rounds += 1
            if rounds > 1000:
                return "hangs"
            unfinished = [f for f in running if not f.done()]
            if use_gather:
                await asyncio.gather(*running)
            elif unfinished:
                await asyncio.wait(unfinished)
            else:
                await asyncio.sleep(0)
        return f"{rounds} round(s)"

    return f"gather: {await drain(True)}; wait/sleep(0): {await drain(False)}"


@probe
async def burst_leaves_together() -> str:
    """``_enqueue``: requests created in one iteration join the group before
    the ``call_soon`` flush runs (tasks start on later iterations, in order)."""
    loop = asyncio.get_running_loop()
    log: list[str] = []

    async def request(i: int) -> None:
        if not log:
            loop.call_soon(log.append, "flush")
        log.append(f"r{i}")

    await asyncio.gather(*(request(i) for i in range(4)))
    await asyncio.sleep(0)  # eager tasks finish before the flush runs
    return " ".join(log)


@probe
async def eager_burst() -> str:
    """The same burst under ``asyncio.eager_task_factory`` (3.12+): tasks run
    inside ``create_task``, the flush still waits for the loop."""
    factory = getattr(asyncio, "eager_task_factory", None)
    if factory is None:
        return "n/a (no eager_task_factory)"
    loop = asyncio.get_running_loop()
    loop.set_task_factory(factory)
    try:
        return await burst_leaves_together()
    finally:
        loop.set_task_factory(None)


@probe
async def soon_vs_later() -> str:
    """Ordering of ``call_soon`` and ``call_later(0)`` scheduled in one
    iteration, and whether a callback scheduled by a callback runs in the same
    iteration (``_run_once`` snapshots the ready queue)."""
    loop = asyncio.get_running_loop()
    log: list[str] = []
    loop.call_later(0, log.append, "later0")
    loop.call_soon(log.append, "soon")
    loop.call_soon(lambda: loop.call_soon(log.append, "nested"))
    loop.call_soon(log.append, "soon2")
    await asyncio.sleep(0)
    first = list(log)
    await asyncio.sleep(0)
    return f"iteration 1: {' '.join(first)}; iteration 2: {' '.join(log[len(first) :])}"


@probe
async def later_min_delay() -> str:
    """``call_later(window)`` fires no earlier than the window; the overshoot
    is the timer resolution (15 ms on Windows)."""
    loop = asyncio.get_running_loop()
    worst = 0.0
    for _ in range(20):
        fut = loop.create_future()
        t0 = time.perf_counter()
        loop.call_later(0.002, fut.set_result, None)
        await fut
        worst = max(worst, time.perf_counter() - t0 - 0.002)
    return f"2 ms window: worst overshoot {worst * 1000:.2f} ms"


@probe
async def semaphore_order() -> str:
    """``_run``: released slots go to waiters in FIFO order; a newcomer that
    calls ``acquire`` while waiters exist queues behind them."""
    sem = asyncio.Semaphore(1)
    log: list[str] = []
    await sem.acquire()

    async def waiter(name: str) -> None:
        async with sem:
            log.append(name)

    tasks = [asyncio.create_task(waiter(f"w{i}")) for i in range(3)]
    await asyncio.sleep(0)  # all three are queued
    sem.release()
    sem.release()  # one extra permit: the newcomer sees value > 0
    if not sem.locked():
        log.append("newcomer-jumped")
    tasks.append(asyncio.create_task(waiter("newcomer")))
    await asyncio.gather(*tasks)
    return " ".join(log)


@probe
async def semaphore_cancel() -> str:
    """A waiter cancelled after being woken (its slot handed over) must pass
    the slot on; a waiter cancelled while queued must not consume one."""
    sem = asyncio.Semaphore(1)
    await sem.acquire()
    log: list[str] = []

    async def waiter(name: str) -> None:
        async with sem:
            log.append(name)

    first = asyncio.create_task(waiter("first"))
    second = asyncio.create_task(waiter("second"))
    await asyncio.sleep(0)
    sem.release()  # wakes `first` (its future is set) ...
    first.cancel()  # ... but it is cancelled before it resumes
    await asyncio.gather(first, second, return_exceptions=True)
    queued = asyncio.create_task(waiter("queued"))
    await sem.acquire()
    await asyncio.sleep(0)
    queued.cancel()
    await asyncio.gather(queued, return_exceptions=True)
    sem.release()
    return f"{' '.join(log)}; value after cancels {sem._value}"


@probe
async def future_callback_lag() -> str:
    """``task.add_done_callback(running.discard)``: the callback runs one
    iteration after the task is done, never inline."""
    running: set[asyncio.Task[None]] = set()
    task = asyncio.create_task(asyncio.sleep(0))
    running.add(task)
    task.add_done_callback(running.discard)
    lag = -1
    for i in range(10):
        if task.done() and lag < 0:
            lag = i
        if not running:
            return f"done at iteration {lag}, discarded at {i}"
        await asyncio.sleep(0)
    return "callback never ran"


@probe
async def executor_completion() -> str:
    """``run_in_executor`` futures complete on the loop thread, in the order
    the worker threads finish (each posts ``call_soon_threadsafe``)."""
    loop = asyncio.get_running_loop()
    log: list[str] = []
    finished = threading.Event()

    def work(name: str, seconds: float) -> str:
        time.sleep(seconds)
        finished.set()
        return name

    with ThreadPoolExecutor(2) as pool:
        slow = loop.run_in_executor(pool, work, "slow", 0.05)
        fast = loop.run_in_executor(pool, work, "fast", 0.01)
        for fut in (slow, fast):
            fut.add_done_callback(lambda f: log.append(f.result()))
        finished.wait()  # blocks the loop thread until `fast` returned
        seen_early = fast.done()
        await asyncio.gather(slow, fast)
    return f"order {' '.join(log)}; done() before the loop ran: {seen_early}"


@probe
async def cancel_running_call() -> str:
    """``asyncio.timeout`` around a call whose thread is running: the
    awaiting task gets ``TimeoutError``, the thread keeps running, the
    concurrent future cannot be cancelled."""
    loop = asyncio.get_running_loop()
    done = threading.Event()

    def work() -> None:
        time.sleep(0.05)
        done.set()

    with ThreadPoolExecutor(1) as pool:
        fut = loop.run_in_executor(pool, work)
        t0 = time.perf_counter()
        try:
            async with asyncio.timeout(0.01):
                await fut
        except TimeoutError:
            outcome = "TimeoutError"
        else:
            outcome = "no error"
        raised_after = time.perf_counter() - t0
        task = asyncio.current_task()
        assert task is not None
        cancelling = task.cancelling()
        thread_done = done.is_set()
        done.wait()
    return (
        f"{outcome} after {raised_after * 1000:.0f} ms, fut.cancelled()={fut.cancelled()},"
        f" thread finished at raise: {thread_done}, cancelling()={cancelling}"
    )


@probe
async def cancel_batch_waiter() -> str:
    """A cancelled waiter's future: ``set_result`` on it raises, so the batch
    must check ``done()`` first; the other waiters are unaffected."""
    loop = asyncio.get_running_loop()
    a, b = loop.create_future(), loop.create_future()

    async def wait_on(fut: asyncio.Future[int]) -> int:
        return await fut

    ta = asyncio.create_task(asyncio.wait_for(a, 0.01))
    tb = asyncio.create_task(wait_on(b))
    await asyncio.sleep(0.03)
    try:
        a.set_result(1)
        a_set = "set_result ok"
    except asyncio.InvalidStateError as exc:
        a_set = f"InvalidStateError: {exc}"
    b.set_result(2)
    results = await asyncio.gather(ta, tb, return_exceptions=True)
    return f"a.cancelled()={a.cancelled()}, {a_set}; results {[type(r).__name__ if isinstance(r, BaseException) else r for r in results]}"


@probe
async def wait_for_order() -> str:
    """Arrival order of a burst when one request is wrapped in ``wait_for``:
    3.11 wraps the coroutine in a second task (one iteration later), 3.12+
    awaits it inside ``timeout()`` (gh-96764), so it keeps its place."""
    log: list[str] = []

    async def request(name: str) -> None:
        log.append(name)

    await asyncio.gather(asyncio.wait_for(request("a"), 1), request("b"))
    return " ".join(log)


@probe
async def executor_shutdown() -> str:
    """``executor.shutdown(wait=True)`` with a thread still running blocks
    the loop thread for the rest of the call; a worker completing after the
    loop closed does not raise."""
    loop = asyncio.get_running_loop()
    pool = ThreadPoolExecutor(1)
    fut = loop.run_in_executor(pool, time.sleep, 0.05)
    t0 = time.perf_counter()
    pool.shutdown(wait=True)
    blocked = time.perf_counter() - t0
    await fut
    return f"shutdown(wait=True) blocked the loop {blocked * 1000:.0f} ms"


def leftover_thread_after_loop_close() -> str:
    """A private executor call still running when ``asyncio.run`` returns:
    the loop closes without waiting, the completion callback is dropped."""
    pool = ThreadPoolExecutor(1)
    done = threading.Event()

    def work() -> None:
        time.sleep(0.05)
        done.set()

    async def main() -> asyncio.Future[None]:
        return asyncio.get_running_loop().run_in_executor(pool, work)

    t0 = time.perf_counter()
    fut = asyncio.run(main())
    returned = time.perf_counter() - t0
    done.wait()
    time.sleep(0.01)
    pool.shutdown()
    return (
        f"asyncio.run returned after {returned * 1000:.0f} ms, fut.done()={fut.done()},"
        f" cancelled={fut.cancelled()}"
    )


def main(argv: list[str]) -> None:
    names = argv or [*PROBES, "leftover_thread_after_loop_close"]
    print(f"python {sys.version.split()[0]} ({sys.platform})")
    for name in names:
        if name == "leftover_thread_after_loop_close":
            result = leftover_thread_after_loop_close()
        else:
            fn = PROBES[name]
            assert inspect.iscoroutinefunction(fn)
            result = asyncio.run(fn())
        print(f"{name}: {result}")


if __name__ == "__main__":
    main(sys.argv[1:])
