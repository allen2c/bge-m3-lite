# asyncio facts behind `AsyncEmbedder` (Python 3.11 versus 3.12+)

`serving.py` schedules with `call_soon`, `call_later`, a `Semaphore`,
`run_in_executor` futures and done-callbacks. Every fact it relies on is
reproduced by `tools/asyncio_probe.py` (standard library only) and pinned by
a test in `tests/test_serving_asyncio.py` or `tests/test_serving.py`; both
suites run on 3.11 and 3.13 in CI. Measured 2026-09-07 on 3.11.15, 3.12.13,
3.13.7 and 3.14.6 (macOS): **the boundary is 3.12** — 3.12, 3.13 and 3.14
print identical probe lines. The 3.13 `Semaphore` rewrite (gh-111693)
changes only the cancel bookkeeping, not the order.

```bash
uv run --no-project -p 3.11 tools/asyncio_probe.py   # any interpreter uv can find
```

## Assumptions, behaviour per version, the test that pins each

| assumption (where) | 3.11 | 3.12+ | CPython change | pinned by |
|---|---|---|---|---|
| `await gather(*done_tasks)` yields, so a drain loop lets done-callbacks run (`close`, v0.6) | yields 1 iteration | **does not yield**: gather of done futures completes eagerly; the v0.6 loop spun forever | gh-104144 (PR #104138, 3.12.0; part of the eager-task work gh-97696) | `test_close_returns_with_a_done_task_awaiting_its_callback`; every `close()` test under `pytest-timeout` |
| `asyncio.wait(futs)` always yields (`close`, v0.6.1) | 2 iterations | 2 iterations | — | same |
| a task's done-callback (`_running.discard`) runs one iteration after `done()`, never inline | done at 2, discarded at 3 | same | — | same |
| a burst created in one iteration joins the pending group before the `call_soon` flush runs (`_enqueue`) | r0 r1 r2 r3 flush | same; also under `eager_task_factory` (tasks run inside `create_task`, the flush still waits for the loop) | `eager_task_factory`: gh-97696 (3.12.0) | `test_micro_batcher_takes_a_burst_as_one_call`, `test_burst_leaves_together_under_eager_tasks` |
| `call_soon` runs before a `call_later(0)` of the same iteration; a callback scheduled by a callback waits for the next iteration | soon soon2 · later0 nested | same | — | `test_micro_batcher_window_and_full_batch_release_early` |
| `call_later(window)` fires no earlier than the window | overshoot 0.6 ms (15 ms on Windows) | same | — | same (tests poll instead of sleeping) |
| released slots go to waiters FIFO, a newcomer queues behind them (`_run`) | w0 w1 w2 newcomer | same | FIFO since 3.11.0: gh-90155 (PR #93222) + gh-97545; 3.13.0 rewrote `acquire` again in gh-111693 (PR #111694), same order | `test_slot_waiters_run_in_arrival_order`, the window test's call order |
| a waiter cancelled while queued takes no slot; one cancelled after being woken passes the slot on | second; value 1 | same | gh-90155 (3.11.0; earlier versions could lose the slot) | `test_cancelled_slot_waiter_leaves_the_queue` |
| `run_in_executor` futures complete on the loop thread, in thread-finish order; `done()` flips only once the loop ran | fast slow; False | same | — | `test_back_pressure_bounds_in_flight_and_counts_queue` |
| cancelling the awaiting task (`asyncio.timeout`) cancels the future at once; the thread runs on; `cancelling()` is 0 afterwards | TimeoutError after 12 ms, thread still running | same | `Task.uncancel` / `cancelling` (3.11.0); `timeout()` uncancels since gh-102780 (3.11.3) | `test_timed_out_request_keeps_its_slot_until_the_call_returns` |
| `set_result` on a cancelled waiter future raises `InvalidStateError` (`_run_batch` checks `done()`) | raises | same | — | `test_cancelled_batch_waiter_does_not_break_the_batch` |
| a request wrapped in `wait_for` arrives with its burst | **one iteration later** (`wait_for` wraps the coroutine in a second task): batch order b a | in place: a b (`wait_for` awaits inside `timeout()`) | gh-96764 (3.12.0) | same test (order-insensitive) |
| `executor.shutdown(wait=True)` with a running thread blocks the loop thread (`close` drains first) | 55 ms for a 50 ms call | same | — | the timed-out test's loop ticker |
| a call still running when the loop closes: its completion is dropped silently | fut never done, no error | same | — | not needed: `close()` drains before the pool shuts down |

## What v0.6.2 changed in `serving.py`

`_run` awaits `asyncio.shield(fut)` and frees the slot from the future's
done-callback (`_call_returned`), not from the awaiting coroutine's
`finally`. Before, a caller cancelled by `asyncio.timeout` released the slot
while `session.run` was still running: `in_flight` undercounted, the next
request queued inside the thread pool behind the abandoned call, and
`close()` blocked the loop in `executor.shutdown(wait=True)` for the rest of
it. Now the slot, `in_flight` and `_running` follow the thread; the cancelled
caller gets its `CancelledError` immediately. Probe: `cancel_running_call`,
`executor_shutdown`. The results are unchanged (the same `encode` call, the
same split), and the closed-loop rates are within noise (`measurements.md`).

## Probe output (the lines that differ)

```
python 3.11.15     gather_done: gather yields 1 iteration(s), wait yields 2
python 3.13.7      gather_done: gather yields 0 iteration(s), wait yields 2
python 3.11.15     close_drain_shape: gather: 1 round(s); wait/sleep(0): 1 round(s)
python 3.13.7      close_drain_shape: gather: hangs; wait/sleep(0): 1 round(s)
python 3.11.15     eager_burst: n/a (no eager_task_factory)
python 3.13.7      eager_burst: r0 r1 r2 r3 flush
python 3.11.15     wait_for_order: b a
python 3.13.7      wait_for_order: a b
```

Identical on every version: `burst_leaves_together: r0 r1 r2 r3 flush`,
`soon_vs_later: iteration 1: soon soon2; iteration 2: later0 nested`,
`semaphore_order: w0 w1 w2 newcomer`, `semaphore_cancel: second; value after
cancels 1`, `future_callback_lag: done at iteration 2, discarded at 3`,
`executor_completion: order fast slow; done() before the loop ran: False`,
`cancel_running_call: TimeoutError after 12 ms, fut.cancelled()=True, thread
finished at raise: False, cancelling()=0`, `cancel_batch_waiter:
a.cancelled()=True, InvalidStateError`, `executor_shutdown: blocked the loop
55–60 ms`, `leftover_thread_after_loop_close: fut.done()=False`.

Rules that follow: never `await gather()` on futures that may already be
done inside a drain loop (use `wait` on the unfinished ones, `sleep(0)`
otherwise); never release a slot from a coroutine that can be cancelled;
tests assert scheduling order only after polling the counters, never after a
fixed `sleep`.
