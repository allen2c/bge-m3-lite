"""asyncio front-end for :class:`BGEM3Embedder`: non-blocking, bounded, batched.

``session.run`` releases the GIL and ``encode`` keeps no mutable state, so one
shared embedder can serve many coroutines. :class:`AsyncEmbedder` runs the
calls in its own thread pool (never the loop thread, never the default
``to_thread`` pool), lets at most ``max_concurrency`` of them run at once and,
with ``batch_window_ms > 0``, merges requests that arrive within the window
into one ``encode`` call (docs/serving.md has the numbers behind the defaults).
"""

from __future__ import annotations

import asyncio
import functools
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from bge_m3_lite.embedder import BATCH_SIZE, MAX_BATCH_TOKENS, BGEM3Embedder

# Measured on the M4 (docs/serving.md): fp32 short queries are GEMM-bound —
# two runs in flight fill the thread gaps, and padding a burst into one call
# is what scales (8 clients: 174 req/s versus 73); int8 runs 2 700 small ops
# per call and scales with runs in flight instead, batching loses there.
DEFAULT_CONCURRENCY = {"fp32": 2, "int8": 4}
DEFAULT_BATCH_WINDOW_MS = {"fp32": 10.0, "int8": 0.0}

_Options = tuple[Any, ...]


class _Pending:
    """Requests with identical options waiting for the batch window to end."""

    def __init__(self, options: _Options) -> None:
        self.options = options
        self.texts: list[str] = []
        self.waiters: list[tuple[asyncio.Future[Any], int, bool]] = []  # n, single
        self.handle: asyncio.Handle | None = None


class AsyncEmbedder:
    """``await``-able :class:`BGEM3Embedder` for FastAPI and other asyncio servers.

    >>> async with AsyncEmbedder(precision="int8") as emb:
    ...     out = await emb.encode("hello")

    ``embedder`` wraps an existing instance (kept open on ``close``); any other
    keyword argument builds one. ``max_concurrency`` bounds the ``encode`` calls
    running at once; further requests wait in ``queue_depth``. With
    ``batch_window_ms > 0`` (the micro-batcher) a request that finds every
    slot busy or claimed is held until a slot frees (at most that long), and
    everything held with the same options (up to ``batch_size`` texts) goes
    into one ``encode`` call: the outputs are what the synchronous ``encode``
    returns for that list, split back per request. A request that finds a free
    slot starts on the next loop iteration together with the rest of its
    burst, so the batcher costs nothing when the server is idle. Both default
    per precision (measured in docs/serving.md): 2 slots and a 10 ms window
    for fp32, 4 slots and no batching for int8.
    """

    def __init__(
        self,
        embedder: BGEM3Embedder | None = None,
        *,
        max_concurrency: int | None = None,
        batch_window_ms: float | None = None,
        batch_size: int = BATCH_SIZE,
        max_batch_tokens: int = MAX_BATCH_TOKENS,
        **embedder_kw: Any,
    ) -> None:
        if embedder is not None and embedder_kw:
            raise TypeError("pass either an embedder or its constructor arguments")
        self._owns_embedder = embedder is None
        self.embedder = BGEM3Embedder(**embedder_kw) if embedder is None else embedder
        precision = self.embedder.precision
        if max_concurrency is None:
            max_concurrency = DEFAULT_CONCURRENCY[precision]
        if batch_window_ms is None:
            batch_window_ms = DEFAULT_BATCH_WINDOW_MS[precision]
        if max_concurrency < 1 or batch_window_ms < 0 or batch_size < 1:
            raise ValueError("max_concurrency, batch_size >= 1; batch_window_ms >= 0")
        self.max_concurrency = max_concurrency
        self.batch_window_ms = batch_window_ms
        self.batch_size = batch_size
        self.max_batch_tokens = max_batch_tokens
        self._executor = ThreadPoolExecutor(
            max_workers=max_concurrency, thread_name_prefix="bge-m3-lite"
        )
        self._slots = asyncio.Semaphore(max_concurrency)
        self._pending: dict[_Options, _Pending] = {}
        self._running: set[asyncio.Future[Any]] = set()
        self._queued = 0  # requests whose call is waiting for a slot
        self._waiting_calls = 0
        self.in_flight = 0  # ``encode`` calls running in the thread pool
        self.closed = False

    @property
    def queue_depth(self) -> int:
        """Requests waiting for a slot or for the batch window."""
        return self._queued + sum(len(p.waiters) for p in self._pending.values())

    # -- public API (same signatures and outputs as BGEM3Embedder) ------------

    async def encode(
        self,
        texts: str | Sequence[str],
        *,
        max_length: int | None = None,
        return_dense: bool = True,
        return_sparse: bool = False,
        return_colbert_vecs: bool = False,
    ) -> dict[str, Any]:
        """:meth:`BGEM3Embedder.encode` without blocking the event loop."""
        if self.closed:
            raise RuntimeError("AsyncEmbedder is closed")
        options: _Options = (
            max_length,
            return_dense,
            return_sparse,
            return_colbert_vecs,
        )
        if self.batch_window_ms > 0:
            return await self._enqueue(texts, options)
        self._claim(1)
        return await self._run(texts, options, 1)

    async def encode_queries(
        self, queries: str | Sequence[str], *, max_length: int | None = None, **kw: Any
    ) -> dict[str, Any]:
        """:meth:`encode` with ``max_length`` defaulting to ``query_max_length``."""
        if max_length is None:
            max_length = self.embedder.query_max_length
        return await self.encode(queries, max_length=max_length, **kw)

    async def encode_corpus(
        self, corpus: str | Sequence[str], *, max_length: int | None = None, **kw: Any
    ) -> dict[str, Any]:
        """:meth:`encode` with ``max_length`` defaulting to ``passage_max_length``."""
        if max_length is None:
            max_length = self.embedder.passage_max_length
        return await self.encode(corpus, max_length=max_length, **kw)

    async def close(self) -> None:
        """Stop taking requests, finish the queued and running ones, release the
        pool (and the embedder if this instance created it)."""
        if self.closed:
            return
        self.closed = True
        for pending in list(self._pending.values()):
            self._flush(pending)
        while self._running or self.queue_depth:
            if self._running:
                await asyncio.gather(*self._running, return_exceptions=True)
            else:
                await asyncio.sleep(0)  # a queued request is taking its slot
        self._executor.shutdown(wait=True)
        if self._owns_embedder:
            self.embedder.close()

    async def __aenter__(self) -> AsyncEmbedder:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    # -- plumbing -------------------------------------------------------------

    def _kwargs(self, options: _Options) -> dict[str, Any]:
        max_length, return_dense, return_sparse, return_colbert_vecs = options
        return dict(
            batch_size=self.batch_size,
            max_batch_tokens=self.max_batch_tokens,
            max_length=max_length,
            return_dense=return_dense,
            return_sparse=return_sparse,
            return_colbert_vecs=return_colbert_vecs,
        )

    def _claim(self, requests: int) -> None:
        """Count a call (of ``requests`` requests) that is about to queue for a
        slot, before its task runs: the batcher must see it immediately."""
        self._queued += requests
        self._waiting_calls += 1

    async def _run(
        self, texts: str | Sequence[str], options: _Options, requests: int
    ) -> Any:
        """One ``encode`` call in the pool, bounded by the semaphore
        (``_claim`` was called for it)."""
        try:
            await self._slots.acquire()
        finally:
            self._waiting_calls -= 1
            self._queued -= requests
        self.in_flight += 1
        try:
            fut = asyncio.get_running_loop().run_in_executor(
                self._executor,
                functools.partial(self.embedder.encode, texts, **self._kwargs(options)),
            )
            self._running.add(fut)
            try:
                return await fut
            finally:
                self._running.discard(fut)
        finally:
            self.in_flight -= 1
            self._slots.release()
            for pending in list(self._pending.values()):  # a slot is free
                self._flush(pending)

    async def _enqueue(self, texts: str | Sequence[str], options: _Options) -> Any:
        """Park the request until the batch window closes or the batch is full."""
        single = isinstance(texts, str)
        items = [texts] if single else list(texts)
        loop = asyncio.get_running_loop()
        pending = self._pending.get(options)
        if pending is None:
            pending = self._pending[options] = _Pending(options)
        fut: asyncio.Future[Any] = loop.create_future()
        pending.texts.extend(items)
        pending.waiters.append((fut, len(items), single))
        if len(pending.texts) >= self.batch_size:
            self._flush(pending)
        elif pending.handle is None:
            if self.in_flight + self._waiting_calls < self.max_concurrency:
                # A slot is free and nobody is waiting for it: start on the
                # next loop iteration, so that requests arriving in this one
                # (a burst) leave together.
                pending.handle = loop.call_soon(self._flush, pending)
            else:  # hold until a slot frees, at most the window
                pending.handle = loop.call_later(
                    self.batch_window_ms / 1000, self._flush, pending
                )
        return await fut

    def _flush(self, pending: _Pending) -> None:
        """Move a pending group into one ``encode`` task that splits the result."""
        if self._pending.pop(pending.options, None) is None:
            return  # already flushed
        if pending.handle is not None:
            pending.handle.cancel()
        self._claim(len(pending.waiters))
        task = asyncio.get_running_loop().create_task(self._run_batch(pending))
        self._running.add(task)
        task.add_done_callback(self._running.discard)

    async def _run_batch(self, pending: _Pending) -> None:
        try:
            out = await self._run(pending.texts, pending.options, len(pending.waiters))
        except BaseException as exc:  # every waiter sees the failure
            for fut, _, _ in pending.waiters:
                if not fut.done():
                    fut.set_exception(exc)
            return
        start = 0
        for fut, n, single in pending.waiters:
            part = {
                k: (None if v is None else v[start : start + n]) for k, v in out.items()
            }
            if single:
                part = {k: (None if v is None else v[0]) for k, v in part.items()}
            start += n
            if not fut.done():
                fut.set_result(part)


__all__ = ["AsyncEmbedder", "DEFAULT_BATCH_WINDOW_MS", "DEFAULT_CONCURRENCY"]
