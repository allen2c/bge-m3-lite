# Serving with asyncio and FastAPI (v0.6)

`AsyncEmbedder` (`bge_m3_lite/serving.py`) is the `await`-able front-end of
one `BGEM3Embedder`: `encode`, `encode_queries`, `encode_corpus` with the
signatures and outputs of the synchronous API, run in a private thread pool,
at most `max_concurrency` calls in flight, the rest queued. Measured facts
it relies on (`measurements.md`): `session.run` releases the GIL, `encode`
keeps no mutable state (one instance is safe across threads), a coroutine
ticking every 10 ms is delayed by 1–4 ms while requests run (20 ms with 4+
600-token passages in flight: the pure-Python tokenizer holds the GIL).

## FastAPI recipe

```python
@asynccontextmanager
async def lifespan(app):
    async with AsyncEmbedder(precision="int8") as emb:  # one model copy per process
        app.state.emb = emb
        yield


@app.post("/embed")
async def embed(texts: list[str]):
    out = await app.state.emb.encode_queries(texts, return_sparse=True)
    return {"dense": out["dense_vecs"].tolist(), "sparse": out["lexical_weights"]}
```

The production shape (two wrappers, queue-depth guard, timeouts, health,
shutdown) is in `fastapi.md`.

Rules: never call the synchronous `encode` inside a coroutine (a short query
blocks the loop for 11–20 ms, a long passage for seconds); no other CPU work
in the embedder's pool (private, sized `max_concurrency`); one
`AsyncEmbedder` per process, created in the lifespan, closed there
(`close()` drains the queue, then releases the pool and the embedder it
built; a wrapped embedder stays open). Uvicorn `--workers N` gives N model
copies: memory is N × (RSS after load + activations, below), and
`BGEM3Embedder(low_memory=True)` shares the weight pages between the workers
at 2× the single-query latency (`../resources.md`). A request cancelled while
its call runs (`asyncio.timeout`, a client gone) keeps its slot until the
thread returns: nothing can stop `session.run`, so `in_flight` stays honest
and `close()` waits in the loop (`asyncio.md` lists every scheduling fact
the class relies on, per Python version).

## Defaults and the micro-batcher

`max_concurrency` defaults to 2 for fp32 and 4 for int8, `batch_window_ms`
to 10 for fp32 and 0 (off) for int8 (`DEFAULT_*` in `serving.py`). The
batcher: a request that finds a free, unclaimed slot starts on the next loop
iteration, together with every request of the same burst (one `gather`, one
`select` wake-up); one that finds every slot busy or claimed is held (at
most the window) until a slot frees, and everything held with the same
options leaves as one `encode` call of at most `batch_size` texts. Results
are exactly what the synchronous `encode(list)` returns for that list, split
per request. Requests only merge within a character-length bucket
(`LENGTH_BUCKETS`: ≤ 128, ≤ 512, ≤ 2048, longer; a request sits in the bucket
of its longest text): a batch is padded to its longest text, so one
600-token passage in a burst of queries used to turn the burst into a batch
of passages (fp32 query p50 in such a burst 1.5–4.4 s, now 37–75 ms,
`measurements.md`). Two earlier designs lost: "always wait the window" costs the
window at low load (×1: −22 %), and "start when a slot is free" never
batches a closed loop, whose requests arrive exactly when a slot frees.

Why these defaults (M4, `measurements.md`): fp32 short queries are
GEMM-bound — two runs in flight fill the thread gaps (+45 % at equal CPU),
more only adds latency, and padding a burst into one call is what scales
(2.5× the sequential rate at 4 clients with p95 under 2× the single-query
latency, 3.6× at 8, at a third of the CPU per request). int8 runs 1 400
small ops per call and scales with runs in flight (2.1× at 4, 2.4× at 8);
batching loses 6–24 % there. Passages gain 25–40 % from 2–4 runs in flight
and nothing from batching on either precision: give a corpus endpoint its
own `AsyncEmbedder(embedder, batch_window_ms=0)` around the same
`BGEM3Embedder` (each wrapper counts its own slots).

## Memory under concurrency

Every run in flight allocates its own activations in the shared arena, so the
`../memory.md` rule applies per in-flight run: **peak ≈ RSS after load +
in-flight × padded tokens per call × 0.07 MiB** (626-token passages: fp32
+41 MiB, int8 +38 MiB per extra run in flight from 1 to 8; short batches
cost the same per token since v0.6.1). The peak is reached once and stays.
Budget for uvicorn: `workers × (1.3 GB fp32 | 0.75 GB int8 + max_concurrency
× max_batch_tokens × 0.07 MiB)`; lower `max_batch_tokens` (default 16384) on
small machines.

## Choosing

- **int8** for a query service on x86 / ARM / Apple Silicon: 2× the short
  queries per second of fp32 at every concurrency, 0.75 GB instead of 1.3
  GB, `max_concurrency=4` (8 buys +10 % at 2× the latency); accuracy in
  `../quantization/measurements.md`.
- **fp32** when exact parity matters: the defaults (2 slots, batching)
  serve short queries at 2.5–3.6× the sequential rate under load; a
  passage endpoint gets a second wrapper with `batch_window_ms=0`.
- Two-core x86 runners gain nothing from any mode (GEMM saturates at one
  query); ARM gains 2× from batching on fp32 (`measurements.md`).
