# Serving: asyncio and FastAPI (v0.6)

`AsyncEmbedder` (`bge_m3_lite/serving.py`) is the `await`-able front-end of
one `BGEM3Embedder`: `encode`, `encode_queries`, `encode_corpus` with the
signatures and outputs of the synchronous API, run in a private thread pool,
at most `max_concurrency` calls in flight, the rest queued. It relies on
(`tools/bench_serving.py`, M4, ORT 1.29, 2026-09-06, nothing else running):
`session.run` releases the GIL, `encode` keeps no mutable state (one instance
is safe across threads), and a coroutine ticking every 10 ms is delayed by at
most 1–4 ms while requests run (up to 20 ms with 4+ 600-token passages in
flight: the pure-Python tokenizer holds the GIL for a few ms each).

## FastAPI

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI
from bge_m3_lite import AsyncEmbedder


@asynccontextmanager
async def lifespan(app):
    async with AsyncEmbedder(precision="int8") as emb:  # one model copy per process
        app.state.emb = emb
        yield


app = FastAPI(lifespan=lifespan)


@app.post("/embed")
async def embed(texts: list[str]):
    out = await app.state.emb.encode_queries(texts, return_sparse=True)
    return {"dense": out["dense_vecs"].tolist(), "sparse": out["lexical_weights"]}


@app.get("/health")
async def health():
    emb = app.state.emb
    return {"in_flight": emb.in_flight, "queue_depth": emb.queue_depth}
```

Rules: never call the synchronous `encode` inside a coroutine (a short query
blocks the loop for 11–20 ms, a long passage for seconds); do not run other
CPU work in the embedder's pool (it is private, sized `max_concurrency`);
one `AsyncEmbedder` per process, created in the lifespan, closed there
(`close()` drains the queue, then releases the pool and the embedder it
built; a wrapped embedder stays open). Uvicorn `--workers N` gives N model
copies: memory is N × (RSS after load + activations, next section), and
`BGEM3Embedder(low_memory=True)` shares the weight pages between the workers
at 2× the single-query latency (`resources.md`).

## Concurrency: one session, `to_thread`, closed-loop clients (M4, 4 threads)

40 × 9-token queries and 16 × 158-token passages; each of *c* clients sends
its next request when the previous one returns; latency as seen by a client.

| short queries | fp32 req/s | p50 / p95 ms | CPU ms/req | int8 req/s | p50 / p95 ms | CPU ms/req |
|---|---|---|---|---|---|---|
| sequential | 48 | 21 / 21 | 72 | 89 | 11 / 12 | 29 |
| `to_thread` ×2 | 68–75 | 27 / 28 | 63–68 | 154 | 13 / 14 | 28 |
| `to_thread` ×4 | 64–71 | 56 / 60 | 96 | 188 | 21 / 24 | 36 |
| `to_thread` ×8 | 68–72 | 110 / 131 | 132 | 208–213 | 36 / 48 | 43 |
| `encode(list)` of 4 / 8 / 40 | 122 / 171 / 238 | 33 / 47 / 168 | 28 / 20 / 15 | 150 / 174 / 196 | 27 / 46 / 204 | 20 / 18 / 18 |
| 4 sessions × 1 thread, ×4 / ×8 | 55 / 68 | 72 / 115 | 72 / 112 | 164 / 222 | 24 / 34 | 24 / 34 |

| 158-token passages | fp32 req/s | p50 ms | CPU ms/req | int8 req/s | p50 ms | CPU ms/req |
|---|---|---|---|---|---|---|
| sequential | 12.8 | 78 | 280 | 8.6–10.4 | 96–125 | 325 |
| `to_thread` ×2 / ×4 / ×8 | 16.3 / 17.5 / 16.7 | 121 / 226 / 471 | 288 / 391 / 521 | 13.2 / 15.3 / 17.4 | 151 / 256 / 449 | 358 / 438 / 526 |
| `encode(list)` of 16 | 12.2–13.3 | 1210 | 279 | 7.6–9.9 | 1600–2100 | 354 |

What this says: fp32 is GEMM-bound — two runs in flight fill the gaps a
9-token query leaves in 4 threads (+45 % at the same CPU per request), more
only adds latency, and padding queries into one call is what scales (2.5× at
4, 3.6× at 8). int8 runs 2 700 small ops per query and scales with runs in
flight instead (2.1× at 4, 2.4× at 8; batching loses at every size). For
passages both precisions gain 25–40 % from 2–4 runs in flight and nothing
from batching. Four 1-thread sessions lose to one 4-thread session except
int8 ×8 (+4 % for 4× the memory).

## Defaults and the micro-batcher

`max_concurrency` defaults to 2 for fp32 and 4 for int8, `batch_window_ms`
to 10 for fp32 and 0 (off) for int8 (`DEFAULT_*` in `serving.py`). The
batcher: a request that finds a free, unclaimed slot starts on the next loop
iteration, together with every request of the same burst (one `gather`, one
`select` wake-up); one that finds every slot busy or claimed is held (at
most the window) until a slot frees, and everything held with the same
options leaves as one `encode` call of at most `batch_size` texts. Results
are exactly what the synchronous `encode(list)` returns for that list, split
per request. Two earlier designs lost: "always wait the window" costs the
window at low load (×1: −22 %), and "start when a slot is free" never
batches a closed loop, whose requests arrive exactly when a slot frees.

| `AsyncEmbedder`, short queries, c closed-loop clients | fp32 req/s | p50 / p95 ms | CPU ms/req | int8 req/s | p50 / p95 ms | CPU ms/req |
|---|---|---|---|---|---|---|
| ×1, batching off / on | 48 / 49 | 21 / 21 → 20 / 21 | 72 / 71 | 87 / 86 | 11 / 12 → 11 / 15 | 30 / 30 |
| ×2, batching off / on | 76 / 87 | 26 / 32 → 22 / 38 | 63 / 38 | 135 / 119 | 13 / 37 → 17 / 18 | 30 / 23 |
| ×4, batching off / on | 77 / 124 | 52 / 54 → 32 / 34 | 62 / 28 | 192 / 147 | 20 / 24 → 27 / 28 | 35 / 20 |
| ×8, batching off / on | 76 / 174 | 105 / 109 → 46 / 47 | 63 / 20 | 181 / 171 | 43 / 48 → 47 / 48 | 37 / 18 |
| 158-token passages ×4, off / on | 16.5 / 13.3 | 240 / 250 → 295 / 320 | 289 / 273 | 15.0 / 9.4 | 263 / 274 → 419 / 455 | 455 / 371 |

fp32 with batching beats every concurrency setting at every load (2.5× the
sequential rate at 4 clients with p95 under 2× the single-query latency,
3.6× at 8) at a third of the CPU per request; int8 loses 6–24 % of its
throughput with batching (for −40 % CPU), so it stays off. Passages lose
with batching on either precision: give a corpus endpoint its own
`AsyncEmbedder(embedder, batch_window_ms=0)` around the same
`BGEM3Embedder` (each wrapper counts its own slots).

## Memory under concurrency

Every run in flight allocates its own activations in the shared arena, so the
`memory.md` rule applies per in-flight run: **peak ≈ RSS after load +
in-flight × padded tokens per call × 0.07 MiB** (626-token passages: fp32
+41 MiB, int8 +38 MiB per extra run in flight from 1 to 8; 0.11 MiB per
token for short batches). The peak is reached once and stays. Budget for
uvicorn: `workers × (1.3 GB fp32 | 0.75 GB int8 + max_concurrency ×
max_batch_tokens × 0.11 MiB)`; lower `max_batch_tokens` (default 16384) on
small machines.

## Choosing

- **int8** for a query service on x86 / ARM / Apple Silicon: 2× the
  requests per second of fp32 at every concurrency, 0.75 GB instead of 1.3
  GB, `max_concurrency=4` (8 buys +10 % at 2× the latency). Accuracy:
  `quantization/measurements.md`.
- **fp32** when exact parity matters: the defaults (2 slots, batching)
  serve short queries at 2.6–3.6× the sequential rate under load; for a
  passage endpoint use a second wrapper with `batch_window_ms=0`.

## GitHub-hosted runners (CI `bench`, 2026-09-06, 9-token queries, req/s at p50 ms)

| runner | graph | sequential | `to_thread` ×2 / ×4 | `encode(list)` 8 / 40 | `AsyncEmbedder` ×4, default |
|---|---|---|---|---|---|
| EPYC 9V74 (4 vCPU, 2 threads) | fp32 | 25 at 40 | 27 / 29 | 28 / 28 | 26 at 154 (batching ±0) |
| | int8 | 35 at 29 | 42 / 50 | 51 / 57 | 51 at 80 |
| Neoverse-N2 (4 cores) | fp32 | 18 at 55 | 17 / 19 | 32 / 35 | 28 at 141 (1.7× the ×4 threads) |
| | int8 | 40 at 25 | 50 / 71 | 92 / 124 | 75 at 55 |
| Apple M1 VM (3 vCPU) | fp32 | 11 at 95 | 15 / 15 | 20 / 21 | 16 at 261 (threads: 17) |
| | int8 | 25 at 38 | 39 / 57 | 40 / 39 | 59 at 65 |

Two-core x86 has no thread gaps to fill and nothing to gain from padding
(GEMM saturates at one query), so every mode sits within ±15 % there; ARM
gains 2× from batching on fp32 and, on int8, from either. Passages: 1.2–3
req/s fp32, 3–6 int8, concurrency ±10 %, batching never wins. The summary
of every `bench` run prints the full tables under "serving".
