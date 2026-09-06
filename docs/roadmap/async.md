# v0.6 plan: asyncio / FastAPI serving

Goal: `bge-m3-lite` is "asyncio ready" — an `async` API that never blocks the
event loop, scales throughput on one CPU without extra dependencies, and a
best-practice document with numbers for the FastAPI case. Runtime dependency
stays `onnxruntime` only (`asyncio`, `threading`, `concurrent.futures` are
stdlib). The first numbers (M4, noisy machine) are in `next.md`.

## What is known

- `session.run` releases the GIL; the tokenizer costs 0.03 ms per query. A
  shared `BGEM3Embedder` is reentrant (no mutable state in `encode`), so
  `await asyncio.to_thread(emb.encode, q)` is correct today.
- One session with ORT's intra-op threads beats N single-thread sessions.
- fp32 short queries are GEMM-bound: concurrency +50 %, one `encode(list)`
  of 40 gives 2.4× — batching wins. int8 short queries are latency-bound
  (2 700 small ops): concurrency 8 gives 2.6×, batching loses.
- Unknown: peak memory under concurrency (each in-flight run allocates its
  own activations in the shared arena), CI-runner (2–4 vCPU) behaviour, and
  the effect of ORT's inter-op / `run_async` path versus Python threads.

## Deliverables

1. `bge_m3_lite/serving.py` (name to be confirmed): `AsyncEmbedder` wrapping
   one `BGEM3Embedder` with
   - `await encode(...)`, `encode_queries`, `encode_corpus` (same signatures
     and outputs as the sync API, run in a dedicated `ThreadPoolExecutor`);
   - a bounded in-flight limit (`max_concurrency`, default measured, likely
     2–4 per session) so bursts queue instead of oversubscribing the CPU;
   - an optional micro-batcher (`batch_window_ms`, `max_batch_tokens`):
     requests arriving within the window are padded into one `encode` call,
     results split back per request; off by default for int8 if the numbers
     hold, on for fp32 short queries;
   - graceful `close()` / `async with`, and `queue_depth` / `in_flight`
     counters for health endpoints.
2. `docs/serving.md` (≤ 100 lines): the FastAPI recipe (lifespan-managed
   embedder, one process per model copy, uvicorn workers × memory maths
   from `memory.md` / `resources.md`), blocking rules (never call the sync
   API in a coroutine, never share the executor with other CPU work), the
   measured table, and when to choose fp32 vs int8, batching vs concurrency.
3. `tools/bench_serving.py`: sequential / `to_thread` at concurrency 1–8 /
   micro-batch / passages, printing req/s, p50/p95, CPU per request, peak
   RSS; wired into the CI `bench` job so the matrix reports it.
4. Tests: event-loop-not-blocked (a ticking coroutine keeps its period
   during `encode`), results identical to the sync API (bit-exact, also
   through the micro-batcher with mixed lengths), back-pressure (in-flight
   never exceeds the limit), clean shutdown.

## Order of work

1. Recreate the bench as `tools/bench_serving.py`, run it alone on the M4
   and on the CI matrix; add peak RSS versus concurrency (`memory.md` rule
   must still hold: budget ≈ in-flight × padded tokens × 0.11 MiB).
2. Decide the executor size and default `max_concurrency` from the data
   (per precision), and whether the micro-batcher earns its complexity for
   fp32 (target: ≥ 2× sequential throughput at p95 < 2× single-query
   latency on the M4).
3. Implement `AsyncEmbedder`, tests, `docs/serving.md`; extend
   `resources.md` with CPU-seconds per request under concurrency.
4. Release v0.6.0 (no model assets change).

## Non-goals

Multi-process pools (document uvicorn workers instead), GPU, streaming,
ORT `run_async` unless it measurably beats threads, any new dependency.
