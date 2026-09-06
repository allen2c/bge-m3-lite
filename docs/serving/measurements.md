# Serving measurements

`tools/bench_serving.py`, ORT 1.29, 2026-09-06; the M4 (4P + 6E cores, 4
threads) was measured alone, the runners by the CI `bench` matrix (every
run prints these tables under "serving" in the job summary). Latency is what
a closed-loop client sees; CPU is process CPU time per request.

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
| `run_async` ×1 / ×2 / ×4 (v0.6.1) | 41 / 58 / 63 | 26 / 34 / 62 | 33 / 51 / 58 | 58 / 105 / 125 | 17 / 18 / 24 | 24 / 24 / 23 |

| 158-token passages | fp32 req/s | p50 ms | CPU ms/req | int8 req/s | p50 ms | CPU ms/req |
|---|---|---|---|---|---|---|
| sequential | 12.8 | 78 | 280 | 8.6–10.4 | 96–125 | 325 |
| `to_thread` ×2 / ×4 / ×8 | 16.3 / 17.5 / 16.7 | 121 / 226 / 471 | 288 / 391 / 521 | 13.2 / 15.3 / 17.4 | 151 / 256 / 449 | 358 / 438 / 526 |
| `encode(list)` of 16 | 12.2–13.3 | 1210 | 279 | 7.6–9.9 | 1600–2100 | 354 |

Reading (details in `recipe.md`): fp32 is GEMM-bound — two runs in flight fill the gaps a
9-token query leaves in 4 threads (+45 % at the same CPU per request), more
only adds latency, and padding queries into one call is what scales (2.5× at
4, 3.6× at 8). int8 runs 2 700 small ops per query and scales with runs in
flight instead (2.1× at 4, 2.4× at 8; batching loses at every size). For
passages both precisions gain 25–40 % from 2–4 runs in flight and nothing
from batching. Four 1-thread sessions lose to one 4-thread session except
int8 ×8 (+4 % for 4× the memory). `session.run_async` (the run posted to
onnxruntime's own pool, awaited through a future; tokenizer and heads on the
loop thread) loses to the thread pool at every concurrency on both
precisions and on every runner (−25–50 % req/s, higher p95), saving only
10–30 % CPU per request: closed in v0.6.1.

## Mixed bursts and the length buckets (v0.6.1, M4, `AsyncEmbedder` defaults)

Bursts of *c* queries sent at once, alone or together with one 602-token
passage; the queries' latency is the number to watch (`bench_serving.py
--only burst`). v0.6.0 merged the passage into the queries' batch:

| query burst | fp32 p50 / p95 ms alone | + passage, v0.6.0 | + passage, v0.6.1 | int8 alone | + passage (no batching) |
|---|---|---|---|---|---|
| ×2 | 20 / 21 | 1505 / 1758 | 37 / 39 | 15 / 15 | 24 / 41 |
| ×4 | 31 / 33 | 2553 / 2590 | 57 / 59 | 21 / 24 | 22 / 25 |
| ×8 | 46 / 67 | 4428 / 4727 | 75 / 77 | 33 / 46 | 32 / 48 |

With the buckets the queries leave as their own call on the second slot;
what remains (1.6–1.9× the query-only latency on fp32) is the passage's GEMM
sharing the four threads, as with any two runs in flight.

## `AsyncEmbedder` (defaults per precision, see `recipe.md`)

| `AsyncEmbedder`, short queries, c closed-loop clients | fp32 req/s | p50 / p95 ms | CPU ms/req | int8 req/s | p50 / p95 ms | CPU ms/req |
|---|---|---|---|---|---|---|
| ×1, batching off / on | 48 / 49 | 21 / 21 → 20 / 21 | 72 / 71 | 87 / 86 | 11 / 12 → 11 / 15 | 30 / 30 |
| ×2, batching off / on | 76 / 87 | 26 / 32 → 22 / 38 | 63 / 38 | 135 / 119 | 13 / 37 → 17 / 18 | 30 / 23 |
| ×4, batching off / on | 77 / 124 | 52 / 54 → 32 / 34 | 62 / 28 | 192 / 147 | 20 / 24 → 27 / 28 | 35 / 20 |
| ×8, batching off / on | 76 / 174 | 105 / 109 → 46 / 47 | 63 / 20 | 181 / 171 | 43 / 48 → 47 / 48 | 37 / 18 |
| 158-token passages ×4, off / on | 16.5 / 13.3 | 240 / 250 → 295 / 320 | 289 / 273 | 15.0 / 9.4 | 263 / 274 → 419 / 455 | 455 / 371 |

## GitHub-hosted runners (CI `bench`, 2026-09-06, 9-token queries, req/s at p50 ms)

| runner | graph | sequential | `to_thread` ×2 / ×4 | `encode(list)` 8 / 40 | `AsyncEmbedder` ×4, default |
|---|---|---|---|---|---|
| EPYC 9V74 (4 vCPU, 2 threads) | fp32 | 25 at 40 | 27 / 29 | 28 / 28 | 26 at 154 (batching ±0) |
| | int8 | 35 at 29 | 42 / 50 | 51 / 57 | 51 at 80 |
| Neoverse-N2 (4 cores) | fp32 | 18 at 55 | 17 / 19 | 32 / 35 | 28 at 141 (1.7× the ×4 threads) |
| | int8 | 40 at 25 | 50 / 71 | 92 / 124 | 75 at 55 |
| Apple M1 VM (3 vCPU) | fp32 | 11 at 95 | 15 / 15 | 20 / 21 | 16 at 261 (threads: 17) |
| | int8 | 25 at 38 | 39 / 57 | 40 / 39 | 59 at 65 |

`run_async` ×1 / ×2 / ×4 on the same runners (2026-09-07, v0.6.1 bench):
EPYC 9V45 fp32 19 / 19 / 19 req/s versus `to_thread` 25 / 36 / 34, int8
33 / 31 / 33 versus 40 / 69 / 79; Neoverse-N2 fp32 8.5 / 14 / 16 versus
18 / 17 / 20, int8 28 / 54 / 63 versus 41 / 52 / 72; M1 VM fp32 14 / 21 /
23 versus 23 / 28 / 22, int8 36 / 67 / 70 versus 48 / 74 / 92.

Two-core x86 has no thread gaps to fill and nothing to gain from padding
(GEMM saturates at one query), so every mode sits within ±15 % there; ARM
gains 2× from batching on fp32 and, on int8, from either. Passages: 1.2–3
req/s fp32, 3–6 int8, concurrency ±10 %, batching never wins. The summary
of every `bench` run prints the full tables under "serving".
