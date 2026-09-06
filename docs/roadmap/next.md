# Roadmap: v0.6.1 (next)

Shipped versions and the facts behind them: `done.md`. Every item below is
measured before it is merged (CI `bench` matrix, `tools/bench_serving.py`
run alone on the M4) and lands only if the numbers say so.

## v0.6.1 — serving follow-ups and the leftovers

| item | what to do | done when |
|---|---|---|
| ORT `run_async` versus threads | `session.run_async` from the loop thread instead of `run_in_executor`; measure req/s, p50/p95, CPU/req and loop lag at concurrency 1–8 on fp32 and int8 (`bench_serving.py` gets a mode) | adopted only if it beats the thread pool on the M4 and two runners; otherwise recorded in `../serving/measurements.md` and closed |
| token-aware micro-batcher | the batcher counts texts, so one long passage in a query batch pads every query to its width; hold a request's token count (tokenize in the worker, or a cheap length estimate) and flush a group at `max_batch_tokens`, or keep long texts out of query batches | mixed query + 600-token burst no slower than the query-only burst; bit-exact test with mixed lengths still green |
| short-batch activation memory | fp32 batches of texts ≤ 256 tokens cost +20 % since v0.5.2 (three hidden-state copies per layer inside the `Loop`); try a scan output or a single-iteration bypass in `fuse` | `128 × 128` peak back to ≤ 1450 MiB (`../memory.md` table) at unchanged tok/s and bit-exact outputs |
| int8 node count / start-up | fold the scalar scale / zero-point chain (2 700 outer nodes, 1.1 s start-up on runners vs 0.4 s fp32); the layer loop halves it but costs memory on int8 | start-up ≤ 0.6 s on the M4 without a memory regression in the `../memory.md` int8 columns |

Serving numbers on the CI runners are in `../serving/measurements.md`
(EPYC 9V74, Neoverse-N2, M1 VM, 2026-09-06).

## Closed decisions (do not reopen without a user report)

- int8 on Xeon (VNNI): v4 is 0.84× fp32 on 128-token batches, short queries
  still 2× faster; one x86 recipe (u8·u8), no per-CPU assets
  (`../quantization/measurements.md`).
- Arena tuning: `kSameAsRequested`, shrinkage and disabling the arena were
  measured and rejected (`../memory.md`, `../resources.md`).

## Later candidates

- Weight-only int8 `MatMulNBits` for Apple Silicon (memory only; int8 GEMM
  is slower than SGEMM there); embeddings served from the mmapped file
  instead of the int8 `Gather` copy; sparse-head rounding experiments with
  the held-out set.
- Rust/maturin kernels only if onnxruntime is still the bottleneck (the
  pure-Python tokenizer and the fixtures are the correctness contract).
