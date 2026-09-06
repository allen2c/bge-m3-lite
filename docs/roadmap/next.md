# Roadmap: what is next

Shipped versions and the facts behind them: `done.md`. Every item is
measured before it is merged (CI `bench` matrix, `tools/bench_serving.py`
and the memory shapes run alone on the M4) and lands only if the numbers say
so. v0.6.1 closed the four open items (`run_async`, length-aware batching,
short-batch memory, int8 start-up); nothing is scheduled.

## Closed decisions (do not reopen without a user report)

- ORT `session.run_async` instead of the thread pool: −25–50 % req/s and a
  higher p95 at every concurrency, on the M4 and three runners, both
  precisions (`../serving/measurements.md`); the thread pool stays.
- The v0.5.2 short-batch memory: an `If` bypass of the single-iteration
  `Loop` doubles the prepacked weights (+867 MiB; onnxruntime prepacks per
  kernel instance, subgraphs included); the row `Loop` of v0.6.1 replaced
  the layout instead (`../memory.md`). 1 024-row windows: same memory, −5 %.
- int8 start-up: the graph optimisation level does not matter (0.67–0.72 s
  at every level for 2 700 nodes); the row `Loop` halves the outer node
  count and the time, so the scalar scale / zero-point chain stays as it is.
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
- Token-count buckets in the micro-batcher (the character buckets misjudge
  CJK text by up to 3×) if a mixed-script service reports padding waste.
- Rust/maturin kernels only if onnxruntime is still the bottleneck (the
  pure-Python tokenizer and the fixtures are the correctness contract).
