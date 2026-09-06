# Roadmap: what is next

Shipped versions and the facts behind them: `done.md`.

## int8 on Xeon (VNNI): measured, closed (2026-09-06)

- Xeon 8573C runner: int8 v4 473–494 tok/s versus fp32 fused 564–576
  (0.84×) on 128-token batches, short queries still 2× faster with int8
  (31 vs 57 ms). Threads and spinning are not the cause (4 threads +8 %,
  spin ±2 %). `MatMulInteger` (v3 path) gains nothing on the EPYCs, u8·s8
  weights gain 18–26 % there but saturate the AVX2 kernel (dense 0.978), so
  they cannot be the single x86 asset (`../quantization/measurements.md`).
- Decision: one x86 recipe (v4, u8·u8); no per-CPU assets, no more Xeon
  hunting on the random `ubuntu-latest` runner. Reopen only if a user
  reports a Xeon-only workload where batch throughput matters more than the
  2× short-query gain.
- Still worth a look on every platform: fold the scalar scale/zero-point
  chain (2 700 nodes: start-up 1.1 s vs 0.4 s fp32; the layer loop halves
  it but costs memory on int8).

## v0.5.x — resource efficiency (v0.5.0–v0.5.2 shipped, see `done.md`, `../resources.md`, `../memory.md`)

- **v0.5.2** shipped (`done.md`, `../memory.md`): chunk 256 and the layer
  tail inside the attention `Loop` for fp32. Left open: the +20 % for batches
  of texts no longer than the chunk (three hidden-state copies per layer in
  the loop; a scan output would need equal chunks), and a memory-exact int8
  layout (the loop costs memory there). The arena high-water mark, not the
  live set, is what `max_batch_tokens` buys; `kSameAsRequested` and
  shrinkage were measured and rejected.
- **v0.6 candidate: asyncio serving** (first numbers, M4, load average 3.5 so
  re-measure alone; `tools/` has no script yet): `session.run` releases the
  GIL (a Python spinner keeps 79–93 % of its rate while `encode` runs), the
  tokenizer costs 0.03 ms per query, so `await asyncio.to_thread(emb.encode,
  q)` on one shared embedder is the natural pattern. One session × 4 threads,
  short queries: fp32 43 req/s sequential → 65 at concurrency 8 (p50 24 →
  118 ms), one `encode(list)` of 40 = 104 req/s; int8 58 → 152 req/s at
  concurrency 8 (p50 18 → 50 ms), while its `encode(list)` is only 66.
  Four 1-thread sessions are worse than one 4-thread session (fp32 52 vs 62
  req/s, int8 89 vs 117; with `low_memory` 18 / 45). Open: peak memory under
  concurrency (each in-flight run has its own activations), an `encode_async`
  / micro-batching helper (stdlib only), and CI-runner numbers.
- **v0.6 (other candidates)** — weight-only int8 `MatMulNBits` for Apple Silicon
  (memory only; int8 GEMM is slower than SGEMM there); embeddings served from
  the mmapped file instead of the int8 `Gather` copy; sparse-head rounding
  experiments with the held-out set.

## Later

- Rust/maturin kernels: only if onnxruntime is still the bottleneck
  (the pure-Python tokenizer and the fixtures are the correctness contract).
