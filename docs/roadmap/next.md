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
- **v0.6 (candidates)** — weight-only int8 `MatMulNBits` for Apple Silicon
  (memory only; int8 GEMM is slower than SGEMM there); embeddings served from
  the mmapped file instead of the int8 `Gather` copy; sparse-head rounding
  experiments with the held-out set.

## Later

- Rust/maturin kernels: only if onnxruntime is still the bottleneck
  (the pure-Python tokenizer and the fixtures are the correctness contract).
