# Roadmap: what is next

Shipped versions and the facts behind them: `done.md`.

## int8 on Xeon (VNNI): the v4 recipe regressed (measured 2026-09-06)

- Xeon 8573C runner: int8 v4 473–494 tok/s versus fp32 fused 564–576
  (0.84×), where v3 (`MatMulInteger` + `Cast` + `Mul`, v0.3.1, 4 threads
  with spinning) did 742 (1.5×). Threads and spinning are not the cause
  (`env_variants` bench on two Xeon draws): 4 threads +8 % for int8 (511
  tok/s) and −3 % for fp32, `BGE_M3_LITE_SPIN=1` ±2 %; the ordering stays.
  Short queries are still 2× faster with int8 there (31 vs 57 ms).
- Kernel candidates, buildable as `quantize` variants: `--matmul-integer`
  (v3 path), `--signed-weights` (u8·s8, the native `VPDPBUSD`/AMX operand
  order; saturates MLAS's AVX2 kernel, so accuracy must be read on the EPYC
  too), and both. CI results: `quantization/measurements.md`.
- Profile the remaining scalar ops (`ReduceMax/ReduceMin` are two extra
  passes over the input); fold the scalar scale/zero-point chain to cut the
  2 700 nodes (start-up 1.1 s vs 0.4 s fp32).

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
