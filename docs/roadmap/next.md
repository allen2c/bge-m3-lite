# Roadmap: what is next

Shipped versions and the facts behind them: `done.md`.

## int8 on Xeon (VNNI): the v4 recipe regressed (measured 2026-09-06)

- The Xeon 8573C runner was finally drawn: int8 v4 494 tok/s versus fp32
  fused 576 (0.86×), where v3 did 742 (1.5×). Per-axis `QuantizeLinear` +
  `MatMulIntegerToFloat` (u8·u8) apparently has no VNNI path that
  `MatMulInteger` (v3) had. Candidates: keep `MatMulInteger` + `Cast`/`Mul`
  on x86, or u8·s8 weights on VNNI only (`VPDPBUSD` does not saturate; the
  AVX2 kernel does), or ORT's per-tensor `DynamicQuantizeMatMul` for the
  projections whose activations tolerate it. Bench with `quantize_variants`
  until `ubuntu-latest` draws the Xeon again (random).
- Profile the remaining scalar ops (`ReduceMax/ReduceMin` are two extra
  passes over the input).
- `attention_chunk` per platform: 512 was chosen on the M4; the CI matrix
  may prefer 1024 (fewer iterations) on 4-vCPU runners.
- Start-up of the int8 graph (1.1 s vs 0.4 s fp32): 2 700 nodes cost ORT
  session time; folding the scalar scale/zero-point chain into fewer ops
  would also help here.

## v0.5.x — resource efficiency (v0.5.0 and v0.5.1 shipped, see `done.md` and `../resources.md`)

- **v0.5.2** — activation memory: measured 0.09–0.12 MiB per padded token
  (`../memory.md`, guidance for `max_batch_tokens` per RSS budget is there).
  Move the per-token tail of each layer (output projection, SkipLayerNorm,
  FFN, SkipLayerNorm) into the attention `Loop` body so the FFN
  intermediates are `chunk × 16 KiB` instead of `padded_tokens × 16 KiB`;
  must stay bit-exact and within 3 % on short inputs. Measure whether the
  arena shrink run option (`memory.enable_memory_arena_shrinkage`, no
  effect in the first test) can return memory between requests (disabling
  the arena does not, `../resources.md`).
- **v0.6 (candidates)** — weight-only int8 `MatMulNBits` for Apple Silicon
  (memory only; int8 GEMM is slower than SGEMM there); embeddings served from
  the mmapped file instead of the int8 `Gather` copy; sparse-head rounding
  experiments with the held-out set.

## Later

- Rust/maturin kernels: only if onnxruntime is still the bottleneck
  (the pure-Python tokenizer and the fixtures are the correctness contract).
