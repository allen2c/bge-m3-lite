# Roadmap

Versions are planned around one measurable goal each. Numbers quoted here were
measured on 2026-09-05 (see `verification.md` and `quantization.md`); every
step is gated by the fixtures in `tests/fixtures/` staying green.

## Baseline facts that shaped the plan

| finding | consequence |
|---|---|
| int8 is 3.9× faster than fp32 on x86 (Xeon 8573C, VNNI) and 4.3× on Neoverse-N2, but only 1.1× on Apple Silicon | int8 accuracy work (v0.3) is worth it; macOS users need fp32 speed-ups instead |
| `onnxruntime.transformers` fuses 24 `Attention`, 48 `SkipLayerNormalization`, 24 `BiasGelu`; outputs unchanged; only 48 tensors (288 MiB, the merged QKV weights) differ from the Hub weights | fused fp32 graph (v0.2) can be built locally from the cached weights in seconds, no 2.3 GB re-download |
| ORT's own `ORT_ENABLE_ALL` already fuses LayerNorm + Gelu but never Attention on this opset-11 export; saving the ORT-optimised graph to disk cuts session creation from 0.38 s to 0.33 s only | "cache the optimised graph" was dropped: the start-up cost is elsewhere |
| warm start-up was 0.86 s: tokenizer parse 0.31 s, ORT session 0.44 s; cold start is bound by reading 2.3 GB from disk | v0.1 caches the parsed vocabulary (0.31 s → 0.05 s); the rest needs a smaller model (int8: 543 MB) |
| a batch of 12 texts × 8192 tokens allocates more than the 400 MiB hidden state: the unfused attention materialises `batch × 16 × seq²` floats | token-budget batching (v0.1) bounds memory by padded tokens instead of text count |

## v0.1.0 — API, start-up, Windows (done)

- `encode_queries` / `encode_corpus` with separate default lengths
  (`query_max_length=512`, `passage_max_length=max_length`), `compute_score`
  returning the same five keys as FlagEmbedding.
- Token-budget batching: `max_batch_tokens` (default 16384 padded tokens) next
  to `batch_size`; mixed short/long inputs no longer OOM.
- Vocabulary cache next to `sentencepiece.bpe.model` (own format, no pickle):
  tokenizer load 0.31 s → 0.05 s.
- Windows: `windows-latest` in the CI matrix, lock-file liveness check without
  `os.kill` (which terminates the target process on Windows).
- Measured x86 / Linux ARM / macOS-runner numbers for fp32 and int8 in the docs.

## v0.2.0 — fused fp32 backbone (done)

- `model_fused.onnx` (graph, 155 KB) + `model_fused.onnx_data` (the 48 merged
  QKV tensors, 288 MiB) as release assets, pinned by size + SHA-256. The graph
  references the unchanged tensors of the cached `model.onnx_data` by offset,
  so no second 2.3 GB download. `bge-m3-lite fuse` rebuilds both
  deterministically (needs the `quant` extra).
- `BGEM3Embedder(fused=True)` is the default for fp32; `fused=False` /
  `encode --raw` keeps the raw export. Measured: +10–14 % at 128/512 tokens on
  M4, +2–5 % on the Linux runners (MLAS GEMM dominates there), +20–30 % on
  the macOS VM runner; outputs identical everywhere.

## v0.3.0 / v0.3.1 — int8 v2 → v3 (done)

- v0.3.0 shipped SmoothQuant + ORT `quantize_dynamic` on the fused graph. It
  measured dense cosine 0.998 on the development M4 but 0.96–0.98 on every CI
  runner: ORT's KleidiAI kernel on Apple Silicon quantises activations per
  row, everything else per tensor. Lesson: **validate int8 on the CI `bench`
  matrix before pinning an asset.**
- v0.3.1 makes the per-row scheme explicit in the graph (`--method rowwise`,
  uint8 activations with a per-row zero point, `MatMulInteger`, QKV via
  `MultiHeadAttention`, uint8 weights because the AVX2 u8·s8 kernel saturates
  its int16 intermediates), plus SmoothQuant α 0.5. Dense cosine 0.9988, sparse
  top-5 10–11/11, ColBERT p5 0.99 on x86, ARM and macOS alike; 1.5× (x86
  VNNI) to 3.5× (ARM) faster than fp32, 20–30 % slower than the per-tensor
  kernels. Details and the discarded variants: `quantization.md`.
- Also: the fused graph declares the `com.microsoft` opset (onnx tooling
  needs it), `quantize` gained `--method/--alpha/--calibration`, and the CI
  `bench` job accepts a `quantize_variants` matrix.

## v0.4.0 — memory, int8 speed, evaluation (in progress)

Everything below is implemented and measured on the M4 (2026-09-06); the
release waits for one CI `bench` run (`fuse_local` + default `quantize`) to
confirm the x86/ARM numbers, then new `model_fused.*` and `model_int8.onnx`
assets are pinned.

| goal | result |
|---|---|
| long inputs must not need 4 GiB | attention in query chunks (`Loop`, 512 rows): 8192 tokens 7.4 GB → 2.5 GB, bit-exact, ≤ 3 % on short inputs (`memory.md`) |
| int8 element-wise overhead | per-axis `QuantizeLinear` + `MatMulIntegerToFloat`: +11 % on the M4, same accuracy; the ORT-fusable per-tensor form is out of reach (per-row zero point unsupported), `--symmetric` gets closer but loses accuracy (`quantization.md`) |
| sparse int8 accuracy | measured, not fixed: the flips are fp32 near-ties (gap 0.001–0.009); last-layer-fp32 variants change ±1 text, so the recipe stays; `--keep-fp32` remains for experiments |
| calibration provenance | 212 hand-written + 360 MIRACL passages, licence and recipe in `calibration.md`; 40-text held-out set disjoint from calibration, reported by `tools/eval_model.py` |
| engineering | `hub.py` retries 429/5xx/timeouts with back-off; the CI `bench` summary prints the CPU model and one accuracy + tok/s row per graph |

## v0.4.x candidates (measure first)

- x86 numbers for v0.4 from the bench matrix; if the Xeon gain is below
  +20 %, profile the remaining scalar ops (`ReduceMax/ReduceMin` are two
  extra passes; `MaxPool`-style fused min/max does not exist in ORT).
- `attention_chunk` per platform: 512 was chosen on the M4; the CI matrix
  may prefer 1024 (fewer iterations) on 4-vCPU runners.
- Start-up of the int8 graph (1.1 s vs 0.4 s fp32): 2 700 nodes cost ORT
  session time; folding the scalar scale/zero-point chain into fewer ops
  would also help here.

## v0.5 ideas (not started)

- Chunk the FFN too (`Loop` over token blocks) so the 16 KiB/token
  intermediate stops scaling with batch length; only matters above 32k
  padded tokens per batch.
- Sparse head in fp32 with a re-quantised last hidden state: the measured
  near-ties suggest dithering rather than precision is the issue; a
  calibrated per-token rounding offset could be tested with the held-out set.
- Weight-only int8 (`--method nbits --bits 8 --accuracy-level 4`) on
  Apple Silicon, where int8 GEMM is slower than fp32 SGEMM: not a speed
  path, only a memory one.

## Later

- Rust/maturin kernels: only if onnxruntime is still the bottleneck
  (the pure-Python tokenizer and the fixtures are the correctness contract).
