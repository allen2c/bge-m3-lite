# Roadmap: v0.1 – v0.4 (shipped)

The versions before the resource work; the baseline facts are in `done.md`.

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
  kernels. Details and the discarded variants: `../quantization/`.
- Also: the fused graph declares the `com.microsoft` opset (onnx tooling
  needs it), `quantize` gained `--method/--alpha/--calibration`, and the CI
  `bench` job accepts a `quantize_variants` matrix.

## v0.4.0 — memory, int8 speed, evaluation (done)

Measured on the M4 and on the CI matrix (2026-09-06, `../quantization/measurements.md`,
`../memory.md`); ships new `model_fused.onnx` (graph only, the data file is
unchanged) and `model_int8.onnx` assets.

| goal | result |
|---|---|
| long inputs must not need 4 GiB | attention in query chunks (`Loop`, 512 rows): 8192 tokens 7.4 GB → 2.5 GB, bit-exact, ≤ 3 % on short inputs (`../memory.md`) |
| int8 element-wise overhead | per-axis `QuantizeLinear` + `MatMulIntegerToFloat`: +11 % on the M4, +6–9 % on EPYC (AVX2), +19 % on the macOS VM, +1–3 % on Neoverse (GEMM-bound); same accuracy. The ORT-fusable per-tensor form is out of reach (per-row zero point unsupported); `--symmetric` gets closer but loses accuracy (`../quantization/recipe.md`) |
| sparse int8 accuracy | measured, not fixed: the flips are fp32 near-ties (gap 0.001–0.009); last-layer-fp32 variants change ±1 text, so the recipe stays; `--keep-fp32` remains for experiments |
| calibration provenance | 212 hand-written + 360 MIRACL passages, licence and recipe in `../calibration.md`; 40-text held-out set disjoint from calibration, reported by `tools/eval_model.py` |
| engineering | `hub.py` retries 429/5xx/timeouts with back-off; the CI `bench` summary prints the CPU model and one accuracy + tok/s row per graph |
