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

## v0.2.0 — fused fp32 backbone (lossless, faster)

- `model_fused.onnx` (graph, 155 KB) + `model_fused.onnx_data` (the 48 merged
  QKV tensors, 288 MiB) as release assets, pinned by size + SHA-256. The graph
  references the unchanged tensors of the cached `model.onnx_data` by offset,
  so no second 2.3 GB download. `bge-m3-lite fuse` rebuilds both
  deterministically (needs the `quant` extra).
- `BGEM3Embedder(fused=True)` is the default for fp32; `fused=False` /
  `encode --raw` keeps the raw export. Measured on M4: +10–14 % at 128/512
  tokens, outputs identical; Linux numbers from the CI `bench` job.

## v0.3.0 — int8 v2 (accuracy)

- Quantise the fused graph (`QAttention`, `MatMulInteger` with per-channel
  weights) so int8 keeps the fusion speed-up.
- Static or SmoothQuant activation calibration on a small multilingual set
  (the 11 fixture texts plus ~200 public sentences across the 6 scripts).
  Targets: dense cosine ≥ 0.999 mean, sparse top-5 identical ≥ 10/11, ColBERT
  ranking identical on all platforms.
- New `model_int8.onnx` release asset; the v0.0.2 asset stays downloadable.

## Later

- Rust/maturin kernels: only if onnxruntime is still the bottleneck after v0.3
  (the pure-Python tokenizer and the fixtures are the correctness contract).
- Long-context memory: chunked attention would need graph surgery; until then
  the token budget is the tool.
