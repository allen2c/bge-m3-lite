# Roadmap: shipped versions

Versions are planned around one measurable goal each. Numbers quoted here were
measured on 2026-09-05 (see `../verification.md` and `../quantization/`); every
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

## v0.5.0 — resource efficiency (done)

Measured on the M4 and the CI matrix (2026-09-06, `../resources.md`); ships
`model_int8.onnx` + `model_int8.onnx_data` (same tensors as v0.4).

| goal | result |
|---|---|
| resident memory of the int8 session | graph + external data like the fused graph: 1838 → 724 MiB after load (ORT no longer keeps the parsed protobuf next to the prepacked weights), 8192-token peak 2286 → 1679 MiB, start-up 0.80 → 0.75 s, outputs identical |
| idle CPU | `session.intra_op.allow_spinning=0` by default: 27–64 ms/s → 0 per session, short-query CPU −35 %, throughput −0–5 %; `BGE_M3_LITE_SPIN=1` / `spin=True` restore it |
| CPU-seconds per token | thread default = performance cores on Apple Silicon (4 on the M4; ORT picked 5): −14 % CPU at equal throughput; one thread gives 2× the tokens per CPU-second of four — documented for multi-worker services |
| measurement | `tools/eval_model.py` prints RSS after load / peak, CPU-s per 1k tokens, short-query wall + CPU, idle CPU and OS thread count; the CI bench summary tabulates them, one process per model |

## v0.5.1 — `low_memory` mode (done)

`BGEM3Embedder(low_memory=True)` / `encode --low-memory` disables MLAS
prepacking (M4, `../resources.md`): fp32 starts in 0.11 s with 140 MiB
private memory (113 MB physical footprint after queries), int8 in 0.63 s /
149 MiB; weight pages are file-backed and shared between processes; batch
throughput −5 %, single short queries 2× slower. Disabling the arena was
measured and rejected: it returns no memory after a long request and costs
100–200 MiB more.

## v0.5.2 — activation memory (done)

`fuse` moves the whole layer tail into the attention `Loop` and the chunk is
256 (`../memory.md`): fp32 peak per padded token 0.10–0.125 → 0.064–0.075
MiB for texts of 512+ tokens (−30–40 %), +20 % for batches of texts no longer
than the chunk; int8 ships chunk 256 without the tail (−7–33 %, the loop
costs memory there). Bit-identical outputs, ±1 % on 16/128-token batches,
−4 % at 512 tokens. `quantize` rebuilds the unchunked fused graph from
`model.onnx` in memory, so the int8 weight file is unchanged since v0.5.0
and `--attention-chunk` / `--layer-loop` take effect. Lesson: onnxruntime's
memory pattern never applies to the first run of a shape; the BFC arena grows
in powers of two, and the peak follows the allocation sequence, so every
layout change must be measured (`kSameAsRequested`, shrinkage, chunk 512 +
loop all lost).

## v0.6.0 — asyncio serving (done)

`AsyncEmbedder` (`../serving.md`, measured with `tools/bench_serving.py` on
the M4 and the CI matrix, 2026-09-06; no model asset changes): `await
encode / encode_queries / encode_corpus` with the synchronous signatures and
outputs, run in a private thread pool with at most `max_concurrency` calls in
flight (default 2 for fp32, 4 for int8), `queue_depth` / `in_flight` for
health endpoints, `async with` / `close()` that drains the queue. Facts: one
4-thread session beats four 1-thread sessions; fp32 short queries are
GEMM-bound (two runs in flight +50 % at equal CPU, then flat; padding 8 into
one call 3.6×), int8 scales with runs in flight (2.1× at 4, 2.4× at 8) and
loses with batching; passages gain 25–40 % from 2–4 runs in flight; each run
in flight adds its own activations (0.07 MiB per padded token). The
micro-batcher (`batch_window_ms`) holds a request only while every slot is
busy or claimed and takes a burst as one call, so it costs nothing at low
load: fp32 short queries 2.5× the sequential rate at 4 clients (p95 34 ms),
3.6× at 8, at a third of the CPU per request; on by default for fp32, off for
int8 (−6–24 %).
