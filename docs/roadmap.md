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

Measured on the M4 and on the CI matrix (2026-09-06, `quantization.md`,
`memory.md`); ships new `model_fused.onnx` (graph only, the data file is
unchanged) and `model_int8.onnx` assets.

| goal | result |
|---|---|
| long inputs must not need 4 GiB | attention in query chunks (`Loop`, 512 rows): 8192 tokens 7.4 GB → 2.5 GB, bit-exact, ≤ 3 % on short inputs (`memory.md`) |
| int8 element-wise overhead | per-axis `QuantizeLinear` + `MatMulIntegerToFloat`: +11 % on the M4, +6–9 % on EPYC (AVX2), +19 % on the macOS VM, +1–3 % on Neoverse (GEMM-bound); same accuracy. The ORT-fusable per-tensor form is out of reach (per-row zero point unsupported); `--symmetric` gets closer but loses accuracy (`quantization.md`) |
| sparse int8 accuracy | measured, not fixed: the flips are fp32 near-ties (gap 0.001–0.009); last-layer-fp32 variants change ±1 text, so the recipe stays; `--keep-fp32` remains for experiments |
| calibration provenance | 212 hand-written + 360 MIRACL passages, licence and recipe in `calibration.md`; 40-text held-out set disjoint from calibration, reported by `tools/eval_model.py` |
| engineering | `hub.py` retries 429/5xx/timeouts with back-off; the CI `bench` summary prints the CPU model and one accuracy + tok/s row per graph |

## v0.4.x candidates (measure first)

- Xeon (VNNI) numbers for v0.4: both bench runs drew the EPYC runner.
  If the gain there is also below +20 %, profile the remaining scalar ops
  (`ReduceMax/ReduceMin` are two extra passes over the input).
- `attention_chunk` per platform: 512 was chosen on the M4; the CI matrix
  may prefer 1024 (fewer iterations) on 4-vCPU runners.
- Start-up of the int8 graph (1.1 s vs 0.4 s fp32): 2 700 nodes cost ORT
  session time; folding the scalar scale/zero-point chain into fewer ops
  would also help here.

## v0.5 — resource efficiency (planned, numbers measured on the M4, 2026-09-06)

Speed is parked after v0.4; the next versions minimise memory and CPU-seconds
per token. Baseline (`onnxruntime` defaults, int8 = single 597 MB file):

| configuration | RSS after load | RSS after 128×16 | 8192-tok peak | start-up | CPU ms per short query | idle CPU |
|---|---|---|---|---|---|---|
| int8 v0.4 as shipped | 1786 MiB | 1904 MiB | 2515 MiB | 0.88 s | 54 | 57 ms/s |
| int8 as graph + external data (mmap) | 653 MiB | 968 MiB | 1451 MiB | 0.69 s | 50 | 41 ms/s |
| + 4 threads (P-cores), no spinning | 653 MiB | 968 MiB | | 0.69 s | 29 | 0 |
| + no prepacking (`low_memory`) | 72 MiB | 677 MiB | | 0.56 s | 68–99 | 0 |
| fp32 fused as shipped | 1217 MiB | 1572 MiB | 2148 MiB | 0.73 s | 87 | 64 ms/s |
| fp32, no spinning | 1217 MiB | 1566 MiB | | 0.36 s | 73 | 0 |
| fp32, no prepacking | 62 MiB | 1588 MiB (file-backed) | | 0.04 s | 198 | |

Findings behind the plan:

- ORT keeps the parsed protobuf of an embedded-weights model *and* the
  prepacked MLAS copy: shipping `model_int8.onnx` as graph + `.onnx_data`
  (like the fused graph) drops 1.1 GB of resident memory at no speed cost.
- ORT's intra-op threads spin after each run: 57–64 ms of CPU per idle
  second per session, and 30–40 % of the CPU time of a short query.
  `session.intra_op.allow_spinning=0` removes it; throughput −0–5 %.
- On the M4 (4P + 6E) four threads give the same throughput as ten with
  22–27 % less CPU time; one thread does 2160 tok per CPU-second versus
  1000 with ten (throughput servers should run several one-thread
  workers). Tokenizer and pooling are < 4 % of CPU.
- Disabling prepacking makes start-up 0.04 s (fp32) / 0.56 s (int8) with
  60–70 MiB resident; the weights are then file-backed and reclaimable, but
  single short queries are 2× slower (MLAS packs B on every call).

Versions:

- **v0.5.0** — int8 asset as graph + external data (new `INT8_FILES` pin,
  CI matrix check of external-data loading on Windows); `allow_spinning=0`
  by default (`BGE_M3_LITE_SPIN=1` to restore); default threads = performance
  cores on macOS, physical cores elsewhere; `docs/resources.md` with the
  table above re-measured on the CI matrix; `tools/eval_model.py` reports
  CPU-seconds per 1k tokens and peak RSS next to tok/s.
- **v0.5.1** — `BGEM3Embedder(low_memory=True)`: no prepacking, arena off,
  for serverless / CLI one-shot use (start-up 0.04–0.5 s, tens of MiB
  resident); `--low-memory` on the CLI; document the 2× short-query cost.
- **v0.5.2** — activation memory: FFN intermediates (`padded_tokens ×
  16 KiB` per layer) in token blocks inside the existing `Loop`, and
  `max_batch_tokens` guidance per RSS budget; measure whether the arena
  shrink run option (`memory.enable_memory_arena_shrinkage`, no effect in
  the first test) can return memory between requests.
- **v0.6 (candidates)** — weight-only int8 `MatMulNBits` for Apple Silicon
  (memory only; int8 GEMM is slower than SGEMM there); embeddings served from
  the mmapped file instead of the int8 `Gather` copy; sparse-head rounding
  experiments with the held-out set.

## Later

- Rust/maturin kernels: only if onnxruntime is still the bottleneck
  (the pure-Python tokenizer and the fixtures are the correctness contract).
