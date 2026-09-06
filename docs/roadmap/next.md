# Roadmap: what is next

Shipped versions and the facts behind them: `done.md`.

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
