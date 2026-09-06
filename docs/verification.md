# Verification

## Accuracy versus FlagEmbedding (fp32, 2026-09-05)

Reference: `BGEM3FlagModel("BAAI/bge-m3", use_fp16=False)` on 11 texts
(8–761 tokens, 6 scripts, one empty string), fixtures in
`tests/fixtures/embeddings_ref.{json,npz}` via `tools/make_embedding_fixtures.py`.

| output | result |
|---|---|
| dense | cosine 1.0, max abs diff 4.5e-7 |
| sparse | identical key sets and top-5 order, max abs diff 2.2e-6 |
| colbert | per-token cosine 1.0, max abs diff 2e-5 (761-token text) |

Run with `BGE_M3_LITE_RUN_SLOW=1 uv run pytest -m slow` (tolerances 1e-4 / 1e-3).

## Platforms

| platform | how | result |
|---|---|---|
| macOS arm64 (M4) | native | fast + slow suites pass |
| Linux aarch64 | Docker `python:3.12-slim` on Apple Silicon; CI `ubuntu-24.04-arm` | fast + slow suites pass, ORT 1.29.0 |
| Linux x86_64 | CI `ubuntu-latest` (Xeon Platinum 8573C) | fast suite + full-model `bench` pass, fp32 exact |
| Windows x86_64 | CI `windows-latest` | fast suite, including the int8 graph unit tests |

`ubuntu-24.04-arm` runners are free for public repositories only. The `bench`
job (`workflow_dispatch`) downloads the model and runs `tools/eval_model.py`
on all three Linux/macOS runners.

## Performance baseline (M4, fp32, ORT default threads)

| workload | throughput |
|---|---|
| 16 tokens × 32 texts | 2100 tok/s (131 texts/s) |
| 128 tokens × 16 | 2007 tok/s |
| 512 tokens × 4 | 1655 tok/s |

## GitHub-hosted runners (`tools/eval_model.py`, 4 vCPU)

v0.5.0 (no spinning, ORT default threads = physical cores; memory and CPU
columns in `resources.md`):

| runner | CPU | fp32 fused 128-tok | int8 v4 128-tok | int8 speed-up |
|---|---|---|---|---|
| `ubuntu-latest` x86_64 | Xeon Platinum 8573C (AVX-512 VNNI) | 576 tok/s | 494 tok/s | 0.86× (regression, `roadmap/next.md`) |
| `ubuntu-24.04-arm` | Neoverse-N2 | 306 tok/s | 1100 tok/s | 3.6× |
| `macos-latest` (VM) | Apple M1, 3 cores | 232 tok/s | 581 tok/s | 2.5× |

v0.4.0 (chunked attention, int8 v4):

| runner | CPU | fp32 fused 128-tok | int8 v4 128-tok | int8 speed-up |
|---|---|---|---|---|
| `ubuntu-latest` x86_64 | AMD EPYC 7763 (AVX2) | 263 tok/s | 368 tok/s | 1.4× |
| `ubuntu-24.04-arm` | Neoverse-N2 | 314 tok/s | 1117 tok/s | 3.6× |
| `macos-latest` (VM) | Apple M1, 3 cores | 142 tok/s | 411 tok/s | 2.9× |

`ubuntu-latest` alternates between the two x86 CPUs; int8 accuracy is the same
on every runner (`quantization/measurements.md`).

fp32 is exact on every runner (dense cosine 1.0, sparse and ColBERT identical).
The macOS runner is a throttled VM; use the M4 numbers above for Apple Silicon.

## Start-up (M4, files in the page cache)

| step | v0.0.2 | v0.1.0 |
|---|---|---|
| imports | 0.07 s | 0.07 s |
| tokenizer (250 000 pieces) | 0.31 s | 0.05 s (vocabulary cache) |
| ORT session, fp32 | 0.44 s | 0.44 s |
| total | 0.86 s | 0.6 s |

Cold start is dominated by reading 2.3 GB of weights. The int8 session
opens in 0.31 s since v0.6.1 (0.69 s before: the outer graph shrank from
2 692 to 1 396 nodes with the layer tail in a `Loop`, `memory.md`); the
optimisation level makes no difference to either (0.67–0.72 s at every level
for the v0.5.2 int8 graph), nor does saving the ORT-optimised graph (0.38 s
→ 0.33 s for fp32), see `roadmap/done.md`.

## Memory

The hidden state of one batch is `padded_tokens × 4 KiB`. Since v0.4 attention
runs in query chunks (256 since v0.5.2) and since v0.6.1 the layer tail in
256-row windows (`memory.md`): one 8192-token text peaks at 1.8 GB RSS
with either backbone (7.4–11.7 GB before v0.4) and any batch costs about
0.07 MiB per padded token on top of the loaded model. `encode(...,
max_batch_tokens=16384)` (default) bounds the padded tokens per batch, so
mixed inputs are safe; lower it for long documents on small machines.
Resident memory per backbone, CPU-seconds per token and the idle cost of a
session: `resources.md` (v0.5: int8 724 MiB after load, no idle CPU).

## Held-out set

`tests/fixtures/heldout_*` (40 texts, `calibration.md`) is evaluated by
`tools/eval_model.py` next to the 11 FlagEmbedding texts; the fp32 fused
graph reproduces its own reference exactly, the int8 numbers are in
`quantization/measurements.md`.
