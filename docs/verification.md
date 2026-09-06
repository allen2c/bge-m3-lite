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

| runner | CPU | fp32 fused 128-tok | int8 v3 128-tok | int8 speed-up |
|---|---|---|---|---|
| `ubuntu-latest` x86_64 | Xeon Platinum 8573C (AVX-512 VNNI) | ~490 tok/s | 742 tok/s | 1.5× |
| `ubuntu-latest` x86_64 | AMD EPYC 7763 (AVX2) | 275 tok/s | 352 tok/s | 1.3× |
| `ubuntu-24.04-arm` | Neoverse-N2 | 316 tok/s | 1120 tok/s | 3.5× |
| `macos-latest` (VM) | Apple Silicon, 3 cores | 127 tok/s | 319 tok/s | 2.5× |

`ubuntu-latest` alternates between the two x86 CPUs; int8 accuracy is the same
on every runner (`quantization.md`).

fp32 is exact on every runner (dense cosine 1.0, sparse and ColBERT identical).
The macOS runner is a throttled VM; use the M4 numbers above for Apple Silicon.

## Start-up (M4, files in the page cache)

| step | v0.0.2 | v0.1.0 |
|---|---|---|
| imports | 0.07 s | 0.07 s |
| tokenizer (250 000 pieces) | 0.31 s | 0.05 s (vocabulary cache) |
| ORT session, fp32 | 0.44 s | 0.44 s |
| total | 0.86 s | 0.6 s |

Cold start is dominated by reading 2.3 GB of weights; the int8 backbone
(569 MB) starts in about a third of the time. Saving the ORT-optimised graph
does not help (0.38 s → 0.33 s for session creation), see `roadmap.md`.

## Memory

The hidden state of one batch is `padded_tokens × 4 KiB`; unfused attention
additionally materialises `batch × 16 × seq² × 4 bytes` per layer (4 GiB for
a single 8192-token text). `encode(..., max_batch_tokens=16384)` (default)
bounds the padded tokens per batch, so mixed inputs are safe; lower it for
long documents on small machines.
