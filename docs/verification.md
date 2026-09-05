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
| Linux aarch64 | Docker `python:3.12-slim` on Apple Silicon | fast + slow suites pass, ORT 1.29.0 |
| Linux x86_64 | Docker with QEMU | fast suite passes, ORT 1.29.0 |
| CI | `.github/workflows/ci.yml` matrix | not yet run on GitHub |

`ubuntu-24.04-arm` runners are free for public repositories only.

## Performance baseline (M4, fp32, ORT default threads)

| workload | throughput |
|---|---|
| 16 tokens × 32 texts | 1970 tok/s (123 texts/s) |
| 128 tokens × 16 | 1780 tok/s |
| 512 tokens × 4 | 1540 tok/s |

Session start ≈ 3 s. One 128-token forward ≈ 77 GFLOP (302 M non-embedding
parameters). Memory per batch ≈ `batch_size × seq_len × 4 KiB` for the hidden
state, plus the 2.3 GB weights.
