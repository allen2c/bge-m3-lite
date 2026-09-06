# AGENTS.md

`bge-m3-lite`: CPU inference for BAAI/bge-m3 (dense + sparse + ColBERT) with
`onnxruntime` as the only runtime dependency; everything else is written here.

## Rules

- Never add a runtime dependency. Dev tools go in `[dependency-groups] dev`,
  build-time ONNX tooling in the `quant` extra, torch/transformers in `ref`.
- Flat layout (`bge_m3_lite/`, `tests/`, `tools/`, `docs/`), no `src/`.
- fp32 outputs must match FlagEmbedding: `tests/fixtures/` is the contract.
- Verify claims empirically (CI `bench` matrix for anything platform-specific)
  and record the numbers in `docs/`; do not rely on memory.

## Commands

```bash
uv sync --group dev --group quant
uv run ruff format . && uv run ruff check . && uv run pyright
uv run pyproject-fmt --check pyproject.toml && uv run pytest
BGE_M3_LITE_RUN_SLOW=1 uv run pytest -m slow    # full model, 2.3 GB cache
```

## Docs (`docs/`)

| file | read it for |
|---|---|
| `architecture.md` | modules, data flow, model files, cache |
| `tokenizer.md` | the from-scratch XLM-R tokenizer |
| `fusion.md` | fused fp32 graph: what ships, how it is built |
| `quantization.md` | int8 backbone: recipe, platform findings, measurements |
| `calibration.md` | calibration texts (sources, licence) and the held-out evaluation set |
| `memory.md` | attention in query chunks: why long inputs no longer need 4 GiB |
| `verification.md` | accuracy, platforms, throughput, start-up, memory |
| `development.md` | fixtures, CI, release and asset upload |
| `roadmap.md` | what was done, why, and what is left |
