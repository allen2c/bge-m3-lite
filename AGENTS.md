# AGENTS.md

`bge-m3-lite`: CPU inference for BAAI/bge-m3 (dense + sparse + ColBERT) with
`onnxruntime` as the only runtime dependency; everything else is written here.

## Rules

- Never add a runtime dependency. Dev tools go in `[dependency-groups] dev`,
  build-time ONNX tooling in the `quant` extra, torch/transformers in `ref`.
- Flat layout (`bge_m3_lite/`, `tests/`, `tools/`, `docs/`), no `src/`.
- fp32 outputs must match FlagEmbedding: `tests/fixtures/` is the contract.
- Verify claims empirically (CI `bench` matrix for anything platform-specific;
  the development Mac is not representative) and record the numbers in
  `docs/`; do not rely on memory. Benchmark with nothing else running.
- Docs stay under 100 lines each (150 max); split into folders instead.

## Commands

```bash
uv sync --group dev --group quant
uv run ruff format . && uv run ruff check . && uv run pyright
uv run pyproject-fmt --check pyproject.toml && uv run pytest
BGE_M3_LITE_RUN_SLOW=1 uv run pytest -m slow    # full model, 2.3 GB cache
```

## Docs (`docs/`)

| path | read it for |
|---|---|
| `architecture.md` | modules, data flow, model files, cache |
| `tokenizer.md` | the from-scratch XLM-R tokenizer |
| `fusion.md` | fused fp32 graph: what ships, how it is built |
| `quantization/recipe.md` | int8 backbone: SmoothQuant + row-wise scheme, variants |
| `quantization/measurements.md` | int8 accuracy and speed per platform |
| `calibration.md` | calibration texts (sources, licence), held-out evaluation set |
| `memory.md` | attention in query chunks; activation memory per padded token, `max_batch_tokens` budget |
| `resources.md` | resident memory, CPU-seconds per token, threads, idle CPU, `low_memory` (v0.5) |
| `verification.md` | accuracy, platforms, throughput, start-up, memory |
| `development.md` | fixtures, CI bench inputs, release and asset upload |
| `roadmap/done.md` | shipped versions and the facts behind them |
| `roadmap/next.md` | open items: v0.5.2 activation memory, int8 on Xeon VNNI |
