# AGENTS.md

`bge-m3-lite`: CPU inference for BAAI/bge-m3 (dense + sparse + ColBERT) with
`onnxruntime` as the only runtime dependency; everything else is written here.

## Rules

- Never add a runtime dependency (`asyncio`, `threading` are stdlib). Dev
  tools go in `[dependency-groups] dev`, build-time ONNX tooling in the
  `quant` extra, torch/transformers in `ref`.
- Flat layout (`bge_m3_lite/`, `tests/`, `tools/`, `docs/`), no `src/`.
- fp32 outputs must match FlagEmbedding: `tests/fixtures/` is the contract;
  `AsyncEmbedder` must stay bit-exact with the synchronous API.
- Verify claims empirically (CI `bench` matrix for anything platform-specific;
  the development Mac is not representative and runners can disagree with
  it) and record the numbers in `docs/`; do not rely on memory. Benchmark
  with nothing else running (`ps` for stale pytest processes first).
- Test on Python 3.11 and 3.13 (asyncio differs; `pytest-timeout` caps a
  test at 120 s; see `docs/development.md`).
- Docs stay under 100 lines each (150 max); split into folders instead.

## Commands

```bash
uv sync --group dev --group quant
uv run ruff format . && uv run ruff check . && uv run pyright
uv run pyproject-fmt --check pyproject.toml && uv run pytest
UV_PROJECT_ENVIRONMENT=.venv313 uv run -p 3.13 --group dev pytest -q
BGE_M3_LITE_RUN_SLOW=1 uv run pytest -m slow    # full model, 2.3 GB cache
uv run tools/bench_serving.py --precision int8  # serving numbers, run alone
```

## Docs (`docs/`)

| path | read it for |
|---|---|
| `architecture.md` | modules, data flow, model files, cache |
| `tokenizer.md` | the from-scratch XLM-R tokenizer |
| `fusion.md` | fused fp32 graph: what ships, how it is built |
| `quantization/recipe.md`, `measurements.md` | int8 backbone: SmoothQuant + row-wise scheme; accuracy and speed per platform |
| `calibration.md` | calibration texts (sources, licence), held-out evaluation set |
| `memory.md` | attention `Loop` (chunk 256) + layer tail in a row `Loop` (`--tail`, `--tail-rows`); activation memory per padded token, `max_batch_tokens` |
| `resources.md` | resident memory, CPU-seconds per token and per request, threads, idle CPU, `low_memory`, start-up |
| `serving/recipe.md`, `measurements.md` | `AsyncEmbedder` for FastAPI: defaults, micro-batcher and its length buckets, workers × memory; req/s tables (M4, CI runners), `run_async` verdict |
| `verification.md` | accuracy, platforms, throughput, start-up, memory |
| `development.md` | fixtures, CI bench inputs, Python 3.13 check, release and asset upload |
| `roadmap/done.md` | shipped versions and the facts behind them |
| `roadmap/next.md` | v0.6.2 plan (asyncio 3.11 vs 3.13), closed decisions, later candidates |
