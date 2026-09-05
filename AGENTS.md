# AGENTS.md

`bge-m3-lite`: CPU inference for BAAI/bge-m3 (dense + sparse + ColBERT).
`onnxruntime` is the **only** runtime dependency; everything else is written
here from scratch. Read `docs/` before changing anything non-trivial.

## Rules

- Never add a runtime dependency. Dev tooling goes in `[dependency-groups] dev`,
  the torch/transformers reference stack in `ref`.
- Flat layout: `bge_m3_lite/`, no `src/`. Tests in `tests/`, scripts in `tools/`.
- Outputs must stay identical to FlagEmbedding; the fixtures in `tests/fixtures/`
  are the contract. Verify facts empirically, do not rely on memory.

## Commands

```bash
uv sync --group dev --group quant   # quant: onnx for pyright + quantize
uv run ruff format . && uv run ruff check . && uv run pyright
uv run pyproject-fmt --check pyproject.toml
uv run pytest                                   # fast
BGE_M3_LITE_RUN_SLOW=1 uv run pytest -m slow    # full model (2.3 GB cache)
```

## Docs

- `docs/architecture.md` – modules, data flow, model files, verified facts
- `docs/tokenizer.md` – how the from-scratch tokenizer matches transformers
- `docs/verification.md` – accuracy, platforms, performance numbers
- `docs/quantization.md` – int8 backbone: how it is built, measured trade-offs
- `docs/development.md` – fixtures, CI, release
- `docs/roadmap.md` – version plan and the measurements behind it
