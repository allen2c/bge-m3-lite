# Development

## Setup and checks

```bash
uv sync --group dev
uv run ruff format . && uv run ruff check . && uv run pyright
uv run pyproject-fmt --check pyproject.toml
uv run pytest                                   # downloads ~7 MB of small files once
BGE_M3_LITE_RUN_SLOW=1 uv run pytest -m slow    # needs the 2.3 GB model in the cache
uv build                                        # pure-Python wheel (~30 KB)
```

## Fixtures (reference stack, never needed at runtime)

```bash
uv sync --group ref                      # torch, transformers, sentencepiece, FlagEmbedding
uv run tools/make_tokenizer_fixtures.py  # tests/fixtures/tokenizer_cases.json
uv run tools/make_embedding_fixtures.py  # tests/fixtures/embeddings_ref.{json,npz}
uv run tools/gen_grapheme_tables.py DIR  # DIR holds the Unicode 16 data files
```

`tests/fixtures/GraphemeBreakTest.txt` is the official Unicode test file.

## Local cross-platform check

```bash
docker run --rm --platform linux/arm64 -v $PWD:/src:ro python:3.12-slim sh -c \
  'pip -q install uv && cp -r /src /w && cd /w && uv sync -q --group dev && uv run pytest -q'
# linux/amd64 works the same way through QEMU (slower).
```

## Release (manual, run by the maintainer)

1. Bump `version` in `pyproject.toml` and `bge_m3_lite/__init__.py`, commit.
2. Run every check above, including the slow suite.
3. Tag and push:
   ```bash
   git tag -a v0.0.1 -m "v0.0.1"
   git push origin main --tags
   ```
4. Build and publish (token from https://pypi.org/manage/account/token/):
   ```bash
   rm -rf dist && uv build
   uv publish --token pypi-...            # or: UV_PUBLISH_TOKEN=... uv publish
   uv publish --index testpypi ...        # optional dry run on TestPyPI first
   ```
   Model files are not part of the wheel; they are pinned by revision + SHA-256
   in `hub.py`.

## Roadmap

- v0.0.2: int8 dynamic quantisation of `model.onnx` as a build-time step,
  validated against the same fixtures (watch sparse ordering), hosted as a
  release asset; optional fp32/int8 switch.
- Cache the ORT-optimised graph to cut the 3 s start-up.
- Fused attention via `onnxruntime.transformers` at build time.
- Rust/maturin kernels only if ORT int8 underperforms on Apple Silicon.
