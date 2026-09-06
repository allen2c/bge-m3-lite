# Development

## Setup and checks

```bash
uv sync --group dev --group quant
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
uv run tools/make_heldout.py             # tests/fixtures/heldout_* from the fp32 fused graph (no ref stack)
uv run tools/eval_model.py MODEL.onnx    # accuracy on both sets + tok/s; CI bench turns this into a table
```

`tests/fixtures/GraphemeBreakTest.txt` is the official Unicode test file.

## Local cross-platform check

```bash
docker run --rm --platform linux/arm64 -v $PWD:/src:ro python:3.12-slim sh -c \
  'pip -q install uv && cp -r /src /w && cd /w && uv sync -q --group dev && uv run pytest -q'
# linux/amd64 works the same way through QEMU (slower).
```

## Release (maintainer)

1. Bump `version` in `pyproject.toml` and `bge_m3_lite/__init__.py`; if a
   model asset changed, pin its size + SHA-256 and release URL in `hub.py`.
2. Run every check above, including the slow suite. New int8 recipes must be
   validated on the CI `bench` matrix first (`workflow_dispatch`, input
   `quantize_variants`, e.g. `--alpha 0.65|--method dynamic`; `fuse_local`
   also builds and times the fused fp32 graph from the checked-out code).
   The job summary shows the CPU model and one accuracy + tok/s row per
   graph: `ubuntu-latest` alternates between a Xeon (VNNI) and an EPYC (AVX2),
   and the development machine (Apple Silicon) is not representative.
3. Tag and push; `release.yml` runs the checks, publishes to PyPI via trusted
   publishing (environment `pypi`) and creates the GitHub release:
   ```bash
   git tag -a v0.4.0 -m v0.4.0 && git push origin v0.4.0
   ```
4. Upload the model assets the new `hub.py` points at (deterministic builds,
   compare the printed digests first). Run `gh` inside the repository: leaving
   it unloads the direnv-provided `GH_TOKEN`.
   ```bash
   bge-m3-lite fuse        # model_fused.onnx + model_fused.onnx_data (hub.FUSED_FILES)
   bge-m3-lite quantize    # model_int8.onnx (hub.INT8_FILE)
   gh release upload v0.4.0 -R allen2c/bge-m3-lite ~/.cache/bge-m3-lite/BAAI--bge-m3/model_fused.onnx ~/.cache/bge-m3-lite/BAAI--bge-m3/model_int8.onnx
   ```
   Until the upload finishes, fresh installs of that version cannot download
   the asset. Unchanged assets stay on their old release (per-file URLs).

One-time setup: PyPI pending publisher (`release.yml`, environment `pypi`)
and the `pypi` environment in the repository settings. Manual fallback:
`uv build && uv publish --token pypi-...`.
