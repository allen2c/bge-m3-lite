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
   git tag -a v0.1.0 -m "v0.1.0"
   git push origin main --tags
   ```
4. Pushing the tag runs `.github/workflows/release.yml`: checks, build, then
   publish to PyPI via trusted publishing and attach the wheel to a GitHub
   release. One-time setup:
   - PyPI → your account → *Publishing* → add a pending publisher:
     project `bge-m3-lite`, owner/repo of this repository, workflow
     `release.yml`, environment `pypi`.
   - GitHub repo → Settings → Environments → create `pypi` (optionally require
     reviewers).
   Manual fallback: `uv build && uv publish --token pypi-...`.
5. Upload `model_int8.onnx` to the same release (see *Release assets*).
   Model files are never part of the wheel; they are pinned by revision +
   SHA-256 in `hub.py`.

## Release assets

`model_int8.onnx` (543 MiB) is not on the Hub. Build it with
`bge-m3-lite quantize`, check the printed SHA-256 against `hub.INT8_FILE`, and
upload it to the GitHub release named in `hub.INT8_RELEASE`:

```bash
gh release create v0.0.2 ~/.cache/bge-m3-lite/BAAI--bge-m3/model_int8.onnx --title v0.0.2
```

## Roadmap

See `roadmap.md` for the version plan and the measurements behind it.
