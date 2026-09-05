# bge-m3-lite

CPU inference for [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) with
**`onnxruntime` as the only dependency**. All three BGE-M3 outputs are supported
and match the official PyTorch implementation (FlagEmbedding) to fp32 precision:

| output | shape | notes |
|---|---|---|
| `dense_vecs` | `(n, 1024)` | CLS pooling, L2-normalised |
| `lexical_weights` | `list[dict[str, float]]` | token-id → weight, max-pooled, specials removed |
| `colbert_vecs` | `list[(len-1, 1024)]` | per-token vectors without `<s>`, L2-normalised |

Everything except the transformer forward pass is implemented in this package
from scratch: the XLM-RoBERTa tokenizer (SentencePiece unigram model, the
`nmt_nfkc` precompiled charsmap, Unicode grapheme segmentation), the torch-free
loader for the sparse / ColBERT heads, the model downloader and the pooling.

Platforms: Apple Silicon, Linux ARM64, Linux x86_64, Windows x86_64 (Python 3.11+).

## Install

```bash
uv add bge-m3-lite        # or: pip install bge-m3-lite
```

## Use

```python
from bge_m3_lite import BGEM3Embedder

embedder = BGEM3Embedder()  # first call downloads ~2.3 GB into ~/.cache/bge-m3-lite
out = embedder.encode(
    ["What is BGE M3?", "BGE M3 是一個多語言嵌入模型。"],
    return_dense=True,
    return_sparse=True,
    return_colbert_vecs=True,
)
out["dense_vecs"].shape  # (2, 1024)
out["lexical_weights"][0]  # {'4865': 0.08, '83': 0.08, ...}
out["colbert_vecs"][0].shape  # (7, 1024)

embedder.convert_id_to_token(out["lexical_weights"][0])
embedder.compute_lexical_matching_score(lw_query, lw_passage)
embedder.colbert_score(q_vecs, p_vecs)

# retrieval helpers: queries default to 512 tokens, passages to max_length (8192)
q = embedder.encode_queries(["What is BGE M3?"])
p = embedder.encode_corpus(["BGE M3 is a multilingual embedding model ..."])
embedder.compute_score([("What is BGE M3?", "BGE M3 is ...")])
# {'colbert': [...], 'sparse': [...], 'dense': [...], 'sparse+dense': [...], 'colbert+sparse+dense': [...]}
```

Passing a single string returns unwrapped values, like FlagEmbedding.
Batches are bounded by `batch_size` texts **and** `max_batch_tokens` padded
tokens (default 16384), so mixing short and 8192-token inputs stays within
memory.
`BGEM3Embedder(precision="int8")` loads a 4× smaller quantised backbone
(see `docs/quantization.md` for the accuracy trade-off).

### CLI

```bash
bge-m3-lite download                     # pre-fetch the model files (2.3 GB + 288 MB fused)
bge-m3-lite info                         # cache state
echo "hello" | bge-m3-lite encode --sparse --colbert --tokens
```

### Environment variables

| variable | effect |
|---|---|
| `BGE_M3_LITE_CACHE` | cache directory (default `~/.cache/bge-m3-lite/BAAI--bge-m3`) |
| `HF_ENDPOINT` | Hugging Face mirror, e.g. `https://hf-mirror.com` |
| `BGE_M3_LITE_OFFLINE=1` | never download, fail if files are missing |
| `BGE_M3_LITE_THREADS` | onnxruntime intra-op threads (default: physical cores) |
| `BGE_M3_LITE_FUSED_URL`, `BGE_M3_LITE_INT8_URL` | mirror for the fused / int8 release assets |

Model files are pinned to a specific Hugging Face revision and verified by
SHA-256 after download.

## Development

See `AGENTS.md` and `docs/` (architecture, tokenizer, verification, development).

## Status

v0.2.0: fp32 with exact parity with FlagEmbedding (fused graph by default),
opt-in int8 backbone, retrieval helpers, token-budget batching, Windows.
Plan: `docs/roadmap.md`.
