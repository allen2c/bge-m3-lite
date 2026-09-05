# Architecture

## Data flow

```
text ─► tokenizer.py ─► input_ids/attention_mask (int64)
     ─► model.py (onnxruntime, model.onnx) ─► token_embeddings (B, S, 1024) fp32
     ─► embedder.py:
          dense   = normalize(h[:, 0])
          sparse  = relu(h @ Wsp + b) → max per token id, drop <s> </s> <pad> <unk>, keep w > 0
          colbert = normalize(h[:, 1:n_tokens] @ Wc + bc)      (keeps </s>, drops padding)
```

Formulas follow `FlagEmbedding/inference/embedder/encoder_only/m3.py` and
`finetune/embedder/encoder_only/m3/modeling.py`. `lexical_weights` keys are
`str(token_id)` for compatibility.

## Modules

| file | role |
|---|---|
| `tokenizer.py` | XLM-R tokenizer: SentencePiece unigram + HF `Precompiled` charsmap |
| `_grapheme.py`, `_grapheme_data.py` | UAX #29 grapheme clusters (tables generated from Unicode 16) |
| `_proto.py` | minimal protobuf reader for `sentencepiece.bpe.model` |
| `_torch_pickle.py` | torch-free loader for the two `.pt` heads (strict class allowlist) |
| `hub.py` | pinned file table, SHA-256, resumable download, cross-process lock |
| `model.py` | `OnnxBackbone`: ORT session, CPU provider, `ORT_ENABLE_ALL` |
| `embedder.py` | `BGEM3Embedder`: batching (longest first), pooling, scoring helpers |
| `cli.py` | `bge-m3-lite download / info / encode` |

## Model files (Hugging Face `BAAI/bge-m3`, revision `5617a9f6`)

| file | size | note |
|---|---|---|
| `onnx/model.onnx` + `model.onnx_data` + `Constant_7_attr__value` | 2.27 GB | official opset-11 export, outputs `token_embeddings`, `sentence_embedding` |
| `sentencepiece.bpe.model` | 5 MB | 250 000 pieces, unigram, `nmt_nfkc` precompiled charsmap |
| `colbert_linear.pt` | 2 MB | Linear(1024→1024), stored fp16 |
| `sparse_linear.pt` | 3.5 KB | Linear(1024→1), stored fp16 |

Cache: `~/.cache/bge-m3-lite/BAAI--bge-m3` (`BGE_M3_LITE_CACHE`), mirror via
`HF_ENDPOINT`, `BGE_M3_LITE_OFFLINE=1` disables downloads. Files under 64 MiB are
re-hashed on every load; the big model data is hashed once at download time.

## Facts verified against the sources (2026-09-05)

- The Hub repo has no safetensors; only `pytorch_model.bin` plus the two heads.
- The official ONNX already exposes the last hidden state, so all three outputs
  can be derived without a third-party ONNX or `onnxruntime-extensions`.
- `onnxruntime` 1.29.0 ships wheels for cp311–cp314 on macOS arm64, manylinux
  aarch64/x86_64 and Windows; its dependencies are numpy, protobuf, flatbuffers,
  packaging. numpy is therefore usable inside this package.
- Pure Python matmul is ~0.07 GFLOP/s versus ~760 GFLOP/s for Accelerate on an
  M4; a from-scratch Python forward pass is not viable, hence onnxruntime.
