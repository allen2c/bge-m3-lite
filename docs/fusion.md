# Fused fp32 backbone

`BGEM3Embedder()` runs a fused version of the Hub ONNX export by default.
The outputs are unchanged (the slow suite compares against FlagEmbedding with
the same tolerances as before); `fused=False` or `encode --raw` keeps the
original graph.

## What is fused

The Hub file is an opset-11 export in which every attention block is a dozen
`MatMul` / `Reshape` / `Transpose` / `Softmax` nodes and LayerNorm is spelled
out with `ReduceMean` / `Sub` / `Pow` / `Sqrt`. onnxruntime's own graph
optimiser (`ORT_ENABLE_ALL`) fuses LayerNorm and Gelu at load time but never
Attention on this export. `onnxruntime.transformers.optimizer` (model type
`bert`, 16 heads, hidden 1024) rewrites it into contrib ops:

| op | count |
|---|---|
| `Attention` | 24 |
| `SkipLayerNormalization` | 48 |
| `BiasGelu` | 24 |
| `MatMul` (FFN + output projections) | 72 |

## How it ships

Of the 293 initializers in the fused graph, 245 are byte-identical to tensors
in the Hub `model.onnx_data`; the 48 others are the per-layer merged QKV
weights `[1024, 3072]` and biases. The release therefore holds two small files
(built by `bge-m3-lite fuse`, deterministic, pinned in `hub.FUSED_FILES`):

| file | size | content |
|---|---|---|
| `model_fused.onnx` | 252 KB | graph (declares the `com.microsoft` opset); shared tensors reference `model.onnx_data` by offset |
| `model_fused.onnx_data` | 288 MiB | the 48 merged QKV tensors |

Both live in the cache next to `model.onnx_data`, which is why the fp32
download (2.3 GB) is still needed. Since v0.4 every `Attention` is rewritten
into `MatMul` + `Split` + a `Loop` over query chunks, and since v0.6.1 the
rest of the layer runs in a second `Loop` over 256 rows of the flattened
batch (`fuse --tail rows`, `memory.md`); the outputs are unchanged and the
data file is the one from v0.3.0. `BGE_M3_LITE_FUSED_URL` overrides the base
URL of the two assets.

```bash
pip install "bge-m3-lite[quant]"   # onnx + onnxruntime.transformers deps
bge-m3-lite fuse                    # rebuilds both files into the cache
```

## Measured (Apple M4, `tools/eval_model.py`)

| graph | dense cos | sparse top-5 | colbert p5 | 16-tok ×32 | 128-tok ×16 | 512-tok ×4 |
|---|---|---|---|---|---|---|
| raw export | 1.0 | 11/11 | 1.0 | 2100 tok/s | 2007 tok/s | 1655 tok/s |
| **fused** | 1.0 | 11/11 | 1.0 | 1860 tok/s | 2214 tok/s | 1858 tok/s |

+10–14 % on 128/512-token inputs, a few percent slower on 16-token batches
(the `Attention` op's overhead is not amortised there). Apple's Accelerate GEMM
dominates on M4; Linux runners (MLAS kernels) are measured by the CI `bench`
job, see `verification.md`.

## Measured on GitHub-hosted runners (4 vCPU, 128-token batches)

| runner | raw | fused |
|---|---|---|
| x86_64 AMD EPYC 7763 | 267 tok/s | 278 tok/s (+4 %) |
| aarch64 Neoverse-N2 | 308 tok/s | 314 tok/s (+2 %) |
| macOS VM | 110 tok/s | 132 tok/s (+20 %) |

On Linux the MLAS GEMM kernels dominate, so the fusion mostly saves the small
ops; the int8 build starts from the fused graph anyway (`quantization/recipe.md`).
