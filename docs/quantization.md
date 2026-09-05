# Quantization (int8, v0.3.0)

`BGEM3Embedder(precision="int8")` uses a quantised backbone; `fp32` stays the
default because it is bit-exact with FlagEmbedding.

## What ships as `int8`

Built by `bge-m3-lite quantize` from the **fused** fp32 graph (`fusion.md`):

1. **SmoothQuant** (α = 0.5) on the 96 projections (QKV, attention output,
   FFN in / out of every layer): per-input-channel scales
   `s_k = max|X_k|^α / max|W_k|^(1-α)` from the activations of 212 calibration
   texts in 30+ languages (`bge_m3_lite/calibration.txt`, 512 tokens max);
   `W ← diag(s) W` and one `Mul` node applies `X / s`. fp32 outputs are
   unchanged by this step (unit-tested); it only tames the outlier channels of
   the LayerNorm outputs before they are quantised.
2. `quantize_dynamic`: per-channel int8 weights for every `MatMul`, the merged
   QKV `Attention` (→ `QAttention`) and the word-embedding `Gather`; uint8
   activations quantised at run time.

Single file `model_int8.onnx`, 543 MiB (fp32: 2.27 GB). The build is
deterministic (same SHA-256 on repeated runs); `hub.INT8_FILE` pins it.
Implemented with numpy on the ONNX graph, no extra dependency beyond the
`quant` extra.

```bash
pip install "bge-m3-lite[quant]"       # onnx, onnx-ir, sympy (build time only)
bge-m3-lite quantize                    # writes model_int8.onnx into the cache
bge-m3-lite quantize --alpha 0.65       # other SmoothQuant strengths
bge-m3-lite quantize --no-smooth        # v0.0.2-style plain dynamic quantisation
bge-m3-lite encode --int8 "text"
```

Downloads come from the GitHub release asset in `hub.INT8_RELEASE`
(override with `BGE_M3_LITE_INT8_URL`); a locally built file in the cache is
used as-is when its size and digest match.

## Measured on Apple M4 (11 fixture texts, `tools/eval_model.py`)

| variant | size | dense cos (min / mean) | sparse top-5 same | colbert token cos p5 | 128-tok tok/s |
|---|---|---|---|---|---|
| fp32 raw export | 2.27 GB | 1.0 / 1.0 | 11/11 | 1.0 | 2007 |
| fp32 fused (default) | +288 MB | 1.0 / 1.0 | 11/11 | 1.0 | 2214 |
| **int8 v2: fused + SmoothQuant α 0.5 (shipped)** | 543 MB | 0.9976 / 0.9982 | 9/11 (Spearman 0.996) | 0.989 | 2386 |
| int8 v1 (v0.0.2): dynamic, raw graph | 543 MB | 0.992 / 0.993 | 7/11 | ~0.9 | 2145 |
| int8 fused, no SmoothQuant | 543 MB | 0.991 / 0.993 | 7/11 | 0.97 | 2508 |
| int8 fused, QKV left in fp32 | 758 MB | 0.998 / 0.998 | 10/11 | 0.99 | 2426 |
| static int8, MinMax calibration | 543 MB | 0.44 / 0.59 | 4/11 | 0.17 | 1763 |
| nbits 8-bit, fp32 compute | 1.31 GB | 0.9994 / 0.9997 | 9/11 | 0.997 | 1730 |
| nbits 4-bit | 1.16 GB | 0.925 / 0.946 | 5/11 | 0.64 | 1830 |

SmoothQuant α sweep (calibrated on the 109 tokenizer fixture texts): 0.5 →
dense 0.9982 / sparse 11/11; 0.65 → 0.9987 / 9/11; 0.8 → 0.9976 / 9/11.
Keeping the word embeddings in fp32 (+730 MB) adds only 0.0001 dense cosine.

## Measured on GitHub-hosted runners (4 vCPU, int8 v1)

| runner | dense cos mean | sparse top-5 same | colbert p5 | fp32 → int8 128-tok |
|---|---|---|---|---|
| x86_64 Xeon 8573C (AVX-512 VNNI) | 0.987 | 4/11 | 0.92 | 491 → 1914 tok/s (**3.9×**) |
| x86_64 AMD EPYC 7763 (no VNNI) | 0.985 | 4/11 | 0.92 | 267 → 499 tok/s (1.9×) |
| aarch64 Neoverse-N2 | 0.988 | 5/11 | 0.95 | 298 → 1272 tok/s (**4.3×**) |
| macOS VM (3 cores) | 0.988 | 6/11 | 0.95 | 77 → 316 tok/s (4.1×) |

Take-aways:

- On Apple Silicon (native) fp32 is already fast (Accelerate); int8 gives
  ~10% speed and 4× less memory. On Linux x86 with VNNI and on ARM the fp32
  kernels are much slower and int8 is 4× faster; without VNNI (Zen 3) 2×.
- The accuracy loss of plain dynamic quantisation comes from per-tensor
  activation quantisation of the LayerNorm outputs feeding the QKV projection
  (leaving only those 24 nodes in fp32 recovers most of it). SmoothQuant
  recovers the same accuracy at no size or speed cost.
- Static (calibrated per-tensor) activation ranges are far worse than dynamic
  ones for this model; 4-bit is not usable without GPTQ/AWQ.
- Sparse weights remain the most sensitive output: key sets are intact, the
  top-5 order still changes in 2 of 11 texts. Use fp32 when exact lexical
  scores matter.

## Building other variants

```bash
bge-m3-lite quantize --method nbits --bits 8 --accuracy-level 0   # near-lossless, same speed
bge-m3-lite quantize --keep-embeddings                            # fp32 word embeddings
bge-m3-lite quantize --raw --no-smooth                            # v0.0.2 recipe
bge-m3-lite quantize --calibration my_texts.txt                   # own calibration set
uv run tools/eval_model.py path/to/model.onnx                     # accuracy + speed report
```

Any file can be used with `BGEM3Embedder(model_path=...)` or `encode --model`.
