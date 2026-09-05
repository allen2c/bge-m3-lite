# Quantization (int8, v0.3.1)

`BGEM3Embedder(precision="int8")` uses a quantised backbone; `fp32` stays the
default because it is bit-exact with FlagEmbedding.

## What ships as `int8`

Built by `bge-m3-lite quantize` from the **fused** fp32 graph (`fusion.md`),
entirely with numpy on the ONNX graph (no extra dependency beyond the `quant`
extra):

1. **SmoothQuant** (α = 0.5) on the 96 projections (QKV, attention output,
   FFN in / out of every layer): per-input-channel scales
   `s_k = max|X_k|^α / max|W_k|^(1-α)` from the activations of 212 calibration
   texts in 30+ languages (`bge_m3_lite/calibration.txt`, 512 tokens max);
   `W ← diag(s) W` and one `Mul` applies `X / s`. fp32 outputs are unchanged
   by this step (unit-tested); it tames the outlier channels of the LayerNorm
   outputs before quantisation.
2. **Row-wise dynamic uint8 activations**, spelled out with standard ops so
   that every platform computes the same thing: per token `scale = (max − min)
   / 255`, zero point, `MatMulInteger` (u8 × s8), and the zero-point
   correction `s · (Q·Wq − z · colsum(Wq))`. Weights are per-column symmetric
   int8, stored as uint8 with a zero point of 128 (`MatMulInteger` removes it):
   MLAS's AVX2 u8·s8 kernel saturates its int16 intermediates (`VPMADDUBSW`),
   u8·u8 does not, and the integer results are otherwise identical. The merged
   QKV `Attention` becomes the same quantised projection +
   `MultiHeadAttention`; the word embeddings are int8 rows with one scale per
   row (lossless in practice).

Single file `model_int8.onnx`, 569 MiB (fp32: 2.27 GB). Deterministic build;
`hub.INT8_FILE` pins size and SHA-256.

```bash
pip install "bge-m3-lite[quant]"       # onnx, onnx-ir, sympy (build time only)
bge-m3-lite quantize                    # writes model_int8.onnx into the cache
bge-m3-lite quantize --alpha 0.65       # other SmoothQuant strengths
bge-m3-lite quantize --weights s8       # u8·s8 GEMM: only correct on VNNI / ARM CPUs
bge-m3-lite quantize --method dynamic   # ORT's per-tensor quantize_dynamic (v0.0.2/v0.3.0 style)
bge-m3-lite encode --int8 "text"
```

Downloads come from the GitHub release asset in `hub.INT8_RELEASE`
(override with `BGE_M3_LITE_INT8_URL`); a locally built file in the cache is
used as-is when its size and digest match.

## Why row-wise: the v0.3.0 lesson

ORT's `quantize_dynamic` quantises activations per *tensor*. On Apple Silicon
its `DynamicQuantizeMatMul` kernel (KleidiAI) silently switches to per-row
scales, which is why v0.3.0 measured dense cosine 0.998 on an M4 but only
0.96–0.98 on every GitHub runner (x86 and ARM alike, and the macOS VM). The
row-wise graph makes the accurate scheme explicit; the cost is a chain of
element-wise ops per projection (≈20–30 % of the int8 throughput on Linux,
more on Apple Silicon where fp32 is already fast). int8 activations without a
zero point (`--symmetric`) are 5× slower on x86: MLAS has no s8·s8 kernel.

## Measured on GitHub-hosted runners (4 vCPU, `tools/eval_model.py`, 2026-09-06)

| runner | variant | dense cos min / mean | sparse top-5 same | colbert p5 | 128-tok tok/s |
|---|---|---|---|---|---|
| x86_64 Xeon (VNNI) | fp32 fused | 1.0 | 11/11 | 1.0 | ~490 |
| | **int8 v3 (shipped)** | **0.9984 / 0.9988** | **11/11** | **0.993** | **742** |
| | int8 v0.3.0 (per-tensor + SmoothQuant) | 0.927 / 0.969 | 7/11 | 0.62 | 1069 |
| | int8 per-tensor, no SmoothQuant (v0.0.2 recipe on the fused graph) | 0.978 / 0.985 | 5/11 | 0.93 | ~1000 |
| aarch64 Neoverse-N2 | fp32 fused | 1.0 | 11/11 | 1.0 | 316 |
| | **int8 v3** | **0.9986 / 0.9988** | **10/11** | **0.990** | **1120** |
| | int8 v0.3.0 | 0.975 / 0.983 | 7/11 | 0.72 | 1323 |
| | int8 per-tensor, no SmoothQuant | 0.982 / 0.988 | 5/11 | 0.94 | 1336 |
| macOS VM (3 cores) | fp32 fused | 1.0 | 11/11 | 1.0 | 127 |
| | **int8 v3** | **0.9986 / 0.9988** | **10/11** | **0.993** | **319** |
| | int8 v0.3.0 | 0.974 / 0.983 | 7/11 | 0.78 | 397 |

Variants that lost: row-wise symmetric int8 (0.9978, 104 tok/s on x86),
`--reduce-range` (0.9978, no speed gain), α = 0.65 (dense 0.9990 but sparse
8/11), static MinMax calibration (0.59), 4-bit (0.95).

On Apple Silicon (native M4) int8 v3 runs at ~1400 tok/s versus 2200 tok/s
for fused fp32: use fp32 there unless memory matters (4× smaller).

Take-aways:

- Sparse weights remain the most sensitive output; use fp32 when exact
  lexical scores matter.
- x86 speed depends on VNNI: the Xeon runner does 1.5× fp32 with v3 (per-tensor
  kernels 2×); an AMD EPYC 7763 (AVX2 only) gets about half of that.

## Building other variants

```bash
bge-m3-lite quantize --method nbits --bits 8 --accuracy-level 0   # weight-only, near-lossless, fp32 speed
bge-m3-lite quantize --keep-embeddings                            # fp32 word embeddings
bge-m3-lite quantize --method dynamic --raw --no-smooth           # v0.0.2 recipe
bge-m3-lite quantize --calibration my_texts.txt                   # own calibration set
uv run tools/eval_model.py path/to/model.onnx                     # accuracy + speed report
```

Any file can be used with `BGEM3Embedder(model_path=...)` or `encode --model`.
