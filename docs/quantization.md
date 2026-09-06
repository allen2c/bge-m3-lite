# Quantization (int8)

`BGEM3Embedder(precision="int8")` uses a quantised backbone; `fp32` stays the
default because it is bit-exact with FlagEmbedding.

## What ships as `int8`

Built by `bge-m3-lite quantize` from the **fused** fp32 graph (`fusion.md`),
entirely with numpy on the ONNX graph (no extra dependency beyond the `quant`
extra):

1. **SmoothQuant** (α = 0.5) on the 96 projections (QKV, attention output,
   FFN in / out of every layer): per-input-channel scales
   `s_k = max|X_k|^α / max|W_k|^(1-α)` from the activations of 572 calibration
   texts (`calibration.md`: 212 hand-written + 360 MIRACL passages, 512 tokens max);
   `W ← diag(s) W` and one `Mul` applies `X / s`. fp32 outputs are unchanged
   by this step (unit-tested); it tames the outlier channels of the LayerNorm
   outputs before quantisation.
2. **Row-wise dynamic uint8 activations**, written with the two ops ORT can
   run per row on every platform: per token `scale = (max − min) / 255` and a
   zero point (`ReduceMax`/`ReduceMin` and a few scalar ops on the `(M, 1)`
   results), one per-axis `QuantizeLinear`, then `MatMulIntegerToFloat`
   (u8 × u8, per-column weight scale, int32 → float and the weight scale
   folded in), and the zero-point correction `s · (Y − z · colsum(Wq))`
   (`MatMulIntegerToFloat` takes a per-row `a_scale` but only a scalar
   `a_zero_point`). Weights are per-column symmetric int8 stored as uint8
   with a zero point of 128: MLAS's AVX2 u8·s8 kernel saturates its int16
   intermediates (`VPMADDUBSW`), u8·u8 does not. The merged QKV `Attention`
   becomes the same quantised projection + chunked `MultiHeadAttention`
   (`memory.md`); the word embeddings are int8 rows with one scale per row.
   The graph is opset 13 (per-axis `QuantizeLinear`); `Unsqueeze`/`ReduceSum`
   of the opset-11 export are converted.

Single file `model_int8.onnx`, 569 MiB (fp32: 2.27 GB). Deterministic build;
`hub.INT8_FILE` pins size and SHA-256.

```bash
pip install "bge-m3-lite[quant]"       # onnx, onnx-ir, sympy (build time only)
bge-m3-lite quantize                    # writes model_int8.onnx into the cache
bge-m3-lite quantize --alpha 0.65       # other SmoothQuant strengths
bge-m3-lite quantize --method dynamic   # ORT's per-tensor quantize_dynamic (v0.0.2/v0.3.0 style)
bge-m3-lite encode --int8 "text"
```

Downloads come from the GitHub release asset in `hub.INT8_RELEASE`
(override with `BGE_M3_LITE_INT8_URL`); a locally built file in the cache is
used as-is when its size and digest match.

## v0.4: fewer element-wise passes

v0.3.1 spelled the per-row scheme out with 20 standard ops per projection
(`Div/Round/Add/Clip/Cast` to quantise, `MatMulInteger/Cast/Mul/Sub/Mul/Mul`
to dequantise), which ORT does not fuse: its `DynamicQuantizeMatMul` and
`MatMulIntegerToFloat` fusions only match the per-tensor
`DynamicQuantizeLinear` pattern, and the contrib kernels reject a per-row
`a_zero_point` ("Per-Channel is not supported yet"). Measured on the M4
(single 2048 × 1024 × 4096 projection, ms): fp32 8.6, v0.3.1 chain 16.9,
per-axis `QuantizeLinear` + `MatMulIntegerToFloat` 15.7, the same with a
fixed zero point of 128 (symmetric) 13.8, ORT's per-tensor
`DynamicQuantizeMatMul` 12.0. v0.4 ships the exact per-row variant
(`--symmetric` builds the faster one: dense cosine 0.9975, sparse 26/40,
not worth it). Full model, M4, 128 tokens × 16: 1375 → 1532 tok/s (+11 %);
on x86 the element-wise share is larger, to be measured on the CI matrix.

## Why row-wise: the v0.3.0 lesson

ORT's `quantize_dynamic` quantises activations per *tensor*. On Apple Silicon
its `DynamicQuantizeMatMul` kernel (KleidiAI) silently switches to per-row
scales, which is why v0.3.0 measured dense cosine 0.998 on an M4 but only
0.96–0.98 on every GitHub runner (x86 and ARM alike, and the macOS VM). The
row-wise graph makes the accurate scheme explicit; the cost is a chain of
element-wise ops per projection (≈20–30 % of the int8 throughput on Linux,
more on Apple Silicon where fp32 is already fast). int8 activations without a
zero point are 5× slower on x86: MLAS has no s8·s8 kernel.

## Measured on the M4 (`tools/eval_model.py`, 2026-09-06, v0.4 recipe)

| variant | 11-set dense min / mean | sparse top-5 | held-out dense min / mean | held-out top-5 | colbert p5 | 128-tok tok/s |
|---|---|---|---|---|---|---|
| fp32 fused (chunked) | 1.0 | 11/11 | 1.0 | 40/40 | 1.0 | 2210 |
| int8 v3 (v0.3.1 asset) | 0.9976 / 0.9987 | 9/11 | 0.9973 / 0.9987 | 29/40 | 0.992 | 1375 |
| **int8 v4 (v0.4)** | 0.9982 / 0.9986 | 10/11 | 0.9977 / 0.9986 | 30/40 | 0.986–0.995 | 1532 |
| int8 v4 `--symmetric` | 0.9968 / 0.9977 | 8/11 | 0.9961 / 0.9975 | 26/40 | 0.985 | 1580 |
| int8 v3 + last FFN fp32 (+12 MB) | 0.9976 / 0.9987 | 10/11 | 0.9974 / 0.9988 | 29/40 | 0.992 | |
| int8 v3 + last layer fp32 (+36 MB) | 0.9977 / 0.9988 | 9/11 | 0.9974 / 0.9988 | 30/40 | 0.992 | slower |
| int8 v3 + last 2 layers fp32 (+72 MB) | 0.9978 / 0.9989 | 8/11 | 0.9976 / 0.9989 | 29/40 | 0.992 | 800–1080 |
| int8 v3, α = 0.4 | 0.9978 / 0.9983 | 10/11 | 0.9968 / 0.9983 | 29/40 | 0.987 | |

Sparse top-5 flips are near-ties: on every flipped text the fp32 gap between
the swapped tokens is 0.001–0.009, the same size as the int8 noise
(`--keep-fp32 REGEX` keeps chosen projections in fp32 for such experiments;
none of the cheap variants above moves sparse accuracy beyond ±1 text).

## Measured on GitHub-hosted runners (4 vCPU, `tools/eval_model.py`, 2026-09-06, v0.3.1 recipe)

| runner | variant | dense cos min / mean | sparse top-5 same | colbert p5 | 128-tok tok/s |
|---|---|---|---|---|---|
| x86_64 Xeon (VNNI) | fp32 fused | 1.0 | 11/11 | 1.0 | ~490 |
| | **int8 v3 (shipped)** | **0.9984 / 0.9988** | **11/11** | **0.993** | **742** |
| | int8 v0.3.0 (per-tensor + SmoothQuant) | 0.927 / 0.969 | 7/11 | 0.62 | 1069 |
| | int8 per-tensor, no SmoothQuant (v0.0.2 recipe on the fused graph) | 0.978 / 0.985 | 5/11 | 0.93 | ~1000 |
| x86_64 AMD EPYC 7763 (AVX2, no VNNI) | fp32 fused | 1.0 | 11/11 | 1.0 | 275 |
| | **int8 v3** | **0.9984 / 0.9988** | **10/11** | **0.993** | **352** |
| | int8 v0.3.0 | 0.927 / 0.969 | 7/11 | 0.62 | 508 |
| aarch64 Neoverse-N2 | fp32 fused | 1.0 | 11/11 | 1.0 | 316 |
| | **int8 v3** | **0.9986 / 0.9988** | **10/11** | **0.990** | **1120** |
| | int8 v0.3.0 | 0.975 / 0.983 | 7/11 | 0.72 | 1323 |
| | int8 per-tensor, no SmoothQuant | 0.982 / 0.988 | 5/11 | 0.94 | 1336 |
| macOS VM (3 cores) | fp32 fused | 1.0 | 11/11 | 1.0 | 127 |
| | **int8 v3** | **0.9986 / 0.9988** | **10/11** | **0.993** | **319** |
| | int8 v0.3.0 | 0.974 / 0.983 | 7/11 | 0.78 | 397 |

Variants that lost (all kept out of the CLI): row-wise symmetric int8
(0.9978, 104 tok/s on x86), 7-bit weights (0.9978, no speed gain), α = 0.65
(dense 0.9990 but sparse 8/11), static MinMax calibration (0.59), 4-bit (0.95).

On Apple Silicon (native M4) int8 v3 runs at ~1400 tok/s versus 2200 tok/s
for fused fp32: use fp32 there unless memory matters (4× smaller).

Take-aways:

- Sparse weights remain the most sensitive output; use fp32 when exact
  lexical scores matter.
- x86 speed depends on VNNI: the Xeon runner does 1.5× fp32 with v3 (per-tensor
  kernels 2×); the AMD EPYC 7763 (AVX2 only) 1.3× (per-tensor 1.9×). The
  u8·u8 weights are what make the AVX2 result correct at all.

## Building other variants

```bash
bge-m3-lite quantize --method nbits --bits 8 --accuracy-level 0   # weight-only, near-lossless, fp32 speed
bge-m3-lite quantize --keep-embeddings                            # fp32 word embeddings
bge-m3-lite quantize --method dynamic --raw --no-smooth           # v0.0.2 recipe
bge-m3-lite quantize --calibration my_texts.txt                   # own calibration set
bge-m3-lite quantize --keep-fp32 'layer\.23/' --keep-fp32 'Attention_23$'  # last layer fp32
bge-m3-lite quantize --attention-chunk 0                          # single MultiHeadAttention per layer
uv run tools/eval_model.py path/to/model.onnx                     # accuracy + speed report
```

Any file can be used with `BGEM3Embedder(model_path=...)` or `encode --model`.
