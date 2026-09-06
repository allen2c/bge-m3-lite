# Quantization (int8): the recipe

`BGEM3Embedder(precision="int8")` uses a quantised backbone; `fp32` stays the
default because it is bit-exact with FlagEmbedding.

## What ships as `int8`

Built by `bge-m3-lite quantize` from the **fused** fp32 graph (`../fusion.md`),
entirely with numpy on the ONNX graph (no extra dependency beyond the `quant`
extra):

1. **SmoothQuant** (α = 0.5) on the 96 projections (QKV, attention output,
   FFN in / out of every layer): per-input-channel scales
   `s_k = max|X_k|^α / max|W_k|^(1-α)` from the activations of 572 calibration
   texts (`../calibration.md`: 212 hand-written + 360 MIRACL passages, 512 tokens max);
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
   (`../memory.md`); the word embeddings are int8 rows with one scale per row.
   The graph is opset 13 (per-axis `QuantizeLinear`); `Unsqueeze`/`ReduceSum`
   of the opset-11 export are converted.

Ships as `model_int8.onnx` (graph, 0.7 MB) + `model_int8.onnx_data` (weights,
569 MiB; fp32: 2.27 GB) since v0.5: external data halves the resident memory
(`../resources.md`). Deterministic build; `hub.INT8_FILES` pins sizes and
SHA-256 digests.

```bash
pip install "bge-m3-lite[quant]"       # onnx, onnx-ir, sympy (build time only)
bge-m3-lite quantize                    # writes model_int8.onnx into the cache
bge-m3-lite quantize --alpha 0.65       # other SmoothQuant strengths
bge-m3-lite quantize --method dynamic   # ORT's per-tensor quantize_dynamic (v0.0.2/v0.3.0 style)
bge-m3-lite encode --int8 "text"
```

Downloads come from the GitHub release assets in `hub.INT8_RELEASE`
(`BGE_M3_LITE_INT8_URL` overrides the base URL); a locally built pair in the
cache is used as-is when sizes and digests match.

## v0.4: fewer element-wise passes

v0.3.1 spelled the per-row scheme out with 20 standard ops per projection
(`Div/Round/Add/Clip/Cast` to quantise, `MatMulInteger/Cast/Mul/Sub/Mul/Mul`
to dequantise), which ORT does not fuse: its `DynamicQuantizeMatMul` and
`MatMulIntegerToFloat` fusions only match the per-tensor
`DynamicQuantizeLinear` pattern, and the contrib kernels reject a per-row
`a_zero_point` ("Per-Channel is not supported yet"). measurements.md
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
