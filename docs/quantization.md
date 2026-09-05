# Quantization (v0.0.2)

`BGEM3Embedder(precision="int8")` uses a quantised backbone; `fp32` stays the
default because it is bit-exact with FlagEmbedding.

## What ships as `int8`

`onnxruntime.quantization.quantize_dynamic`, per-channel int8 weights for every
`MatMul` and the word-embedding `Gather`, uint8 activations quantised at run
time. Single file `model_int8.onnx`, 543 MiB (fp32: 2.27 GB). The build is
deterministic; `hub.INT8_FILE` pins its size and SHA-256.

```bash
pip install "bge-m3-lite[quant]"       # onnx, onnx-ir, sympy (build time only)
bge-m3-lite quantize                    # writes model_int8.onnx into the cache
bge-m3-lite encode --int8 "text"
```

Downloads come from the GitHub release asset in `hub.INT8_RELEASE`
(override with `BGE_M3_LITE_INT8_URL`); a locally built file in the cache is
used as-is when its size and digest match.

## Measured on Apple M4 (11 fixture texts, `tools/eval_model.py`)

| variant | size | dense cos (min / mean) | sparse top-5 same | colbert token cos p5 | 128-tok tok/s |
|---|---|---|---|---|---|
| fp32 | 2.27 GB | 1.0 / 1.0 | 11/11 | 1.0 | 1780 |
| dynamic int8, per-tensor | 1.30 GB | 0.983 / 0.987 | 4/11 | – | 1720 |
| **dynamic int8, per-channel + Gather (shipped)** | 543 MB | 0.992 / 0.993 | 7/11 | ~0.9 | 2145 |
| nbits 8-bit, block 128, int8 compute | 1.31 GB | 0.998 / 0.998 | 8/11 | 0.98 | 1500 |
| nbits 8-bit, block 128, fp32 compute | 1.31 GB | 0.9994 / 0.9997 | 9/11 | 0.997 | 1730 |
| nbits 4-bit, block 128 | 1.16 GB | 0.925 / 0.946 | 5/11 | 0.64 | 1830 |

## Measured on Linux aarch64 (Docker on the same M4, Linux ORT build, no Accelerate)

| variant | dense cos mean | sparse top-5 same | colbert p5 | 128-tok tok/s |
|---|---|---|---|---|
| fp32 | 1.0 | 11/11 | 1.0 | 641 |
| dynamic int8 (shipped) | 0.987 | 6/11 | 0.95 | 1627 (**2.5×**) |

## Measured on GitHub-hosted runners (4 vCPU, `tools/eval_model.py`)

| runner | dense cos mean | sparse top-5 same | colbert p5 | fp32 → int8 128-tok |
|---|---|---|---|---|
| x86_64 Xeon 8573C (VNNI) | 0.987 | 4/11 | 0.92 | 491 → 1914 tok/s (**3.9×**) |
| aarch64 Neoverse-N2 | 0.988 | 5/11 | 0.95 | 298 → 1272 tok/s (**4.3×**) |
| macOS VM (3 cores) | 0.988 | 6/11 | 0.95 | 77 → 316 tok/s (4.1×) |

Take-aways:

- On Apple Silicon (native) fp32 is already fast (Accelerate); int8 gives ~10%
  speed and 4× less memory. On Linux x86 (VNNI) and ARM the fp32 kernels are
  much slower and int8 is 4× faster, which is why v0.3 invests in int8 accuracy
  (`roadmap.md`).
- The accuracy loss of dynamic quantisation comes from per-tensor activation
  quantisation (outlier channels), not from the weights: weight-only 8-bit with
  fp32 compute is near-lossless but no faster.
- 4-bit is not usable for this model without calibration (GPTQ/AWQ).
- Sparse weights are the most sensitive output: key sets stay mostly intact but
  the top-5 order changes in a third of the texts. Use fp32 when lexical scores
  matter.

## Building other variants

```bash
bge-m3-lite quantize --method nbits --bits 8 --accuracy-level 0   # near-lossless, same speed
bge-m3-lite quantize --keep-embeddings                            # fp32 word embeddings
uv run tools/eval_model.py path/to/model.onnx                     # accuracy + speed report
```

Any file can be used with `BGEM3Embedder(model_path=...)` or `encode --model`.
