# Quantization (int8): measurements

Recipe and variants: `recipe.md`. All numbers from `tools/eval_model.py`
(11 FlagEmbedding texts + the 40-text held-out set, `../calibration.md`).

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

## v0.4 recipe on GitHub-hosted runners (4 vCPU, `tools/eval_model.py`, 2026-09-06)

| runner | graph | dense min / mean (11 / held-out) | sparse top-5 (11 / held-out) | 128-tok tok/s |
|---|---|---|---|---|
| AMD EPYC 7763 (AVX2) | fp32 fused, chunked | 1.0 | 11/11, 40/40 | 263 |
| | int8 v3 (v0.3.1) | 0.9983 / 0.9979 | 8/11, 29/40 | 346 |
| | **int8 v4** | **0.9982 / 0.9971** | **9/11, 29/40** | **368** (376 with `--attention-chunk 0`) |
| | int8 v4 `--symmetric` | 0.9969 / 0.9961 | 9/11, 26/40 | 383 |
| Neoverse-N2 | fp32 fused, chunked | 1.0 | 11/11, 40/40 | 314 |
| | int8 v3 | 0.9977 / 0.9974 | 10/11, 28/40 | 1103 |
| | **int8 v4** | **0.9981 / 0.9978** | **7/11, 29/40** | **1117** (1134 unchunked) |
| macOS VM (Apple M1, 3 vCPU) | fp32 fused, chunked | 1.0 | 11/11, 40/40 | 142 |
| | int8 v3 | 0.9976 / 0.9973 | 9/11, 28/40 | 344 |
| | **int8 v4** | **0.9981 / 0.9978** | **8/11, 24/40** | **411** |

The Xeon (VNNI) runner was not drawn in these runs; its v0.3.1 numbers are
below. Sparse top-5 counts move by ±3 between platforms and recipes for the
same dense cosine: they are the near-ties described above, not a trend.

## v0.3.1 recipe on GitHub-hosted runners (4 vCPU, 2026-09-06)

| runner | variant | dense cos min / mean | sparse top-5 same | colbert p5 | 128-tok tok/s |
|---|---|---|---|---|---|
| x86_64 Xeon 8573C (VNNI) | fp32 fused | 1.0 | 11/11 | 1.0 | ~490 |
| | int8 v3 | 0.9984 / 0.9988 | 11/11 | 0.993 | 742 |
| | int8 v0.3.0 (per-tensor + SmoothQuant) | 0.927 / 0.969 | 7/11 | 0.62 | 1069 |
| x86_64 AMD EPYC 7763 (AVX2) | fp32 fused | 1.0 | 11/11 | 1.0 | 275 |
| | int8 v3 | 0.9984 / 0.9988 | 10/11 | 0.993 | 352 |
| aarch64 Neoverse-N2 | fp32 fused | 1.0 | 11/11 | 1.0 | 316 |
| | int8 v3 | 0.9986 / 0.9988 | 10/11 | 0.990 | 1120 |
| | int8 v0.3.0 | 0.975 / 0.983 | 7/11 | 0.72 | 1323 |
| macOS VM (3 cores) | fp32 fused | 1.0 | 11/11 | 1.0 | 127 |
| | int8 v3 | 0.9986 / 0.9988 | 10/11 | 0.993 | 319 |

Variants that lost (all kept out of the CLI): row-wise symmetric int8
(0.9978, 104 tok/s on x86), 7-bit weights (0.9978, no speed gain), α = 0.65
(dense 0.9990 but sparse 8/11), static MinMax calibration (0.59), 4-bit (0.95).

On Apple Silicon (native M4) int8 v3 runs at ~1400 tok/s versus 2200 tok/s
for fused fp32: use fp32 there unless memory matters (4× smaller).

Sparse weights remain the most sensitive output (use fp32 when exact lexical
scores matter); x86 speed depends on VNNI (Xeon 1.5× fp32, EPYC 1.3×), and
the u8·u8 weights are what make the AVX2 result correct at all.
