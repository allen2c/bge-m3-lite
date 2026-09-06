# Long inputs: attention in query chunks

## The problem

ORT's CPU `Attention` and `MultiHeadAttention` kernels allocate the full score
matrix `batch × heads × S × S` in fp32 (`attention_cpu_base.h`,
`ApplyAttention`): 4 GiB for one 8192-token text, on top of the weights.
Neither op tiles it, no session option changes that, and the only flash-style
CPU path in ORT 1.29 (`GroupQueryAttention`) is gated to causal decoders.
Measured on an M4 (`ru_maxrss`, one text, ORT 1.29):

| graph | 512 tok | 2048 tok | 8192 tok |
|---|---|---|---|
| raw export (`model.onnx`) | 1.4 GB | 2.1 GB | 13.4 GB |
| fused fp32, `Attention` op | 1.3 GB | 1.8 GB | 7.4 GB |
| int8 v3, `MultiHeadAttention` op | 1.9 GB | 2.0 GB | 7.5 GB |

## The fix (v0.4): `Loop` over query chunks

Softmax is per query row, so attention over `chunk` query rows against the
full K/V is exact. Both build steps (`fuse`, `quantize --method rowwise`)
emit, per layer, `MatMul` (packed QKV) → `Split` → `Loop` whose body slices
`chunk` rows of Q and runs `MultiHeadAttention` against the whole K/V, then
concatenates (`quantize.attention_nodes`). The trip count is
`ceil(S / chunk)`, computed in the graph: texts up to `chunk` tokens run a
single iteration. The score buffer becomes `batch × heads × chunk × S`.

| graph, one 8192-token text, M4 (2026-09-06) | peak RSS | 8192-tok wall | 128-tok × 16 tok/s |
|---|---|---|---|
| fused fp32, `Attention` op (v0.3) | 7.4 GB | 22 s | 2288 |
| fused fp32, `Loop` chunk 512 (v0.4) | 2.5 GB | 25 s | 2210 |
| int8, one `MultiHeadAttention` per layer | 7.5–11.7 GB | 22 s | 1548 |
| int8, `Loop` chunk 512 (v0.4) | 2.5 GB | 22 s | 1532 |

(peak RSS of the int8 graph varies with how many encodes ran before: the
ORT arena keeps the largest attention buffer.) Short inputs lose 1–3 %, the
fixed alternative (N static slices per layer, no `Loop`) 9–28 %, so it was
dropped. Outputs are bit-identical to the unchunked graph (dense, sparse and
ColBERT on a mixed-length padded batch; the fp32 fixtures still match
FlagEmbedding exactly).

`--attention-chunk N` on `fuse` and `quantize` sets the chunk (512 in
v0.4–v0.5.1, 256 since v0.5.2; `0` restores the single op). At 256 the
per-layer score buffer for an 8192-token text is `16 × 256 × 8192 × 4 B` =
128 MiB.

## What still costs memory

- Weights: fp32 is mmap'd from `model.onnx_data` (RSS grows as pages are
  touched); int8 loads its 569 MB file fully, plus ORT's arena for the
  2 700-node graph (1.8 GB resident after loading).
- Hidden states and FFN intermediates: `padded_tokens × 4 KiB` and
  `padded_tokens × 16 KiB` per layer, i.e. 160 MiB at 8192 tokens (v0.5.2
  bounds the FFN part to `batch × chunk × 16 KiB`, below).
- `encode(..., max_batch_tokens=16384)` still bounds the padded tokens per
  batch; lower it on small machines.

## v0.5.2: the whole layer inside the `Loop` (M4, one `encode`, peak − RSS after load)

`fuse` now moves the rest of each layer (output projection, SkipLayerNorm,
FFN, SkipLayerNorm; all per token) into the attention `Loop` body, slicing the
residual like the query rows (`quantize.layer_tail_into_loop`; `fuse
--no-layer-loop` keeps the v0.4 layout, `quantize --layer-loop` applies it to
int8). Outputs are bit-identical (dense, sparse, ColBERT; fp32 and int8),
16/128-token batches within ±1 %, 512-token texts −4 % (two iterations of
256), one 8192-token text +10 % wall. CI runners agree (128-token tok/s, v0.5.1
→ v0.5.2 graph): EPYC 7763 269 → 268, Neoverse-N2 317 → 312, M1 VM 339 → 364.

What actually sets the peak: onnxruntime's memory pattern only applies from the
second run of a shape; the first run of a new shape allocates from the BFC
arena, which grows in powers of two (1 + 32 + 32 + 64 + 128 + 256 + 512 MiB
for one 8192-token text) and never shrinks (`memory.enable_memory_arena_shrinkage`
and `kSameAsRequested` measured: no gain, or worse). So the per-token cost is
the arena's high-water mark, not the sum of live buffers, and it moves with the
allocation *sequence*: the same rewrite at chunk 512 costs +15–23 %, at chunk 256
it saves a third for texts longer than the chunk.

| tokens × texts | fp32 v0.5.1 (chunk 512) | fp32 v0.5.2 (256, tail in loop) | int8 v0.5.1 | int8 v0.5.2 (256) | int8 256 + tail in loop |
|---|---|---|---|---|---|
| 1024 × 1 | 104 MiB (104 KiB/tok) | 72 (72) | | | |
| 8192 × 1 | 835 (104) | 567 (71) | 690 (86) | 637 (80) | 716 (90) |
| 2048 × 4 | 766 (96) | 512 (64) | | | |
| 1024 × 16 | 1526 (95) | 1025 (64) | 1233 (77) | 1150 (72) | 1309 (82) |
| 512 × 32 | 1997 (125) | 1197 (75) | 1739 (109) | 1162 (73) | 1385 (87) |
| 256 × 64 | 1443 (90) | 1731 (108) | 1192 (74) | 1192 (74) | 1479 (92) |
| 128 × 128 | 1442 (90) | 1731 (108) | | | |

Texts no longer than the chunk (one iteration) pay for the extra copies the
loop makes of the hidden state (residual slice, accumulator, loop output):
+20 % for fp32. The int8 graph (30 small ops per projection) loses with the
tail inside the loop at every shape, so int8 ships chunk 256 without it
(start-up would drop from 0.75 s to 0.35 s with it; 512-token texts −3 %).
Rule of thumb since v0.5.2: **peak ≈ RSS after load + 0.11 MiB × padded tokens
per batch** for short texts, **0.07–0.08 MiB** for texts of 512+ tokens; a
budget of `H` MiB allows `max_batch_tokens ≈ H / 0.11`. The arena keeps the
largest buffer set, so the peak is reached once and stays. Concurrent runs
(`serving/recipe.md`) each add their own set: 626-token texts cost +41 MiB (fp32) /
+38 MiB (int8) per extra run in flight, i.e. the 0.07 MiB rule per run.
