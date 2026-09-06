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

- Weights: fp32 is mmap'd from `model.onnx_data` and prepacked by MLAS
  (1.3 GB resident after loading, `resources.md`); int8 maps its 569 MB
  file the same way (0.7 GB).
- Hidden states and FFN intermediates: `padded_tokens × 4 KiB` and
  `padded_tokens × 16 KiB` per layer, i.e. 160 MiB at 8192 tokens (since
  v0.6.1 the FFN part is bounded to `256 rows × 16 KiB`, below).
- `encode(..., max_batch_tokens=16384)` still bounds the padded tokens per
  batch; lower it on small machines.

## v0.5.2: the whole layer inside the attention `Loop` (superseded)

`fuse --tail loop` moves the rest of each layer (output projection,
SkipLayerNorm, FFN, SkipLayerNorm; all per token) into the attention `Loop`
body, slicing the residual like the query rows (`quantize.layer_tail_into_loop`).
Bit-identical outputs; 16/128-token batches within ±1 %, 512-token texts −4 %,
one 8192-token text +10 % wall. What sets the peak: onnxruntime's memory
pattern only applies from the second run of a shape; the first run of a new
shape allocates from the BFC arena, which grows in powers of two and never
shrinks (`memory.enable_memory_arena_shrinkage` and `kSameAsRequested`
measured: no gain, or worse). So the per-token cost is the arena's high-water
mark, set by the allocation *sequence*: the same rewrite at chunk 512 cost
+15–23 %, at chunk 256 it saved a third for texts longer than the chunk and
cost +20 % for batches of texts no longer than the chunk (three hidden-state
copies per iteration: residual slice, `Concat` accumulator, loop output) —
the int8 graph lost at every shape and shipped without it.

## v0.6.1: the layer tail in a `Loop` over rows of the flattened batch

`fuse` and `quantize` (`--tail rows`, the default; `loop` and `none` keep
the older layouts) run the per-token tail of every layer in a second `Loop`
over 256 rows of the `(1, batch × seq, hidden)` activations
(`quantize.layer_tail_row_loop`); the attention `Loop` keeps chunking the
query rows. The FFN intermediate is then `256 × 16 KiB` for every batch
shape, where the v0.5.2 body still ran the tail on `batch × chunk` rows
(`128 × 128`: 256 MiB per intermediate). The body's result is a *scan
output*: onnxruntime writes each window into one buffer, no `Concat`
accumulator, no copy out of the loop. Scan outputs need one shape per
iteration, so the last window is shifted back to 256 full rows (up to 255
rows recomputed) and the outer graph reassembles the rows with two `Slice`
and a `Concat`. Outputs are bit-identical on the M4 (fixtures, held-out
set, fp32 and int8; the int8 weight file is unchanged since v0.5.0); on x86
MLAS picks its GEMM kernel by row count, so the rows recomputed in the
shifted window can differ in the last bit (fixtures exact at 1e-4 on every
runner, `verification.md`).

| tokens × texts, M4, one `encode`, peak − RSS after load | fp32 v0.5.2 | fp32 v0.6.1 | int8 v0.5.2 | int8 v0.6.1 |
|---|---|---|---|---|
| 128 × 128 | 1734 MiB (108 KiB/tok) | **941** (59) | 1207 (75) | **1076** (67) |
| 256 × 64 | 1735 (108) | 1063 (66) | 1192 (74) | 1077 (67) |
| 512 × 32 | 1199 (75) | 975 (61) | 1174 (73) | 1046 (65) |
| 1024 × 16 | 1029 (64) | 965 (60) | 1150 (72) | 1034 (65) |
| 8192 × 1 | 570 (71) | 540 (68) | 648 (81) | 584 (73) |
| 128 × 16 | 220 (110) | 120 (60) | 161 (80) | 145 (72) |

Throughput on the M4 is unchanged within noise on every shape (128 × 128
2072 → 2142 tok/s fp32; 8192 × 1 345 → 343; int8 1369 → 1306 / 314 → 314).
CI runners (128-token batches, built on the runner): fp32 −2–3 % (EPYC 7763
270 → 264, Neoverse-N2 316 → 306); int8 −2–9 % (Neoverse 1146 → 1039, Xeon
8573C 488 → 477, M1 VM 911 → 918) at unchanged short-query latency; 256
rows is the optimum there (`--tail-rows`: 64 / 128 / 1024 rows lose 8–20 %
on int8, the window's intermediates leave the L2 either way; fp32 64 rows
−13 % on the M4; peak memory is the same for every window). `quantize
--tail none` restores the v0.5.2 int8 throughput at 0.7 s start-up and
+10 % memory. Start-up: the int8 outer graph shrinks from
2 692 to 1 396 nodes and the session opens in 0.31 s instead of 0.69 s on
the M4 (fp32 0.32 s either way, `resources.md`). Rule of thumb since
v0.6.1: **peak ≈ RSS after load + 0.06–0.07 MiB × padded tokens per batch**
(fp32 and int8, every shape); a budget of `H` MiB allows `max_batch_tokens
≈ H / 0.07`. The arena keeps the largest buffer set, so the peak is reached
once and stays; concurrent runs (`serving/recipe.md`) each add their own.

Measured and rejected on the way: an `If` that runs the v0.5.2 body without
the slices for single-iteration inputs — both branches hold the tail's
`MatMul`s and onnxruntime prepacks weights per kernel instance, subgraphs
included, so RSS after load rose by 867 MiB (the 24 layers' projections)
and the peak did not move; 1 024-row windows — same memory, −5 % tok/s.
