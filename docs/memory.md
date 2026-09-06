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

`--attention-chunk N` on `fuse` and `quantize` sets the chunk (default 512;
`0` restores the single op). At 512 the per-layer buffer for an 8192-token
text is `16 × 512 × 8192 × 4 B` = 256 MiB.

## What still costs memory

- Weights: fp32 is mmap'd from `model.onnx_data` (RSS grows as pages are
  touched); int8 loads its 569 MB file fully, plus ORT's arena for the
  2 700-node graph (1.8 GB resident after loading).
- Hidden states and FFN intermediates: `padded_tokens × 4 KiB` and
  `padded_tokens × 16 KiB` per layer, i.e. 160 MiB at 8192 tokens.
- `encode(..., max_batch_tokens=16384)` still bounds the padded tokens per
  batch; lower it on small machines.

## Activation memory per padded token (M4, v0.5.1, one `encode` call)

Peak RSS minus RSS after load, `max_batch_tokens` = tokens × texts:

| batch | tokens | int8 | fp32 fused |
|---|---|---|---|
| 1024 × 1 | 1024 | +91 MiB | +108 MiB |
| 8192 × 1 | 8192 | +704 MiB (86 KiB/tok) | +840 MiB (103 KiB/tok) |
| 1024 × 8 | 8192 | +628 MiB | +767 MiB |
| 1024 × 16 | 16384 | +1246 MiB | +1528 MiB |
| 512 × 32 | 16384 | +1752 MiB (107 KiB/tok) | +2001 MiB (122 KiB/tok) |

Rule of thumb: **peak ≈ RSS after load + 0.09–0.12 MiB × padded tokens per
batch**; the default `max_batch_tokens=16384` therefore needs 1.3–2 GB of
headroom, and a budget of `H` MiB allows `max_batch_tokens ≈ H / 0.12`. The
ORT arena keeps the largest buffer set, so the peak is reached once and
stays. Roughly a third of the per-token cost is the FFN intermediate
(2 × 16 KiB), the target of the planned v0.5.2 token-block FFN.
