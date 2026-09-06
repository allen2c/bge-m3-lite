# Resource usage: memory and CPU-seconds

v0.5 minimises what a session costs when it is *not* busy and what a token
costs when it is. Numbers from `tools/eval_model.py` (which prints all of
them) and the same code in a per-configuration harness; M4 (4P + 6E cores),
ORT 1.29, 2026-09-06, nothing else running. Short query = one 9-token text
per `encode` call; RSS = resident set size of the process.

## What v0.5.0 changed and why

| configuration (int8 backbone) | start-up | RSS after load | after 128 × 16 | 128 × 16 tok/s | CPU-s / 1k tok | short query wall / CPU | idle CPU |
|---|---|---|---|---|---|---|---|
| v0.4: one 597 MB file, ORT defaults | 0.80 s | 1838 MiB | 1867 MiB (8192-tok peak 2286) | 1528 | 3.26 | 10 / 51 ms | 27–57 ms/s |
| graph + `model_int8.onnx_data`, ORT defaults | 0.74 s | 724 MiB | 936 MiB | 1535 | 3.21 | 11 / 52 ms | 32 ms/s |
| + no spinning (`allow_spinning=0`) | 0.74 s | 724 MiB | 935 MiB | 1591 | 2.70 | 12 / 34 ms | 0 |
| **+ 4 threads (P-cores) = v0.5 default** | 0.75 s | 724 MiB | 935 MiB (peak 1679) | 1582 | 2.32 | 11 / 29 ms | 0 |

1. **External data.** onnxruntime keeps the parsed protobuf of a model with
   embedded weights *and* the prepacked MLAS copies of every weight. Shipping
   the int8 backbone as graph (0.7 MB) + `.onnx_data` (569 MB), exactly like
   the fused fp32 graph, drops 1.1 GB of resident memory and 0.06 s of
   start-up at identical outputs (every tensor is byte-identical to v0.4).
2. **No spinning.** ORT's intra-op workers spin-wait after each run:
   27–64 ms of CPU per idle second per session (varies with the run before),
   and 30–40 % of the CPU time of a short query. `session.intra_op.allow_spinning=0`
   removes it for 0–5 % throughput. `BGE_M3_LITE_SPIN=1` or
   `BGEM3Embedder(spin=True)` restores the old behaviour.
3. **Performance cores.** `hw.perflevel0.logicalcpu` (4 on the M4) is the
   thread default on Apple Silicon; onnxruntime's own default spawned 5.
   Elsewhere the default stays 0 (onnxruntime: physical cores).
   `BGE_M3_LITE_THREADS` / `num_threads=` override it.

## Threads: throughput versus CPU time (M4, no spinning)

| threads | int8 128 × 16 tok/s | int8 CPU-s / 1k tok | fp32 tok/s | fp32 CPU-s / 1k tok | short query wall (int8 / fp32) |
|---|---|---|---|---|---|
| 1 | 539 | 1.86 | 1179 | 0.85 | 21 / 24 ms |
| 4 (default) | 1582 | 2.32 | 2143 | 1.73 | 11 / 20 ms |
| 5 (ORT default) | 1622 | 2.67 | | | 12 ms |
| 10 (all cores) | 1878 | 3.98 | 2473 | 3.43 | 12 / 19 ms |

One thread delivers 2× the tokens per CPU-second of four and 2–4× that of
ten: a throughput service should run several one-thread workers
(`BGEM3Embedder(num_threads=1)`) rather than one session on all cores. All
ten cores buy +16–19 % throughput for +70–100 % CPU. Tokenizer and pooling
are < 4 % of the CPU time; the rest is the backbone.

## fp32 fused graph (already external data)

| configuration | start-up | RSS after load | after 128 × 16 | 8192-tok peak | tok/s | CPU-s / 1k tok | short query | idle |
|---|---|---|---|---|---|---|---|---|
| v0.4 (ORT defaults) | 0.39 s | 1283 MiB | 1554 MiB | | 2258 | 2.20 | 17 / 88 ms | 33 ms/s |
| v0.5 default | 0.38 s | 1283 MiB | 1554 MiB | 2392 MiB | 2143 | 1.73 | 20 / 71 ms | 0 |
| int8 v0.5 default | 0.75 s | 724 MiB | 935 MiB | 1679 MiB | 1582 | 2.32 | 11 / 29 ms | 0 |

## `low_memory=True` (v0.5.1): weights stay in the mapped file

`BGEM3Embedder(low_memory=True)` / `encode --low-memory` sets
`session.disable_prepacking=1`: MLAS does not build its packed copy of every
weight, so the weights are served from the mmapped `.onnx_data` files
(M4, v0.5 defaults otherwise):

| backbone | mode | start-up | private memory after load | 128 × 16 tok/s | short query wall / CPU |
|---|---|---|---|---|---|
| fp32 fused | default | 0.38 s | 1283 MiB | 2143 | 20 / 71 ms |
| fp32 fused | low_memory | **0.11 s** | **140 MiB** (113 MB `phys_footprint` after queries) | 2038 | 41 / 152 ms |
| int8 | default | 0.75 s | 724 MiB | 1582 | 11 / 29 ms |
| int8 | low_memory | **0.63 s** | **149 MiB** | 1505 | 21 / 68 ms |

The weight pages then show up in RSS once touched (fp32: 1.3 GB, int8:
450 MiB) but they are file-backed: shared between processes through the page
cache (two processes at 1.3 GB RSS each added 0.5 MB of file-backed pages to
the system) and reclaimable under pressure. Batch throughput is unchanged
because packing is amortised; every *short* query packs B again, hence 2×
the latency and CPU. Right for serverless, one-shot CLI calls and many
one-thread workers on one machine; wrong for a resident service answering
single queries.

Not done: disabling the arena. It does not return memory after a long
request (RSS after a 4096-token text stays at 903 MiB int8 / 1778 fp32
versus 704 / 1614 with the arena) and makes nothing faster.

## GitHub-hosted runners (CI `bench`, 4 vCPU, v0.5.0 defaults, 2026-09-06)

| runner | backbone | start-up | RSS load / peak | 128 × 16 tok/s | CPU-s / 1k tok | short query wall / CPU | idle |
|---|---|---|---|---|---|---|---|
| Xeon 8573C (2 cores × 2 SMT, ORT default 2 threads) | fp32 fused | 0.84 s | 1271 / 1695 MiB | 576 | 3.44 | 56 / 108 ms | 0 |
| | int8 (built on the runner) | 1.38 s | 417 / 930 MiB | 494 | 3.97 | 29 / 50 ms | 0 |
| Neoverse-N2 (4 cores, 4 threads) | fp32 fused | 0.88 s | 1270 / 1684 MiB | 306 | 12.7 | 55 / 195 ms | 0 |
| | int8 | 1.84 s | 416 / 920 MiB | 1100 | 3.21 | 26 / 59 ms | 0 |
| Apple M1 VM (3 vCPU, 3 threads) | fp32 fused | 4.6 s | 1279 / 1631 MiB | 232 | 11.2 | 65 / 142 ms | 0 |
| | int8 | 1.35 s | 419 / 718 MiB | 581 | 4.54 | 44 / 86 ms | 0 |

`os-threads` in the log confirms onnxruntime's default of physical cores: the
SMT Xeon runner starts two fewer workers than the ARM one. The raw export
`model.onnx` costs the same RSS as the fused graph (1275 MiB) at 5–10 % less
throughput. Xeon: int8 v4 is *slower* than fp32 there (494 vs 576 tok/s; the
v0.3.1 recipe did 742), see `quantization/measurements.md`.
