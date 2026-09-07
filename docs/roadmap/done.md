# Roadmap: shipped versions

v0.1–v0.4 are in `done-early.md`.

Versions are planned around one measurable goal each. Numbers quoted here were
measured on 2026-09-05 (see `../verification.md` and `../quantization/`); every
step is gated by the fixtures in `tests/fixtures/` staying green.

## Baseline facts that shaped the plan

| finding | consequence |
|---|---|
| int8 is 3.9× faster than fp32 on x86 (Xeon 8573C, VNNI) and 4.3× on Neoverse-N2, but only 1.1× on Apple Silicon | int8 accuracy work (v0.3) is worth it; macOS users need fp32 speed-ups instead |
| `onnxruntime.transformers` fuses 24 `Attention`, 48 `SkipLayerNormalization`, 24 `BiasGelu`; outputs unchanged; only 48 tensors (288 MiB, the merged QKV weights) differ from the Hub weights | fused fp32 graph (v0.2) can be built locally from the cached weights in seconds, no 2.3 GB re-download |
| ORT's own `ORT_ENABLE_ALL` already fuses LayerNorm + Gelu but never Attention on this opset-11 export; saving the ORT-optimised graph to disk cuts session creation from 0.38 s to 0.33 s only | "cache the optimised graph" was dropped: the start-up cost is elsewhere |
| warm start-up was 0.86 s: tokenizer parse 0.31 s, ORT session 0.44 s; cold start is bound by reading 2.3 GB from disk | v0.1 caches the parsed vocabulary (0.31 s → 0.05 s); the rest needs a smaller model (int8: 543 MB) |
| a batch of 12 texts × 8192 tokens allocates more than the 400 MiB hidden state: the unfused attention materialises `batch × 16 × seq²` floats | token-budget batching (v0.1) bounds memory by padded tokens instead of text count |

## v0.5.0 — resource efficiency (done)

Measured on the M4 and the CI matrix (2026-09-06, `../resources.md`); ships
`model_int8.onnx` + `model_int8.onnx_data` (same tensors as v0.4).

| goal | result |
|---|---|
| resident memory of the int8 session | graph + external data like the fused graph: 1838 → 724 MiB after load (ORT no longer keeps the parsed protobuf next to the prepacked weights), 8192-token peak 2286 → 1679 MiB, start-up 0.80 → 0.75 s, outputs identical |
| idle CPU | `session.intra_op.allow_spinning=0` by default: 27–64 ms/s → 0 per session, short-query CPU −35 %, throughput −0–5 %; `BGE_M3_LITE_SPIN=1` / `spin=True` restore it |
| CPU-seconds per token | thread default = performance cores on Apple Silicon (4 on the M4; ORT picked 5): −14 % CPU at equal throughput; one thread gives 2× the tokens per CPU-second of four — documented for multi-worker services |
| measurement | `tools/eval_model.py` prints RSS after load / peak, CPU-s per 1k tokens, short-query wall + CPU, idle CPU and OS thread count; the CI bench summary tabulates them, one process per model |

## v0.5.1 — `low_memory` mode (done)

`BGEM3Embedder(low_memory=True)` / `encode --low-memory` disables MLAS
prepacking (M4, `../resources.md`): fp32 starts in 0.11 s with 140 MiB
private memory (113 MB physical footprint after queries), int8 in 0.63 s /
149 MiB; weight pages are file-backed and shared between processes; batch
throughput −5 %, single short queries 2× slower. Disabling the arena was
measured and rejected: it returns no memory after a long request and costs
100–200 MiB more.

## v0.5.2 — activation memory (done)

`fuse` moves the whole layer tail into the attention `Loop` and the chunk is
256 (`../memory.md`): fp32 peak per padded token 0.10–0.125 → 0.064–0.075
MiB for texts of 512+ tokens (−30–40 %), +20 % for batches of texts no longer
than the chunk; int8 ships chunk 256 without the tail (−7–33 %, the loop
costs memory there). Bit-identical outputs, ±1 % on 16/128-token batches,
−4 % at 512 tokens. `quantize` rebuilds the unchunked fused graph from
`model.onnx` in memory, so the int8 weight file is unchanged since v0.5.0
and `--attention-chunk` / `--layer-loop` take effect. Lesson: onnxruntime's
memory pattern never applies to the first run of a shape; the BFC arena grows
in powers of two, and the peak follows the allocation sequence, so every
layout change must be measured (`kSameAsRequested`, shrinkage, chunk 512 +
loop all lost).

## v0.6.0 — asyncio serving (done)

`AsyncEmbedder` (`../serving/recipe.md`, measured with `tools/bench_serving.py` on
the M4 and the CI matrix, 2026-09-06; no model asset changes): `await
encode / encode_queries / encode_corpus` with the synchronous signatures and
outputs, run in a private thread pool with at most `max_concurrency` calls in
flight (default 2 for fp32, 4 for int8), `queue_depth` / `in_flight` for
health endpoints, `async with` / `close()` that drains the queue. Facts: one
4-thread session beats four 1-thread sessions; fp32 short queries are
GEMM-bound (two runs in flight +50 % at equal CPU, then flat; padding 8 into
one call 3.6×), int8 scales with runs in flight (2.1× at 4, 2.4× at 8) and
loses with batching; passages gain 25–40 % from 2–4 runs in flight; each run
in flight adds its own activations (0.07 MiB per padded token). The
micro-batcher (`batch_window_ms`) holds a request only while every slot is
busy or claimed and takes a burst as one call, so it costs nothing at low
load: fp32 short queries 2.5× the sequential rate at 4 clients (p95 34 ms),
3.6× at 8, at a third of the CPU per request; on by default for fp32, off for
int8 (−6–24 %).

## v0.6.1 — the leftovers, measured (done)

Measured on the M4 and the CI matrix (2026-09-07, `../memory.md`,
`../serving/measurements.md`); ships new `model_fused.onnx` and
`model_int8.onnx` graphs (both weight files unchanged, outputs
bit-identical).

| item | result |
|---|---|
| `session.run_async` versus the thread pool | `bench_serving.py --only run_async`: loses everywhere (M4 fp32 ×1 41 vs 53 req/s, int8 ×4 125 vs 176; runners −25–50 %), closed |
| length-aware micro-batcher | requests merge only within a character-length bucket (≤ 128 / 512 / 2048 chars): a query burst next to a 600-token passage answers in 37–75 ms instead of 1.5–4.4 s (fp32); int8 unchanged (no batching) |
| short-batch activation memory | the layer tail runs in a scan-output `Loop` over 256 rows of the flattened batch (`fuse`/`quantize --tail rows`): fp32 `128 × 128` 1734 → 941 MiB, every shape −5–46 %, 0.06–0.07 MiB per padded token for any batch; tok/s unchanged; the `If` bypass was measured and rejected (double prepacking) |
| int8 start-up | the same row loop halves the outer graph (2 692 → 1 396 nodes): session 0.69 → 0.31 s on the M4 (1.7 → 0.8 s on the runners), int8 memory −10 % at every shape, short-query latency unchanged, 128-token batches −2–9 % on the runners (256-row windows are the optimum; `--tail none` restores it); no change to the quantisation recipe |

## v0.6.2 — asyncio on Python 3.11 versus 3.12+ (done)

No model asset changes. Every scheduling fact `serving.py` relies on is
reproduced by `tools/asyncio_probe.py` on 3.11 / 3.12 / 3.13 / 3.14 and
recorded with the test that pins it in `../serving/asyncio.md`. The boundary
is 3.12, not 3.13: `gather()` of done futures no longer yields (the v0.6
`close()` busy-loop), `wait_for` keeps its coroutine in the caller's task
(arrival order inside a batch), the eager task factory exists; 3.13's
`Semaphore` rewrite changes nothing observable (FIFO hand-over and the cancel
bookkeeping behave the same on 3.11.15). One code change: a request
cancelled while its call runs (`asyncio.timeout`) now keeps its slot until
the thread returns (`asyncio.shield`); before, the slot was released early,
`in_flight` undercounted and `close()` blocked the loop in
`executor.shutdown`. Six new tests, green on both versions.
