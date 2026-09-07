# FastAPI integration

The shape that follows from `recipe.md` (defaults, memory) and `asyncio.md`
(what cancellation and `close()` really do). One model per process, two
`AsyncEmbedder` wrappers around it (queries batch, passages do not), a
queue-depth guard before a timeout, everything closed by the lifespan.

```python
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from bge_m3_lite import AsyncEmbedder, BGEM3Embedder

QUERY_TIMEOUT_S, PASSAGE_TIMEOUT_S = 2.0, 30.0  # wider than the call itself
MAX_QUEUE = 64  # beyond this answer 503 instead of stretching p95


@asynccontextmanager
async def lifespan(app: FastAPI):
    model = BGEM3Embedder(precision="int8")  # one copy per uvicorn worker
    async with (
        AsyncEmbedder(model) as queries,  # per-precision defaults
        AsyncEmbedder(model, batch_window_ms=0) as corpus,  # passages: no batching
    ):
        app.state.queries, app.state.corpus = queries, corpus
        yield
    model.close()  # close() drained both wrappers in the loop first


app = FastAPI(lifespan=lifespan)


class EmbedIn(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=64)
    sparse: bool = False
    colbert: bool = False


def guard(emb: AsyncEmbedder) -> None:
    if emb.queue_depth >= MAX_QUEUE:
        raise HTTPException(503, "embedder busy", headers={"Retry-After": "1"})


async def call(emb: AsyncEmbedder, method, body: EmbedIn, timeout: float):
    guard(emb)
    try:
        async with asyncio.timeout(timeout):
            return await method(
                body.texts, return_sparse=body.sparse, return_colbert_vecs=body.colbert
            )
    except TimeoutError:  # the thread still finishes the call and holds its slot
        raise HTTPException(504, "embedding timed out") from None


@app.post("/embed/query")
async def embed_query(body: EmbedIn, request: Request):
    emb = request.app.state.queries
    return to_json(await call(emb, emb.encode_queries, body, QUERY_TIMEOUT_S))


@app.post("/embed/passage")
async def embed_passage(body: EmbedIn, request: Request):
    emb = request.app.state.corpus
    return to_json(await call(emb, emb.encode_corpus, body, PASSAGE_TIMEOUT_S))


@app.get("/health")
async def health(request: Request):
    s = request.app.state
    return {
        name: {"in_flight": e.in_flight, "queue_depth": e.queue_depth}
        for name, e in (("queries", s.queries), ("corpus", s.corpus))
    }


def to_json(out):
    cv = out["colbert_vecs"]
    return {
        "dense": out["dense_vecs"].tolist(),
        "sparse": out["lexical_weights"],
        "colbert": None if cv is None else [v.tolist() for v in cv],
    }
```

## Why each piece is there

- **Two wrappers, one model**: each counts its own slots; queries get the
  micro-batcher (fp32) or four runs in flight (int8), passages gain nothing
  from batching and would pad a query burst (`measurements.md`).
- **`guard` before `timeout`**: a timeout does not save CPU — the thread runs
  `session.run` to the end and keeps its slot (`asyncio.md`), so the queue
  depth is the first line of defence and the timeout the last. Size the
  timeout above the real call time for the longest allowed input.
- **Never the synchronous `encode` in a coroutine** (11–20 ms of blocked
  loop per short query, seconds per passage) and no other CPU work in
  `asyncio.to_thread` competing with the embedder's private pool.
- **Shutdown by the lifespan only**: `close()` waits for every running call
  inside the loop, then shuts the pool down; calling the executor yourself
  from a coroutine blocks the server. Uvicorn finishes the lifespan on
  SIGTERM, so in-flight requests complete.
- **Workers from memory**: per worker `1.3 GB fp32 | 0.75 GB int8 +
  max_concurrency × max_batch_tokens × 0.07 MiB`; lower `max_batch_tokens`
  on small machines, or `BGEM3Embedder(low_memory=True)` to share weight
  pages between workers at 2× the single-query latency (`../resources.md`).
- **Python 3.11 and 3.12+ behave the same here**; if the app adds its own
  "wait until this set of tasks is empty" loop, remember that `gather` of
  finished tasks no longer yields on 3.12+ (`asyncio.md`).
