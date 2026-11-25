"""HTTP API.

    uvicorn rag.api:app --port 8080

Health endpoints are split the way Kubernetes expects:
- /health/live  : the process is up (restart it if this fails)
- /health/ready : models are loaded and the index is non-empty (only then
                  should the Service send it traffic)

Loading models takes several seconds; without a readiness check, Kubernetes
would route requests to a pod that cannot answer yet.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from rag.pipeline import Pipeline
from rag.retrieve import Retriever

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    pipe = Pipeline()
    pipe.retriever.reranker  # load the cross-encoder now, not on the first request  # noqa: B018
    pipe.store.sparse_search("warmup", 1)  # build the BM25 index now
    state["pipe"] = pipe
    yield
    state.clear()


app = FastAPI(title="docs-rag-assistant", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    k: int = Field(default=5, ge=1, le=20)
    mode: str = "hybrid_rerank"


class Passage(BaseModel):
    source: str
    heading_path: str
    score: float
    text: str


class AskResponse(BaseModel):
    answer: str | None
    passages: list[Passage]
    timings_ms: dict[str, float]


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.get("/health/live")
def live() -> dict:
    return {"status": "ok"}


@app.get("/health/ready")
def ready() -> dict:
    pipe: Pipeline | None = state.get("pipe")
    if pipe is None or pipe.store.count() == 0:
        raise HTTPException(503, "not ready: models loading or index empty")
    return {"status": "ready", "chunks": pipe.store.count()}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    if req.mode not in Retriever.MODES:
        raise HTTPException(422, f"mode must be one of {Retriever.MODES}")
    ans = state["pipe"].ask(req.question, k=req.k, mode=req.mode)
    passages = [
        Passage(source=h.source, heading_path=h.heading_path, score=h.score, text=h.text) for h in ans.retrieval.hits
    ]
    return AskResponse(answer=ans.answer, passages=passages, timings_ms=ans.retrieval.timings_ms)
