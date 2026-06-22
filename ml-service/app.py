"""
Pure embedding + rerank microservice.
No project knowledge, no prompt assembly, no state beyond model weights.
The Rust daemon owns everything else.
"""
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer, CrossEncoder

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_models = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    _models["embed"] = SentenceTransformer(EMBED_MODEL)
    _models["rerank"] = CrossEncoder(RERANK_MODEL)
    _models["dim"] = _models["embed"].get_sentence_embedding_dimension()
    yield
    _models.clear()


app = FastAPI(lifespan=lifespan)


class EmbedRequest(BaseModel):
    # Inputs are opaque text payloads. We never interpret them as instructions.
    inputs: List[str] = Field(..., max_length=512)
    normalize: bool = True


class EmbedResponse(BaseModel):
    model: str
    dim: int
    vectors: List[List[float]]


class RerankRequest(BaseModel):
    query: str
    documents: List[str] = Field(..., max_length=256)
    top_k: int | None = None


class RerankItem(BaseModel):
    index: int
    score: float


class RerankResponse(BaseModel):
    model: str
    results: List[RerankItem]


@app.get("/health")
def health():
    return {"status": "ok", "embed_model": EMBED_MODEL, "dim": _models.get("dim")}


@app.post("/embed", response_model=EmbedResponse)
def embed(req: EmbedRequest):
    vecs = _models["embed"].encode(
        req.inputs,
        normalize_embeddings=req.normalize,
        convert_to_numpy=True,
        batch_size=32,
    )
    return EmbedResponse(
        model=EMBED_MODEL,
        dim=_models["dim"],
        vectors=vecs.tolist(),
    )


@app.post("/rerank", response_model=RerankResponse)
def rerank(req: RerankRequest):
    pairs = [[req.query, d] for d in req.documents]
    scores = _models["rerank"].predict(pairs).tolist()
    ranked = sorted(
        ({"index": i, "score": float(s)} for i, s in enumerate(scores)),
        key=lambda x: x["score"],
        reverse=True,
    )
    if req.top_k:
        ranked = ranked[: req.top_k]
    return RerankResponse(model=RERANK_MODEL, results=[RerankItem(**r) for r in ranked])