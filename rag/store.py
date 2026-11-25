"""Vector store (Qdrant) plus a keyword index (BM25) built from the same data.

Qdrant holds every chunk's vector *and* its payload (text, source file,
heading path, hash of the file it came from). Treating Qdrant as the single
source of truth means:

- the BM25 index is rebuilt from Qdrant payloads, so the two can never
  disagree about which chunks exist;
- incremental indexing compares file hashes stored in Qdrant against files
  on disk, so no separate manifest can drift out of sync.

Rebuilding BM25 takes milliseconds for a few thousand chunks; the expensive
part (embedding) is incremental and cached.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)
from rank_bm25 import BM25Okapi

from rag.chunking import Chunk
from rag.config import Settings

# Keeps identifiers such as "max-model-len", "fp8_e4m3" or "v1.completions"
# whole. Exact identifiers are where keyword search beats embeddings, and the
# reason this system is hybrid rather than dense-only.
TOKEN = re.compile(r"[a-z0-9_]+(?:[-.][a-z0-9_]+)*")


def tokenize(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


@dataclass
class Hit:
    id: str
    score: float
    source: str
    heading_path: str
    text: str


def point_id(source: str, index: int) -> str:
    """Deterministic ID, so re-indexing a file overwrites rather than duplicates."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source}#{index}"))


class Store:
    def __init__(self, settings: Settings):
        self.s = settings
        if settings.qdrant_url:
            self.client = QdrantClient(url=settings.qdrant_url)
        else:
            settings.qdrant_path.mkdir(parents=True, exist_ok=True)
            self.client = QdrantClient(path=str(settings.qdrant_path))
        self._bm25: BM25Okapi | None = None
        self._bm25_hits: list[Hit] = []

    # ---------- writes ----------

    def ensure_collection(self, dim: int, recreate: bool = False) -> None:
        name = self.s.collection
        if recreate and self.client.collection_exists(name):
            self.client.delete_collection(name)
        if not self.client.collection_exists(name):
            self.client.create_collection(name, vectors_config=VectorParams(size=dim, distance=Distance.COSINE))
            if self.s.qdrant_url:  # payload indexes are a server feature
                self.client.create_payload_index(name, "source", field_schema=PayloadSchemaType.KEYWORD)

    def indexed_files(self) -> dict[str, str]:
        """Map source file -> hash of the version currently indexed."""
        if not self.client.collection_exists(self.s.collection):
            return {}
        files: dict[str, str] = {}
        offset = None
        while True:
            points, offset = self.client.scroll(
                self.s.collection, limit=1000, offset=offset, with_payload=["source", "file_hash"], with_vectors=False
            )
            for p in points:
                files[p.payload["source"]] = p.payload["file_hash"]
            if offset is None:
                return files

    def delete_file(self, source: str) -> None:
        same_file = Filter(must=[FieldCondition(key="source", match=MatchValue(value=source))])
        self.client.delete(self.s.collection, points_selector=FilterSelector(filter=same_file))

    def upsert(self, chunks: list[Chunk], vectors: np.ndarray, file_hashes: dict[str, str]) -> None:
        points = [
            PointStruct(
                id=point_id(c.source, c.index),
                vector=v.tolist(),
                payload={
                    "source": c.source,
                    "index": c.index,
                    "heading_path": c.heading_path,
                    "text": c.text,
                    "embed_text": c.embed_text,
                    "file_hash": file_hashes[c.source],
                },
            )
            for c, v in zip(chunks, vectors, strict=True)
        ]
        for start in range(0, len(points), 256):
            self.client.upsert(self.s.collection, points=points[start : start + 256])
        self._bm25 = None  # invalidate; rebuilt lazily on next keyword search

    # ---------- reads ----------

    def count(self) -> int:
        if not self.client.collection_exists(self.s.collection):
            return 0
        return self.client.count(self.s.collection).count

    def dense_search(self, vector: np.ndarray, k: int) -> list[Hit]:
        res = self.client.query_points(self.s.collection, query=vector.tolist(), limit=k, with_payload=True)
        return [self._hit(p.id, p.score, p.payload) for p in res.points]

    def sparse_search(self, query: str, k: int) -> list[Hit]:
        if self._bm25 is None:
            self._build_bm25()
        if not self._bm25_hits:
            return []
        scores = self._bm25.get_scores(tokenize(query))
        top = np.argsort(scores)[::-1][:k]
        return [
            Hit(**{**self._bm25_hits[i].__dict__, "score": float(scores[i])}) for i in top if scores[i] > 0
        ]

    def _build_bm25(self) -> None:
        hits: list[Hit] = []
        offset = None
        while True:
            points, offset = self.client.scroll(
                self.s.collection, limit=1000, offset=offset, with_payload=True, with_vectors=False
            )
            hits.extend(self._hit(p.id, 0.0, p.payload) for p in points)
            if offset is None:
                break
        self._bm25_hits = hits
        corpus = [tokenize(f"{h.heading_path}\n{h.text}") for h in hits]
        self._bm25 = BM25Okapi(corpus) if corpus else None

    @staticmethod
    def _hit(pid, score: float, payload: dict) -> Hit:
        return Hit(
            id=str(pid),
            score=float(score),
            source=payload["source"],
            heading_path=payload.get("heading_path", ""),
            text=payload["text"],
        )
