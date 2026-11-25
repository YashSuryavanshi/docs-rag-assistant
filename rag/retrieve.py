"""Hybrid retrieval: dense + BM25, fused with Reciprocal Rank Fusion, then reranked.

Pipeline for one query:

    query ──► dense search (top 30)  ─┐
          └─► BM25 search  (top 30)  ─┴─► RRF fusion ──► cross-encoder rerank ──► top 5

Why each stage exists:

- Dense search finds passages that mean the same thing in different words
  ("reuse KV cache across requests" -> "Automatic Prefix Caching").
- BM25 finds exact identifiers that embeddings blur ("--max-num-seqs",
  "VLLM_USE_V1", error strings).
- RRF merges the two ranked lists using only ranks, not raw scores. Cosine
  similarities and BM25 scores live on different scales, so averaging them
  would be meaningless; ranks are comparable. score(d) = sum 1 / (k + rank).
- The cross-encoder reads query and passage *together*, which is far more
  accurate than comparing two independently computed vectors, but too slow to
  run over the whole corpus. So it only re-scores the ~30 fused candidates.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from sentence_transformers import CrossEncoder

from rag.config import Settings
from rag.embeddings import Embedder
from rag.store import Hit, Store


def reciprocal_rank_fusion(rankings: list[list[Hit]], k: int = 60) -> list[Hit]:
    scores: dict[str, float] = {}
    first_seen: dict[str, Hit] = {}
    for ranking in rankings:
        for rank, hit in enumerate(ranking, start=1):
            scores[hit.id] = scores.get(hit.id, 0.0) + 1.0 / (k + rank)
            first_seen.setdefault(hit.id, hit)
    fused = sorted(scores, key=scores.get, reverse=True)
    return [Hit(**{**first_seen[i].__dict__, "score": scores[i]}) for i in fused]


@dataclass
class RetrievalResult:
    hits: list[Hit]
    timings_ms: dict[str, float] = field(default_factory=dict)


class Retriever:
    MODES = ("dense", "sparse", "hybrid", "hybrid_rerank")

    def __init__(self, settings: Settings, embedder: Embedder, store: Store):
        self.s = settings
        self.embedder = embedder
        self.store = store
        self._reranker: CrossEncoder | None = None

    @property
    def reranker(self) -> CrossEncoder:
        if self._reranker is None:
            self._reranker = CrossEncoder(self.s.reranker_model)
        return self._reranker

    def retrieve(self, query: str, k: int | None = None, mode: str = "hybrid_rerank") -> RetrievalResult:
        """`mode` exists so the evaluation can measure what each stage contributes."""
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}")
        k = k or self.s.final_k
        t: dict[str, float] = {}

        dense: list[Hit] = []
        sparse: list[Hit] = []
        if mode != "sparse":
            t0 = time.perf_counter()
            dense = self.store.dense_search(self.embedder.embed_query(query), self.s.dense_k)
            t["dense"] = (time.perf_counter() - t0) * 1000
        if mode != "dense":
            t0 = time.perf_counter()
            sparse = self.store.sparse_search(query, self.s.sparse_k)
            t["sparse"] = (time.perf_counter() - t0) * 1000

        if mode == "dense":
            return RetrievalResult(dense[:k], t)
        if mode == "sparse":
            return RetrievalResult(sparse[:k], t)

        fused = reciprocal_rank_fusion([dense, sparse], k=self.s.rrf_k)
        if mode == "hybrid":
            return RetrievalResult(fused[:k], t)

        t0 = time.perf_counter()
        candidates = fused[: self.s.rerank_k]
        scores = self.reranker.predict([(query, f"{h.heading_path}\n{h.text}") for h in candidates])
        ranked = sorted(zip(candidates, scores, strict=True), key=lambda pair: pair[1], reverse=True)
        t["rerank"] = (time.perf_counter() - t0) * 1000
        return RetrievalResult([Hit(**{**h.__dict__, "score": float(s)}) for h, s in ranked[:k]], t)
