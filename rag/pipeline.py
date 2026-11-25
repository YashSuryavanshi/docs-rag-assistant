"""Builds the components once and wires them together.

Models take seconds to load, so the CLI, the API and the evaluation scripts
all construct one Pipeline and reuse it.
"""

from __future__ import annotations

from dataclasses import dataclass

from rag.config import Settings, get_settings
from rag.embeddings import Embedder
from rag.generate import Generator
from rag.ingest import IndexReport, build_index
from rag.retrieve import RetrievalResult, Retriever
from rag.store import Store


@dataclass
class Answer:
    question: str
    answer: str | None
    retrieval: RetrievalResult


class Pipeline:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.embedder = Embedder(
            self.settings.embedding_model, self.settings.embedding_cache, self.settings.query_instruction
        )
        self.store = Store(self.settings)
        self.retriever = Retriever(self.settings, self.embedder, self.store)
        self.generator = Generator(self.settings)

    def index(self, full: bool = False) -> IndexReport:
        return build_index(self.settings, self.embedder, self.store, full=full)

    def ask(self, question: str, k: int | None = None, mode: str = "hybrid_rerank") -> Answer:
        result = self.retriever.retrieve(question, k=k, mode=mode)
        return Answer(question, self.generator.answer(question, result.hits), result)
