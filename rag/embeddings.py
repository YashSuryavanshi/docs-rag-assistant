"""Sentence embeddings with a persistent cache.

Embedding is the expensive step of indexing. The cache is keyed by
sha256(model name + text), so:

- re-indexing after a docs update only embeds chunks whose text changed;
- changing chunk size re-embeds only chunks that actually differ;
- switching models never returns stale vectors (the model name is in the key).

SQLite is used because it ships with Python, is safe for one writer, and
needs no server.
"""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingCache:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path)
        self._db.execute("CREATE TABLE IF NOT EXISTS emb (key TEXT PRIMARY KEY, vec BLOB NOT NULL)")

    @staticmethod
    def key(model: str, text: str) -> str:
        return hashlib.sha256(f"{model}\x00{text}".encode()).hexdigest()

    def get_many(self, keys: list[str]) -> dict[str, np.ndarray]:
        found: dict[str, np.ndarray] = {}
        for start in range(0, len(keys), 500):  # stay under SQLite's variable limit
            batch = keys[start : start + 500]
            marks = ",".join("?" * len(batch))
            for k, blob in self._db.execute(f"SELECT key, vec FROM emb WHERE key IN ({marks})", batch):
                found[k] = np.frombuffer(blob, dtype=np.float32)
        return found

    def put_many(self, items: dict[str, np.ndarray]) -> None:
        self._db.executemany(
            "INSERT OR REPLACE INTO emb (key, vec) VALUES (?, ?)",
            [(k, v.astype(np.float32).tobytes()) for k, v in items.items()],
        )
        self._db.commit()


class Embedder:
    def __init__(self, model_name: str, cache_path: Path, query_instruction: str = ""):
        self.model_name = model_name
        self.query_instruction = query_instruction
        self._model = SentenceTransformer(model_name)
        self._cache = EmbeddingCache(cache_path)
        self.cache_hits = 0
        self.cache_misses = 0

    @property
    def dim(self) -> int:
        return self._model.get_sentence_embedding_dimension()

    def embed_passages(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        keys = [EmbeddingCache.key(self.model_name, t) for t in texts]
        cached = self._cache.get_many(keys)
        missing = [i for i, k in enumerate(keys) if k not in cached]
        self.cache_hits += len(texts) - len(missing)
        self.cache_misses += len(missing)

        if missing:
            vecs = self._model.encode(
                [texts[i] for i in missing],
                batch_size=batch_size,
                normalize_embeddings=True,  # cosine similarity == dot product
                show_progress_bar=len(missing) > 200,
            )
            new = {keys[i]: v for i, v in zip(missing, vecs, strict=True)}
            self._cache.put_many(new)
            cached.update(new)

        return np.stack([cached[k] for k in keys])

    def embed_query(self, query: str) -> np.ndarray:
        # Queries are not cached: they are cheap and rarely repeat exactly.
        return self._model.encode(self.query_instruction + query, normalize_embeddings=True)
