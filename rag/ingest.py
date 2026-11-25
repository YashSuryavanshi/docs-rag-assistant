"""Incremental indexing.

Each file is hashed. Compared with the hashes stored in Qdrant, a file is:

- new or changed -> its old chunks are deleted, it is re-chunked and upserted;
- unchanged      -> skipped entirely;
- removed        -> its chunks are deleted.

Re-embedding is further limited by the embedding cache: even a changed file
only pays for the chunks whose text actually changed.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from pathlib import Path

from rag.chunking import CHUNKER_VERSION, Chunk, chunk_markdown
from rag.config import Settings
from rag.embeddings import Embedder
from rag.store import Store


@dataclass
class IndexReport:
    files_total: int
    files_added: int
    files_changed: int
    files_removed: int
    files_unchanged: int
    chunks_written: int
    embed_cache_hits: int
    embed_cache_misses: int
    seconds: float

    def __str__(self) -> str:
        return (
            f"files: {self.files_total} total | {self.files_added} added, {self.files_changed} changed, "
            f"{self.files_removed} removed, {self.files_unchanged} unchanged\n"
            f"chunks written: {self.chunks_written} | embeddings: {self.embed_cache_hits} from cache, "
            f"{self.embed_cache_misses} computed | {self.seconds:.1f}s"
        )


def file_hash(text: str, settings: Settings) -> str:
    # Chunking parameters and chunker code version are part of the hash:
    # changing either must re-chunk every file even though the files
    # themselves did not change.
    salt = f"v{CHUNKER_VERSION}:{settings.chunk_size}:{settings.chunk_overlap}:{settings.embedding_model}"
    return hashlib.sha256(f"{salt}\x00{text}".encode()).hexdigest()


def load_corpus(corpus_dir: Path) -> dict[str, str]:
    files = {}
    for path in sorted(corpus_dir.rglob("*.md")):
        files[path.relative_to(corpus_dir).as_posix()] = path.read_text(encoding="utf-8", errors="replace")
    return files


def build_index(settings: Settings, embedder: Embedder, store: Store, full: bool = False) -> IndexReport:
    start = time.perf_counter()
    hits0, misses0 = embedder.cache_hits, embedder.cache_misses

    store.ensure_collection(embedder.dim, recreate=full)
    corpus = load_corpus(settings.corpus_dir)
    # A file that yields no chunks (a stub page, an include-only template) is never stored
    # in Qdrant, so it would be reported as "added" on every run. Drop such files up front.
    corpus = {
        src: text
        for src, text in corpus.items()
        if chunk_markdown(src, text, settings.chunk_size, settings.chunk_overlap)
    }
    hashes = {src: file_hash(text, settings) for src, text in corpus.items()}
    indexed = {} if full else store.indexed_files()

    added = [s for s in corpus if s not in indexed]
    changed = [s for s in corpus if s in indexed and indexed[s] != hashes[s]]
    removed = [s for s in indexed if s not in corpus]

    for src in changed + removed:
        store.delete_file(src)

    chunks: list[Chunk] = []
    for src in added + changed:
        chunks.extend(chunk_markdown(src, corpus[src], settings.chunk_size, settings.chunk_overlap))

    if chunks:
        vectors = embedder.embed_passages([c.embed_text for c in chunks])
        store.upsert(chunks, vectors, hashes)

    return IndexReport(
        files_total=len(corpus),
        files_added=len(added),
        files_changed=len(changed),
        files_removed=len(removed),
        files_unchanged=len(corpus) - len(added) - len(changed),
        chunks_written=len(chunks),
        embed_cache_hits=embedder.cache_hits - hits0,
        embed_cache_misses=embedder.cache_misses - misses0,
        seconds=time.perf_counter() - start,
    )
