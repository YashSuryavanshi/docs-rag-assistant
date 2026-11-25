"""Incremental indexing with a fake embedder, so the test needs no model download."""

import numpy as np

from rag.config import get_settings
from rag.ingest import build_index
from rag.store import Store


class FakeEmbedder:
    dim = 8

    def __init__(self):
        self.cache_hits = 0
        self.cache_misses = 0
        self.embedded = 0

    def embed_passages(self, texts):
        self.embedded += len(texts)
        self.cache_misses += len(texts)
        rng = np.random.default_rng(len(texts))
        v = rng.normal(size=(len(texts), self.dim)).astype(np.float32)
        return v / np.linalg.norm(v, axis=1, keepdims=True)


def test_files_with_no_chunks_do_not_show_up_as_added_every_run(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "real.md").write_text("# Real\n\nsome content")
    (corpus / "stub.md").write_text('--8<-- "examples/snippet.py"\n')  # include directive only

    s = get_settings(corpus_dir=corpus, qdrant_path=tmp_path / "q", qdrant_url=None, collection="t")
    store, emb = Store(s), FakeEmbedder()

    first = build_index(s, emb, store)
    assert (first.files_total, first.files_added) == (1, 1)

    second = build_index(s, emb, store)
    assert (second.files_added, second.files_changed, second.chunks_written) == (0, 0, 0)
    store.client.close()


def test_only_changed_files_are_reindexed(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.md").write_text("# A\n\nalpha text")
    (corpus / "b.md").write_text("# B\n\nbeta text")

    s = get_settings(corpus_dir=corpus, qdrant_path=tmp_path / "q", qdrant_url=None, collection="t")
    store, emb = Store(s), FakeEmbedder()

    first = build_index(s, emb, store)
    assert (first.files_added, first.files_changed) == (2, 0)

    second = build_index(s, emb, store)
    assert (second.files_added, second.files_changed, second.chunks_written) == (0, 0, 0)

    (corpus / "b.md").write_text("# B\n\nbeta text, edited")
    (corpus / "a.md").unlink()
    third = build_index(s, emb, store)
    assert (third.files_changed, third.files_removed, third.files_unchanged) == (1, 1, 0)
    assert set(store.indexed_files()) == {"b.md"}
    store.client.close()
