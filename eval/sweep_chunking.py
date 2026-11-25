"""Measure how chunk size and overlap affect retrieval, instead of guessing.

    python -m eval.sweep_chunking

Each configuration gets its own throwaway index under data/index/sweep/. The
embedding cache is shared, so a chunk that comes out identical under two
configurations is embedded once.
"""

from __future__ import annotations

import itertools
import statistics

from eval.metrics import METRICS
from eval.run_eval import EVAL_DIR, evaluate, load_dataset
from rag.config import PROJECT_ROOT, get_settings
from rag.embeddings import Embedder
from rag.generate import Generator
from rag.ingest import build_index
from rag.pipeline import Pipeline
from rag.retrieve import Retriever
from rag.store import Store

SIZES = [100, 200, 300, 450]
OVERLAPS = [0, 40]
MODE = "hybrid_rerank"


def main() -> None:
    dataset = load_dataset(EVAL_DIR / "dataset.jsonl")
    base = get_settings()
    embedder = Embedder(base.embedding_model, base.embedding_cache, base.query_instruction)

    rows = []
    for size, overlap in itertools.product(SIZES, OVERLAPS):
        s = get_settings(
            chunk_size=size,
            chunk_overlap=overlap,
            qdrant_path=PROJECT_ROOT / "data" / "index" / "sweep" / f"{size}_{overlap}",
            collection=f"sweep_{size}_{overlap}",
            qdrant_url=None,
        )
        store = Store(s)
        report = build_index(s, embedder, store, full=True)

        # Reuse the loaded models; only the store differs per configuration.
        pipe = Pipeline.__new__(Pipeline)
        pipe.settings, pipe.embedder, pipe.store = s, embedder, store
        pipe.retriever, pipe.generator = Retriever(s, embedder, store), Generator(s)

        summary = evaluate(pipe, dataset, MODE, k=s.final_k)["summary"]
        rows.append((size, overlap, store.count(), summary))
        print(f"size={size:<4} overlap={overlap:<3} chunks={store.count():<5} {report.seconds:5.1f}s  "
              + "  ".join(f"{m}={summary[m]:.3f}" for m in METRICS))
        store.client.close()

    best = max(rows, key=lambda r: (r[3]["context_precision"], r[3]["mrr"]))
    print(f"\nbest by context_precision: chunk_size={best[0]}, chunk_overlap={best[1]} "
          f"(precision {best[3]['context_precision']:.3f}, mrr {best[3]['mrr']:.3f})")
    print(f"mean context_precision across configs: {statistics.mean(r[3]['context_precision'] for r in rows):.3f}")


if __name__ == "__main__":
    main()
