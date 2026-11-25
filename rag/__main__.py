"""Command line interface.

    python -m rag index            # incremental: only new/changed files
    python -m rag index --full     # rebuild from scratch
    python -m rag ask "How do I enable prefix caching?"
    python -m rag ask "..." --mode dense    # compare retrieval stages
"""

from __future__ import annotations

import argparse

from rag.pipeline import Pipeline
from rag.retrieve import Retriever


def main() -> None:
    parser = argparse.ArgumentParser(prog="rag")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_index = sub.add_parser("index", help="build or update the index")
    p_index.add_argument("--full", action="store_true", help="drop and rebuild the collection")

    p_ask = sub.add_parser("ask", help="ask a question")
    p_ask.add_argument("question")
    p_ask.add_argument("-k", type=int, default=None, help="passages to return")
    p_ask.add_argument("--mode", choices=Retriever.MODES, default="hybrid_rerank")

    args = parser.parse_args()
    pipe = Pipeline()

    if args.cmd == "index":
        print(pipe.index(full=args.full))
        print(f"collection now holds {pipe.store.count()} chunks")
        return

    ans = pipe.ask(args.question, k=args.k, mode=args.mode)
    for i, hit in enumerate(ans.retrieval.hits, start=1):
        preview = " ".join(hit.text.split())[:220]
        print(f"[{i}] {hit.score:.3f}  {hit.source}  |  {hit.heading_path}\n    {preview}\n")
    timings = ", ".join(f"{k} {v:.0f} ms" for k, v in ans.retrieval.timings_ms.items())
    print(f"timings: {timings}")
    if ans.answer is not None:
        print(f"\nANSWER:\n{ans.answer}")
    else:
        print("\n(no LLM configured; set RAG_LLM_BASE_URL and RAG_LLM_MODEL to generate answers)")


if __name__ == "__main__":
    main()
