"""Retrieval metrics, computed against hand-labelled relevant files.

Each eval question lists the corpus file(s) that answer it. A retrieved chunk
counts as relevant if it comes from one of those files.

- hit@k             : did ANY relevant chunk appear in the top k?
- mrr               : 1 / rank of the first relevant chunk (0 if none).
                      Rewards putting the answer at the top.
- context_precision : average precision over the top k. High when relevant
                      chunks sit above irrelevant ones. This is the metric
                      that drops when chunks are too small (fragments of many
                      files) or too large (one relevant line buried in noise).
- context_recall    : fraction of the relevant files covered in the top k.
                      Matters for questions whose answer spans several pages.
"""

from __future__ import annotations


def hit_at_k(retrieved: list[str], relevant: set[str]) -> float:
    return 1.0 if any(s in relevant for s in retrieved) else 0.0


def mrr(retrieved: list[str], relevant: set[str]) -> float:
    for rank, s in enumerate(retrieved, start=1):
        if s in relevant:
            return 1.0 / rank
    return 0.0


def context_precision(retrieved: list[str], relevant: set[str]) -> float:
    hits, total = 0, 0.0
    for rank, s in enumerate(retrieved, start=1):
        if s in relevant:
            hits += 1
            total += hits / rank
    return total / hits if hits else 0.0


def context_recall(retrieved: list[str], relevant: set[str]) -> float:
    return len(relevant & set(retrieved)) / len(relevant) if relevant else 0.0


METRICS = {
    "hit@k": hit_at_k,
    "mrr": mrr,
    "context_precision": context_precision,
    "context_recall": context_recall,
}


def score(retrieved: list[str], relevant: set[str]) -> dict[str, float]:
    return {name: fn(retrieved, relevant) for name, fn in METRICS.items()}
