"""Evaluate retrieval quality, and fail if it regresses (used as the CI gate).

    python -m eval.run_eval                      # all retrieval modes, compared
    python -m eval.run_eval --mode hybrid_rerank --gate
    python -m eval.run_eval --faithfulness       # also judge generated answers (needs an LLM)

--gate exits with status 1 if any metric of the gated mode falls below
eval/thresholds.yaml, which is how GitHub Actions blocks a merge that makes
retrieval worse.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import yaml

from eval.faithfulness import faithfulness
from eval.metrics import METRICS, score
from rag.pipeline import Pipeline
from rag.retrieve import Retriever

EVAL_DIR = Path(__file__).resolve().parent


def check_gate(summary: dict[str, float], minimums: dict[str, float]) -> list[str]:
    """Return one message per metric that is below its floor (empty list = gate passes)."""
    return [f"{m}: {summary[m]:.3f} < {floor}" for m, floor in minimums.items() if summary[m] < floor]


def load_dataset(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def evaluate(pipe: Pipeline, dataset: list[dict], mode: str, k: int) -> dict:
    per_q, latencies = [], []
    for item in dataset:
        t0 = time.perf_counter()
        res = pipe.retriever.retrieve(item["question"], k=k, mode=mode)
        latencies.append((time.perf_counter() - t0) * 1000)
        sources = [h.source for h in res.hits]
        per_q.append({"id": item["id"], "sources": sources, **score(sources, set(item["relevant"]))})
    summary = {m: statistics.mean(q[m] for q in per_q) for m in METRICS}
    summary["latency_p50_ms"] = statistics.median(latencies)
    return {"mode": mode, "k": k, "summary": summary, "per_question": per_q}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=EVAL_DIR / "dataset.jsonl")
    parser.add_argument("--mode", choices=Retriever.MODES, default=None, help="default: compare all modes")
    parser.add_argument("-k", type=int, default=5)
    parser.add_argument("--gate", action="store_true", help="exit 1 if below eval/thresholds.yaml")
    parser.add_argument("--faithfulness", action="store_true", help="judge generated answers with the LLM")
    parser.add_argument("--out", type=Path, default=EVAL_DIR / "results" / "latest.json")
    args = parser.parse_args()

    dataset = load_dataset(args.dataset)
    pipe = Pipeline()
    if pipe.store.count() == 0:
        sys.exit("Index is empty. Run `python -m rag index` first.")

    modes = [args.mode] if args.mode else list(Retriever.MODES)
    reports = [evaluate(pipe, dataset, m, args.k) for m in modes]

    print(f"\n{len(dataset)} questions, top-{args.k}\n")
    header = f"{'mode':<15}" + "".join(f"{m:>19}" for m in [*METRICS, "latency_p50_ms"])
    print(header)
    print("-" * len(header))
    for r in reports:
        print(f"{r['mode']:<15}" + "".join(f"{r['summary'][m]:>19.3f}" for m in [*METRICS, "latency_p50_ms"]))

    if args.faithfulness:
        if not pipe.generator.enabled:
            sys.exit("--faithfulness needs RAG_LLM_BASE_URL and RAG_LLM_MODEL")
        scores = []
        for item in dataset:
            ans = pipe.ask(item["question"])
            f = faithfulness(pipe.generator, ans.answer or "", ans.retrieval.hits)
            if f is not None:
                scores.append(f)
        print(f"\nfaithfulness (hybrid_rerank): {statistics.mean(scores):.3f} over {len(scores)} answers")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(reports, indent=2))

    if args.gate:
        thresholds = yaml.safe_load((EVAL_DIR / "thresholds.yaml").read_text())
        gated = next((r for r in reports if r["mode"] == thresholds["mode"]), None)
        if gated is None:
            sys.exit(f"--gate needs mode {thresholds['mode']!r} to be evaluated (use --mode or run all modes)")
        failures = check_gate(gated["summary"], thresholds["min"])
        if failures:
            print("\nQUALITY GATE FAILED\n  " + "\n  ".join(failures))
            sys.exit(1)
        print("\nquality gate passed")


if __name__ == "__main__":
    main()
