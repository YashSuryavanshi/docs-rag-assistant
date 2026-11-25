"""Unit tests for the parts of retrieval that do not need models."""

from eval.metrics import context_precision, context_recall, hit_at_k, mrr
from rag.retrieve import reciprocal_rank_fusion
from rag.store import Hit, tokenize


def h(i: str) -> Hit:
    return Hit(id=i, score=0.0, source=f"{i}.md", heading_path="", text="")


def test_rrf_rewards_agreement_between_rankings():
    dense = [h("a"), h("b"), h("c")]
    sparse = [h("c"), h("b"), h("d")]
    fused = [x.id for x in reciprocal_rank_fusion([dense, sparse], k=60)]
    # "b" is 2nd in both lists, so it beats "a" which appears in only one.
    assert fused.index("b") < fused.index("a")
    assert set(fused) == {"a", "b", "c", "d"}


def test_tokenizer_keeps_cli_flags_and_identifiers_whole():
    toks = tokenize("Use --max-num-seqs and kv_cache_dtype=fp8_e4m3 in v1.completions")
    assert "max-num-seqs" in toks
    assert "fp8_e4m3" in toks
    assert "v1.completions" in toks


def test_metrics():
    retrieved = ["x.md", "a.md", "y.md", "b.md"]
    relevant = {"a.md", "b.md"}
    assert hit_at_k(retrieved, relevant) == 1.0
    assert mrr(retrieved, relevant) == 0.5
    assert context_recall(retrieved, relevant) == 1.0
    # precision@2 = 1/2, precision@4 = 2/4 -> mean 0.5
    assert context_precision(retrieved, relevant) == 0.5
    assert hit_at_k(["z.md"], relevant) == 0.0
