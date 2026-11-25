import yaml

from eval.run_eval import EVAL_DIR, check_gate

BASELINE = {"hit@k": 1.0, "mrr": 0.905, "context_precision": 0.862, "context_recall": 0.917}


def test_gate_passes_at_baseline():
    thresholds = yaml.safe_load((EVAL_DIR / "thresholds.yaml").read_text())
    assert check_gate(BASELINE, thresholds["min"]) == []


def test_gate_names_every_metric_that_regressed():
    thresholds = yaml.safe_load((EVAL_DIR / "thresholds.yaml").read_text())
    regressed = {**BASELINE, "mrr": 0.60, "context_precision": 0.55}
    failures = check_gate(regressed, thresholds["min"])
    assert len(failures) == 2
    assert any(f.startswith("mrr") for f in failures)
    assert any(f.startswith("context_precision") for f in failures)


def test_gate_floor_is_inclusive():
    assert check_gate({"mrr": 0.85}, {"mrr": 0.85}) == []
