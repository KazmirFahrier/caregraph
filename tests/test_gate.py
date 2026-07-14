from caregraph.eval.gate import Thresholds, evaluate_gate
from caregraph.eval.ragas_adapter import Sample, evaluate


def test_gate_passes_on_good_metrics() -> None:
    report = evaluate_gate(
        {"faithfulness": 1.0, "refusal_accuracy": 1.0, "redteam_pass_rate": 1.0}
    )
    assert report.passed
    assert not report.failures


def test_gate_blocks_on_faithfulness_regression() -> None:
    report = evaluate_gate(
        {"faithfulness": 0.80, "refusal_accuracy": 1.0, "redteam_pass_rate": 1.0},
        Thresholds(min_faithfulness=0.90),
    )
    assert not report.passed
    assert any("faithfulness" in f for f in report.failures)


def test_gate_blocks_on_redteam_failure() -> None:
    report = evaluate_gate(
        {"faithfulness": 1.0, "refusal_accuracy": 1.0, "redteam_pass_rate": 0.75}
    )
    assert not report.passed


def test_ragas_adapter_offline_fallback() -> None:
    samples = [
        Sample(
            question="What BP target for CKD?",
            answer="The blood pressure target is less than 130 over 80 mmHg.",
            contexts=["The blood pressure target is less than 130 over 80 mmHg."],
            ground_truth="less than 130/80 mmHg",
        )
    ]
    scores = evaluate(samples)
    assert scores["faithfulness"] >= 0.5
    assert set(scores) == {"faithfulness", "answer_relevancy", "context_precision"}
