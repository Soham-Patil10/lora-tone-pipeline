"""Tests for scoring: metric ranges, aggregation, prompt-template parity."""
import pytest

from src.evaluation.metrics import ExampleScore, format_valid
from src.utils.prompt_templates import build_messages


def test_example_score_overall_weighting():
    s = ExampleScore("x", "billing", tone_score=1.0, content_preservation=1.0,
                     format_valid=1.0, rouge_l=0.5)
    assert s.overall == pytest.approx(1.0)
    s0 = ExampleScore("y", "billing", 0.0, 0.0, 0.0, 0.0)
    assert s0.overall == pytest.approx(0.0)


@pytest.mark.xfail(reason="stub: implement format_valid")
def test_format_valid_rejects_leaked_pii_tags():
    assert format_valid("Hi <NAME>, your refund is processed.") == 0.0


def test_build_messages_is_deterministic_and_shared():
    ex = {"instruction": "Rewrite it.", "input": "Customer message:\nhi\n\nDraft reply:\nno"}
    a = build_messages(ex)
    b = build_messages(ex)
    assert a == b
    assert a[0]["role"] == "system" and a[1]["role"] == "user"


@pytest.mark.xfail(reason="stub: implement aggregate")
def test_aggregate_reports_per_category():
    from src.evaluation.metrics import aggregate

    scores = [
        ExampleScore("1", "billing", 0.8, 0.9, 1.0, 0.4),
        ExampleScore("2", "technical", 0.6, 0.7, 1.0, 0.3),
    ]
    out = aggregate(scores)
    assert "overall" in out and "billing" in str(out)
