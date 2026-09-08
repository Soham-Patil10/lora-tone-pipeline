"""Tests for the data pipeline: schema, splits, no leakage."""
import pytest

from src.data_pipeline.clean import to_schema_row
from src.data_pipeline.split import group_key


def test_schema_row_has_required_keys():
    raw = {
        "customer_message": "Where is my order?",
        "draft_reply": "It shipped. Track it with the link.",
        "gold_reply": "Thanks for reaching out...",
        "category": "technical",
        "source": "doc_1",
    }
    row = to_schema_row(raw)
    assert set(row) == {"instruction", "input", "output", "meta"}
    assert row["meta"]["category"] == "technical"


@pytest.mark.xfail(reason="stub: implement stratified_group_split")
def test_no_group_leakage_between_train_and_test():
    from src.data_pipeline.split import stratified_group_split

    rows = [
        {"meta": {"category": "billing", "source": f"doc_{i%5}"}, "output": str(i)}
        for i in range(200)
    ]
    splits = stratified_group_split(rows, [0.8, 0.1, 0.1], seed=42)
    train_groups = {group_key(r) for r in splits["train"]}
    test_groups = {group_key(r) for r in splits["test"]}
    assert train_groups.isdisjoint(test_groups)


@pytest.mark.xfail(reason="stub: implement stratified_group_split")
def test_every_split_covers_every_category():
    from src.data_pipeline.split import stratified_group_split

    cats = ["billing", "technical", "angry_complaint"]
    rows = [
        {"meta": {"category": cats[i % 3], "source": f"doc_{i}"}, "output": str(i)}
        for i in range(300)
    ]
    splits = stratified_group_split(rows, [0.8, 0.1, 0.1], seed=42)
    for part in splits.values():
        assert {r["meta"]["category"] for r in part} == set(cats)
