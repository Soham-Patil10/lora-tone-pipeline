"""Phase 1.4 - handcrafted evaluation benchmark (30-50 hard examples).

This is separate from the test split. It is the benchmark that proves the
fine-tune is *better*, not just *different*. Each example is deliberately chosen
to stress a failure mode.

Schema (data/eval_benchmark/handcrafted_eval.jsonl):
  {
    "id": "angry_complaint_003",
    "instruction": "...",
    "input": "Customer message:\n...\n\nDraft reply:\n...",
    "reference": "gold on-brand reply",
    "category": "angry_complaint",
    "failure_mode": "base model mirrors the customer's anger / gets defensive",
    "rubric": {
      "tone": "empathetic, calm, non-defensive",
      "content_preservation": "keeps the $20 credit offer, no new promises",
      "format": "<= 120 words, ends with an offer of further help"
    }
  }

This file is hand-authored. This module only validates it and prints coverage.
Run: python -m src.data_pipeline.build_eval_set --config config.yaml
"""
from __future__ import annotations

from collections import Counter

from src.utils.io import load_config, read_jsonl

REQUIRED_KEYS = {"id", "instruction", "input", "reference", "category", "failure_mode", "rubric"}


def validate(rows: list[dict], categories: list[str]) -> None:
    ids = set()
    for r in rows:
        missing = REQUIRED_KEYS - r.keys()
        assert not missing, f"{r.get('id', '?')}: missing {missing}"
        assert r["id"] not in ids, f"duplicate id {r['id']}"
        assert r["category"] in categories, f"{r['id']}: bad category {r['category']}"
        ids.add(r["id"])


def main(config: str = "config.yaml") -> None:
    cfg = load_config(config)
    rows = read_jsonl(cfg["paths"]["eval_benchmark"])
    validate(rows, cfg["eval"]["categories"])
    assert 30 <= len(rows) <= 60, f"expected 30-60 examples, got {len(rows)}"
    print(f"{len(rows)} benchmark examples")
    for cat, n in Counter(r["category"] for r in rows).most_common():
        print(f"  {cat:16s} {n}")


if __name__ == "__main__":
    import typer

    typer.run(main)
