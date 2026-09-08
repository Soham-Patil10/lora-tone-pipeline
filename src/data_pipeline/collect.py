"""Phase 1.1 - gather raw (customer_message, draft_reply, gold_reply) triples.

Sources for a customer-support tone-matching dataset:
  * Public support-conversation corpora (e.g. Bitext customer-support,
    MultiWOZ, Ubuntu dialogue) -> take the customer turn as ``customer_message``.
  * Generate a deliberately flat/blunt ``draft_reply`` (rule-based or a cheap
    model) that is factually correct but off-brand.
  * Write / curate the on-brand ``gold_reply`` (this is the label - the scarce,
    high-value part; do it by hand or with a strong model + human review).

Output: data/raw/collected.jsonl with rows:
  {"customer_message", "draft_reply", "gold_reply", "category", "source"}

Run:  python -m src.data_pipeline.collect --config config.yaml
"""
from __future__ import annotations

from pathlib import Path

from src.utils.io import load_config, write_jsonl

CATEGORIES = ["billing", "technical", "angry_complaint", "cancellation", "praise", "edge_case"]


def load_source_conversations(cfg: dict) -> list[dict]:
    """Return list of {customer_message, category, source}. TODO: implement per source."""
    raise NotImplementedError


def synthesize_draft_reply(customer_message: str, category: str) -> str:
    """Produce a correct-but-off-brand draft (terse, no empathy). TODO."""
    raise NotImplementedError


def curate_gold_reply(customer_message: str, draft_reply: str, category: str) -> str:
    """The on-brand target. TODO: strong-model draft + manual review queue."""
    raise NotImplementedError


def main(config: str = "config.yaml") -> None:
    cfg = load_config(config)
    rows: list[dict] = []
    for conv in load_source_conversations(cfg):
        draft = synthesize_draft_reply(conv["customer_message"], conv["category"])
        gold = curate_gold_reply(conv["customer_message"], draft, conv["category"])
        rows.append({**conv, "draft_reply": draft, "gold_reply": gold})
    out = Path(cfg["paths"]["raw_dir"]) / "collected.jsonl"
    write_jsonl(rows, out)
    print(f"wrote {len(rows)} raw rows -> {out}")


if __name__ == "__main__":
    import typer

    typer.run(main)
