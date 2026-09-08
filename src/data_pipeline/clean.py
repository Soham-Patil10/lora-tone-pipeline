"""Phase 1.2 - clean raw rows and convert to instruction-following schema.

Steps:
  1. Normalise whitespace / unicode / PII scrub (names, emails, order IDs -> tags).
  2. Drop rows with empty fields or gold == draft.
  3. Near-duplicate removal via sentence-embedding cosine similarity
     (>= cfg.data.dedup_threshold).
  4. Length / format filters (cfg.data.max_seq_len budget).
  5. Emit schema rows (see data/schema.md):
       {"instruction", "input", "output", "meta": {...}}

Run: python -m src.data_pipeline.clean --config config.yaml
Input:  data/raw/collected.jsonl
Output: data/processed/clean.jsonl
"""
from __future__ import annotations

from pathlib import Path

from src.utils.io import load_config, read_jsonl, write_jsonl

INSTRUCTION = (
    "Rewrite the draft support reply so it matches the company brand voice. "
    "Keep every fact and the policy outcome exactly the same."
)


def normalize_text(text: str) -> str:
    raise NotImplementedError


def scrub_pii(text: str) -> str:
    """Replace names/emails/order numbers with <NAME> <EMAIL> <ORDER_ID>. TODO."""
    raise NotImplementedError


def dedupe(rows: list[dict], threshold: float) -> list[dict]:
    """Embed output text, greedily drop near-duplicates. TODO (sentence-transformers)."""
    raise NotImplementedError


def to_schema_row(raw: dict) -> dict:
    return {
        "instruction": INSTRUCTION,
        "input": (
            f"Customer message:\n{normalize_text(raw['customer_message'])}\n\n"
            f"Draft reply:\n{normalize_text(raw['draft_reply'])}"
        ),
        "output": normalize_text(raw["gold_reply"]),
        "meta": {"category": raw["category"], "source": raw.get("source", "unknown")},
    }


def main(config: str = "config.yaml") -> None:
    cfg = load_config(config)
    raw = read_jsonl(Path(cfg["paths"]["raw_dir"]) / "collected.jsonl")
    rows = [to_schema_row(r) for r in raw]
    rows = [r for r in rows if r["output"] and r["output"] != r["input"]]
    rows = dedupe(rows, cfg["data"]["dedup_threshold"])
    out = Path(cfg["paths"]["processed_dir"]) / "clean.jsonl"
    write_jsonl(rows, out)
    print(f"kept {len(rows)} clean rows -> {out}")


if __name__ == "__main__":
    import typer

    typer.run(main)
