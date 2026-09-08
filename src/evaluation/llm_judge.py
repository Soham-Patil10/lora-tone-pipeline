"""Phase 3.3 - LLM-as-judge blind pairwise comparison (base vs fine-tuned).

For each benchmark example, show GPT-4o the customer message + draft + both
candidate replies in RANDOM order (blind). Ask for:
  * a winner (A / B / tie)
  * a 1-5 quality score for each on brand-tone adherence
  * one sentence of reasoning

Aggregate: fine-tuned win-rate, mean score delta, and the reasoning corpus.
Cache every call keyed by a hash of the prompt (results/judge_cache.jsonl) so
reruns are cheap and deterministic.

Run: python -m src.evaluation.llm_judge --config config.yaml
Output: reports/predictions/judge.jsonl, reports/predictions/judge_summary.json
"""
from __future__ import annotations

import random

from src.utils.io import load_config, read_jsonl, write_jsonl

JUDGE_SYSTEM = (
    "You are evaluating customer-support replies for adherence to a brand voice: "
    "warm, empathetic, professional, concise, solution-oriented, non-defensive. "
    "Judge ONLY tone and helpfulness, not length. Be strict and consistent."
)

JUDGE_USER_TEMPLATE = """Customer message:
{customer_message}

Draft reply (internal, off-brand):
{draft_reply}

Reply A:
{reply_a}

Reply B:
{reply_b}

Return JSON: {{"winner": "A"|"B"|"tie", "score_a": 1-5, "score_b": 1-5, "reason": "..."}}
"""


def judge_pair(customer_message: str, draft_reply: str, reply_a: str, reply_b: str, cfg: dict) -> dict:
    """One cached GPT-4o call. TODO: openai client + disk cache by prompt hash."""
    raise NotImplementedError


def main(config: str = "config.yaml") -> None:
    cfg = load_config(config)
    rng = random.Random(cfg["project"]["seed"])
    base = {r["id"]: r for r in read_jsonl("reports/predictions/baseline.jsonl")}
    ft = {r["id"]: r for r in read_jsonl("reports/predictions/finetuned.jsonl")}
    bench = read_jsonl(cfg["paths"]["eval_benchmark"])

    out = []
    for ex in bench:
        swap = rng.random() < 0.5  # blind: A/B order randomised
        a, b = (base[ex["id"]], ft[ex["id"]]) if not swap else (ft[ex["id"]], base[ex["id"]])
        verdict = judge_pair(ex["input"], ex.get("draft_reply", ""), a["prediction"], b["prediction"], cfg)
        verdict["finetuned_is"] = "B" if not swap else "A"
        out.append({"id": ex["id"], **verdict})
    write_jsonl(out, "reports/predictions/judge.jsonl")
    # TODO: summarise win-rate + mean score delta -> judge_summary.json


if __name__ == "__main__":
    import typer

    typer.run(main)
