"""Phase 3.1 - establish the baseline BEFORE any fine-tuning claims.

Runs the untouched base model (same prompt template, same decoding params) over
the full handcrafted benchmark + test split. Stores per-example outputs and
scores. These numbers are the denominator for every improvement claim.

Run: python -m src.evaluation.run_baseline --config config.yaml
Output: reports/predictions/baseline.jsonl   (id, category, prediction, ExampleScore)
        reports/predictions/baseline_summary.json
"""
from __future__ import annotations

from src.evaluation.metrics import aggregate, score_example
from src.utils.io import load_config, read_jsonl, write_jsonl
from src.utils.prompt_templates import build_messages


def load_model(cfg: dict, adapter_dir: str | None):
    """adapter_dir=None -> plain base model. Shared by baseline + finetuned. TODO."""
    raise NotImplementedError


def generate(model, tokenizer, messages: list[dict], cfg: dict) -> str:
    """Deterministic decode (temperature 0, fixed max_new_tokens). TODO."""
    raise NotImplementedError


def evaluate_model(cfg: dict, adapter_dir: str | None, tag: str) -> dict:
    model, tokenizer = load_model(cfg, adapter_dir)
    rows = read_jsonl(cfg["paths"]["eval_benchmark"]) + read_jsonl(cfg["data"]["test_file"])
    preds, scores = [], []
    for ex in rows:
        pred = generate(model, tokenizer, build_messages(ex, cfg["eval"]["prompt_template"]), cfg)
        s = score_example(ex, pred)
        preds.append({"id": ex.get("id"), "category": s.category, "prediction": pred, **vars(s)})
        scores.append(s)
    write_jsonl(preds, f"reports/predictions/{tag}.jsonl")
    return aggregate(scores)


def main(config: str = "config.yaml") -> None:
    cfg = load_config(config)
    summary = evaluate_model(cfg, adapter_dir=None, tag="baseline")
    print(summary)


if __name__ == "__main__":
    import typer

    typer.run(main)
