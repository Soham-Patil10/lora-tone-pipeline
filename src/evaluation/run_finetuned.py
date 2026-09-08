"""Phase 3.2 - evaluate the best fine-tuned adapter on the SAME benchmark.

Reuses evaluate_model() from run_baseline so scoring is byte-identical. Then
diffs against the stored baseline: overall delta, per-category delta, list of
wins (fine-tune helped) and regressions (fine-tune hurt), each with the example
text so they can be quoted in reports/baseline_vs_finetuned.md.

Run: python -m src.evaluation.run_finetuned --config config.yaml
Output: reports/predictions/finetuned.jsonl
        reports/predictions/finetuned_summary.json
        reports/baseline_vs_finetuned.md  (regenerated)
"""
from __future__ import annotations

from src.evaluation.run_baseline import evaluate_model
from src.utils.io import load_config, read_jsonl


def diff_against_baseline(baseline_path: str, finetuned_path: str) -> dict:
    """Return {overall_delta, per_category_delta, wins:[...], regressions:[...]}. TODO."""
    base = read_jsonl(baseline_path)
    ft = read_jsonl(finetuned_path)
    _ = (base, ft)
    raise NotImplementedError


def render_comparison_md(diff: dict, out_path: str) -> None:
    raise NotImplementedError


def main(config: str = "config.yaml") -> None:
    cfg = load_config(config)
    summary = evaluate_model(cfg, adapter_dir=cfg["paths"]["adapter_out_dir"], tag="finetuned")
    diff = diff_against_baseline(
        "reports/predictions/baseline.jsonl", "reports/predictions/finetuned.jsonl"
    )
    render_comparison_md(diff, "reports/baseline_vs_finetuned.md")
    print(summary)


if __name__ == "__main__":
    import typer

    typer.run(main)
