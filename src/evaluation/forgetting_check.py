"""Phase 3.4 - catastrophic forgetting check.

Fine-tuning on a narrow style task can erode general ability. Run a fixed subset
of standard benchmarks on BOTH the base model and the fine-tuned (base+adapter)
model and report the delta.

Suites (cfg.eval.forgetting.suites): arc_easy, hellaswag, and an IFEval subset
(instruction following). Driven by lm-evaluation-harness.

Decision rule: if any suite regresses more than cfg.eval.forgetting.max_regression_pct,
flag it in reports/forgetting_analysis.md and recommend fewer epochs / lower rank.

Run: python -m src.evaluation.forgetting_check --config config.yaml
"""
from __future__ import annotations

from src.utils.io import load_config


def run_harness(model_id: str, adapter_dir: str | None, suites: list[str]) -> dict:
    """Return {suite: primary_metric}. TODO: shell out to lm_eval or use its API."""
    raise NotImplementedError


def render_report(base: dict, ft: dict, max_regression_pct: float, out_path: str) -> bool:
    """Write forgetting_analysis.md; return True if a regression exceeds the bound. TODO."""
    raise NotImplementedError


def main(config: str = "config.yaml") -> None:
    cfg = load_config(config)
    suites = cfg["eval"]["forgetting"]["suites"]
    base = run_harness(cfg["model"]["base_model"], None, suites)
    ft = run_harness(cfg["model"]["base_model"], cfg["paths"]["adapter_out_dir"], suites)
    regressed = render_report(
        base, ft, cfg["eval"]["forgetting"]["max_regression_pct"], "reports/forgetting_analysis.md"
    )
    print("REGRESSION DETECTED" if regressed else "no significant forgetting")


if __name__ == "__main__":
    import typer

    typer.run(main)
