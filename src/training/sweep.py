"""Phase 2.4 - hyperparameter sweep over cfg.sweep.grid.

Varies rank (8/16/32), learning_rate (1e-4/2e-4/5e-4), epochs (1/3/5).
Each cell = one call into src.training.train.main with overrides, logged as a
separate tracker run. Produces experiments/sweep_results/comparison_table.csv:

    run_id, lora.rank, train.learning_rate, train.num_train_epochs,
    val_loss, val_tone_score, val_content_preservation, gpu_mem_gb, minutes

The winning config (max cfg.sweep.select_metric on val) is copied into
config.yaml for the final run.

Run: python -m src.training.sweep --config config.yaml
"""
from __future__ import annotations

import itertools
from pathlib import Path

from src.utils.io import load_config


def expand_grid(grid: dict) -> list[dict]:
    keys = list(grid)
    return [dict(zip(keys, combo)) for combo in itertools.product(*grid.values())]


def run_cell(config_path: str, overrides: dict) -> dict:
    """Invoke training with overrides, return the row of metrics. TODO."""
    raise NotImplementedError


def main(config: str = "config.yaml") -> None:
    cfg = load_config(config)
    cells = expand_grid(cfg["sweep"]["grid"])[: cfg["sweep"]["max_runs"]]
    rows = [run_cell(config, {**{k: v for k, v in c.items()}}) for c in cells]
    out = Path("experiments/sweep_results/comparison_table.csv")
    _write_csv(rows, out)
    best = max(rows, key=lambda r: r[cfg["sweep"]["select_metric"]])
    print(f"best: {best}")


def _write_csv(rows: list[dict], path: Path) -> None:
    raise NotImplementedError


if __name__ == "__main__":
    import typer

    typer.run(main)
