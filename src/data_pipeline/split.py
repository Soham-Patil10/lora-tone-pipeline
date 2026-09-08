"""Phase 1.3 - stratified train/val/test split with leakage guards.

  * Split cfg.data.split_ratios (default 80/10/10).
  * Stratify by meta.category so every split covers every category.
  * Group-aware: rows sharing a ``source`` document/conversation id must not be
    split across train and test (prevents leakage).
  * Test set is sacred: written once, hash recorded, never re-generated silently.

Run: python -m src.data_pipeline.split --config config.yaml
Input:  data/processed/clean.jsonl
Output: data/processed/{train,val,test}.jsonl  + split_manifest.json (hashes, counts)
"""
from __future__ import annotations

from pathlib import Path

from src.utils.io import load_config, read_jsonl, set_seed, write_jsonl


def group_key(row: dict) -> str:
    return str(row["meta"].get("source", id(row)))


def stratified_group_split(rows: list[dict], ratios: list[float], seed: int) -> dict[str, list[dict]]:
    """Return {"train":..., "val":..., "test":...}. TODO: sklearn GroupShuffleSplit per category."""
    raise NotImplementedError


def write_manifest(splits: dict[str, list[dict]], path: Path) -> None:
    """Record per-split count, category histogram, and sha256 of sorted outputs. TODO."""
    raise NotImplementedError


def main(config: str = "config.yaml") -> None:
    cfg = load_config(config)
    set_seed(cfg["project"]["seed"])
    rows = read_jsonl(Path(cfg["paths"]["processed_dir"]) / "clean.jsonl")
    assert len(rows) >= cfg["data"]["min_examples"], "not enough examples"
    splits = stratified_group_split(rows, cfg["data"]["split_ratios"], cfg["project"]["seed"])
    proc = Path(cfg["paths"]["processed_dir"])
    for name, part in splits.items():
        write_jsonl(part, proc / f"{name}.jsonl")
        print(f"{name}: {len(part)}")
    write_manifest(splits, proc / "split_manifest.json")


if __name__ == "__main__":
    import typer

    typer.run(main)
