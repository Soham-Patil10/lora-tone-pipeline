"""Config loading, JSONL/JSON helpers, seeding, run directories.

Everything downstream imports from here so behaviour stays consistent.
"""
from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Iterable, Iterator

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
def load_config(path: str | Path = "config.yaml", overrides: list[str] | None = None) -> dict[str, Any]:
    """Load YAML config and apply ``key.subkey=value`` CLI overrides.

    TODO: parse ``overrides`` (e.g. ["lora.rank=32"]) and set nested keys,
    coercing values to int/float/bool/str.
    """
    cfg = yaml.safe_load(Path(path).read_text())
    for item in overrides or []:
        dotted, _, raw = item.partition("=")
        _set_nested(cfg, dotted, _coerce(raw))
    return cfg


def _set_nested(d: dict, dotted: str, value: Any) -> None:
    raise NotImplementedError


def _coerce(raw: str) -> Any:
    raise NotImplementedError


# --------------------------------------------------------------------------- #
# JSONL
# --------------------------------------------------------------------------- #
def read_jsonl(path: str | Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def iter_jsonl(path: str | Path) -> Iterator[dict]:
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def write_jsonl(rows: Iterable[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
def set_seed(seed: int) -> None:
    """Seed python, numpy, torch (+cuda). Call at the top of every entrypoint."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # TODO: numpy + torch manual_seed / cuda.manual_seed_all, set deterministic flags
    raise NotImplementedError


def new_run_dir(base: str | Path, name: str) -> Path:
    """Create ``base/<name>-<timestamp>`` and return it."""
    raise NotImplementedError
