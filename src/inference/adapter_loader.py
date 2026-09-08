"""Phase 4.1 - package + load the LoRA adapter.

The adapter is small (<100MB) vs the base (16GB+). Keep them separate so one
base can serve many domain adapters.

  export_adapter()  - copy trained adapter + tokenizer + config snapshot +
                      metrics card into a versioned, checksummed folder.
  merged_export()   - optional: merge_and_unload() to a standalone model for
                      engines without hot-swap adapter support.
"""
from __future__ import annotations

from pathlib import Path


def export_adapter(adapter_dir: str, out_dir: str, metrics_summary: dict) -> Path:
    """Bundle adapter for deployment; write an ADAPTER_CARD.md with eval numbers. TODO."""
    raise NotImplementedError


def load_peft_model(base_model: str, adapter_dir: str, load_in_4bit: bool = True):
    """Return (model, tokenizer) with the adapter attached (PeftModel). TODO."""
    raise NotImplementedError


def merged_export(base_model: str, adapter_dir: str, out_dir: str) -> Path:
    """merge_and_unload() -> full-weights model for vLLM/Ollama/GGUF. TODO."""
    raise NotImplementedError
