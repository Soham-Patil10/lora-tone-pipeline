"""Builds the quantization + LoRA config objects from config.yaml.

Rationale (document this in reports/experiment_report.md):
  * target q_proj, v_proj first  -> cheapest adapter that still moves style;
    the sweep widens to k/o/mlp if tone_score plateaus.
  * rank 16 / alpha 32           -> alpha = 2*rank is a stable starting ratio.
  * dropout 0.05                 -> light regularisation on a small dataset.
  * 4-bit NF4 (QLoRA)            -> fits a 7-8B base on a single 16-24GB GPU.
"""
from __future__ import annotations

from typing import Any


def build_bnb_config(cfg: dict) -> Any:
    """transformers.BitsAndBytesConfig for 4-bit NF4 load. TODO."""
    raise NotImplementedError


def build_lora_config(cfg: dict) -> Any:
    """peft.LoraConfig from cfg['lora']. TODO."""
    raise NotImplementedError


def load_base_model_and_tokenizer(cfg: dict):
    """Return (model, tokenizer): 4-bit base, prepared for k-bit training,
    pad token set, chat template resolved. TODO."""
    raise NotImplementedError
