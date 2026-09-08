"""Phase 2 - configurable LoRA training entrypoint.

Reproducible from config.yaml alone:
    python -m src.training.train --config config.yaml
    python -m src.training.train --config config.yaml --set lora.rank=32 train.learning_rate=1e-4

Pipeline:
  1. load_config + set_seed
  2. load 4-bit base + tokenizer, attach LoRA adapter
  3. build datasets: render_training_text(), mask prompt tokens in the collator
  4. TRL SFTTrainer with eval on val split every cfg.train.eval_steps
  5. callbacks: tracking, GPU memory, early stopping, best-checkpoint logging
  6. save best adapter -> cfg.paths.adapter_out_dir (+ tokenizer + config snapshot)
"""
from __future__ import annotations

from src.training.callbacks import BestCheckpointLogger, GPUMemoryCallback, TrackingCallback
from src.training.lora_config import build_lora_config, load_base_model_and_tokenizer
from src.utils.io import load_config, read_jsonl, set_seed
from src.utils.prompt_templates import render_training_text


def build_datasets(cfg: dict, tokenizer):
    """Return (train_ds, val_ds) of tokenised, prompt-masked examples. TODO."""
    _ = (read_jsonl(cfg["data"]["train_file"]), read_jsonl(cfg["data"]["val_file"]))
    raise NotImplementedError


def compute_val_metrics(eval_preds, tokenizer):
    """val_tone_score alongside the built-in val_loss. TODO: decode preds,
    call src.evaluation.metrics.tone_score against the val references."""
    raise NotImplementedError


def main(config: str = "config.yaml", set_: list[str] = None) -> None:
    cfg = load_config(config, overrides=set_)
    set_seed(cfg["project"]["seed"])

    model, tokenizer = load_base_model_and_tokenizer(cfg)
    model = _attach_adapter(model, build_lora_config(cfg))
    train_ds, val_ds = build_datasets(cfg, tokenizer)

    trainer = _make_sft_trainer(
        cfg, model, tokenizer, train_ds, val_ds,
        callbacks=[TrackingCallback(cfg), GPUMemoryCallback(), BestCheckpointLogger()],
    )
    trainer.train()
    trainer.model.save_pretrained(cfg["paths"]["adapter_out_dir"])
    tokenizer.save_pretrained(cfg["paths"]["adapter_out_dir"])
    _snapshot_config(cfg, cfg["paths"]["adapter_out_dir"])


def _attach_adapter(model, lora_cfg):
    raise NotImplementedError


def _make_sft_trainer(cfg, model, tokenizer, train_ds, val_ds, callbacks):
    raise NotImplementedError


def _snapshot_config(cfg: dict, out_dir: str) -> None:
    raise NotImplementedError


if __name__ == "__main__":
    import typer

    typer.run(main)
