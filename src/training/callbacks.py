"""TrainerCallbacks: experiment tracking, GPU memory, checkpoint selection.

Required logs (per the project brief):
  * all hyperparameters (logged once from config.yaml at run start)
  * training loss curve
  * validation metrics at every eval step (val_loss + val_tone_score)
  * GPU memory usage
  * training duration
  * which checkpoint was selected as best and why
"""
from __future__ import annotations

from transformers import TrainerCallback


class TrackingCallback(TrainerCallback):
    """Push config + metrics to W&B (or MLflow) via a thin wrapper. TODO."""

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg

    def on_train_begin(self, args, state, control, **kwargs):
        raise NotImplementedError  # log flattened config.yaml, git sha, env

    def on_log(self, args, state, control, logs=None, **kwargs):
        raise NotImplementedError  # forward loss/lr

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        raise NotImplementedError  # forward val_loss, val_tone_score


class GPUMemoryCallback(TrainerCallback):
    """Log torch.cuda.max_memory_allocated each eval step. TODO."""

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        raise NotImplementedError


class BestCheckpointLogger(TrainerCallback):
    """On train end, record the selected checkpoint + the metric that chose it. TODO."""

    def on_train_end(self, args, state, control, **kwargs):
        raise NotImplementedError
