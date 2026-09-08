"""Phase 4.3 - A/B comparison endpoint. This powers the demo.

POST /compare
  body:  {"customer_message": "...", "draft_reply": "..."}
  return:
    {
      "base":      {"reply": "...", "scores": {tone_score, content_preservation, ...}},
      "finetuned": {"reply": "...", "scores": {...}},
      "deltas":    {"tone_score": +0.31, "overall": +0.24},
      "latency_ms": {"base": 820, "finetuned": 840}
    }

Runs both models through the identical prompt template and the automated eval
metrics so the improvement is visible inline.

Run: uvicorn src.inference.ab_compare_api:app --port 8001
"""
from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

from src.evaluation.metrics import score_example
from src.utils.io import load_config
from src.utils.prompt_templates import build_messages

app = FastAPI(title="cs-tone-matching A/B compare")
_STATE: dict = {}


class CompareRequest(BaseModel):
    customer_message: str
    draft_reply: str


@app.on_event("startup")
def _startup() -> None:
    cfg = load_config()
    _STATE["cfg"] = cfg
    _STATE["base"] = _load(cfg, adapter=None)          # TODO
    _STATE["finetuned"] = _load(cfg, adapter=cfg["paths"]["adapter_out_dir"])  # TODO


def _load(cfg: dict, adapter: str | None):
    raise NotImplementedError


def _generate(model_key: str, req: CompareRequest) -> tuple[str, float]:
    """Return (reply, latency_ms). TODO."""
    raise NotImplementedError


@app.post("/compare")
def compare(req: CompareRequest) -> dict:
    ex = {
        "instruction": "Rewrite the draft support reply in the brand voice.",
        "input": f"Customer message:\n{req.customer_message}\n\nDraft reply:\n{req.draft_reply}",
        "output": "",
        "meta": {"category": "live"},
    }
    _ = build_messages(ex)
    _ = score_example  # scored per side once replies exist
    raise NotImplementedError


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
