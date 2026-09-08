"""Phase 4.2 - inference server (base model + LoRA adapter) exposing a chat API.

Engine selected by cfg.serve.engine:
  * vllm   - production batched inference; load base with enable_lora=True and
             register the adapter as a LoRARequest.
  * ollama - simpler local path; serve a merged GGUF export.

Endpoints:
  POST /v1/chat/completions   OpenAI-compatible (streaming + non-streaming)
  GET  /healthz
  GET  /metrics               tokens/sec, latency P50/P95, concurrent requests

Run: uvicorn src.inference.serve:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

from fastapi import FastAPI

from src.utils.io import load_config

app = FastAPI(title="cs-tone-matching inference")
_STATE: dict = {}


@app.on_event("startup")
def _startup() -> None:
    cfg = load_config()
    _STATE["cfg"] = cfg
    _STATE["engine"] = _build_engine(cfg)   # TODO: vLLM LLM(...) or Ollama client


def _build_engine(cfg: dict):
    raise NotImplementedError


@app.get("/healthz")
def healthz() -> dict:
    return {"ok": "engine" in _STATE}


@app.post("/v1/chat/completions")
async def chat_completions(body: dict):
    """Apply prompt template, generate with the adapter, record latency/throughput. TODO."""
    raise NotImplementedError


@app.get("/metrics")
def metrics() -> dict:
    """tokens_per_sec, latency_p50_ms, latency_p95_ms, max_concurrent. TODO."""
    raise NotImplementedError
