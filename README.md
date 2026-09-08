# Fine-Tuning Pipeline with LoRA — Customer Support Tone Matching

An end-to-end, reproducible pipeline that LoRA-fine-tunes an open 7–8B model to
rewrite factually-correct customer-support drafts into a consistent brand voice,
then **proves** the improvement with a held-out benchmark, an LLM-as-judge, and a
catastrophic-forgetting check — with full experiment tracking.

> Headline (fill from `reports/experiment_report.md`):
> *"I fine-tuned \<base model\> on \<N\> support examples using LoRA and improved
> brand-tone adherence from X% to Y% while keeping general capability within Z%
> of baseline."*

## Why fine-tuning (not few-shot / RAG)

Tone matching is a **behaviour** problem, not a knowledge problem. Few-shot
prompting drifts over long tickets and adds latency/cost per call; RAG adds
nothing when there's no missing fact. A small LoRA adapter bakes the voice in.
See `reports/experiment_report.md §1`.

## Pipeline

```
raw tickets ──▶ data_pipeline ──▶ train/val/test + handcrafted benchmark
                                   │
                                   ▼
                        training (LoRA + QLoRA) ──▶ W&B  ──▶ best adapter
                                   │
                                   ▼
        evaluation: baseline vs fine-tuned · LLM-judge · forgetting check
                                   │
                                   ▼
              inference: vLLM serve + /compare A/B endpoint ──▶ demo
```

## Repo layout

| Path | What |
|---|---|
| `config.yaml` | single source of truth — every run reproduces from it |
| `data/schema.md` | dataset format + brand-voice definition |
| `data/processed/` | `train/val/test.jsonl` (instruction-following) |
| `data/eval_benchmark/` | 30–50 handcrafted hard cases |
| `src/data_pipeline/` | `collect → clean → split → build_eval_set` |
| `src/training/` | `lora_config`, `train`, `sweep`, `callbacks` |
| `src/evaluation/` | `metrics`, `run_baseline`, `run_finetuned`, `llm_judge`, `forgetting_check` |
| `src/inference/` | `adapter_loader`, `serve`, `ab_compare_api` |
| `src/utils/` | `io` (config/seed/jsonl), `prompt_templates` (shared prompt) |
| `experiments/` | W&B logs + sweep `comparison_table.csv` |
| `reports/` | experiment report + auto-generated comparison / forgetting docs |
| `demo/` | < 4 min demo script |
| `docker/` | train + serve images, compose stack |

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env    # HF_TOKEN, WANDB_API_KEY, OPENAI_API_KEY

# 1. data
python -m src.data_pipeline.collect       --config config.yaml
python -m src.data_pipeline.clean         --config config.yaml
python -m src.data_pipeline.split         --config config.yaml
python -m src.data_pipeline.build_eval_set --config config.yaml

# 2. train (+ sweep)
python -m src.training.sweep --config config.yaml       # 3+ configs -> comparison table
python -m src.training.train --config config.yaml       # final run with winning config

# 3. evaluate
python -m src.evaluation.run_baseline    --config config.yaml
python -m src.evaluation.run_finetuned   --config config.yaml
python -m src.evaluation.llm_judge       --config config.yaml
python -m src.evaluation.forgetting_check --config config.yaml

# 4. serve + demo
uvicorn src.inference.serve:app --port 8000
uvicorn src.inference.ab_compare_api:app --port 8001
```

## Status

Skeleton. Function bodies marked `raise NotImplementedError` / `TODO`. Tests in
`tests/` are `xfail` stubs that lock in the intended contracts.
