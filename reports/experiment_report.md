# Experiment Report — Customer Support Tone Matching (LoRA)

> Fill each section from artifacts in this repo. Every number must be traceable
> to a file under `reports/predictions/`, `experiments/`, or a W&B run.

## 1. Problem & why fine-tuning

- The task: rewrite factually-correct support drafts into the brand voice.
- Why not few-shot prompting: <cost / latency / consistency drift over long
  tickets / prompt-length limits — back with numbers>.
- Why not RAG: <no knowledge gap; this is a style/behaviour gap>.
- What "better" means here: higher `tone_score` and judge win-rate with
  `content_preservation` held ≥ baseline.

## 2. Dataset

| | count |
|---|---|
| raw collected | |
| after cleaning / dedup | |
| train / val / test | / / |
| handcrafted benchmark | |

- Sources: <...>
- Curation method: <draft synthesis + gold curation + review process>
- Category distribution: <table>
- Leakage controls: grouped split by `meta.source`; manifest hashes in
  `data/processed/split_manifest.json`.

## 3. Hyperparameter sweep

Paste `experiments/sweep_results/comparison_table.csv`.

- Final config chosen: rank=__, lr=__, epochs=__
- Why: <highest val_tone_score with no content_preservation regression; note
  the overfitting seen at epochs=5>.

## 4. Base vs fine-tuned (held-out benchmark + test split)

| metric | base | fine-tuned | Δ |
|---|---|---|---|
| tone_score | | | |
| content_preservation | | | |
| format_valid | | | |
| overall | | | |
| LLM-judge win-rate (fine-tuned) | — | | |

- Per-category breakdown: <table>
- 2–3 wins (with text): see `reports/baseline_vs_finetuned.md`
- 1–2 regressions (with text): <...>

## 5. Catastrophic forgetting

From `reports/forgetting_analysis.md`:

| suite | base | fine-tuned | Δ% |
|---|---|---|---|
| arc_easy | | | |
| hellaswag | | | |
| ifeval_subset | | | |

Verdict: <within ±5% bound / regression + mitigation>.

## 6. Inference

- Adapter size: __ MB (base: __ GB)
- Serving: <vLLM / Ollama>, tokens/sec __, latency P50 __ ms / P95 __ ms,
  max concurrent __.

## 7. Headline

> "I fine-tuned <base model> on <N> customer-support examples using LoRA and
> improved brand-tone adherence from __% to __% (LLM-judge win-rate __%), while
> keeping content preservation at __% and general benchmark performance within
> __% of baseline."
