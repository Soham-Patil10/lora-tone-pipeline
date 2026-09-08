# Demo Script (< 4 minutes)

Goal: make the improvement obvious in under four minutes. Record at 1080p, no
dead air, captions for every number.

## 0:00–0:25 — The problem
- One line: "Support agents write correct replies that sound wrong. I taught a
  7B model our brand voice with LoRA."
- Show a real ticket + a blunt draft reply.

## 0:25–1:10 — Base model fails
- Hit `POST /compare` with that ticket.
- Show the **base** reply: correct facts, off-brand (curt / defensive / robotic).
- Point at the automated `tone_score` (low).

## 1:10–2:00 — Fine-tuned model handles it
- Same response payload, show the **fine-tuned** reply side by side.
- Callouts: acknowledges feelings first, keeps the exact $ amount / policy,
  closes with an offer of help.
- Show `deltas`: tone_score +__, content_preservation flat, latency ~equal.

## 2:00–2:50 — It's rigorous, not a vibe
- W&B dashboard: training loss + val curves, the 3-config sweep table,
  which checkpoint was chosen and why.
- `reports/baseline_vs_finetuned.md`: overall + per-category deltas,
  LLM-judge blind win-rate.
- `reports/forgetting_analysis.md`: general benchmarks within 5%.

## 2:50–3:30 — It's deployable
- `adapters/` folder: adapter is __ MB vs __ GB base — swap without reloading.
- vLLM server up: tokens/sec, P50/P95 latency.

## 3:30–3:55 — Headline
- On screen: the filled-in headline sentence from `reports/experiment_report.md`,
  every number linked to a file in the repo.

## Assets to capture beforehand
- [ ] 3 strong before/after ticket pairs (one angry, one billing, one edge case)
- [ ] W&B run URL (public)
- [ ] terminal with server running + `curl` snippets ready
- [ ] repo open at `reports/`
