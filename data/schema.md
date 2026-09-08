# Dataset schema

Domain: **customer support tone matching**
Task: given a customer message and a factually-correct but off-brand *draft
reply*, produce a reply that matches the company brand voice **without changing
any fact or the policy outcome**.

## Brand voice (the thing being taught)

> Warm, empathetic, professional. Acknowledge feelings first, then give a clear
> next step. Plain language, short sentences, active voice. No jargon, no blame,
> no over-apologising. Always close with a concrete offer of further help.

Keep this in sync with `src/utils/prompt_templates.py:BRAND_TONE_GUIDE`.

## Training / split rows — `data/processed/{train,val,test}.jsonl`

Instruction-following format, one JSON object per line:

```json
{
  "instruction": "Rewrite the draft support reply so it matches the company brand voice. Keep every fact and the policy outcome exactly the same.",
  "input": "Customer message:\n<verbatim customer text>\n\nDraft reply:\n<off-brand but correct draft>",
  "output": "<on-brand rewrite — the label>",
  "meta": {
    "category": "billing | technical | angry_complaint | cancellation | praise | edge_case",
    "source": "<conversation/document id — used for leakage-safe splitting>"
  }
}
```

Rules
- `output` preserves every concrete commitment in the draft (amounts, dates,
  yes/no decision). No new promises.
- PII scrubbed to tags: `<NAME> <EMAIL> <ORDER_ID> <PHONE>`.
- 500–2,000 rows total; 80/10/10 split, stratified by `meta.category`,
  grouped by `meta.source`.
- Test split is written once; hashes recorded in `split_manifest.json`.

## Handcrafted benchmark — `data/eval_benchmark/handcrafted_eval.jsonl`

30–50 hard, hand-authored examples (separate from the test split). Adds
`id`, `reference`, `failure_mode`, `rubric`:

```json
{
  "id": "angry_complaint_003",
  "instruction": "...",
  "input": "Customer message:\n...\n\nDraft reply:\n...",
  "reference": "<gold on-brand reply>",
  "category": "angry_complaint",
  "failure_mode": "base model mirrors the customer's anger / gets defensive",
  "rubric": {
    "tone": "empathetic, calm, non-defensive",
    "content_preservation": "keeps the $20 credit offer; no new promises",
    "format": "<= 120 words; ends with an offer of further help"
  }
}
```

## Raw rows — `data/raw/collected.jsonl` (git-ignored)

`{customer_message, draft_reply, gold_reply, category, source}` — produced by
`src/data_pipeline/collect.py`, consumed by `clean.py`.
