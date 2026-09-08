# Data task — EDA + prep for "customer support tone matching"

**Owner:** <groupmate>
**Goal of the whole project:** fine-tune a small open model to rewrite a
*correct-but-blunt* support reply into a *warm, human* reply **without changing
any fact**. Your job is the data: explore a ready-made dataset, tell us if it
works, and (phase 2) turn it into training files.

You do **not** need to write support replies from scratch. Read this whole file
once before starting. Questions → ping <owner>.

---

## Part 0 — Setup (15 min)

1. Make a free [huggingface.co](https://huggingface.co) account → Settings →
   Access Tokens → **New token (Read)**. Put it in a file called `.env` in the
   repo root:
   ```
   HF_TOKEN=hf_xxxxxxxxxxxxx
   ```
2. Environment:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate            # Windows
   pip install datasets pandas matplotlib jupyter
   ```
3. Open the starter notebook: `jupyter notebook notebooks/eda.ipynb`
   (the dataset already loads in the first cell).

---

## The dataset

**Bitext Customer Support Dataset** —
`bitext/Bitext-customer-support-llm-chatbot-training-dataset` on Hugging Face.

| | |
|---|---|
| Size | ~27,000 rows, English, one table |
| `instruction` | the customer's message |
| `response` | a support agent's reply (already fairly polished) |
| `category` | ~11 buckets: ORDER, REFUND, INVOICE, PAYMENT, CANCELLATION, ACCOUNT, DELIVERY, SHIPPING, FEEDBACK, CONTACT, NEWSLETTER |
| `intent` | 27 finer labels |
| `flags` | letter codes for phrasing/politeness style |
| License | permissive, **attribution required** — read the dataset card and note it |

Backups if we need more (esp. angry customers / compliments), don't touch unless asked:
- `MohammadOthman/mo-customer-support-tweets-945k` (real brand tweets, messier)
- Kaggle: "Customer Support on Twitter" (raw version of the above)

---

## Why this dataset helps us

Every row already has a **customer message + an agent reply**. Our plan:

1. Use the existing `response` as the **"good reply"** (our target).
2. Auto-generate a **"blunt draft"** from it later (script strips the warmth).

That means **the big manual writing job disappears** — *if* the existing
replies are actually warm/human enough to use as targets. **Confirming that is
the main question your EDA answers.**

---

## Part 1 — EDA (this week)

Work in `notebooks/eda.ipynb`. For each step, leave your findings in a markdown
cell right below the code.

1. **Load & eyeball.** Row count, columns, print 20 random rows. What does a
   typical message/reply look like?
2. **Category balance.** Count rows per `category`, bar chart. Which categories
   are large? Which are tiny or missing?
3. **Lengths.** Histogram of word counts for `instruction` and for `response`.
   Note anything empty or absurdly long.
4. **Reply quality — the key one.** Read ~30 `response` values. Score each
   quickly: is it *warm and human* (good), or *stiff / templated / robotic*
   (bad)? Roughly what % are good? Paste 3 good and 3 bad examples.
5. **Junk check.** Duplicate rows? Leftover personal info (names, emails, order
   numbers)? Template placeholders like `{{Order Number}}`?
6. **Map to our 6 buckets.** We need: `billing`, `technical`,
   `angry_complaint`, `cancellation`, `praise`, `edge_case`. Which Bitext
   categories feed each? Make the mapping table. Flag buckets with too few
   examples (angry / praise are likely thin here).

### Deliverable — a half-page report (in the notebook or a `.md`)

- Total rows; estimated **usable** rows
- Table: examples available per one of our **6 buckets**
- Avg message length, avg reply length
- **Are the existing replies good enough to use as targets? yes / no / mostly**
- Problems found: missing buckets, PII to scrub, duplicates, placeholders
- 6 pasted example rows (3 good, 3 bad)

Send that to <owner>. We decide the next step from it.

---

## Part 2 — Build the training files (only after we agree on the plan)

You'll fill in 3 short functions in `src/data_pipeline/` (we'll pair on this):

| File | What it does |
|---|---|
| `collect.py` | pull Bitext rows → `data/raw/collected.jsonl` as `{customer_message, draft_reply, gold_reply, category, source}` |
| `clean.py` | scrub PII to tags (`<NAME> <EMAIL> <ORDER_ID>`), drop dupes/junk, convert to the training format |
| `split.py` | 80/10/10 split, balanced by category, no shared `source` across splits |

Plus hand-write **30–50 hard "exam" examples** in
`data/eval_benchmark/handcrafted_eval.jsonl` (4 samples already there as a
template — copy the shape).

### Definition of done

- [ ] ~1,000 clean examples, roughly balanced across the 6 buckets
- [ ] every `gold_reply` keeps the draft's facts/amounts, adds no new promises
- [ ] no real personal info left in the text
- [ ] 30–50 benchmark examples written
- [ ] `python -m src.data_pipeline.split` runs clean and prints category counts
- [ ] `python -m src.data_pipeline.build_eval_set` passes validation
- [ ] `pytest tests/test_data_pipeline.py` green
- [ ] data card added to `data/schema.md`: source, license/attribution, counts, method

---

## Reference — the exact training row format

See `data/schema.md`. One example:

```json
{
  "instruction": "Rewrite the draft support reply so it matches the company brand voice. Keep every fact and the policy outcome exactly the same.",
  "input": "Customer message:\nI was charged twice for March.\n\nDraft reply:\nDuplicate charge confirmed. $12 refunded in 5-7 days.",
  "output": "Hi <NAME>, I'm sorry about the double charge - I can see it here. I've refunded the $12 and it'll be back on your card within 5-7 business days. If it hasn't arrived by then, just reply and I'll chase it for you.",
  "meta": { "category": "billing", "source": "bitext_04123" }
}
```
