"""Single source of truth for how examples become model prompts.

The SAME template must be used for training, baseline eval, and fine-tuned eval,
otherwise the comparison is invalid. Keyed by name so config.eval.prompt_template
can select one.
"""
from __future__ import annotations

from typing import Callable

# The brand voice the fine-tune is teaching. Keep in sync with data/schema.md.
BRAND_TONE_GUIDE = (
    "Voice: warm, empathetic, and professional. Acknowledge the customer's "
    "feelings first, then give a clear next step. Use plain language, short "
    "sentences, and an active voice. No corporate jargon, no blame, no "
    "over-apologising. Always close with a concrete offer of further help."
)

SYSTEM_PROMPT = (
    "You are a senior customer support specialist. You rewrite draft replies so "
    "they match the company brand voice while preserving every fact and the "
    "policy decision in the draft.\n\n" + BRAND_TONE_GUIDE
)


def _default_user_prompt(example: dict) -> str:
    """example keys: instruction, input (contains customer_message + draft_reply)."""
    return (
        f"{example['instruction']}\n\n"
        f"---\n{example['input']}\n---\n\n"
        "Rewrite the draft reply in the brand voice."
    )


TEMPLATES: dict[str, Callable[[dict], str]] = {
    "default": _default_user_prompt,
}


def build_messages(example: dict, template: str = "default") -> list[dict]:
    """Return chat-format messages (no tokenisation). Used everywhere."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": TEMPLATES[template](example)},
    ]


def render_training_text(example: dict, tokenizer, template: str = "default") -> str:
    """Full prompt + target, formatted with the tokenizer chat template.

    TODO: apply_chat_template(messages + assistant turn=example['output'],
    tokenize=False). Mask the prompt tokens in the collator, not here.
    """
    raise NotImplementedError
