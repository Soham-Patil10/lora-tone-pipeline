"""Task-specific automated metrics for tone matching.

Perplexity is NOT enough. We score three axes and combine:

  tone_score            in [0,1]  - does the output match the brand voice?
                                    classifier or rubric-based scorer over
                                    {empathy, clarity, non-defensiveness,
                                     active-voice, closing-offer}.
  content_preservation  in [0,1]  - are all facts / the policy outcome from the
                                    draft still present and unchanged?
                                    (NLI entailment both directions, or
                                     embedding sim of extracted claims.)
  format_valid          in {0,1}  - length budget, ends with an offer, no
                                    placeholder text, no leaked <NAME> tags.
  rouge_l               in [0,1]  - lexical overlap with the reference (weak
                                    signal, reported for completeness).

overall = 0.5*tone_score + 0.4*content_preservation + 0.1*format_valid
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ExampleScore:
    id: str
    category: str
    tone_score: float
    content_preservation: float
    format_valid: float
    rouge_l: float

    @property
    def overall(self) -> float:
        return 0.5 * self.tone_score + 0.4 * self.content_preservation + 0.1 * self.format_valid


def tone_score(prediction: str) -> float:
    """TODO: fine-tuned tone classifier (DistilRoBERTa) OR weighted rubric checks."""
    raise NotImplementedError


def content_preservation(draft_reply: str, prediction: str) -> float:
    """TODO: bidirectional NLI / claim-set overlap between draft and prediction."""
    raise NotImplementedError


def format_valid(prediction: str, max_words: int = 160) -> float:
    raise NotImplementedError


def rouge_l(prediction: str, reference: str) -> float:
    raise NotImplementedError


def score_example(ex: dict, prediction: str) -> ExampleScore:
    """ex is a benchmark/test row; prediction is the model output."""
    raise NotImplementedError


def aggregate(scores: list[ExampleScore]) -> dict:
    """Overall + per-category means, returned as a flat dict for logging. TODO."""
    raise NotImplementedError
