"""HotpotQA answer scoring, following the official evaluation script (EM and token F1)."""

from __future__ import annotations

import re
import string
from collections import Counter

_SPECIAL = {"yes", "no", "noanswer"}


def normalize_answer(text: str) -> str:
    """Lowercase, drop punctuation and articles, collapse whitespace."""
    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(prediction: str | None, gold: str) -> float:
    if prediction is None:
        return 0.0
    return float(normalize_answer(prediction) == normalize_answer(gold))


def f1_score(prediction: str | None, gold: str) -> float:
    if prediction is None:
        return 0.0
    pred, ref = normalize_answer(prediction), normalize_answer(gold)
    if (pred in _SPECIAL or ref in _SPECIAL) and pred != ref:
        return 0.0
    common = Counter(pred.split()) & Counter(ref.split())
    same = sum(common.values())
    if same == 0:
        return 0.0
    precision, recall = same / len(pred.split()), same / len(ref.split())
    return 2 * precision * recall / (precision + recall)
