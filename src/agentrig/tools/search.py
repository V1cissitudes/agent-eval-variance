"""BM25 search over the 10 context paragraphs bundled with each HotpotQA (distractor) question."""

from __future__ import annotations

import re
from typing import Any

from rank_bm25 import BM25Okapi

_TOKEN = re.compile(r"\w+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class ParagraphSearch:
    """Deterministic: ties are broken by paragraph order, so the same query always returns the
    same paragraphs. That keeps tool output from adding variance of its own."""

    name = "search"

    def __init__(self, context: dict[str, Any], top_k: int = 2) -> None:
        self.titles: list[str] = list(context["title"])
        self.texts: list[str] = ["".join(sents).strip() for sents in context["sentences"]]
        self.top_k = top_k
        corpus = [tokenize(f"{t} {x}") for t, x in zip(self.titles, self.texts, strict=True)]
        self._bm25 = BM25Okapi(corpus)

    def rank(self, query: str) -> list[int]:
        scores = self._bm25.get_scores(tokenize(query))
        return sorted(range(len(self.titles)), key=lambda i: (-scores[i], i))[: self.top_k]

    def run(self, argument: str) -> str:
        return "\n".join(f"[{self.titles[i]}] {self.texts[i]}" for i in self.rank(argument))
