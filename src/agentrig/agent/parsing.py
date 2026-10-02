"""Parse one model turn into an action, a final answer, or a format error."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

_ACTION = re.compile(r"^\s*Action\s*:\s*(\w+)\s*\[(.*)\]\s*$", re.IGNORECASE | re.MULTILINE)
_FINAL = re.compile(r"^\s*Final\s+Answer\s*:\s*(.*?)\s*$", re.IGNORECASE | re.MULTILINE)
_THOUGHT = re.compile(r"^\s*Thought\s*:\s*(.*?)\s*$", re.IGNORECASE | re.MULTILINE)


@dataclass(frozen=True)
class Parsed:
    kind: Literal["action", "final", "error"]
    thought: str | None = None
    tool: str | None = None
    argument: str | None = None
    answer: str | None = None
    error: str | None = None


def _clean(text: str) -> str:
    """Drop markdown emphasis such as **Action:** that small models like to add."""
    return text.replace("**", "").replace("__", "")


def parse_output(text: str) -> Parsed:
    """If both an Action and a Final Answer appear, whichever comes first wins."""
    text = _clean(text)
    thought_match = _THOUGHT.search(text)
    thought = thought_match.group(1) if thought_match else None
    action, final = _ACTION.search(text), _FINAL.search(text)
    if final and (not action or final.start() < action.start()):
        answer = final.group(1)
        if not answer:
            return Parsed("error", thought, error="empty final answer")
        return Parsed("final", thought, answer=answer)
    if action:
        argument = action.group(2).strip()
        if not argument:
            return Parsed("error", thought, error="empty action argument")
        return Parsed("action", thought, tool=action.group(1).lower(), argument=argument)
    return Parsed("error", thought, error="no Action or Final Answer line")
