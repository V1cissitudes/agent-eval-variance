"""Chat message history for the agent. The input at step t equals the initial messages plus
every earlier (assistant output, observation) pair, so trajectories store only the new parts."""

from __future__ import annotations

from agentrig.agent.prompts import QUESTION_TEMPLATE, SYSTEM_PROMPT

Message = dict[str, str]


def initial_messages(question: str) -> list[Message]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": QUESTION_TEMPLATE.format(question=question)},
    ]


def append_turn(messages: list[Message], assistant_text: str, observation: str) -> None:
    """Add one step. Thinking text is never fed back (Qwen3 drops earlier reasoning too)."""
    messages.append({"role": "assistant", "content": assistant_text.strip()})
    messages.append({"role": "user", "content": observation})
