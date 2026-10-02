"""ReAct agent loop: call model -> parse -> run tool -> append observation -> repeat.

The model call is injected (`llm`), so the loop has no network code and can be tested with
scripted replies. A run ends with a final answer or when max_steps is reached.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

from agentrig.agent.messages import Message, append_turn, initial_messages
from agentrig.agent.parsing import Parsed, parse_output
from agentrig.agent.prompts import (
    FORMAT_ERROR_OBSERVATION,
    OBSERVATION_TEMPLATE,
    UNKNOWN_TOOL_OBSERVATION,
)
from agentrig.llm.client import LLMReply
from agentrig.tools.base import Tool

LLM = Callable[[list[Message], int], LLMReply]


@dataclass
class StepRecord:
    index: int
    content: str
    reasoning: str
    parsed: dict[str, Any]
    observation: str | None
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int
    reasoning_tokens: int | None
    latency_s: float


@dataclass
class AgentResult:
    final_answer: str | None
    termination: str  # "final_answer" | "max_steps"
    initial_messages: list[Message]
    steps: list[StepRecord] = field(default_factory=list)


def observe(parsed: Parsed, tools: dict[str, Tool]) -> str:
    """Turn a non-final parse into the observation shown to the model."""
    if parsed.kind == "error":
        return FORMAT_ERROR_OBSERVATION
    tool = tools.get(parsed.tool or "")
    if tool is None:
        return UNKNOWN_TOOL_OBSERVATION.format(tool=parsed.tool)
    return OBSERVATION_TEMPLATE.format(text=tool.run(parsed.argument or ""))


def make_step(index: int, reply: LLMReply, parsed: Parsed, observation: str | None) -> StepRecord:
    return StepRecord(
        index=index,
        content=reply.content,
        reasoning=reply.reasoning,
        parsed=asdict(parsed),
        observation=observation,
        finish_reason=reply.finish_reason,
        prompt_tokens=reply.prompt_tokens,
        completion_tokens=reply.completion_tokens,
        reasoning_tokens=reply.reasoning_tokens,
        latency_s=reply.latency_s,
    )


def run_agent(question: str, tools: dict[str, Tool], llm: LLM, max_steps: int) -> AgentResult:
    """Answering without any search is accepted (and visible as zero search steps)."""
    messages = initial_messages(question)
    result = AgentResult(None, "max_steps", [dict(m) for m in messages])
    for index in range(max_steps):
        reply = llm(messages, index)
        parsed = parse_output(reply.content)
        if parsed.kind == "final":
            result.steps.append(make_step(index, reply, parsed, None))
            result.final_answer, result.termination = parsed.answer, "final_answer"
            return result
        observation = observe(parsed, tools)
        result.steps.append(make_step(index, reply, parsed, observation))
        append_turn(messages, reply.content, observation)
    return result
