"""Build an OpenAI-compatible client and make single chat calls with full sampling control."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from openai import OpenAI

from agentrig.config import Endpoint


@dataclass(frozen=True)
class Sampling:
    """Per-request sampling settings. Everything sent to the server is recorded in trajectories."""

    temperature: float
    top_p: float = 1.0
    max_tokens: int = 1024
    seed: int | None = None
    top_k: int | None = None  # None = not sent (server default applies)


@dataclass(frozen=True)
class LLMReply:
    content: str
    reasoning: str
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int
    reasoning_tokens: int | None
    latency_s: float


def make_client(endpoint: Endpoint, timeout: float = 300.0) -> OpenAI:
    return OpenAI(base_url=endpoint.base_url, api_key=endpoint.api_key, timeout=timeout)


def thinking_kwargs(endpoint: Endpoint, enable_thinking: bool) -> dict[str, Any]:
    """Translate the enable_thinking switch into the parameter this server understands."""
    if endpoint.thinking_control == "chat_template_kwargs":
        return {"extra_body": {"chat_template_kwargs": {"enable_thinking": enable_thinking}}}
    if endpoint.thinking_control == "reasoning_effort":
        # Ollama: reasoning_effort="none" cleanly disables thinking on hybrid Qwen3 models.
        # Omitting it keeps the model's default (thinking on).
        return {} if enable_thinking else {"reasoning_effort": "none"}
    raise ValueError(f"Unknown thinking_control {endpoint.thinking_control!r}")


def extract_reasoning(message: Any) -> str:
    """vLLM >= 0.30 and Ollama use `reasoning`; older vLLM used `reasoning_content`."""
    for field in ("reasoning", "reasoning_content"):
        value = getattr(message, field, None)
        if value:
            return value
    return ""


def build_request(
    endpoint: Endpoint,
    model: str,
    messages: list[dict[str, str]],
    sampling: Sampling,
    enable_thinking: bool,
    stop: list[str] | None = None,
) -> dict[str, Any]:
    """Assemble chat.completions kwargs; extra_body from thinking and top_k is merged."""
    request: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": sampling.temperature,
        "top_p": sampling.top_p,
        "max_tokens": sampling.max_tokens,
    }
    if sampling.seed is not None:
        request["seed"] = sampling.seed
    if stop:
        request["stop"] = stop
    extra = thinking_kwargs(endpoint, enable_thinking)
    extra_body = dict(extra.pop("extra_body", {}))
    if sampling.top_k is not None:
        extra_body["top_k"] = sampling.top_k
    request.update(extra)
    if extra_body:
        request["extra_body"] = extra_body
    return request


def chat(client: OpenAI, request: dict[str, Any]) -> LLMReply:
    """Send one request and flatten the response into an LLMReply."""
    start = time.perf_counter()
    resp = client.chat.completions.create(**request)
    latency = time.perf_counter() - start
    choice = resp.choices[0]
    usage = resp.usage
    details = getattr(usage, "completion_tokens_details", None) if usage else None
    return LLMReply(
        content=choice.message.content or "",
        reasoning=extract_reasoning(choice.message),
        finish_reason=choice.finish_reason or "",
        prompt_tokens=usage.prompt_tokens if usage else 0,
        completion_tokens=usage.completion_tokens if usage else 0,
        reasoning_tokens=getattr(details, "reasoning_tokens", None) if details else None,
        latency_s=latency,
    )
