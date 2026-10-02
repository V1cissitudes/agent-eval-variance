import pytest

from agentrig.config import expand_env, load_endpoint, load_model
from agentrig.llm.client import thinking_kwargs


def test_expand_env_uses_value_and_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENTRIG_TEST_VAR", "abc")
    assert expand_env("x/${AGENTRIG_TEST_VAR}/y") == "x/abc/y"
    assert expand_env("${AGENTRIG_UNSET_VAR:-fallback}") == "fallback"


def test_expand_env_missing_required(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AGENTRIG_UNSET_VAR", raising=False)
    with pytest.raises(KeyError, match="AGENTRIG_UNSET_VAR"):
        expand_env("${AGENTRIG_UNSET_VAR}")


def test_vllm_endpoint_reads_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VLLM_BASE_URL", "http://gpu-box:8000/v1")
    ep = load_endpoint("vllm_5070")
    assert ep.base_url == "http://gpu-box:8000/v1"
    assert ep.api_key  # falls back to EMPTY


def test_thinking_off_by_default() -> None:
    assert load_model("qwen3-4b")["enable_thinking"] is False


def test_thinking_kwargs_per_platform(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VLLM_BASE_URL", "http://gpu-box:8000/v1")
    assert thinking_kwargs(load_endpoint("ollama_mac"), False) == {"reasoning_effort": "none"}
    vllm = thinking_kwargs(load_endpoint("vllm_5070"), False)
    assert vllm["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False
