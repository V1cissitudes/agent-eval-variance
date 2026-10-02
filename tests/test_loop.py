from agentrig.agent.loop import run_agent
from agentrig.llm.client import LLMReply
from agentrig.tools.search import ParagraphSearch

CONTEXT = {"title": ["Oak Park"], "sentences": [["Hemingway was born in Oak Park."]]}


def scripted(*outputs: str):
    """Fake model: returns the given outputs in order and records what it was shown."""
    seen: list[list[dict[str, str]]] = []

    def llm(messages, step):
        seen.append([dict(m) for m in messages])
        return LLMReply(outputs[step], "", "stop", 10 * (step + 1), 5, None, 0.01)

    return llm, seen


def tools():
    tool = ParagraphSearch(CONTEXT, top_k=1)
    return {tool.name: tool}


def test_search_then_answer() -> None:
    llm, seen = scripted(
        "Thought: search\nAction: search[Hemingway born]", "Thought: ok\nFinal Answer: Oak Park"
    )
    result = run_agent("Where was Hemingway born?", tools(), llm, max_steps=5)
    assert (result.final_answer, result.termination, len(result.steps)) == (
        "Oak Park",
        "final_answer",
        2,
    )
    assert seen[1][-1]["content"].startswith("Observation: [Oak Park]")


def test_answer_without_search_is_accepted() -> None:
    llm, _ = scripted("Thought: I know\nFinal Answer: Oak Park")
    result = run_agent("q", tools(), llm, max_steps=5)
    assert result.termination == "final_answer"
    assert [s.parsed["kind"] for s in result.steps] == ["final"]


def test_format_error_gets_feedback_and_max_steps_ends_run() -> None:
    llm, seen = scripted("no format", "still no format")
    result = run_agent("q", tools(), llm, max_steps=2)
    assert (result.final_answer, result.termination) == (None, "max_steps")
    assert "Invalid format" in seen[1][-1]["content"]
