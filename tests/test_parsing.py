from agentrig.agent.parsing import parse_output


def test_action() -> None:
    p = parse_output("Thought: look it up\nAction: search[Ernest Hemingway]")
    assert (p.kind, p.tool, p.argument, p.thought) == (
        "action",
        "search",
        "Ernest Hemingway",
        "look it up",
    )


def test_final_answer() -> None:
    p = parse_output("Thought: done\nFinal Answer: Oak Park, Illinois")
    assert (p.kind, p.answer) == ("final", "Oak Park, Illinois")


def test_first_of_action_and_final_wins() -> None:
    assert parse_output("Action: search[x]\nFinal Answer: y").kind == "action"
    assert parse_output("Final Answer: y\nAction: search[x]").kind == "final"


def test_markdown_and_case_are_tolerated() -> None:
    p = parse_output("**Thought:** hmm\n**action:** Search[ Paris ]")
    assert (p.kind, p.tool, p.argument) == ("action", "search", "Paris")


def test_format_errors() -> None:
    assert parse_output("I think the answer is Paris.").kind == "error"
    assert parse_output("Action: search[]").error == "empty action argument"
    assert parse_output("Final Answer:   ").error == "empty final answer"
