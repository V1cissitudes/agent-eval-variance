from agentrig.tools.search import ParagraphSearch

CONTEXT = {
    "title": ["Paris", "Berlin", "Oak Park"],
    "sentences": [
        ["Paris is the capital of France."],
        ["Berlin is the capital of Germany."],
        ["Oak Park is a village in Illinois.", " Hemingway was born there."],
    ],
}


def test_relevant_paragraph_ranks_first() -> None:
    search = ParagraphSearch(CONTEXT, top_k=2)
    assert search.rank("where was Hemingway born")[0] == 2
    assert search.run("capital of Germany").startswith("[Berlin]")


def test_deterministic_and_top_k() -> None:
    search = ParagraphSearch(CONTEXT, top_k=2)
    assert search.run("capital") == search.run("capital")
    assert len(search.run("capital").splitlines()) == 2
