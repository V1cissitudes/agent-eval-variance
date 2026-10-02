from agentrig.eval.metrics import exact_match, f1_score, normalize_answer


def test_normalize() -> None:
    assert normalize_answer("The  Eiffel Tower!") == "eiffel tower"


def test_exact_match() -> None:
    assert exact_match("the Eiffel tower.", "Eiffel Tower") == 1.0
    assert exact_match(None, "x") == 0.0


def test_f1_partial_and_yes_no() -> None:
    assert abs(f1_score("Oak Park", "Oak Park, Illinois") - 0.8) < 1e-9
    assert f1_score("yes", "no") == 0.0
    assert f1_score("no", "no") == 1.0
