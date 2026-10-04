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


def test_clopper_pearson_matches_known_values() -> None:
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "analysis" / "03_same_seed_divergence.py"
    spec = importlib.util.spec_from_file_location("div", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    lo, hi = mod.clopper_pearson(0, 200)
    assert lo == 0.0 and abs(hi - (1 - 0.025 ** (1 / 200))) < 1e-6  # exact upper bound for 0/n
    lo, hi = mod.clopper_pearson(3, 200)
    assert abs(lo - 0.003104) < 1e-5 and abs(hi - 0.043208) < 1e-5  # R: binom.test(3, 200)
    assert mod.clopper_pearson(5, 1100)[1] < 0.02  # large n must not overflow
