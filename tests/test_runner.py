from agentrig.agent.prompts import STOP_SEQUENCES
from agentrig.config import Endpoint
from agentrig.eval.runner import condition_key, derive_seed, expand_conditions
from agentrig.llm.client import Sampling, build_request

VLLM = Endpoint("vllm_5070", "rtx5070-vllm", "http://x/v1", "EMPTY", "chat_template_kwargs")


def test_derive_seed_is_stable_and_varies() -> None:
    assert derive_seed(1, "q", 0) == derive_seed(1, "q", 0)
    assert derive_seed(1, "q", 0) != derive_seed(1, "q", 1)


def test_expand_conditions_legacy() -> None:
    cfg = {"temperatures": [0.0, 0.7], "enable_thinking": [False, True], "n_repeats": 3}
    conds = expand_conditions(cfg, "on")
    assert len(conds) == 4
    assert condition_key(conds[0]) == "T0.0_think-off_cache-on"
    assert conds[0]["n_repeats"] == 3 and conds[0]["arm"] == "default" and "name" not in conds[0]


def test_expand_conditions_arms_override_defaults() -> None:
    cfg = {
        "top_p": 0.8,
        "top_k": 20,
        "n_repeats": 6,
        "replicates_per_seed": 2,
        "arms": [
            {"name": "greedy", "temperature": 0.0, "n_repeats": 2, "replicates_per_seed": 2},
            {"name": "qwen", "temperature": 0.7},
            {"name": "topp1", "temperature": 0.7, "top_p": 1.0, "top_k": None},
        ],
    }
    greedy, qwen, topp1 = expand_conditions(cfg, "off")
    assert (greedy["n_repeats"], qwen["n_repeats"]) == (2, 6)
    assert (qwen["top_p"], qwen["top_k"]) == (0.8, 20)
    assert (topp1["top_p"], topp1["top_k"]) == (1.0, None)
    assert condition_key(topp1) == "topp1_T0.7_think-off_cache-off"


def test_build_request_merges_extra_body() -> None:
    sampling = Sampling(temperature=0.7, top_p=1.0, max_tokens=64, seed=5, top_k=20)
    req = build_request(VLLM, "m", [], sampling, False, STOP_SEQUENCES)
    assert req["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}, "top_k": 20}
    assert req["seed"] == 5 and req["stop"] == STOP_SEQUENCES


def test_same_seed_replicates_share_a_seed() -> None:
    k = 2  # replicates_per_seed
    seeds = [derive_seed(1, "q", divmod(r, k)[0]) for r in range(4)]
    assert seeds[0] == seeds[1] and seeds[2] == seeds[3] and seeds[0] != seeds[2]


def test_completed_run_ids_skips_truncated_line(tmp_path) -> None:
    from agentrig.eval.trajectory import completed_run_ids

    path = tmp_path / "runs.jsonl"
    path.write_text('{"run_id": "a", "error": null}\n{"run_id": "b", "err', encoding="utf-8")
    assert completed_run_ids(path) == {"a"}
