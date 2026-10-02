from agentrig.agent.prompts import STOP_SEQUENCES
from agentrig.config import Endpoint
from agentrig.eval.runner import condition_key, derive_seed, expand_conditions
from agentrig.llm.client import Sampling, build_request

VLLM = Endpoint("vllm_5070", "rtx5070-vllm", "http://x/v1", "EMPTY", "chat_template_kwargs")


def test_derive_seed_is_stable_and_varies() -> None:
    assert derive_seed(1, "q", 0) == derive_seed(1, "q", 0)
    assert derive_seed(1, "q", 0) != derive_seed(1, "q", 1)


def test_expand_conditions() -> None:
    cfg = {"temperatures": [0.0, 0.7], "enable_thinking": [False, True]}
    conds = expand_conditions(cfg, "on")
    assert len(conds) == 4
    assert condition_key(conds[0]) == "T0.0_think-off_cache-on"


def test_build_request_merges_extra_body() -> None:
    sampling = Sampling(temperature=0.7, top_p=1.0, max_tokens=64, seed=5, top_k=20)
    req = build_request(VLLM, "m", [], sampling, False, STOP_SEQUENCES)
    assert req["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}, "top_k": 20}
    assert req["seed"] == 5 and req["stop"] == STOP_SEQUENCES


def test_same_seed_replicates_share_a_seed() -> None:
    k = 2  # replicates_per_seed
    seeds = [derive_seed(1, "q", divmod(r, k)[0]) for r in range(4)]
    assert seeds[0] == seeds[1] and seeds[2] == seeds[3] and seeds[0] != seeds[2]
