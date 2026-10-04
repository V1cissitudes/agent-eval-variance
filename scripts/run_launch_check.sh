#!/usr/bin/env bash
# Launch-reproducibility check (exploratory, 2026-10-04; analysis/08_cross_launch.py): the first 50
# greedy questions of E01a/E01b re-run on freshly started servers, per precision and cache setting,
# with KV cache fixed at 1,361 blocks. FP8/AWQ launch exactly as in E01b; the BF16 launch command
# differs from E01a's and loads a different compiled graph (docs/experiments/E01b_results.md).
# Then restores BF16 cache-on.
# Run on the 5070 inside tmux, after E01b has finished:
#   tmux new -d -s launch "bash -lc \"~/code/agent-eval-variance/scripts/run_launch_check.sh 2>&1 \
#     | tee -a ~/code/agent-eval-variance/logs/launch_check.log\""
# Resumable: completed runs are skipped (each condition still restarts its server).
set -euo pipefail
cd ~/code/agent-eval-variance
COMMON="--gpu-mem 0.90 --max-len 8192 --max-num-seqs 1 --max-num-batched-tokens 512"

restart_server() {  # $1 model  $2 revision ("" for none)  $3 cache on|off  $4 extra args  $5 tag
  tmux kill-session -t vllm 2>/dev/null || true
  for _ in $(seq 1 30); do ss -ltn | grep -q ":8000 " || break; sleep 2; done
  local rev=""; [[ -n "$2" ]] && rev="--revision $2"
  local ts; ts=$(date +%Y%m%d_%H%M%S)
  tmux new -d -s vllm "bash -lc \"cd ~/code/agent-eval-variance && HF_HUB_OFFLINE=1 scripts/serve_vllm.sh $1 $rev --prefix-cache $3 $COMMON $4 2>&1 | tee logs/vllm_${ts}_$5.log\""
  for _ in $(seq 1 60); do curl -s -m 5 localhost:8000/v1/models >/dev/null && break; sleep 10; done
  curl -s -m 5 localhost:8000/v1/models >/dev/null || { echo "server did not start: $5"; exit 1; }
  echo "[$(date -Iseconds)] server ready: $5"
}

run_condition() {  # $1 model key  $2 hf id  $3 revision  $4 cache  $5 precision tag (config suffix)
  restart_server "$2" "$3" "$4" "--kv-blocks 1361" "launch_$1_cache-$4"
  uv run --locked --no-sync python scripts/check_endpoint.py --endpoint vllm_5070 --model "$1"
  uv run --locked --no-sync python scripts/run_experiment.py \
    "configs/experiments/dev_E01b_launch_$5.yaml" --prefix-cache "$4"
  echo "[$(date -Iseconds)] finished $1 cache-$4: $(wc -l < "runs/dev_E01b_launch/vllm_5070__$1__cache-$4.jsonl") runs"
}

FP8_REV=96b30dc13593a244a5e59e84687309f53c375cfa
AWQ_REV=74d4bd2bd4bff9cafc9345221320bffb08b406a3
echo "[$(date -Iseconds)] start launch check"
run_condition qwen3-4b     Qwen/Qwen3-4B     ""         off bf16
run_condition qwen3-4b     Qwen/Qwen3-4B     ""         on  bf16
run_condition qwen3-4b-fp8 Qwen/Qwen3-4B-FP8 "$FP8_REV" off fp8
run_condition qwen3-4b-fp8 Qwen/Qwen3-4B-FP8 "$FP8_REV" on  fp8
run_condition qwen3-4b-awq Qwen/Qwen3-4B-AWQ "$AWQ_REV" off awq
run_condition qwen3-4b-awq Qwen/Qwen3-4B-AWQ "$AWQ_REV" on  awq
restart_server Qwen/Qwen3-4B "" on "" bf16_cache-on_default
echo "[$(date -Iseconds)] launch check done"
