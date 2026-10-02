#!/usr/bin/env bash
# Start a vLLM OpenAI-compatible server on the RTX 5070 (WSL2).
# Usage: scripts/serve_vllm.sh <hf_model_id> [--prefix-cache on|off] [--port 8000]
#                              [--dtype bfloat16] [--gpu-mem 0.90] [--max-len 8192]
#                              [--max-num-seqs N] [--max-num-batched-tokens N]
# Prefix caching is a server-side setting, so each cache condition needs its own server run.
# --reasoning-parser qwen3 only moves thinking text into the `reasoning` response field
# (same shape as Ollama); whether thinking is generated is set per request (enable_thinking).
set -euo pipefail

usage="usage: serve_vllm.sh <hf_model_id> [--prefix-cache on|off] [--port N] [--dtype T] [--gpu-mem F] [--max-len N] [--max-num-seqs N] [--max-num-batched-tokens N]"
MODEL="${1:?$usage}"
shift
CACHE="on"
PORT="8000"
DTYPE="bfloat16"
GPU_MEM="0.90"
MAX_LEN="8192"
MAX_NUM_SEQS=""
MAX_NUM_BATCHED_TOKENS=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --prefix-cache) CACHE="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --dtype) DTYPE="$2"; shift 2 ;;
    --gpu-mem) GPU_MEM="$2"; shift 2 ;;
    --max-len) MAX_LEN="$2"; shift 2 ;;
    --max-num-seqs) MAX_NUM_SEQS="$2"; shift 2 ;;
    --max-num-batched-tokens) MAX_NUM_BATCHED_TOKENS="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; echo "$usage" >&2; exit 1 ;;
  esac
done

case "$CACHE" in
  on) CACHE_FLAG="--enable-prefix-caching" ;;
  off) CACHE_FLAG="--no-enable-prefix-caching" ;;  # explicit, not just omitting the "on" flag
  *) echo "--prefix-cache must be on or off" >&2; exit 1 ;;
esac

# --locked: fail instead of silently re-locking if uv.lock is out of date.
CMD=(uv run --locked --extra vllm vllm serve "$MODEL"
  --host 0.0.0.0 --port "$PORT" --dtype "$DTYPE" "$CACHE_FLAG"
  --reasoning-parser qwen3
  --max-model-len "$MAX_LEN" --gpu-memory-utilization "$GPU_MEM")

if [[ -n "$MAX_NUM_SEQS" ]]; then
  CMD+=(--max-num-seqs "$MAX_NUM_SEQS")
fi
if [[ -n "$MAX_NUM_BATCHED_TOKENS" ]]; then
  CMD+=(--max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS")
fi

# vLLM-native sampler instead of FlashInfer: FlashInfer JIT needs a CUDA toolkit (nvcc) matching the
# wheels. The sampler backend is part
# of the serving configuration, so it is fixed here rather than set by hand. Override only on purpose.
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"

echo "[$(date -Iseconds)] VLLM_USE_FLASHINFER_SAMPLER=$VLLM_USE_FLASHINFER_SAMPLER ${CMD[*]}"
exec "${CMD[@]}"
