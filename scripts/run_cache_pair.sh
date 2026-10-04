#!/usr/bin/env bash
# Run a fixed experiment with cache on, cache off, then restore cache on.
# Repeating with the SAME config and serving settings is safe: the runner skips successful runs.
# Failure keeps sessions, logs and all trajectories; there is no automatic retry or data deletion.
# Usage: scripts/run_cache_pair.sh [--dry-run] <experiment.yaml> [serve options...]
set -Eeuo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
DRY_RUN=0
CONFIG=""
PORT=8000
DTYPE=""
REVISION=""
REVISION_SET=0
KV_BLOCKS=""
GPU_MEM=0.90
MAX_LEN=8192
MAX_NUM_SEQS=1
MAX_NUM_BATCHED_TOKENS=512
START_TIMEOUT=${CACHE_PAIR_START_TIMEOUT:-600}
STOP_TIMEOUT=${CACHE_PAIR_STOP_TIMEOUT:-120}
RUN_TIMEOUT=${CACHE_PAIR_RUN_TIMEOUT:-86400}
PHASE=arguments

status() { printf '[%s] %s\n' "$(date -Iseconds)" "$*"; }
fail() { status "FAILED ($PHASE): $*; sessions, logs and data are retained" >&2; exit 1; }
trap 'code=$?; status "FAILED ($PHASE, exit $code): state retained; no retry or recovery attempted" >&2; exit "$code"' ERR
trap 'status "Interrupted ($PHASE): state retained" >&2; exit 130' INT TERM
usage() {
  echo "usage: scripts/run_cache_pair.sh [--dry-run] <experiment.yaml> [--port N] [--revision COMMIT] [--kv-blocks N] [--dtype T] [--gpu-mem F] [--max-len N] [--max-num-seqs N] [--max-num-batched-tokens N]"
}
while (($#)); do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --help|-h) usage; exit 0 ;;
    --revision|--kv-blocks|--port|--dtype|--gpu-mem|--max-len|--max-num-seqs|--max-num-batched-tokens)
      (($# >= 2)) || fail "missing value for $1"
      case "$1" in
        --revision) REVISION=$2; REVISION_SET=1 ;; --kv-blocks) KV_BLOCKS=$2 ;;
        --port) PORT=$2 ;; --dtype) DTYPE=$2 ;; --gpu-mem) GPU_MEM=$2 ;;
        --max-len) MAX_LEN=$2 ;; --max-num-seqs) MAX_NUM_SEQS=$2 ;;
        --max-num-batched-tokens) MAX_NUM_BATCHED_TOKENS=$2 ;;
      esac
      shift 2 ;;
    -*) fail "unsupported option $1 (prefix cache is controlled by this script)" ;;
    *) [[ -z "$CONFIG" ]] || fail "unexpected positional argument $1"; CONFIG=$1; shift ;;
  esac
done
[[ -n "$CONFIG" && -f "$CONFIG" ]] || { usage; fail "experiment config not found"; }
for value in "$PORT" "$MAX_LEN" "$MAX_NUM_SEQS" "$MAX_NUM_BATCHED_TOKENS" "$START_TIMEOUT" "$STOP_TIMEOUT" "$RUN_TIMEOUT"; do
  [[ "$value" =~ ^[1-9][0-9]*$ && ${#value} -le 8 ]] || fail "expected a bounded positive integer"
done
((PORT <= 65535)) || fail "port must be at most 65535"
[[ "$GPU_MEM" =~ ^(0\.[0-9]+|1(\.0+)?)$ && ! "$GPU_MEM" =~ ^0\.0+$ ]] || fail "GPU fraction must be in (0, 1]"
UV_BIN=$(command -v uv) || fail "uv is not installed"
CONTROL=("$UV_BIN" run --locked --no-sync python scripts/cache_pair_control.py)
CONFIG_INFO=$("${CONTROL[@]}" config --config "$CONFIG")
INFO=()  # portable replacement for `mapfile` (absent in macOS bash 3.2)
while IFS= read -r line; do INFO+=("$line"); done <<< "$CONFIG_INFO"
MODEL=${INFO[0]}
ENDPOINT=${INFO[1]}
EXPERIMENT=${INFO[2]}
((REVISION_SET)) || REVISION=${INFO[3]}
DTYPE=${DTYPE:-${INFO[4]}}
KV_BLOCKS=${KV_BLOCKS:-${INFO[5]}}
[[ "$DTYPE" == bfloat16 || "$DTYPE" == float16 ]] || fail "dtype must be explicit"
[[ "$KV_BLOCKS" =~ ^[1-9][0-9]*$ && ${#KV_BLOCKS} -le 8 ]] || fail "invalid KV block count"
[[ -z "$REVISION" || "$REVISION" =~ ^[a-zA-Z0-9_./-]+$ ]] || fail "invalid revision"
[[ -z "$REVISION" || "$REVISION" != -* ]] || fail "invalid revision"
SERVE_ARGS=(--port "$PORT" --dtype "$DTYPE" --gpu-mem "$GPU_MEM" --max-len "$MAX_LEN"
  --max-num-seqs "$MAX_NUM_SEQS" --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
  --kv-blocks "$KV_BLOCKS")
[[ -z "$REVISION" ]] || SERVE_ARGS+=(--revision "$REVISION")
VERIFY_ARGS=(--endpoint "$ENDPOINT" --model "$MODEL" "${SERVE_ARGS[@]}")
RUN_SESSION=cache-pair-run
RUN_DIR="logs/cache-pair_$(date +%Y%m%d_%H%M%S)_$$"

print_command() { printf '[%s] ' "$(date -Iseconds)"; printf '%q ' "$@"; printf '\n'; }
serve_command() { SERVE_CMD=(env UV_NO_SYNC=1 HF_HUB_OFFLINE=1 scripts/serve_vllm.sh "$MODEL" "${SERVE_ARGS[@]}" --prefix-cache "$1"); }
run_command() {
  RUN_CMD=(env UV_NO_SYNC=1 "VLLM_BASE_URL=http://127.0.0.1:$PORT/v1"
    "$UV_BIN" run --locked --no-sync python scripts/run_experiment.py "$CONFIG" --endpoint "$ENDPOINT" --prefix-cache "$1")
}
launch_session() {
  local session=$1 label=$2
  shift 2
  local quoted command
  printf -v quoted '%q ' "$@"
  command="set +e; set -o pipefail; $quoted 2>&1 | tee $(printf '%q' "$RUN_DIR/$label.log"); result=\$?; printf '%s\\n' \"\$result\" > $(printf '%q' "$RUN_DIR/$label.exit"); exit \"\$result\""
  local launch
  printf -v launch 'exec bash --noprofile --norc -c %q' "$command"
  if ((DRY_RUN)); then
    print_command tmux new-session -d -s "$session" -c "$ROOT" 'bash --noprofile --norc'
    print_command tmux set-window-option -t "$session:0" remain-on-exit on
    print_command tmux send-keys -t "$session:0" -l "$launch"
    print_command tmux send-keys -t "$session:0" C-m
    return
  fi
  tmux new-session -d -s "$session" -c "$ROOT" -e "PATH=$PATH" 'bash --noprofile --norc' 9>&-
  tmux set-window-option -t "$session:0" remain-on-exit on
  printf -v command 'exec bash --noprofile --norc -c %q' "$command"
  tmux send-keys -t "$session:0" -l "$command"
  tmux send-keys -t "$session:0" C-m
}
if ((DRY_RUN)); then
  status "DRY RUN: commands only; no tmux, HTTP, GPU, log or trajectory operations"
  status "Preflight refuses active e00, e01a, e01b, chain or cache-pair-run sessions; takes an exclusive lock"
  for cache in on off on; do
    serve_command "$cache"
    label="server-$cache"
    [[ ${RESTORING:-0} == 0 ]] || label=server-restored-on
    launch_session vllm "$label" "${SERVE_CMD[@]}"
    print_command "${CONTROL[@]}" server "${VERIFY_ARGS[@]}" --cache "$cache"
    print_command "${CONTROL[@]}" owned --port "$PORT"
    if [[ "$cache" == on && ${RESTORING:-0} == 1 ]]; then break; fi
    print_command "${CONTROL[@]}" runs --config "$CONFIG" --endpoint "$ENDPOINT" --cache "$cache" --existing
    run_command "$cache"
    launch_session "$RUN_SESSION" "experiment-$cache" "${RUN_CMD[@]}"
    print_command "${CONTROL[@]}" runs --config "$CONFIG" --endpoint "$ENDPOINT" --cache "$cache"
    print_command tmux kill-session -t "=$RUN_SESSION"
    print_command tmux send-keys -t vllm:0 C-c
    print_command "${CONTROL[@]}" released --port "$PORT"
    print_command tmux kill-session -t '=vllm'
    [[ "$cache" != off ]] || RESTORING=1
  done
  status "Logs on a real run: $RUN_DIR; an existing cache-on server is reused only after verification"
  exit 0
fi

PHASE=preflight
for tool in tmux ss nvidia-smi curl flock tee; do command -v "$tool" >/dev/null || fail "missing tool $tool"; done
for session in e00 e01a e01b chain "$RUN_SESSION"; do
  if tmux has-session -t "=$session" 2>/dev/null; then fail "active session $session"; fi
done
"${CONTROL[@]}" idle
for cache in on off; do
  "${CONTROL[@]}" runs --config "$CONFIG" --endpoint "$ENDPOINT" --cache "$cache" --existing
done
mkdir -p logs
exec 9>logs/cache-pair.lock
flock -n 9 || fail "another cache pair is running"
mkdir -p "$RUN_DIR"
status "Experiment=$EXPERIMENT; model=$MODEL; logs=$RUN_DIR"

has_server() { tmux has-session -t '=vllm' 2>/dev/null; }

wait_ready() {
  local cache=$1 label=$2 deadline=$((SECONDS + START_TIMEOUT))
  until curl --fail --silent --max-time 3 "http://127.0.0.1:$PORT/health" >/dev/null; do
    [[ ! -f "$RUN_DIR/$label.exit" ]] || fail "server exited before readiness"
    has_server || fail "server session disappeared"
    ((SECONDS < deadline)) || fail "server readiness timed out"
    sleep 2
  done
  "${CONTROL[@]}" server "${VERIFY_ARGS[@]}" --cache "$cache"
  "${CONTROL[@]}" owned --port "$PORT"
  status "Server ready; actual cache=$cache and all serving parameters verified"
}
start_server() {
  local cache=$1 label=$2
  PHASE="start-$cache"
  serve_command "$cache"
  status "Starting cache=$cache server"
  launch_session vllm "$label" "${SERVE_CMD[@]}"
  wait_ready "$cache" "$label"
}
stop_server() {
  PHASE=stop-server
  "${CONTROL[@]}" owned --port "$PORT"
  status "Sending SIGINT to the managed server; no forced kill"
  tmux send-keys -t vllm:0 C-c
  local deadline=$((SECONDS + STOP_TIMEOUT))
  until release_checked; do
    ((SECONDS < deadline)) || fail "port/GPU release timed out"
    sleep 2
  done
  if has_server; then tmux kill-session -t '=vllm'; fi
  status "Server stopped; port and GPU compute allocations released"
}
release_checked() {
  local code
  if "${CONTROL[@]}" released --port "$PORT"; then return 0; else code=$?; fi
  [[ "$code" == 1 ]] || fail "release query failed"
  return 1
}
run_experiment() {
  local cache=$1 label="experiment-$1" deadline=$((SECONDS + RUN_TIMEOUT))
  PHASE="experiment-$cache"
  "${CONTROL[@]}" runs --config "$CONFIG" --endpoint "$ENDPOINT" --cache "$cache" --existing
  run_command "$cache"
  status "Starting experiment cache=$cache in $RUN_SESSION"
  launch_session "$RUN_SESSION" "$label" "${RUN_CMD[@]}"
  until [[ -f "$RUN_DIR/$label.exit" ]]; do
    tmux has-session -t "=$RUN_SESSION" 2>/dev/null || fail "experiment session disappeared"
    ((SECONDS < deadline)) || fail "experiment timed out"
    sleep 2
  done
  [[ $(cat "$RUN_DIR/$label.exit") == 0 ]] || fail "experiment process failed"
  "${CONTROL[@]}" runs --config "$CONFIG" --endpoint "$ENDPOINT" --cache "$cache"
  tmux kill-session -t "=$RUN_SESSION"
  status "Experiment cache=$cache completed and verified"
}

if has_server; then
  PHASE=verify-existing-cache-on
  "${CONTROL[@]}" server "${VERIFY_ARGS[@]}" --cache on
  "${CONTROL[@]}" owned --port "$PORT"
  status "Reusing the verified idle cache-on server"
else
  "${CONTROL[@]}" released --port "$PORT" || fail "port or GPU is already in use"
  start_server on server-on
fi
run_experiment on
stop_server
start_server off server-off
run_experiment off
stop_server
start_server on server-restored-on
PHASE=complete
status "Both conditions completed; cache-on service restored in vllm; logs=$RUN_DIR"
