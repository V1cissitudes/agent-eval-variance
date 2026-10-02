#!/usr/bin/env bash
# Pull runs/ from the RTX 5070 (WSL2, over Tailscale) to this machine. One-way, never deletes.
# Usage: scripts/sync_runs.sh [--dry-run]
# Needs RUNS_REMOTE_USER, RUNS_REMOTE_HOST, RUNS_REMOTE_PATH (environment or .env).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
if [[ -f "$ROOT/.env" ]]; then
  set -a; source "$ROOT/.env"; set +a
fi

: "${RUNS_REMOTE_USER:?set RUNS_REMOTE_USER (see .env.example)}"
: "${RUNS_REMOTE_HOST:?set RUNS_REMOTE_HOST (see .env.example)}"
: "${RUNS_REMOTE_PATH:?set RUNS_REMOTE_PATH (see .env.example)}"
if [[ "$RUNS_REMOTE_PATH" == "$HOME"* ]]; then
  echo "RUNS_REMOTE_PATH ($RUNS_REMOTE_PATH) looks like a local path; use the absolute path on the 5070" >&2
  exit 1
fi

EXTRA=()
if [[ "${1:-}" == "--dry-run" ]]; then
  EXTRA+=(--dry-run)
fi

mkdir -p "$ROOT/runs"
# No --delete: if something is removed on the 5070, the Mac copy survives.
# ${EXTRA[@]+...}: macOS bash 3.2 treats an empty array as unbound under set -u
rsync -avzh --partial ${EXTRA[@]+"${EXTRA[@]}"} \
  "${RUNS_REMOTE_USER}@${RUNS_REMOTE_HOST}:${RUNS_REMOTE_PATH%/}/" "$ROOT/runs/"
