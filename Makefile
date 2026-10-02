.PHONY: figures test lint format check-endpoint sync

figures:  ## Regenerate results/ and figures/ from analysis/NN_*.py
	uv run --locked python analysis/run_all.py

test:
	uv run --locked pytest

lint:
	uv run --locked ruff check .
	uv run --locked ruff format --check .

format:
	uv run --locked ruff check --fix .
	uv run --locked ruff format .

check-endpoint:
	uv run --locked python scripts/check_endpoint.py

sync:
	scripts/sync_runs.sh
