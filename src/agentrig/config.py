"""Load YAML configs and resolve ${VAR} / ${VAR:-default} from the environment (.env supported)."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"
_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


@dataclass(frozen=True)
class Endpoint:
    name: str
    platform: str
    base_url: str
    api_key: str
    thinking_control: str


def expand_env(value: str) -> str:
    """Replace ${VAR} and ${VAR:-default}; raise if a required variable is unset."""

    def repl(match: re.Match[str]) -> str:
        var, default = match.group(1), match.group(2)
        resolved = os.environ.get(var) or default
        if resolved is None:
            raise KeyError(f"Environment variable {var} is not set (see .env.example)")
        return resolved

    return _ENV_PATTERN.sub(repl, value)


def _expand(obj: Any) -> Any:
    if isinstance(obj, str):
        return expand_env(obj)
    if isinstance(obj, dict):
        return {k: _expand(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand(v) for v in obj]
    return obj


def load_yaml(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_endpoint(name: str | None = None, config_dir: Path = CONFIG_DIR) -> Endpoint:
    """Pick an endpoint: explicit name > AGENTRIG_ENDPOINT > `default` in endpoints.yaml."""
    load_dotenv()
    raw = load_yaml(config_dir / "endpoints.yaml")
    chosen = name or os.environ.get("AGENTRIG_ENDPOINT") or raw["default"]
    if chosen not in raw["endpoints"]:
        raise KeyError(f"Unknown endpoint {chosen!r}; options: {sorted(raw['endpoints'])}")
    fields = _expand(raw["endpoints"][chosen])
    return Endpoint(name=chosen, **fields)


def load_model(name: str, config_dir: Path = CONFIG_DIR) -> dict[str, Any]:
    """Return one entry from models.yaml."""
    models = load_yaml(config_dir / "models.yaml")["models"]
    if name not in models:
        raise KeyError(f"Unknown model {name!r}; options: {sorted(models)}")
    return models[name]
