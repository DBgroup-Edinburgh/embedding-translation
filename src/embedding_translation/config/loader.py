"""YAML → pydantic loader with layered precedence.

Precedence (highest first):
    1. CLI overrides (caller passes them as a dict to `load`)
    2. project config:  ./.embedding_translation/config.yaml
    3. user config:     ~/.embedding_translation/config.yaml
    4. package defaults (pydantic field defaults)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, TypeVar

import yaml
from loguru import logger
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

USER_CONFIG_PATH = Path.home() / ".embedding_translation" / "config.yaml"
PROJECT_CONFIG_PATH = Path(".") / ".embedding_translation" / "config.yaml"


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r") as f:
        data = yaml.safe_load(f)
    return data or {}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge `override` into `base`. Override wins on scalar keys."""
    out = dict(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load(
    config_cls: type[T],
    explicit_path: Path | str | None = None,
    overrides: dict[str, Any] | None = None,
) -> T:
    """Load a config of type `config_cls` with full precedence applied.

    If `explicit_path` is given, it is treated as the project-level file and
    replaces the default ./.embedding_translation/config.yaml lookup.
    """
    user_data = _read_yaml(USER_CONFIG_PATH)
    project_path = Path(explicit_path) if explicit_path else PROJECT_CONFIG_PATH
    project_data = _read_yaml(project_path)
    override_data = overrides or {}

    merged = _deep_merge(_deep_merge(user_data, project_data), override_data)
    logger.debug(
        f"Loading {config_cls.__name__} — user={USER_CONFIG_PATH.exists()}, "
        f"project={project_path.exists()}, overrides={bool(overrides)}"
    )
    return config_cls.model_validate(merged) if merged else config_cls()


def save(cfg: BaseModel, path: Path | str) -> None:
    """Write a config to YAML."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        yaml.safe_dump(cfg.model_dump(), f, sort_keys=False)
    logger.info(f"Wrote config to {path}")
