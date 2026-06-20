"""Configuration loader for the underwriting rules engine.

Provides utilities to load YAML-based configuration, resolve environment
variable overrides, and expose sensible defaults for LLM and filesystem
paths.
"""

import os
from pathlib import Path
from typing import Any, Dict

import yaml


def get_env(key: str, default: str = "") -> str:
    """Return the value of an environment variable, falling back to *default*.

    Args:
        key: Environment variable name.
        default: Fallback value when the variable is unset or empty.

    Returns:
        The resolved string value.
    """
    return os.environ.get(key, default)


def load_config(config_path: str = "./config.yaml") -> Dict[str, Any]:
    """Load configuration from a YAML file and apply environment overrides.

    The loader reads *config_path* (default ``./config.yaml``) and returns
    the parsed dictionary.  Any top-level key that contains ``BASE_URL``,
    ``API_KEY``, ``MODEL``, or ``PATH`` substrings (case-insensitive) will be
    overridden by a matching environment variable when present.

    Args:
        config_path: Filesystem path to the YAML configuration file.

    Returns:
        A dictionary containing the merged configuration.

    Raises:
        FileNotFoundError: When *config_path* does not exist.
        yaml.YAMLError: When the file cannot be parsed as YAML.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with path.open("r", encoding="utf-8") as fh:
        config: Dict[str, Any] = yaml.safe_load(fh) or {}

    _apply_env_overrides(config)
    return config


def _apply_env_overrides(config: Dict[str, Any]) -> None:
    """Mutate *config* in-place with environment variable overrides.

    Scans every value in the top-level ``llm`` and ``paths`` sections (and
    any nested keys) for substrings that indicate an environment override
    candidate.
    """
    override_keys = {"base_url", "api_key", "model", "path"}

    def _walk(d: Dict[str, Any]) -> None:
        for key, value in d.items():
            key_lower = key.lower()
            if isinstance(value, str):
                for override in override_keys:
                    if override in key_lower:
                        env_val = get_env(key.upper(), "")
                        if env_val:
                            d[key] = env_val
                        break
            elif isinstance(value, dict):
                _walk(value)

    _walk(config)
