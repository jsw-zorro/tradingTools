"""YAML config loader for StrategyLab."""

from pathlib import Path

import yaml

_CONFIG_DIR = Path(__file__).resolve().parent.parent.parent / "config"


def _load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def load_settings() -> dict:
    """Load global settings from config/settings.yaml."""
    return _load_yaml(_CONFIG_DIR / "settings.yaml")


def load_strategy_config(strategy_name: str) -> dict:
    """Load strategy-specific config from config/strategies/<name>.yaml."""
    path = _CONFIG_DIR / "strategies" / f"{strategy_name}.yaml"
    if not path.exists():
        return {}
    return _load_yaml(path)


def get_config_dir() -> Path:
    return _CONFIG_DIR
