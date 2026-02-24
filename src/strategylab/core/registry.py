"""Strategy plugin registry."""

from strategylab.core.base_strategy import BaseStrategy

_STRATEGIES: dict[str, type[BaseStrategy]] = {}


def register(cls: type[BaseStrategy]) -> type[BaseStrategy]:
    """Class decorator that registers a strategy by its .name attribute."""
    instance = cls()
    _STRATEGIES[instance.name] = cls
    return cls


def get_strategy(name: str) -> BaseStrategy:
    """Instantiate and return a strategy by name."""
    if name not in _STRATEGIES:
        raise KeyError(f"Strategy '{name}' not found. Available: {list(_STRATEGIES)}")
    return _STRATEGIES[name]()


def get_all_strategies() -> dict[str, BaseStrategy]:
    """Return instances of all registered strategies."""
    return {name: cls() for name, cls in _STRATEGIES.items()}


def list_strategy_names() -> list[str]:
    """Return names of all registered strategies."""
    return list(_STRATEGIES.keys())
