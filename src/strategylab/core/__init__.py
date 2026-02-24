"""Core abstractions for StrategyLab."""

from strategylab.core.base_strategy import BaseStrategy
from strategylab.core.models import ExitResult, Position, Signal, TradeRecord
from strategylab.core.registry import get_all_strategies, get_strategy, list_strategy_names, register

__all__ = [
    "BaseStrategy",
    "Signal",
    "Position",
    "ExitResult",
    "TradeRecord",
    "register",
    "get_strategy",
    "get_all_strategies",
    "list_strategy_names",
]
