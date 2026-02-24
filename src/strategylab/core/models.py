"""Shared data models used by all strategies."""

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class Signal:
    date: pd.Timestamp
    strategy_name: str
    signal_type: str
    strength: str  # "moderate" | "strong" | "extreme"
    metadata: dict = field(default_factory=dict)


@dataclass
class Position:
    entry_date: pd.Timestamp
    strategy_name: str
    instrument: str
    direction: str  # "long" | "short"
    entry_price: float
    quantity: float
    cost_basis: float
    signal: Signal
    metadata: dict = field(default_factory=dict)


@dataclass
class ExitResult:
    exit_date: pd.Timestamp
    exit_reason: str
    exit_price: float
    pnl: float
    pnl_pct: float
    days_held: int


@dataclass
class TradeRecord:
    position: Position
    exit: ExitResult
