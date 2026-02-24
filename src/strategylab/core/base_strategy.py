"""Abstract base class for all trading strategies."""

from abc import ABC, abstractmethod

import pandas as pd

from strategylab.core.models import ExitResult, Position, Signal


class BaseStrategy(ABC):
    """Every strategy plugin must implement this interface."""

    name: str
    description: str
    required_tickers: list[str]

    @abstractmethod
    def detect_signals(
        self, data: dict[str, pd.DataFrame], params: dict
    ) -> list[Signal]:
        """Scan data and return any entry signals."""
        ...

    @abstractmethod
    def construct_position(
        self,
        signal: Signal,
        data: dict[str, pd.DataFrame],
        params: dict,
        portfolio_value: float,
    ) -> Position | None:
        """Build a Position from a signal, or None if conditions aren't met."""
        ...

    @abstractmethod
    def check_exit(
        self,
        position: Position,
        current_date: pd.Timestamp,
        data: dict[str, pd.DataFrame],
        params: dict,
    ) -> ExitResult | None:
        """Return an ExitResult if the position should be closed, else None."""
        ...

    @abstractmethod
    def format_alert(self, signal: Signal, recommendation: dict) -> dict:
        """Return {subject, body_text, body_html} for an email alert."""
        ...

    @abstractmethod
    def get_param_grid(self) -> dict[str, list]:
        """Return the parameter sweep grid for this strategy."""
        ...

    @abstractmethod
    def get_default_params(self) -> dict:
        """Return the default parameter set."""
        ...

    def get_real_time_check_interval(self) -> int:
        """Seconds between live checks. Override per strategy."""
        return 300
