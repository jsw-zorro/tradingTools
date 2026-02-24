"""Tests for the backtest engine."""

import pandas as pd
import pytest

from strategylab.backtest.engine import BacktestResult, run_backtest
from strategylab.backtest.metrics import calculate_metrics
from strategylab.core.base_strategy import BaseStrategy
from strategylab.core.models import ExitResult, Position, Signal


class MockStrategy(BaseStrategy):
    """Simple mock strategy for testing the engine."""

    name = "mock"
    description = "Mock strategy for testing"
    required_tickers = ["TEST"]

    def __init__(self, signals=None, exit_after_days=5):
        self._signals = signals or []
        self._exit_after_days = exit_after_days

    def detect_signals(self, data, params):
        return self._signals

    def construct_position(self, signal, data, params, portfolio_value):
        return Position(
            entry_date=signal.date,
            strategy_name=self.name,
            instrument="TEST_OPTION",
            direction="long",
            entry_price=5.0,
            quantity=2,
            cost_basis=1000.0,
            signal=signal,
            metadata={"strike": 100, "dte": 30},
        )

    def check_exit(self, position, current_date, data, params):
        days_held = (current_date - position.entry_date).days
        if days_held >= self._exit_after_days:
            return ExitResult(
                exit_date=current_date,
                exit_reason="max_hold",
                exit_price=7.0,
                pnl=400.0,
                pnl_pct=40.0,
                days_held=days_held,
            )
        return None

    def format_alert(self, signal, recommendation):
        return {"subject": "test", "body_text": "test", "body_html": ""}

    def get_param_grid(self):
        return {"exit_days": [3, 5, 7]}

    def get_default_params(self):
        return {"exit_days": 5, "max_open_positions": 3}


def _make_data(start="2023-01-01", periods=60):
    dates = pd.bdate_range(start, periods=periods)
    df = pd.DataFrame(
        {"Open": 100, "High": 105, "Low": 95, "Close": 100, "Volume": 1000000},
        index=dates,
    )
    return {"TEST": df}


class TestBacktestEngine:
    def test_no_signals_no_trades(self):
        data = _make_data()
        strategy = MockStrategy(signals=[])
        result = run_backtest(strategy, data, {}, initial_capital=100000)
        assert result.num_trades == 0
        assert result.final_equity == pytest.approx(100000, abs=1)

    def test_single_trade(self):
        data = _make_data()
        dates = list(data["TEST"].index)
        signal = Signal(
            date=dates[5],
            strategy_name="mock",
            signal_type="test",
            strength="moderate",
        )
        strategy = MockStrategy(signals=[signal], exit_after_days=5)
        result = run_backtest(strategy, data, {}, initial_capital=100000)
        assert result.num_trades == 1
        assert result.trades[0].exit.exit_reason == "max_hold"

    def test_multiple_trades(self):
        data = _make_data(periods=120)
        dates = list(data["TEST"].index)
        signals = [
            Signal(date=dates[5], strategy_name="mock", signal_type="test", strength="moderate"),
            Signal(date=dates[30], strategy_name="mock", signal_type="test", strength="strong"),
            Signal(date=dates[60], strategy_name="mock", signal_type="test", strength="extreme"),
        ]
        strategy = MockStrategy(signals=signals, exit_after_days=10)
        result = run_backtest(strategy, data, {}, initial_capital=100000)
        assert result.num_trades == 3

    def test_equity_curve_length(self):
        data = _make_data(periods=30)
        strategy = MockStrategy(signals=[])
        result = run_backtest(strategy, data, {})
        assert len(result.equity_curve) == 30


class TestMetrics:
    def test_empty_result(self):
        result = BacktestResult("test", {}, [], pd.Series(dtype=float), 100000)
        metrics = calculate_metrics(result)
        assert metrics["num_trades"] == 0
        assert metrics["win_rate"] == 0

    def test_winning_trades(self):
        data = _make_data(periods=60)
        dates = list(data["TEST"].index)
        signal = Signal(date=dates[5], strategy_name="mock", signal_type="test", strength="moderate")
        strategy = MockStrategy(signals=[signal], exit_after_days=5)
        result = run_backtest(strategy, data, {}, initial_capital=100000)
        metrics = calculate_metrics(result)
        assert metrics["num_trades"] == 1
        assert "composite_score" in metrics
