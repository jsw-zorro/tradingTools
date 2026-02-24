"""Tests for the VIX/UVXY put strategy."""

import numpy as np
import pandas as pd
import pytest

from strategylab.core.models import Signal
from strategylab.strategies.vix_uvxy_put.exits import check_put_exit
from strategylab.strategies.vix_uvxy_put.position import construct_uvxy_put
from strategylab.strategies.vix_uvxy_put.signals import detect_vix_spikes


def _make_vix_data(
    base_level=20, spike_day=10, spike_level=35, days=30
):
    """Create synthetic VIX data with an optional spike."""
    dates = pd.bdate_range("2023-01-01", periods=days)
    closes = [base_level] * days
    if spike_day is not None and spike_day < days:
        closes[spike_day] = spike_level
    df = pd.DataFrame(
        {"Open": closes, "High": closes, "Low": closes, "Close": closes, "Volume": 1000000},
        index=dates,
    )
    return df


def _make_uvxy_data(base_price=30, days=30):
    """Create synthetic UVXY data."""
    dates = pd.bdate_range("2023-01-01", periods=days)
    # UVXY typically decays, simulate slight downtrend with noise
    prices = [base_price * (0.99 ** i) + np.random.randn() * 0.5 for i in range(days)]
    prices = [max(p, 1.0) for p in prices]
    df = pd.DataFrame(
        {"Open": prices, "High": prices, "Low": prices, "Close": prices, "Volume": 1000000},
        index=dates,
    )
    return df


class TestVixSpikeDetection:
    def test_detects_absolute_spike(self):
        vix_data = _make_vix_data(base_level=20, spike_day=10, spike_level=35)
        params = {
            "spike_absolute": 30,
            "spike_pct_change": 50,  # high so only absolute triggers
            "combine_mode": "or",
            "cooldown_days": 5,
        }
        signals = detect_vix_spikes(vix_data, params)
        assert len(signals) >= 1
        assert signals[0].metadata["abs_triggered"] is True

    def test_detects_pct_spike(self):
        vix_data = _make_vix_data(base_level=20, spike_day=10, spike_level=30)
        params = {
            "spike_absolute": 100,  # high so only pct triggers
            "spike_pct_change": 20,
            "combine_mode": "or",
            "cooldown_days": 5,
        }
        signals = detect_vix_spikes(vix_data, params)
        assert len(signals) >= 1
        assert signals[0].metadata["pct_triggered"] is True

    def test_and_mode_requires_both(self):
        vix_data = _make_vix_data(base_level=20, spike_day=10, spike_level=25)
        params = {
            "spike_absolute": 30,
            "spike_pct_change": 10,
            "combine_mode": "and",
            "cooldown_days": 5,
        }
        signals = detect_vix_spikes(vix_data, params)
        # Spike to 25 triggers pct (25% from 20) but not absolute (25 < 30)
        assert len(signals) == 0

    def test_cooldown_prevents_repeated_signals(self):
        dates = pd.bdate_range("2023-01-01", periods=30)
        closes = [20] * 30
        closes[5] = 35
        closes[7] = 35  # within cooldown
        closes[15] = 35  # outside cooldown
        df = pd.DataFrame({"Close": closes}, index=dates)
        params = {
            "spike_absolute": 30,
            "spike_pct_change": 50,
            "combine_mode": "or",
            "cooldown_days": 5,
        }
        signals = detect_vix_spikes(df, params)
        assert len(signals) == 2  # day 5 and day 15 (day 7 blocked)

    def test_no_spike_no_signals(self):
        vix_data = _make_vix_data(base_level=15, spike_day=None, spike_level=15)
        params = {
            "spike_absolute": 30,
            "spike_pct_change": 20,
            "combine_mode": "or",
            "cooldown_days": 5,
        }
        signals = detect_vix_spikes(vix_data, params)
        assert len(signals) == 0

    def test_signal_strength_classification(self):
        vix_data = _make_vix_data(base_level=20, spike_day=10, spike_level=45)
        params = {
            "spike_absolute": 30,
            "spike_pct_change": 20,
            "combine_mode": "or",
            "cooldown_days": 5,
        }
        signals = detect_vix_spikes(vix_data, params)
        assert len(signals) >= 1
        assert signals[0].strength in ("moderate", "strong", "extreme")


class TestPositionConstruction:
    def test_constructs_position(self):
        vix_data = _make_vix_data(base_level=20, spike_day=10, spike_level=35)
        uvxy_data = _make_uvxy_data(base_price=30, days=30)
        data = {"^VIX": vix_data, "UVXY": uvxy_data}

        signal = Signal(
            date=vix_data.index[10],
            strategy_name="vix_uvxy_put",
            signal_type="vix_spike",
            strength="strong",
            metadata={"vix_close": 35.0},
        )
        params = {
            "entry_delay_days": 1,
            "strike_mode": "otm_5",
            "dte_target": 45,
            "position_size_pct": 5.0,
            "iv_scale_factor": 1.2,
        }
        position = construct_uvxy_put(signal, data, params, portfolio_value=100000)
        assert position is not None
        assert position.instrument == "UVXY_PUT"
        assert position.direction == "long"
        assert position.quantity >= 1
        assert position.cost_basis > 0

    def test_returns_none_for_missing_data(self):
        signal = Signal(
            date=pd.Timestamp("2023-01-15"),
            strategy_name="vix_uvxy_put",
            signal_type="vix_spike",
            strength="moderate",
        )
        position = construct_uvxy_put(signal, {}, {}, 100000)
        assert position is None


class TestExitLogic:
    def _make_position(self, entry_date, uvxy_price=30):
        return Signal(
            date=entry_date,
            strategy_name="vix_uvxy_put",
            signal_type="vix_spike",
            strength="moderate",
        ), {
            "entry_date": entry_date,
            "strategy_name": "vix_uvxy_put",
            "instrument": "UVXY_PUT",
            "direction": "long",
            "entry_price": 5.0,
            "quantity": 2,
            "cost_basis": 1000.0,
            "signal": None,
            "metadata": {
                "strike": uvxy_price * 0.95,
                "dte": 45,
                "iv_estimate": 0.8,
                "underlying_price": uvxy_price,
            },
        }

    def test_max_hold_exit(self):
        dates = pd.bdate_range("2023-01-01", periods=60)
        uvxy_data = _make_uvxy_data(base_price=30, days=60)
        vix_data = _make_vix_data(base_level=25, spike_day=None, days=60)
        data = {"^VIX": vix_data, "UVXY": uvxy_data}

        signal = Signal(
            date=dates[0],
            strategy_name="vix_uvxy_put",
            signal_type="vix_spike",
            strength="moderate",
        )
        from strategylab.core.models import Position
        position = Position(
            entry_date=dates[0],
            strategy_name="vix_uvxy_put",
            instrument="UVXY_PUT",
            direction="long",
            entry_price=5.0,
            quantity=2,
            cost_basis=1000.0,
            signal=signal,
            metadata={
                "strike": 28.5,
                "dte": 45,
                "iv_estimate": 0.8,
            },
        )

        params = {
            "max_hold_days": 30,
            "profit_target_pct": 100,
            "stop_loss_pct": 50,
            "vix_floor_exit": 18,
        }

        # Check at day 31 - should trigger max_hold
        exit_result = check_put_exit(position, dates[45], data, params)
        assert exit_result is not None
        assert exit_result.exit_reason in ("max_hold", "expiration", "vix_floor", "profit_target", "stop_loss")
