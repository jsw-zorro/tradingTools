"""Tests for the unified options chain interface."""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from strategylab.data.options import black_scholes_put
from strategylab.data.options_chain import OptionsChain, clear_chain_cache


@pytest.fixture(autouse=True)
def _clear_cache():
    """Clear chain cache between tests."""
    clear_chain_cache()
    yield
    clear_chain_cache()


def _make_chain_data(underlying="UVXY", days=30, base_price=30):
    """Create synthetic options chain data like QC would provide."""
    dates = pd.bdate_range("2023-01-01", periods=days)
    records = []
    for date in dates:
        underlying_price = base_price * (0.99 ** (dates.get_loc(date)))
        for dte_offset in [30, 45, 60]:
            expiry = date + pd.Timedelta(days=dte_offset)
            for strike_pct in [0.90, 0.95, 1.00, 1.05]:
                strike = round(underlying_price * strike_pct, 1)
                # Rough synthetic price
                T = dte_offset / 365.0
                put_price = black_scholes_put(underlying_price, strike, T, 0.05, 0.8)
                records.append({
                    "date": date,
                    "expiry": expiry,
                    "strike": strike,
                    "right": "put",
                    "open": put_price * 0.98,
                    "high": put_price * 1.05,
                    "low": put_price * 0.95,
                    "close": put_price,
                    "volume": np.random.randint(10, 1000),
                    "open_interest": np.random.randint(100, 5000),
                    "implied_volatility": 0.8 + np.random.randn() * 0.05,
                })
    return pd.DataFrame(records)


class TestOptionsChainFallback:
    """Test that OptionsChain falls back to Black-Scholes when no QC data."""

    def test_no_data_uses_bs(self):
        chain = OptionsChain("FAKE_TICKER")
        # Force loaded state with no data
        chain._loaded = True
        chain._has_real_data = False

        price = chain.get_put_price(
            date=pd.Timestamp("2023-06-15"),
            strike=100,
            underlying_price=100,
            iv=0.3,
            dte=45,
        )
        expected = black_scholes_put(100, 100, 45 / 365, 0.05, 0.3)
        assert price == pytest.approx(expected, rel=0.01)

    def test_has_real_data_flag_false_by_default(self):
        chain = OptionsChain("NONEXISTENT")
        chain._loaded = True
        chain._has_real_data = False
        assert chain.has_real_data is False


class TestOptionsChainWithData:
    """Test OptionsChain with synthetic QC-like data."""

    def _make_chain_with_data(self):
        chain = OptionsChain("UVXY")
        chain._chain_data = _make_chain_data()
        chain._loaded = True
        chain._has_real_data = True
        return chain

    def test_exact_lookup(self):
        chain = self._make_chain_with_data()
        df = chain._chain_data
        # Pick an exact row
        row = df.iloc[0]
        price = chain.get_put_price(
            date=row["date"],
            strike=row["strike"],
            expiry=row["expiry"],
        )
        assert price == pytest.approx(row["close"], rel=0.01)

    def test_nearest_match(self):
        chain = self._make_chain_with_data()
        df = chain._chain_data
        # Use a date that exists but with a slightly off strike
        date = df["date"].iloc[0]
        strike = df["strike"].iloc[0] + 0.001  # tiny offset
        price = chain.get_put_price(
            date=date,
            strike=strike,
            expiry=df["expiry"].iloc[0],
        )
        assert price > 0

    def test_get_chain_snapshot(self):
        chain = self._make_chain_with_data()
        date = chain._chain_data["date"].iloc[0]
        snapshot = chain.get_chain_snapshot(date, "put")
        assert snapshot is not None
        assert len(snapshot) > 0
        assert "strike" in snapshot.columns
        assert "expiry" in snapshot.columns

    def test_find_best_contract(self):
        chain = self._make_chain_with_data()
        date = chain._chain_data["date"].iloc[0]
        contract = chain.find_best_contract(
            date=date,
            target_strike_pct=0.95,
            underlying_price=30,
            target_dte=45,
            right="put",
        )
        assert contract is not None
        assert "strike" in contract
        assert "close" in contract
        assert contract["close"] > 0
        assert contract["dte"] > 0

    def test_get_iv_from_real_data(self):
        chain = self._make_chain_with_data()
        df = chain._chain_data
        row = df.iloc[0]
        iv = chain.get_iv(row["date"], row["strike"], row["expiry"], "put")
        assert iv is not None
        assert iv > 0

    def test_fallback_when_date_not_in_data(self):
        chain = self._make_chain_with_data()
        # Use a date way outside the data range
        price = chain.get_put_price(
            date=pd.Timestamp("2020-01-01"),
            strike=30,
            underlying_price=30,
            iv=0.8,
            dte=45,
        )
        # Should fall back to BS since date is way outside data
        expected = black_scholes_put(30, 30, 45 / 365, 0.05, 0.8)
        assert price == pytest.approx(expected, rel=0.01)


class TestOptionsChainCallPrice:
    def test_call_price_fallback(self):
        chain = OptionsChain("FAKE")
        chain._loaded = True
        chain._has_real_data = False
        price = chain.get_call_price(
            date=pd.Timestamp("2023-06-15"),
            strike=100,
            underlying_price=100,
            iv=0.3,
            dte=45,
        )
        assert price > 0
