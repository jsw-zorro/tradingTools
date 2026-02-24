"""Tests for Black-Scholes options pricing."""

import numpy as np
import pytest

from strategylab.data.options import (
    black_scholes_call,
    black_scholes_put,
    estimate_iv,
    put_delta,
)


class TestBlackScholesPut:
    def test_atm_put_has_positive_value(self):
        price = black_scholes_put(S=100, K=100, T=0.25, r=0.05, sigma=0.3)
        assert price > 0

    def test_deep_itm_put(self):
        price = black_scholes_put(S=50, K=100, T=0.25, r=0.05, sigma=0.3)
        assert price > 45  # deep ITM, close to intrinsic

    def test_deep_otm_put(self):
        price = black_scholes_put(S=200, K=100, T=0.25, r=0.05, sigma=0.3)
        assert price < 1  # very unlikely to be ITM

    def test_expired_put_itm(self):
        price = black_scholes_put(S=90, K=100, T=0, r=0.05, sigma=0.3)
        assert price == pytest.approx(10, abs=0.01)

    def test_expired_put_otm(self):
        price = black_scholes_put(S=110, K=100, T=0, r=0.05, sigma=0.3)
        assert price == 0.0

    def test_higher_vol_increases_price(self):
        low_vol = black_scholes_put(S=100, K=100, T=0.25, r=0.05, sigma=0.2)
        high_vol = black_scholes_put(S=100, K=100, T=0.25, r=0.05, sigma=0.5)
        assert high_vol > low_vol

    def test_longer_tte_increases_price(self):
        short = black_scholes_put(S=100, K=100, T=0.1, r=0.05, sigma=0.3)
        long = black_scholes_put(S=100, K=100, T=1.0, r=0.05, sigma=0.3)
        assert long > short


class TestBlackScholesCall:
    def test_atm_call_has_positive_value(self):
        price = black_scholes_call(S=100, K=100, T=0.25, r=0.05, sigma=0.3)
        assert price > 0

    def test_put_call_parity(self):
        S, K, T, r, sigma = 100, 100, 0.5, 0.05, 0.3
        call = black_scholes_call(S, K, T, r, sigma)
        put = black_scholes_put(S, K, T, r, sigma)
        # C - P = S - K*exp(-rT)
        lhs = call - put
        rhs = S - K * np.exp(-r * T)
        assert lhs == pytest.approx(rhs, abs=0.01)


class TestEstimateIV:
    def test_returns_positive(self):
        prices = np.array([100 + np.random.randn() * 2 for _ in range(60)])
        prices = np.abs(prices)  # ensure positive
        iv = estimate_iv(prices, window=30)
        assert iv > 0

    def test_short_series_fallback(self):
        iv = estimate_iv(np.array([100.0]), window=30)
        assert iv == 0.5  # fallback

    def test_fewer_than_window(self):
        prices = np.array([100.0, 101.0, 99.0, 102.0])
        iv = estimate_iv(prices, window=30)
        assert iv > 0


class TestPutDelta:
    def test_atm_put_delta(self):
        delta = put_delta(S=100, K=100, T=0.25, r=0.05, sigma=0.3)
        assert -0.6 < delta < -0.3  # ATM put delta near -0.5

    def test_deep_itm_put_delta(self):
        delta = put_delta(S=50, K=100, T=0.25, r=0.05, sigma=0.3)
        assert delta < -0.9

    def test_deep_otm_put_delta(self):
        delta = put_delta(S=200, K=100, T=0.25, r=0.05, sigma=0.3)
        assert delta > -0.1
