"""Black-Scholes pricing helpers for options estimation."""

import numpy as np
from scipy.stats import norm


def black_scholes_put(
    S: float, K: float, T: float, r: float, sigma: float
) -> float:
    """Calculate Black-Scholes price for a European put option.

    Args:
        S: Current underlying price
        K: Strike price
        T: Time to expiration in years
        r: Risk-free interest rate
        sigma: Implied volatility (annualized)

    Returns:
        Put option price
    """
    if T <= 0 or sigma <= 0 or S <= 0:
        return max(K - S, 0.0)

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return float(max(put_price, 0.0))


def black_scholes_call(
    S: float, K: float, T: float, r: float, sigma: float
) -> float:
    """Calculate Black-Scholes price for a European call option."""
    if T <= 0 or sigma <= 0 or S <= 0:
        return max(S - K, 0.0)

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return float(max(call_price, 0.0))


def estimate_iv(
    prices: np.ndarray, window: int = 30, annualization: float = 252.0
) -> float:
    """Estimate implied volatility from historical prices as rolling realized vol.

    Args:
        prices: Array of closing prices
        window: Rolling window in trading days
        annualization: Trading days per year

    Returns:
        Annualized volatility estimate
    """
    if len(prices) < window + 1:
        if len(prices) < 2:
            return 0.5  # fallback
        window = len(prices) - 1

    log_returns = np.diff(np.log(prices[-window - 1 :]))
    return float(np.std(log_returns) * np.sqrt(annualization))


def put_delta(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Calculate put option delta."""
    if T <= 0 or sigma <= 0 or S <= 0:
        return -1.0 if S < K else 0.0

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return float(norm.cdf(d1) - 1.0)
