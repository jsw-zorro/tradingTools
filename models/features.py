"""Feature engineering from 60-day OHLCV windows."""

import logging

import numpy as np
import pandas as pd

from config import SECTORS, TICKER_SECTORS

logger = logging.getLogger(__name__)


def _safe_div(a: float, b: float) -> float:
    """Division that returns 0.0 on zero/nan denominator."""
    if b == 0 or np.isnan(b):
        return 0.0
    return a / b


def compute_features_for_sample(sample: dict) -> dict[str, float]:
    """Extract features from a sample's 60-day window.

    All price-level features are expressed as % of current price or as
    z-scores so multi-stock training works.
    """
    window: pd.DataFrame = sample["window"]
    context: dict[str, pd.DataFrame] = sample["context"]
    ticker: str = sample["ticker"]

    close = window["Close"].values
    high = window["High"].values
    low = window["Low"].values
    opn = window["Open"].values
    volume = window["Volume"].values

    if len(close) < 20:
        logger.warning(f"Window too short ({len(close)}) for {ticker}")
        return {}

    current_price = close[-1]
    returns = np.diff(np.log(close))  # log returns

    features: dict[str, float] = {}

    # --- Price returns ---
    for n, label in [(1, "1d"), (5, "5d"), (10, "10d"), (20, "20d"), (50, "50d")]:
        if len(close) > n:
            features[f"return_{label}"] = (close[-1] - close[-1 - n]) / close[-1 - n]
        else:
            features[f"return_{label}"] = 0.0

    # --- Distance from SMAs (as % of price) ---
    for n in [5, 10, 20, 50]:
        if len(close) >= n:
            sma = np.mean(close[-n:])
            features[f"dist_sma_{n}"] = _safe_div(current_price - sma, current_price)
        else:
            features[f"dist_sma_{n}"] = 0.0

    # --- SMA crossover ---
    sma5 = np.mean(close[-5:])
    sma20 = np.mean(close[-20:])
    features["sma_5_20_cross"] = 1.0 if sma5 > sma20 else 0.0

    # --- Price position in full window range ---
    high_all = np.max(high)
    low_all = np.min(low)
    rng = high_all - low_all
    features["price_in_range"] = _safe_div(current_price - low_all, rng) if rng > 0 else 0.5

    # --- Volatility ---
    for n, label in [(5, "5d"), (10, "10d"), (20, "20d"), (50, "50d")]:
        if len(returns) >= n:
            features[f"vol_{label}"] = np.std(returns[-n:])
        else:
            features[f"vol_{label}"] = 0.0

    # ATR (14-day)
    n_atr = min(14, len(close) - 1)
    tr = np.maximum(
        high[-n_atr:] - low[-n_atr:],
        np.maximum(
            np.abs(high[-n_atr:] - close[-n_atr - 1 : -1]),
            np.abs(low[-n_atr:] - close[-n_atr - 1 : -1]),
        ),
    )
    features["atr_14"] = _safe_div(np.mean(tr), current_price)

    # Bollinger Band %B
    sma20_val = np.mean(close[-20:])
    std20 = np.std(close[-20:])
    upper = sma20_val + 2 * std20
    lower = sma20_val - 2 * std20
    bb_range = upper - lower
    features["bb_pctb"] = _safe_div(current_price - lower, bb_range) if bb_range > 0 else 0.5

    # Garman-Klass volatility
    n_gk = min(20, len(close))
    gk_terms = (
        0.5 * np.log(high[-n_gk:] / low[-n_gk:]) ** 2
        - (2 * np.log(2) - 1) * np.log(close[-n_gk:] / opn[-n_gk:]) ** 2
    )
    features["gk_vol"] = float(np.sqrt(np.mean(gk_terms) * 252))

    # --- Momentum ---
    # RSI (14-day)
    n_rsi = min(14, len(returns))
    gains = np.where(returns[-n_rsi:] > 0, returns[-n_rsi:], 0)
    losses = np.where(returns[-n_rsi:] < 0, -returns[-n_rsi:], 0)
    avg_gain = np.mean(gains) if len(gains) > 0 else 0
    avg_loss = np.mean(losses) if len(losses) > 0 else 1e-10
    rs = _safe_div(avg_gain, avg_loss)
    features["rsi_14"] = 100 - 100 / (1 + rs)

    # MACD (12, 26, 9)
    if len(close) >= 26:
        ema12 = pd.Series(close).ewm(span=12, adjust=False).mean().values
        ema26 = pd.Series(close).ewm(span=26, adjust=False).mean().values
        macd_line = ema12 - ema26
        signal_line = pd.Series(macd_line).ewm(span=9, adjust=False).mean().values
        features["macd_signal"] = _safe_div(macd_line[-1], current_price)
        features["macd_hist"] = _safe_div(macd_line[-1] - signal_line[-1], current_price)
    else:
        features["macd_signal"] = 0.0
        features["macd_hist"] = 0.0

    # Rate of change
    for n, label in [(5, "5d"), (10, "10d")]:
        if len(close) > n:
            features[f"roc_{label}"] = (close[-1] - close[-1 - n]) / close[-1 - n]
        else:
            features[f"roc_{label}"] = 0.0

    # --- Volume ---
    avg_vol_20 = np.mean(volume[-20:]) if len(volume) >= 20 else np.mean(volume)
    features["rel_volume"] = _safe_div(volume[-1], avg_vol_20) if avg_vol_20 > 0 else 1.0

    # Volume trend (5-day linear slope)
    if len(volume) >= 5:
        x = np.arange(5, dtype=float)
        vol_5 = volume[-5:].astype(float)
        if np.std(vol_5) > 0:
            slope = np.polyfit(x, vol_5 / np.mean(vol_5), 1)[0]
            features["vol_trend_5d"] = float(slope)
        else:
            features["vol_trend_5d"] = 0.0
    else:
        features["vol_trend_5d"] = 0.0

    # OBV 5-day change (normalized)
    if len(close) >= 6:
        obv_sign = np.sign(np.diff(close[-6:]))
        obv_delta = np.sum(obv_sign * volume[-5:])
        features["obv_5d_change"] = _safe_div(obv_delta, avg_vol_20) if avg_vol_20 > 0 else 0.0
    else:
        features["obv_5d_change"] = 0.0

    # --- Context (SPY, VIX) ---
    spy_data = context.get("SPY")
    if spy_data is not None and len(spy_data) >= 5:
        spy_close = spy_data["Close"].values
        features["spy_return_5d"] = (spy_close[-1] - spy_close[-5]) / spy_close[-5]
        if len(spy_close) >= 20:
            features["spy_return_20d"] = (spy_close[-1] - spy_close[-20]) / spy_close[-20]
        else:
            features["spy_return_20d"] = 0.0
    else:
        features["spy_return_5d"] = 0.0
        features["spy_return_20d"] = 0.0

    vix_data = context.get("VIX")
    if vix_data is not None and len(vix_data) >= 5:
        vix_close = vix_data["Close"].values
        features["vix_level"] = float(vix_close[-1])
        features["vix_change_5d"] = float(vix_close[-1] - vix_close[-5])
        if len(vix_close) >= 20:
            features["vix_change_20d"] = float(vix_close[-1] - vix_close[-20])
        else:
            features["vix_change_20d"] = 0.0
    else:
        features["vix_level"] = 20.0  # fallback neutral
        features["vix_change_5d"] = 0.0
        features["vix_change_20d"] = 0.0

    # --- Prediction horizon ---
    # days_to_friday tells the model how far out it's predicting (1-5 calendar days)
    features["days_to_friday"] = sample.get("days_to_friday", 0)
    features["cutoff_dow"] = sample.get("cutoff_dow", 0)

    # --- Calendar ---
    friday_date = sample.get("friday_date")
    if friday_date is not None:
        features["week_of_month"] = min((friday_date.day - 1) // 7 + 1, 5)
        features["month"] = friday_date.month
    else:
        # For live inference, use the upcoming Friday
        now = pd.Timestamp.now()
        days_until_friday = (4 - now.dayofweek) % 7
        upcoming_friday = now + pd.Timedelta(days=days_until_friday)
        features["week_of_month"] = min((upcoming_friday.day - 1) // 7 + 1, 5)
        features["month"] = upcoming_friday.month

    # --- Sector one-hot (11 GICS sectors, replaces 100+ ticker one-hot) ---
    ticker_sector = TICKER_SECTORS.get(ticker, "")
    for s in SECTORS:
        features[f"sector_{s}"] = 1.0 if ticker_sector == s else 0.0

    return features


def build_feature_dataframe(
    samples: list[dict],
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Convert list of samples into features DataFrame + target return + base prices.

    Target is the *return* (friday_close / current_price - 1) so the model
    learns relative patterns, not absolute price levels.

    Returns (X, y_return, base_prices) where:
      - X: feature matrix
      - y_return: friday_close / current_price - 1
      - base_prices: current_price at cutoff (for converting back to $)
    """
    rows = []
    targets_return = []
    base_prices = []
    for s in samples:
        feat = compute_features_for_sample(s)
        if not feat:
            continue
        current_price = s["window"]["Close"].values[-1]
        friday_return = (s["target"] / current_price) - 1.0
        rows.append(feat)
        targets_return.append(friday_return)
        base_prices.append(current_price)

    X = pd.DataFrame(rows)
    y = pd.Series(targets_return, name="friday_return")
    bp = pd.Series(base_prices, name="base_price")

    # Fill any remaining NaN with 0
    X = X.fillna(0.0)

    logger.info(f"Feature matrix: {X.shape[0]} samples x {X.shape[1]} features")
    return X, y, bp


def compute_avg_weekly_range(
    samples: list[dict],
) -> dict[str, float]:
    """Compute average weekly (Friday) close range per ticker from training data.

    Used as a normalizer for conformal prediction.
    """
    from collections import defaultdict

    weekly_ranges: dict[str, list[float]] = defaultdict(list)
    for s in samples:
        window = s["window"]
        high_30 = np.max(window["High"].values)
        low_30 = np.min(window["Low"].values)
        weekly_ranges[s["ticker"]].append(high_30 - low_30)

    return {t: float(np.mean(r)) for t, r in weekly_ranges.items()}
