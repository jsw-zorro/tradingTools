"""Data fetching and sample construction for Friday close prediction."""

import logging
from datetime import timedelta

import numpy as np
import pandas as pd
import yfinance as yf

from config import (
    ALL_TICKERS,
    CAL_FRAC,
    CONTEXT_TICKERS,
    CUTOFF_DOWS,
    DATA_END,
    DATA_START,
    LOOKBACK_DAYS,
    TARGET_TICKERS,
    TEST_FRAC,
    TRAIN_FRAC,
)

logger = logging.getLogger(__name__)


def fetch_all_data(
    start: str = DATA_START,
    end: str = DATA_END,
) -> dict[str, pd.DataFrame]:
    """Fetch daily OHLCV for all tickers via yfinance.

    Returns dict mapping ticker -> DataFrame with columns
    [Open, High, Low, Close, Volume] indexed by date.
    """
    data = {}
    for ticker in ALL_TICKERS:
        logger.info(f"Fetching {ticker} ...")
        df = yf.download(ticker, start=start, end=end, progress=False)
        if df.empty:
            logger.warning(f"No data for {ticker}, skipping")
            continue
        # yfinance may return MultiIndex columns; flatten
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        data[ticker] = df
    return data


def _get_fridays(df: pd.DataFrame) -> pd.DatetimeIndex:
    """Return dates that are Fridays in the dataframe index."""
    return df.index[df.index.dayofweek == 4]


def build_samples(
    data: dict[str, pd.DataFrame],
) -> list[dict]:
    """Build training samples: one per (Friday, cutoff_day) per target ticker.

    For each Friday, generates up to 4 samples with cutoff on Mon/Tue/Wed/Thu.
    The model learns to predict from any day of the week via the
    ``days_to_friday`` feature.

    Each sample contains:
      - ticker: str
      - friday_date: pd.Timestamp
      - cutoff_dow: int  (0=Mon … 3=Thu)
      - days_to_friday: int  (calendar days from cutoff to Friday)
      - target: float (Friday close price)
      - window: pd.DataFrame  (30-day OHLCV ending at the cutoff day)
      - context: dict[str, pd.DataFrame]  (SPY/VIX aligned to window)
    """
    samples = []
    spy_df = data.get("SPY")
    vix_df = data.get("^VIX")

    for ticker in TARGET_TICKERS:
        df = data.get(ticker)
        if df is None:
            continue
        fridays = _get_fridays(df)

        for friday in fridays:
            week_start = friday - timedelta(days=friday.dayofweek)  # Monday

            for cutoff_dow in CUTOFF_DOWS:
                target_calendar_day = week_start + timedelta(days=cutoff_dow)

                # Find the actual last trading day on or before the target day
                available = df.index[df.index <= target_calendar_day]
                if len(available) < LOOKBACK_DAYS:
                    continue
                cutoff_date = available[-1]

                # Extract 30-day window ending at cutoff
                cutoff_idx = df.index.get_loc(cutoff_date)
                if cutoff_idx < LOOKBACK_DAYS - 1:
                    continue
                window_start_idx = cutoff_idx - LOOKBACK_DAYS + 1
                window = df.iloc[window_start_idx : cutoff_idx + 1].copy()

                # Context windows (SPY, VIX) aligned to same dates
                context = {}
                for ctx_ticker, ctx_df in [("SPY", spy_df), ("VIX", vix_df)]:
                    if ctx_df is None:
                        continue
                    ctx_window = ctx_df.loc[
                        (ctx_df.index >= window.index[0])
                        & (ctx_df.index <= window.index[-1])
                    ]
                    context[ctx_ticker] = ctx_window

                days_to_friday = (friday - cutoff_date).days

                samples.append(
                    {
                        "ticker": ticker,
                        "friday_date": friday,
                        "cutoff_dow": cutoff_dow,
                        "days_to_friday": days_to_friday,
                        "target": float(df.loc[friday, "Close"]),
                        "window": window,
                        "context": context,
                    }
                )

    # Sort by date for walk-forward splitting
    samples.sort(key=lambda s: (s["friday_date"], s["cutoff_dow"]))
    logger.info(
        f"Built {len(samples)} samples "
        f"({len(samples) // len(CUTOFF_DOWS):.0f} Fridays x {len(CUTOFF_DOWS)} cutoffs) "
        f"across {len(TARGET_TICKERS)} tickers"
    )
    return samples


def walk_forward_split(
    samples: list[dict],
) -> tuple[list[dict], list[dict], list[dict]]:
    """Split samples into train/calibration/test by time order."""
    n = len(samples)
    train_end = int(n * TRAIN_FRAC)
    cal_end = int(n * (TRAIN_FRAC + CAL_FRAC))

    train = samples[:train_end]
    cal = samples[train_end:cal_end]
    test = samples[cal_end:]

    logger.info(
        f"Split: train={len(train)}, calibration={len(cal)}, test={len(test)}"
    )
    return train, cal, test


def fetch_recent_data(
    ticker: str,
    lookback: int = LOOKBACK_DAYS,
) -> dict:
    """Fetch recent data for live inference.

    Returns dict with 'window' (ticker OHLCV), 'context' (SPY/VIX),
    and 'ticker'.
    """
    end = pd.Timestamp.now().normalize() + timedelta(days=1)
    start = end - timedelta(days=lookback * 2)  # fetch extra to ensure enough trading days

    df = yf.download(ticker, start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"), progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index().tail(lookback)

    context = {}
    for ctx_ticker_raw, ctx_ticker_clean in [("SPY", "SPY"), ("^VIX", "VIX")]:
        ctx_df = yf.download(ctx_ticker_raw, start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"), progress=False)
        if isinstance(ctx_df.columns, pd.MultiIndex):
            ctx_df.columns = ctx_df.columns.get_level_values(0)
        ctx_df = ctx_df[["Open", "High", "Low", "Close", "Volume"]].copy()
        ctx_df.index = pd.to_datetime(ctx_df.index)
        ctx_df = ctx_df.loc[
            (ctx_df.index >= df.index[0]) & (ctx_df.index <= df.index[-1])
        ]
        context[ctx_ticker_clean] = ctx_df

    return {"ticker": ticker, "window": df, "context": context}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    data = fetch_all_data()
    for t, df in data.items():
        print(f"{t}: {len(df)} rows, {df.index[0].date()} to {df.index[-1].date()}")
    samples = build_samples(data)
    train, cal, test = walk_forward_split(samples)
    print(f"\nTrain: {train[0]['friday_date'].date()} - {train[-1]['friday_date'].date()}")
    print(f"Cal:   {cal[0]['friday_date'].date()} - {cal[-1]['friday_date'].date()}")
    print(f"Test:  {test[0]['friday_date'].date()} - {test[-1]['friday_date'].date()}")
