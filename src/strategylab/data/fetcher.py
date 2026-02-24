"""yfinance wrappers for fetching market data."""

import logging

import pandas as pd
import yfinance as yf

from strategylab.data.cache import load_cached, save_cached

logger = logging.getLogger(__name__)


def fetch_ticker(
    ticker: str,
    start: str | None = None,
    end: str | None = None,
    period: str = "5y",
    use_cache: bool = True,
) -> pd.DataFrame:
    """Fetch OHLCV data for a ticker. Uses parquet cache when available."""
    if use_cache:
        cached = load_cached(ticker, start, end)
        if cached is not None:
            logger.info("Loaded %s from cache (%d rows)", ticker, len(cached))
            return cached

    logger.info("Fetching %s from yfinance", ticker)
    data = yf.download(
        ticker,
        start=start,
        end=end,
        period=period if start is None else None,
        auto_adjust=True,
        progress=False,
    )

    if data.empty:
        logger.warning("No data returned for %s", ticker)
        return data

    # yfinance may return MultiIndex columns for single ticker - flatten
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data.index = pd.to_datetime(data.index)
    data.index.name = "Date"

    if use_cache:
        save_cached(ticker, data)

    return data


def fetch_multiple(
    tickers: list[str],
    start: str | None = None,
    end: str | None = None,
    period: str = "5y",
    use_cache: bool = True,
) -> dict[str, pd.DataFrame]:
    """Fetch data for multiple tickers, returned as {ticker: DataFrame}."""
    result = {}
    for ticker in tickers:
        result[ticker] = fetch_ticker(
            ticker, start=start, end=end, period=period, use_cache=use_cache
        )
    return result


def fetch_latest(tickers: list[str]) -> dict[str, pd.DataFrame]:
    """Fetch the most recent trading day data for live monitoring."""
    return fetch_multiple(tickers, period="5d", use_cache=False)
