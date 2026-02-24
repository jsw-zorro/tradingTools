"""Local parquet caching for market data."""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

_CACHE_DIR = Path(__file__).resolve().parent.parent.parent.parent / "output" / "data_cache"


def _cache_path(ticker: str) -> Path:
    return _CACHE_DIR / f"{ticker.upper().replace('^', '_')}.parquet"


def load_cached(
    ticker: str, start: str | None = None, end: str | None = None
) -> pd.DataFrame | None:
    """Load cached data if it exists and covers the requested range."""
    path = _cache_path(ticker)
    if not path.exists():
        return None

    try:
        df = pd.read_parquet(path)
        df.index = pd.to_datetime(df.index)

        if start and pd.Timestamp(start) < df.index.min():
            return None
        if end and pd.Timestamp(end) > df.index.max():
            return None

        if start:
            df = df[df.index >= pd.Timestamp(start)]
        if end:
            df = df[df.index <= pd.Timestamp(end)]

        return df
    except Exception as e:
        logger.warning("Cache read failed for %s: %s", ticker, e)
        return None


def save_cached(ticker: str, data: pd.DataFrame) -> None:
    """Save data to parquet cache."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _cache_path(ticker)
    data.to_parquet(path)
    logger.info("Cached %s (%d rows) to %s", ticker, len(data), path)


def clear_cache() -> None:
    """Remove all cached data files."""
    if _CACHE_DIR.exists():
        for f in _CACHE_DIR.glob("*.parquet"):
            f.unlink()
        logger.info("Cache cleared")
