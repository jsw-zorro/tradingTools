"""Live option chain data via yfinance for put-selling analysis."""

import logging

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


def round_to_strike(price: float) -> float:
    """Round price down to nearest standard option strike interval.

    Options use $1 strikes under $25, $2.50 under $50, $5 above $50.
    We round DOWN because for put-selling we want strikes below the bound.
    """
    if price < 25:
        return float(int(price))
    elif price < 50:
        return float(int(price / 2.5) * 2.5)
    else:
        return float(int(price / 5) * 5)


def get_friday_expiry(ticker: str, friday_date: pd.Timestamp) -> str | None:
    """Find the nearest available option expiry on or after the target Friday.

    Returns expiry date string (YYYY-MM-DD) or None if no options available.
    """
    try:
        t = yf.Ticker(ticker)
        expirations = t.options  # list of date strings
        if not expirations:
            logger.warning(f"No option expirations found for {ticker}")
            return None

        target = friday_date.strftime("%Y-%m-%d")
        # Find exact match or nearest expiry on/after target
        for exp in sorted(expirations):
            if exp >= target:
                return exp

        # If no expiry on/after target, return the last available
        return expirations[-1]
    except Exception as e:
        logger.warning(f"Failed to get expirations for {ticker}: {e}")
        return None


def fetch_put_premiums(
    ticker: str,
    expiry: str,
    strikes: list[float],
) -> dict[float, dict]:
    """Fetch put option bid/ask/mid for given strikes.

    Returns dict mapping strike -> {bid, ask, mid, volume, open_interest}.
    Missing strikes are omitted from the result.
    """
    try:
        t = yf.Ticker(ticker)
        chain = t.option_chain(expiry)
        puts = chain.puts
    except Exception as e:
        logger.warning(f"Failed to fetch option chain for {ticker} {expiry}: {e}")
        return {}

    result = {}
    for strike in strikes:
        match = puts[puts["strike"] == strike]
        if match.empty:
            # Try nearest available strike
            if not puts.empty:
                closest_idx = (puts["strike"] - strike).abs().idxmin()
                closest = puts.loc[closest_idx]
                if abs(closest["strike"] - strike) <= 5:
                    strike = float(closest["strike"])
                    match = puts[puts["strike"] == strike]

        if not match.empty:
            row = match.iloc[0]
            bid = float(row.get("bid", 0) or 0)
            ask = float(row.get("ask", 0) or 0)
            mid = (bid + ask) / 2 if (bid + ask) > 0 else 0
            result[strike] = {
                "bid": bid,
                "ask": ask,
                "mid": mid,
                "volume": int(row.get("volume", 0) or 0),
                "open_interest": int(row.get("openInterest", 0) or 0),
            }

    return result
