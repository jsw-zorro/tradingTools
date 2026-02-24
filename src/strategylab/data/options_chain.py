"""Unified options pricing interface.

Looks up real prices from QuantConnect data when available,
falls back to Black-Scholes estimation otherwise.
"""

import logging

import numpy as np
import pandas as pd

from strategylab.data.options import black_scholes_put, estimate_iv
from strategylab.data.qc_fetcher import load_cached_options

logger = logging.getLogger(__name__)


class OptionsChain:
    """Unified interface for options pricing, backed by QC data or Black-Scholes.

    Usage:
        chain = OptionsChain("UVXY")
        price = chain.get_put_price(date, strike, expiry, underlying_price, iv)
    """

    def __init__(self, underlying: str):
        self.underlying = underlying
        self._chain_data: pd.DataFrame | None = None
        self._loaded = False
        self._has_real_data = False

    def _ensure_loaded(self):
        if self._loaded:
            return
        self._loaded = True
        self._chain_data = load_cached_options(self.underlying)
        if self._chain_data is not None and not self._chain_data.empty:
            self._has_real_data = True
            logger.info(
                "Loaded real options data for %s (%d records, %s to %s)",
                self.underlying,
                len(self._chain_data),
                self._chain_data["date"].min().date(),
                self._chain_data["date"].max().date(),
            )
        else:
            logger.info(
                "No QC options data for %s, using Black-Scholes estimation",
                self.underlying,
            )

    @property
    def has_real_data(self) -> bool:
        self._ensure_loaded()
        return self._has_real_data

    def get_put_price(
        self,
        date: pd.Timestamp,
        strike: float,
        expiry: pd.Timestamp | None = None,
        underlying_price: float | None = None,
        iv: float | None = None,
        dte: int | None = None,
    ) -> float:
        """Get put option price, preferring real data over Black-Scholes.

        Args:
            date: Valuation date.
            strike: Strike price.
            expiry: Option expiry date (required for real data lookup).
            underlying_price: Current underlying price (required for BS fallback).
            iv: Implied volatility (used for BS fallback if no real IV available).
            dte: Days to expiry (alternative to expiry, used for BS if expiry not given).

        Returns:
            Option price (per share, not per contract).
        """
        self._ensure_loaded()

        # Try real data first
        if self._has_real_data:
            real_price = self._lookup_real_price(date, strike, expiry, "put")
            if real_price is not None:
                return real_price

            # If we have real data but not for this exact contract, try nearest match
            nearest = self._find_nearest_contract(date, strike, expiry, "put")
            if nearest is not None:
                return nearest

        # Fall back to Black-Scholes
        return self._bs_fallback(
            underlying_price, strike, expiry, date, iv, dte, option_type="put"
        )

    def get_call_price(
        self,
        date: pd.Timestamp,
        strike: float,
        expiry: pd.Timestamp | None = None,
        underlying_price: float | None = None,
        iv: float | None = None,
        dte: int | None = None,
    ) -> float:
        """Get call option price, preferring real data over Black-Scholes."""
        self._ensure_loaded()

        if self._has_real_data:
            real_price = self._lookup_real_price(date, strike, expiry, "call")
            if real_price is not None:
                return real_price

            nearest = self._find_nearest_contract(date, strike, expiry, "call")
            if nearest is not None:
                return nearest

        from strategylab.data.options import black_scholes_call
        return self._bs_fallback(
            underlying_price, strike, expiry, date, iv, dte, option_type="call"
        )

    def get_iv(self, date: pd.Timestamp, strike: float, expiry: pd.Timestamp, right: str = "put") -> float | None:
        """Get implied volatility from real data if available."""
        self._ensure_loaded()
        if not self._has_real_data or "implied_volatility" not in self._chain_data.columns:
            return None

        row = self._exact_match(date, strike, expiry, right)
        if row is not None and not pd.isna(row.get("implied_volatility")):
            return float(row["implied_volatility"])
        return None

    def get_chain_snapshot(
        self, date: pd.Timestamp, right: str = "put"
    ) -> pd.DataFrame | None:
        """Get the full options chain for a given date.

        Returns DataFrame with columns: expiry, strike, close, volume,
        open_interest, [implied_volatility]
        """
        self._ensure_loaded()
        if not self._has_real_data:
            return None

        df = self._chain_data
        mask = (df["date"] == date) & (df["right"] == right)
        snapshot = df[mask].copy()
        if snapshot.empty:
            # Try nearest trading day
            available = df[df["right"] == right]["date"].unique()
            if len(available) == 0:
                return None
            nearest_date = available[np.argmin(np.abs(available - date))]
            if abs((nearest_date - date).days) <= 3:
                mask = (df["date"] == nearest_date) & (df["right"] == right)
                snapshot = df[mask].copy()

        return snapshot if not snapshot.empty else None

    def find_best_contract(
        self,
        date: pd.Timestamp,
        target_strike_pct: float,
        underlying_price: float,
        target_dte: int,
        right: str = "put",
    ) -> dict | None:
        """Find the best matching contract from real data.

        Args:
            date: Trade date.
            target_strike_pct: Target strike as fraction of underlying (e.g. 0.95 for 5% OTM put).
            underlying_price: Current underlying price.
            target_dte: Desired days to expiry.
            right: "put" or "call".

        Returns:
            Dict with {strike, expiry, close, volume, open_interest, dte} or None.
        """
        snapshot = self.get_chain_snapshot(date, right)
        if snapshot is None or snapshot.empty:
            return None

        target_strike = underlying_price * target_strike_pct
        target_expiry = date + pd.Timedelta(days=target_dte)

        # Score each contract by distance from target strike and expiry
        snapshot = snapshot.copy()
        snapshot["strike_dist"] = abs(snapshot["strike"] - target_strike) / underlying_price
        snapshot["dte"] = (snapshot["expiry"] - date).dt.days
        snapshot["expiry_dist"] = abs(snapshot["dte"] - target_dte) / target_dte

        # Filter: only consider contracts with DTE > 0 and reasonable volume
        valid = snapshot[(snapshot["dte"] > 0) & (snapshot["close"] > 0)]
        if valid.empty:
            valid = snapshot[snapshot["dte"] > 0]  # relax volume filter
        if valid.empty:
            return None

        # Combined distance score (weighted)
        valid = valid.copy()
        valid["score"] = valid["strike_dist"] * 0.6 + valid["expiry_dist"] * 0.4

        best = valid.loc[valid["score"].idxmin()]
        return {
            "strike": float(best["strike"]),
            "expiry": best["expiry"],
            "close": float(best["close"]),
            "volume": int(best.get("volume", 0)),
            "open_interest": int(best.get("open_interest", 0)),
            "dte": int(best["dte"]),
        }

    def _lookup_real_price(
        self, date: pd.Timestamp, strike: float, expiry: pd.Timestamp | None, right: str
    ) -> float | None:
        """Exact lookup in chain data."""
        row = self._exact_match(date, strike, expiry, right)
        if row is not None:
            price = row.get("close", 0)
            if price > 0:
                return float(price)
        return None

    def _exact_match(
        self, date: pd.Timestamp, strike: float, expiry: pd.Timestamp | None, right: str
    ) -> dict | None:
        """Find exact match in chain data."""
        df = self._chain_data
        mask = (df["date"] == date) & (df["right"] == right)

        # Strike match with tolerance (options strikes are rounded)
        mask = mask & (abs(df["strike"] - strike) < 0.01)

        if expiry is not None:
            mask = mask & (df["expiry"] == expiry)

        matches = df[mask]
        if matches.empty:
            return None
        return matches.iloc[0].to_dict()

    def _find_nearest_contract(
        self, date: pd.Timestamp, strike: float, expiry: pd.Timestamp | None, right: str
    ) -> float | None:
        """Find nearest matching contract when exact match unavailable."""
        df = self._chain_data
        mask = (df["right"] == right)

        # Find nearest date (within 3 trading days)
        available_dates = df[mask]["date"].unique()
        if len(available_dates) == 0:
            return None
        date_dists = np.abs(available_dates - date)
        nearest_date = available_dates[np.argmin(date_dists)]
        if abs((nearest_date - date).days) > 3:
            return None

        mask = mask & (df["date"] == nearest_date)

        # Find nearest strike (within 10%)
        subset = df[mask]
        if subset.empty:
            return None

        strike_dists = abs(subset["strike"] - strike)
        nearest_idx = strike_dists.idxmin()
        nearest_row = subset.loc[nearest_idx]

        if abs(nearest_row["strike"] - strike) / strike > 0.10:
            return None

        # If expiry specified, check DTE is similar
        if expiry is not None and "expiry" in nearest_row.index:
            dte_diff = abs((nearest_row["expiry"] - expiry).days)
            if dte_diff > 15:
                return None

        price = nearest_row.get("close", 0)
        return float(price) if price > 0 else None

    def _bs_fallback(
        self,
        underlying_price: float | None,
        strike: float,
        expiry: pd.Timestamp | None,
        date: pd.Timestamp,
        iv: float | None,
        dte: int | None,
        option_type: str,
    ) -> float:
        """Fall back to Black-Scholes pricing."""
        if underlying_price is None or underlying_price <= 0:
            return 0.0

        if dte is None:
            if expiry is not None:
                dte = max((expiry - date).days, 0)
            else:
                dte = 45  # default

        T = dte / 365.0
        sigma = iv if iv and iv > 0 else 0.8  # default high vol for leveraged ETF
        r = 0.05

        if option_type == "put":
            return black_scholes_put(underlying_price, strike, T, r, sigma)
        else:
            from strategylab.data.options import black_scholes_call
            return black_scholes_call(underlying_price, strike, T, r, sigma)


# Module-level cache of OptionsChain instances
_chain_cache: dict[str, OptionsChain] = {}


def get_options_chain(underlying: str) -> OptionsChain:
    """Get or create an OptionsChain instance for an underlying."""
    if underlying not in _chain_cache:
        _chain_cache[underlying] = OptionsChain(underlying)
    return _chain_cache[underlying]


def clear_chain_cache():
    """Clear the module-level chain cache (useful for testing)."""
    _chain_cache.clear()
