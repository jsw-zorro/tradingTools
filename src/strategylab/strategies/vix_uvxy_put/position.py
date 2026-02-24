"""UVXY put position construction."""

import numpy as np
import pandas as pd

from strategylab.core.models import Position, Signal
from strategylab.data.options import estimate_iv
from strategylab.data.options_chain import get_options_chain

STRATEGY_NAME = "vix_uvxy_put"


def construct_uvxy_put(
    signal: Signal,
    data: dict[str, pd.DataFrame],
    params: dict,
    portfolio_value: float,
) -> Position | None:
    """Construct a UVXY put position from a VIX spike signal.

    Uses real options chain data from QuantConnect when available,
    falls back to Black-Scholes estimation otherwise.

    Args:
        signal: The triggering VIX spike signal.
        data: {"^VIX": df, "UVXY": df, ...}
        params: Strategy parameters including strike_mode, dte_target, etc.
        portfolio_value: Current portfolio value for position sizing.

    Returns:
        Position or None if data is insufficient.
    """
    uvxy_data = data.get("UVXY")
    if uvxy_data is None or uvxy_data.empty:
        return None

    entry_delay = params.get("entry_delay_days", 1)
    entry_date = signal.date + pd.Timedelta(days=entry_delay)

    # Find the actual trading day on or after entry_date
    available_dates = uvxy_data.index[uvxy_data.index >= entry_date]
    if available_dates.empty:
        return None
    entry_date = available_dates[0]

    uvxy_price = float(uvxy_data.loc[entry_date, "Close"])
    if uvxy_price <= 0:
        return None

    chain = get_options_chain("UVXY")
    dte = params.get("dte_target", 45)
    strike_mode = params.get("strike_mode", "otm_5")

    # Try to find a real contract from QC data
    if chain.has_real_data:
        target_pct = _strike_mode_to_pct(strike_mode)
        contract = chain.find_best_contract(
            date=entry_date,
            target_strike_pct=target_pct,
            underlying_price=uvxy_price,
            target_dte=dte,
            right="put",
        )

        if contract is not None:
            strike = contract["strike"]
            option_price = contract["close"]
            actual_dte = contract["dte"]
            expiry = contract["expiry"]

            # Try to get real IV
            real_iv = chain.get_iv(entry_date, strike, expiry, "put")
            iv = real_iv if real_iv else _estimate_fallback_iv(uvxy_data, entry_date, params)

            return _build_position(
                entry_date=entry_date,
                signal=signal,
                strike=strike,
                option_price=option_price,
                dte=actual_dte,
                expiry=str(expiry.date()) if hasattr(expiry, "date") else str(expiry),
                uvxy_price=uvxy_price,
                iv=iv,
                realized_vol=None,
                portfolio_value=portfolio_value,
                params=params,
                pricing_source="quantconnect",
                volume=contract.get("volume", 0),
                open_interest=contract.get("open_interest", 0),
            )

    # Black-Scholes fallback
    strike = _calculate_strike(uvxy_price, strike_mode)

    hist_prices = uvxy_data.loc[uvxy_data.index <= entry_date, "Close"].values
    realized_vol = estimate_iv(hist_prices, window=30)
    iv = realized_vol * params.get("iv_scale_factor", 1.2)

    expiry_date = entry_date + pd.Timedelta(days=dte)
    option_price = chain.get_put_price(
        date=entry_date,
        strike=strike,
        expiry=expiry_date,
        underlying_price=uvxy_price,
        iv=iv,
        dte=dte,
    )

    if option_price <= 0.01:
        return None

    return _build_position(
        entry_date=entry_date,
        signal=signal,
        strike=strike,
        option_price=option_price,
        dte=dte,
        expiry=str(expiry_date.date()),
        uvxy_price=uvxy_price,
        iv=iv,
        realized_vol=realized_vol,
        portfolio_value=portfolio_value,
        params=params,
        pricing_source="black_scholes",
    )


def _build_position(
    *,
    entry_date: pd.Timestamp,
    signal: Signal,
    strike: float,
    option_price: float,
    dte: int,
    expiry: str,
    uvxy_price: float,
    iv: float,
    realized_vol: float | None,
    portfolio_value: float,
    params: dict,
    pricing_source: str,
    volume: int = 0,
    open_interest: int = 0,
) -> Position | None:
    """Build the Position object with sizing logic."""
    if option_price <= 0.01:
        return None

    max_risk = portfolio_value * params.get("position_size_pct", 5.0) / 100.0
    contract_cost = option_price * 100
    quantity = max(1, int(max_risk / contract_cost))
    cost_basis = quantity * contract_cost

    metadata = {
        "strike": strike,
        "dte": dte,
        "expiry": expiry,
        "underlying_price": uvxy_price,
        "iv_estimate": iv,
        "pricing_source": pricing_source,
    }
    if realized_vol is not None:
        metadata["realized_vol"] = realized_vol
    if volume:
        metadata["volume"] = volume
    if open_interest:
        metadata["open_interest"] = open_interest

    return Position(
        entry_date=pd.Timestamp(entry_date),
        strategy_name=STRATEGY_NAME,
        instrument="UVXY_PUT",
        direction="long",
        entry_price=option_price,
        quantity=quantity,
        cost_basis=cost_basis,
        signal=signal,
        metadata=metadata,
    )


def _calculate_strike(underlying_price: float, strike_mode: str) -> float:
    """Calculate strike price based on mode."""
    pct = _strike_mode_to_pct(strike_mode)
    return round(underlying_price * pct, 1)


def _strike_mode_to_pct(strike_mode: str) -> float:
    """Convert strike mode string to fraction of underlying price."""
    if strike_mode == "atm":
        return 1.0
    elif strike_mode == "otm_5":
        return 0.95
    elif strike_mode == "otm_10":
        return 0.90
    else:
        return 0.95


def _estimate_fallback_iv(
    uvxy_data: pd.DataFrame, entry_date: pd.Timestamp, params: dict
) -> float:
    """Estimate IV from realized vol when real IV not available."""
    hist_prices = uvxy_data.loc[uvxy_data.index <= entry_date, "Close"].values
    realized_vol = estimate_iv(hist_prices, window=30)
    return realized_vol * params.get("iv_scale_factor", 1.2)
