"""Exit criteria for UVXY put positions."""

import pandas as pd

from strategylab.core.models import ExitResult, Position
from strategylab.data.options_chain import get_options_chain


def check_put_exit(
    position: Position,
    current_date: pd.Timestamp,
    data: dict[str, pd.DataFrame],
    params: dict,
) -> ExitResult | None:
    """Check if a UVXY put position should be exited.

    Exit conditions (checked in order):
    1. Max hold days reached
    2. VIX floor - VIX has dropped below threshold
    3. Profit target hit
    4. Stop loss hit
    """
    days_held = (current_date - position.entry_date).days
    max_hold = params.get("max_hold_days", 30)
    profit_target_pct = params.get("profit_target_pct", 100)
    stop_loss_pct = params.get("stop_loss_pct", 50)
    vix_floor = params.get("vix_floor_exit", 18)

    # Current option value estimation
    current_price = _estimate_current_price(position, current_date, data, params)
    pnl = (current_price * position.quantity * 100) - position.cost_basis
    pnl_pct = (pnl / position.cost_basis) * 100 if position.cost_basis > 0 else 0.0

    # 1. Max hold / expiration
    dte = position.metadata.get("dte", 45)
    if days_held >= min(max_hold, dte):
        reason = "expiration" if days_held >= dte else "max_hold"
        return ExitResult(
            exit_date=current_date,
            exit_reason=reason,
            exit_price=current_price,
            pnl=pnl,
            pnl_pct=pnl_pct,
            days_held=days_held,
        )

    # 2. VIX floor check
    vix_data = data.get("^VIX")
    if vix_data is not None and current_date in vix_data.index:
        current_vix = float(vix_data.loc[current_date, "Close"])
        if current_vix <= vix_floor:
            return ExitResult(
                exit_date=current_date,
                exit_reason="vix_floor",
                exit_price=current_price,
                pnl=pnl,
                pnl_pct=pnl_pct,
                days_held=days_held,
            )

    # 3. Profit target
    if pnl_pct >= profit_target_pct:
        return ExitResult(
            exit_date=current_date,
            exit_reason="profit_target",
            exit_price=current_price,
            pnl=pnl,
            pnl_pct=pnl_pct,
            days_held=days_held,
        )

    # 4. Stop loss
    if pnl_pct <= -stop_loss_pct:
        return ExitResult(
            exit_date=current_date,
            exit_reason="stop_loss",
            exit_price=current_price,
            pnl=pnl,
            pnl_pct=pnl_pct,
            days_held=days_held,
        )

    return None


def _estimate_current_price(
    position: Position,
    current_date: pd.Timestamp,
    data: dict[str, pd.DataFrame],
    params: dict,
) -> float:
    """Estimate current put option price using QC data or Black-Scholes."""
    uvxy_data = data.get("UVXY")
    if uvxy_data is None or current_date not in uvxy_data.index:
        return position.entry_price

    current_underlying = float(uvxy_data.loc[current_date, "Close"])
    strike = position.metadata.get("strike", current_underlying)
    dte = position.metadata.get("dte", 45)
    days_held = (current_date - position.entry_date).days
    remaining_dte = max(dte - days_held, 0)
    iv = position.metadata.get("iv_estimate", 0.8)

    # Parse expiry from metadata if available
    expiry = None
    expiry_str = position.metadata.get("expiry")
    if expiry_str:
        try:
            expiry = pd.Timestamp(expiry_str)
        except (ValueError, TypeError):
            pass

    chain = get_options_chain("UVXY")
    return chain.get_put_price(
        date=current_date,
        strike=strike,
        expiry=expiry,
        underlying_price=current_underlying,
        iv=iv,
        dte=remaining_dte,
    )
