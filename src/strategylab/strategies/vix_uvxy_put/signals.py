"""VIX spike detection logic."""

import pandas as pd

from strategylab.core.models import Signal

STRATEGY_NAME = "vix_uvxy_put"


def detect_vix_spikes(
    vix_data: pd.DataFrame, params: dict
) -> list[Signal]:
    """Detect VIX spikes based on absolute level and/or percentage change.

    Args:
        vix_data: DataFrame with at least a 'Close' column, indexed by date.
        params: dict with keys:
            - spike_absolute: VIX level threshold
            - spike_pct_change: single-day % change threshold
            - combine_mode: "or" | "and"
            - cooldown_days: minimum days between signals
    """
    signals = []
    spike_abs = params["spike_absolute"]
    spike_pct = params["spike_pct_change"]
    combine_mode = params.get("combine_mode", "or")
    cooldown = params.get("cooldown_days", 5)

    closes = vix_data["Close"]
    pct_changes = closes.pct_change() * 100

    last_signal_date = None

    for i in range(1, len(vix_data)):
        date = vix_data.index[i]
        vix_close = closes.iloc[i]
        daily_pct = pct_changes.iloc[i]

        # Cooldown check
        if last_signal_date is not None:
            days_since = (date - last_signal_date).days
            if days_since < cooldown:
                continue

        abs_triggered = vix_close >= spike_abs
        pct_triggered = daily_pct >= spike_pct

        if combine_mode == "and":
            triggered = abs_triggered and pct_triggered
        else:
            triggered = abs_triggered or pct_triggered

        if not triggered:
            continue

        # Determine signal strength
        strength = _classify_strength(vix_close, daily_pct, spike_abs, spike_pct)

        signals.append(
            Signal(
                date=pd.Timestamp(date),
                strategy_name=STRATEGY_NAME,
                signal_type="vix_spike",
                strength=strength,
                metadata={
                    "vix_close": float(vix_close),
                    "vix_pct_change": float(daily_pct),
                    "abs_triggered": bool(abs_triggered),
                    "pct_triggered": bool(pct_triggered),
                },
            )
        )
        last_signal_date = date

    return signals


def _classify_strength(
    vix_close: float, daily_pct: float, spike_abs: float, spike_pct: float
) -> str:
    """Classify spike strength as moderate, strong, or extreme."""
    extreme_abs = spike_abs * 1.3
    extreme_pct = spike_pct * 2.0
    strong_abs = spike_abs * 1.15
    strong_pct = spike_pct * 1.5

    if vix_close >= extreme_abs or daily_pct >= extreme_pct:
        return "extreme"
    elif vix_close >= strong_abs or daily_pct >= strong_pct:
        return "strong"
    else:
        return "moderate"
