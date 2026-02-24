"""VIX/UVXY Put strategy - buy UVXY puts after VIX spikes."""

import pandas as pd

from strategylab.config import load_strategy_config
from strategylab.core.base_strategy import BaseStrategy
from strategylab.core.models import ExitResult, Position, Signal
from strategylab.core.registry import register
from strategylab.strategies.vix_uvxy_put.exits import check_put_exit
from strategylab.strategies.vix_uvxy_put.position import construct_uvxy_put
from strategylab.strategies.vix_uvxy_put.signals import detect_vix_spikes


@register
class VixUvxyPutStrategy(BaseStrategy):
    """Buy UVXY puts after VIX spikes, betting on mean reversion."""

    name = "vix_uvxy_put"
    description = (
        "Detects VIX spikes and buys UVXY puts, profiting from "
        "volatility mean reversion and UVXY decay."
    )
    required_tickers = ["^VIX", "UVXY"]

    def __init__(self):
        self._config = load_strategy_config(self.name)

    def detect_signals(
        self, data: dict[str, pd.DataFrame], params: dict
    ) -> list[Signal]:
        vix_data = data.get("^VIX")
        if vix_data is None or vix_data.empty:
            return []
        return detect_vix_spikes(vix_data, params)

    def construct_position(
        self,
        signal: Signal,
        data: dict[str, pd.DataFrame],
        params: dict,
        portfolio_value: float,
    ) -> Position | None:
        return construct_uvxy_put(signal, data, params, portfolio_value)

    def check_exit(
        self,
        position: Position,
        current_date: pd.Timestamp,
        data: dict[str, pd.DataFrame],
        params: dict,
    ) -> ExitResult | None:
        return check_put_exit(position, current_date, data, params)

    def format_alert(self, signal: Signal, recommendation: dict) -> dict:
        vix_close = signal.metadata.get("vix_close", "N/A")
        vix_pct = signal.metadata.get("vix_pct_change", 0)
        strength = signal.strength.upper()

        subject = f"[StrategyLab] VIX Spike Alert ({strength}) - VIX at {vix_close:.1f}"

        strike = recommendation.get("strike", "TBD")
        dte = recommendation.get("dte", 45)
        contracts = recommendation.get("contracts", 1)
        max_cost = recommendation.get("max_cost", "TBD")

        body_text = (
            f"VIX SPIKE DETECTED - {strength}\n"
            f"{'=' * 50}\n\n"
            f"VIX Close: {vix_close:.2f}\n"
            f"Daily Change: {vix_pct:+.1f}%\n"
            f"Signal Date: {signal.date.strftime('%Y-%m-%d')}\n\n"
            f"RECOMMENDED ACTION:\n"
            f"  Buy {contracts} UVXY PUT contract(s)\n"
            f"  Strike: ${strike}\n"
            f"  Expiration: ~{dte} DTE\n"
            f"  Max Cost: ~${max_cost}\n\n"
            f"ROBINHOOD STEPS:\n"
            f"  1. Open Robinhood > Search 'UVXY'\n"
            f"  2. Tap 'Trade Options'\n"
            f"  3. Select expiration ~{dte} days out\n"
            f"  4. Select PUT at ${strike} strike\n"
            f"  5. Buy {contracts} contract(s)\n"
            f"  6. Set limit order near the ask\n\n"
            f"EXIT PLAN:\n"
            f"  - Profit target: {recommendation.get('profit_target_pct', 100)}%\n"
            f"  - Stop loss: {recommendation.get('stop_loss_pct', 50)}%\n"
            f"  - Max hold: {recommendation.get('max_hold_days', 30)} days\n"
        )

        body_html = (
            f"<h2>VIX Spike Alert - {strength}</h2>"
            f"<table border='1' cellpadding='8' style='border-collapse:collapse'>"
            f"<tr><td><b>VIX Close</b></td><td>{vix_close:.2f}</td></tr>"
            f"<tr><td><b>Daily Change</b></td><td>{vix_pct:+.1f}%</td></tr>"
            f"<tr><td><b>Date</b></td><td>{signal.date.strftime('%Y-%m-%d')}</td></tr>"
            f"</table>"
            f"<h3>Recommended Action</h3>"
            f"<p>Buy <b>{contracts}</b> UVXY PUT(s) at <b>${strike}</b> strike, "
            f"~<b>{dte}</b> DTE</p>"
            f"<p>Max cost: ~<b>${max_cost}</b></p>"
            f"<h3>Robinhood Steps</h3>"
            f"<ol>"
            f"<li>Open Robinhood &gt; Search 'UVXY'</li>"
            f"<li>Tap 'Trade Options'</li>"
            f"<li>Select expiration ~{dte} days out</li>"
            f"<li>Select PUT at ${strike} strike</li>"
            f"<li>Buy {contracts} contract(s)</li>"
            f"<li>Set limit order near the ask</li>"
            f"</ol>"
            f"<h3>Exit Plan</h3>"
            f"<ul>"
            f"<li>Profit target: {recommendation.get('profit_target_pct', 100)}%</li>"
            f"<li>Stop loss: {recommendation.get('stop_loss_pct', 50)}%</li>"
            f"<li>Max hold: {recommendation.get('max_hold_days', 30)} days</li>"
            f"</ul>"
        )

        return {"subject": subject, "body_text": body_text, "body_html": body_html}

    def get_param_grid(self) -> dict[str, list]:
        sweep = self._config.get("sweep", {})
        if sweep:
            return sweep
        return {
            "spike_absolute": [25, 28, 30, 33, 35],
            "spike_pct_change": [10, 15, 20, 25, 30],
            "combine_mode": ["or", "and"],
            "strike_mode": ["atm", "otm_5", "otm_10"],
            "dte_target": [30, 45, 60, 90],
            "entry_delay_days": [0, 1, 2],
            "profit_target_pct": [50, 100, 150, 200],
            "stop_loss_pct": [50, 75],
            "max_hold_days": [15, 20, 30, 45],
        }

    def get_default_params(self) -> dict:
        defaults = self._config.get("default_params", {})
        if defaults:
            return defaults
        return {
            "spike_absolute": 30,
            "spike_pct_change": 20,
            "combine_mode": "or",
            "cooldown_days": 5,
            "strike_mode": "otm_5",
            "dte_target": 45,
            "entry_delay_days": 1,
            "position_size_pct": 5.0,
            "max_open_positions": 3,
            "profit_target_pct": 100,
            "stop_loss_pct": 50,
            "max_hold_days": 30,
            "vix_floor_exit": 18,
            "iv_scale_factor": 1.2,
        }
