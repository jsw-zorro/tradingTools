"""Polling loop that monitors enabled strategies."""

import logging
import time

from strategylab.config import load_settings
from strategylab.core.registry import get_strategy
from strategylab.data.fetcher import fetch_latest
from strategylab.monitor.alert import send_alert
from strategylab.monitor.scheduler import is_market_open, sleep_until_market_open

logger = logging.getLogger(__name__)


def run_check(strategy_name: str) -> list[dict]:
    """Run a single check for one strategy. Returns list of alerts sent."""
    strategy = get_strategy(strategy_name)
    params = strategy.get_default_params()

    logger.info("Checking %s...", strategy_name)
    data = fetch_latest(strategy.required_tickers)

    # Check for empty data
    for ticker, df in data.items():
        if df.empty:
            logger.warning("No data for %s, skipping check", ticker)
            return []

    signals = strategy.detect_signals(data, params)
    alerts_sent = []

    for signal in signals:
        # Build recommendation context
        recommendation = {
            "strike": "check current chain",
            "dte": params.get("dte_target", 45),
            "contracts": 1,
            "max_cost": "varies",
            "profit_target_pct": params.get("profit_target_pct", 100),
            "stop_loss_pct": params.get("stop_loss_pct", 50),
            "max_hold_days": params.get("max_hold_days", 30),
        }

        alert_content = strategy.format_alert(signal, recommendation)

        success = send_alert(
            subject=alert_content["subject"],
            body_text=alert_content["body_text"],
            body_html=alert_content.get("body_html"),
        )

        if success:
            alerts_sent.append({
                "signal": signal,
                "subject": alert_content["subject"],
            })
            logger.info("Alert sent for %s signal on %s", strategy_name, signal.date)
        else:
            logger.error("Failed to send alert for %s", strategy_name)

    if not signals:
        logger.info("No signals detected for %s", strategy_name)

    return alerts_sent


def run_monitor(market_hours_only: bool = True) -> None:
    """Run the continuous monitoring loop for all enabled strategies."""
    # Force import to trigger strategy registration
    import strategylab.strategies  # noqa: F401

    settings = load_settings()
    enabled = settings.get("monitor", {}).get("enabled_strategies", [])

    if not enabled:
        logger.error("No strategies enabled in config/settings.yaml")
        return

    logger.info("Starting monitor for strategies: %s", enabled)

    while True:
        if market_hours_only and not is_market_open():
            sleep_until_market_open()
            continue

        for strategy_name in enabled:
            try:
                run_check(strategy_name)
            except Exception as e:
                logger.error("Error checking %s: %s", strategy_name, e)

        # Get the minimum check interval across enabled strategies
        min_interval = 300
        for name in enabled:
            try:
                strategy = get_strategy(name)
                min_interval = min(min_interval, strategy.get_real_time_check_interval())
            except Exception:
                pass

        logger.info("Sleeping %d seconds until next check", min_interval)
        time.sleep(min_interval)
