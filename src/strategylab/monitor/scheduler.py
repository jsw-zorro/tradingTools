"""Market-hours scheduling utilities."""

import logging
from datetime import datetime, time

import pytz

from strategylab.config import load_settings

logger = logging.getLogger(__name__)


def is_market_open() -> bool:
    """Check if the US stock market is currently open."""
    settings = load_settings()
    monitor_config = settings.get("monitor", {})
    tz_name = monitor_config.get("timezone", "US/Eastern")
    tz = pytz.timezone(tz_name)
    now = datetime.now(tz)

    # Weekend check
    if now.weekday() >= 5:
        return False

    market_open_str = monitor_config.get("market_open", "09:30")
    market_close_str = monitor_config.get("market_close", "16:00")

    open_h, open_m = map(int, market_open_str.split(":"))
    close_h, close_m = map(int, market_close_str.split(":"))

    market_open = time(open_h, open_m)
    market_close = time(close_h, close_m)

    return market_open <= now.time() <= market_close


def get_next_market_open() -> datetime:
    """Get the next market open time."""
    settings = load_settings()
    monitor_config = settings.get("monitor", {})
    tz_name = monitor_config.get("timezone", "US/Eastern")
    tz = pytz.timezone(tz_name)
    now = datetime.now(tz)

    market_open_str = monitor_config.get("market_open", "09:30")
    open_h, open_m = map(int, market_open_str.split(":"))

    # If before market open today (weekday), return today's open
    if now.weekday() < 5 and now.time() < time(open_h, open_m):
        return now.replace(hour=open_h, minute=open_m, second=0, microsecond=0)

    # Otherwise, find the next weekday
    days_ahead = 1
    next_day = now
    while True:
        next_day = now.replace(hour=open_h, minute=open_m, second=0, microsecond=0)
        next_day += __import__("datetime").timedelta(days=days_ahead)
        if next_day.weekday() < 5:
            return next_day
        days_ahead += 1


def sleep_until_market_open() -> None:
    """Sleep until the next market open. Returns immediately if market is open."""
    import time as time_mod

    if is_market_open():
        return

    next_open = get_next_market_open()
    settings = load_settings()
    tz_name = settings.get("monitor", {}).get("timezone", "US/Eastern")
    tz = pytz.timezone(tz_name)
    now = datetime.now(tz)
    wait_seconds = (next_open - now).total_seconds()

    if wait_seconds > 0:
        logger.info("Market closed. Sleeping until %s (%.0f seconds)", next_open, wait_seconds)
        time_mod.sleep(wait_seconds)
