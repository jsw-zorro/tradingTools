"""Backtest performance metrics."""

import numpy as np
import pandas as pd

from strategylab.backtest.engine import BacktestResult


def calculate_metrics(result: BacktestResult) -> dict:
    """Calculate comprehensive performance metrics from a backtest result."""
    metrics = {
        "strategy": result.strategy_name,
        "initial_capital": result.initial_capital,
        "final_equity": result.final_equity,
        "total_return_pct": result.total_return_pct,
        "num_trades": result.num_trades,
    }

    if result.num_trades == 0:
        metrics.update({
            "win_rate": 0.0,
            "avg_pnl": 0.0,
            "avg_pnl_pct": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "profit_factor": 0.0,
            "max_drawdown_pct": 0.0,
            "sharpe_ratio": 0.0,
            "avg_days_held": 0.0,
            "composite_score": 0.0,
        })
        return metrics

    # Trade-level metrics
    pnls = [t.exit.pnl for t in result.trades]
    pnl_pcts = [t.exit.pnl_pct for t in result.trades]
    days_held = [t.exit.days_held for t in result.trades]

    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    metrics["win_rate"] = len(wins) / len(pnls) * 100 if pnls else 0
    metrics["avg_pnl"] = float(np.mean(pnls))
    metrics["avg_pnl_pct"] = float(np.mean(pnl_pcts))
    metrics["avg_win"] = float(np.mean(wins)) if wins else 0
    metrics["avg_loss"] = float(np.mean(losses)) if losses else 0
    metrics["avg_days_held"] = float(np.mean(days_held))

    # Profit factor
    gross_wins = sum(wins) if wins else 0
    gross_losses = abs(sum(losses)) if losses else 0
    metrics["profit_factor"] = gross_wins / gross_losses if gross_losses > 0 else float("inf") if gross_wins > 0 else 0

    # Drawdown
    metrics["max_drawdown_pct"] = _max_drawdown(result.equity_curve)

    # Sharpe ratio (annualized)
    metrics["sharpe_ratio"] = _sharpe_ratio(result.equity_curve)

    # Exit reason distribution
    reasons = {}
    for t in result.trades:
        reasons[t.exit.exit_reason] = reasons.get(t.exit.exit_reason, 0) + 1
    metrics["exit_reasons"] = reasons

    # Composite score
    metrics["composite_score"] = _composite_score(metrics)

    return metrics


def _max_drawdown(equity_curve: pd.Series) -> float:
    """Calculate maximum drawdown as a percentage."""
    if equity_curve.empty:
        return 0.0
    peak = equity_curve.expanding().max()
    drawdown = (equity_curve - peak) / peak * 100
    return float(drawdown.min())


def _sharpe_ratio(equity_curve: pd.Series, risk_free_rate: float = 0.05) -> float:
    """Calculate annualized Sharpe ratio from equity curve."""
    if len(equity_curve) < 2:
        return 0.0
    daily_returns = equity_curve.pct_change().dropna()
    if daily_returns.std() == 0:
        return 0.0
    excess_return = daily_returns.mean() - risk_free_rate / 252
    return float(excess_return / daily_returns.std() * np.sqrt(252))


def _composite_score(metrics: dict) -> float:
    """Weighted composite score for ranking parameter sets.

    Weights: Sharpe 40%, win rate 20%, profit factor 20%,
             trade count 10%, drawdown 10%.
    """
    sharpe = max(min(metrics.get("sharpe_ratio", 0), 5), -5)  # clamp
    win_rate = metrics.get("win_rate", 0) / 100  # normalize to 0-1
    pf = min(metrics.get("profit_factor", 0), 10)  # clamp
    trades = min(metrics.get("num_trades", 0), 50) / 50  # normalize
    dd = max(metrics.get("max_drawdown_pct", 0), -100) / -100  # normalize, higher is worse

    score = (
        0.40 * (sharpe / 5 + 1) / 2  # normalize sharpe to 0-1ish
        + 0.20 * win_rate
        + 0.20 * (pf / 10)
        + 0.10 * trades
        + 0.10 * (1 + dd)  # dd is negative, so 1+dd penalizes large drawdowns
    )
    return round(score * 100, 2)
