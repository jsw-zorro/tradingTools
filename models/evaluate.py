"""Evaluation metrics and charts for Friday close range predictor."""

import logging
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import ARTIFACTS_DIR, RANGE_WIDTH

logger = logging.getLogger(__name__)


def compute_metrics(
    y_true: np.ndarray,
    lower_bounds: np.ndarray,
    y_pred: np.ndarray,
    tickers: list[str],
    range_width: float = RANGE_WIDTH,
) -> dict:
    """Compute coverage, violation rate, sharpness, MAE, RMSE.

    Returns dict with overall + per-ticker breakdown.
    """
    upper_bounds = lower_bounds + range_width
    in_range = (y_true >= lower_bounds) & (y_true <= upper_bounds)
    below_lower = y_true < lower_bounds

    results = {
        "overall": {
            "coverage": float(np.mean(in_range)),
            "lower_violation_rate": float(np.mean(below_lower)),
            "sharpness_mean": float(np.mean(y_true - lower_bounds)),
            "sharpness_median": float(np.median(y_true - lower_bounds)),
            "mae": float(np.mean(np.abs(y_true - y_pred))),
            "rmse": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
            "n_samples": len(y_true),
        },
        "per_ticker": {},
    }

    # Per-ticker breakdown
    unique_tickers = sorted(set(tickers))
    for t in unique_tickers:
        mask = np.array([tk == t for tk in tickers])
        if not np.any(mask):
            continue
        t_true = y_true[mask]
        t_pred = y_pred[mask]
        t_lb = lower_bounds[mask]
        t_in_range = in_range[mask]
        t_below = below_lower[mask]

        results["per_ticker"][t] = {
            "coverage": float(np.mean(t_in_range)),
            "lower_violation_rate": float(np.mean(t_below)),
            "sharpness_mean": float(np.mean(t_true - t_lb)),
            "mae": float(np.mean(np.abs(t_true - t_pred))),
            "rmse": float(np.sqrt(np.mean((t_true - t_pred) ** 2))),
            "n_samples": int(np.sum(mask)),
        }

    return results


def print_metrics(results: dict) -> None:
    """Pretty-print evaluation metrics."""
    o = results["overall"]
    print("\n" + "=" * 60)
    print("OVERALL METRICS")
    print("=" * 60)
    print(f"  Coverage:              {o['coverage']:.1%}  (target: >=85%)")
    print(f"  Lower bound violation: {o['lower_violation_rate']:.1%}  (target: <=5%)")
    print(f"  Sharpness (mean):      ${o['sharpness_mean']:.2f}")
    print(f"  Sharpness (median):    ${o['sharpness_median']:.2f}")
    print(f"  Point MAE:             ${o['mae']:.2f}")
    print(f"  Point RMSE:            ${o['rmse']:.2f}")
    print(f"  N samples:             {o['n_samples']}")

    print("\n" + "-" * 60)
    print("PER-TICKER BREAKDOWN")
    print("-" * 60)
    header = f"{'Ticker':<8} {'Coverage':>10} {'Violation':>10} {'Sharpness':>10} {'MAE':>8} {'RMSE':>8} {'N':>5}"
    print(header)
    print("-" * len(header))
    for t, m in sorted(results["per_ticker"].items()):
        print(
            f"{t:<8} {m['coverage']:>9.1%} {m['lower_violation_rate']:>9.1%} "
            f"${m['sharpness_mean']:>8.2f} ${m['mae']:>6.2f} ${m['rmse']:>6.2f} {m['n_samples']:>5}"
        )
    print()


def plot_predictions(
    y_true: np.ndarray,
    lower_bounds: np.ndarray,
    y_pred: np.ndarray,
    dates: list,
    tickers: list[str],
    range_width: float = RANGE_WIDTH,
    save_dir: Path | None = None,
) -> None:
    """Plot actual close vs predicted range for each stock."""
    if save_dir is None:
        save_dir = ARTIFACTS_DIR / "plots"
    save_dir.mkdir(parents=True, exist_ok=True)

    unique_tickers = sorted(set(tickers))

    for t in unique_tickers:
        mask = [tk == t for tk in tickers]
        mask = np.array(mask)
        if not np.any(mask):
            continue

        t_dates = [d for d, m in zip(dates, mask) if m]
        t_true = y_true[mask]
        t_pred = y_pred[mask]
        t_lb = lower_bounds[mask]
        t_ub = t_lb + range_width

        fig, ax = plt.subplots(figsize=(14, 6))
        ax.fill_between(
            range(len(t_dates)),
            t_lb,
            t_ub,
            alpha=0.3,
            color="blue",
            label="Predicted range",
        )
        ax.plot(range(len(t_dates)), t_true, "k.-", markersize=3, label="Actual close")
        ax.plot(range(len(t_dates)), t_pred, "r--", alpha=0.5, linewidth=0.8, label="Point prediction")

        # Mark violations
        violations = t_true < t_lb
        if np.any(violations):
            viol_idx = np.where(violations)[0]
            ax.scatter(viol_idx, t_true[violations], color="red", s=20, zorder=5, label="Violations")

        # X-axis labels (show every ~10th date)
        step = max(1, len(t_dates) // 15)
        tick_positions = list(range(0, len(t_dates), step))
        tick_labels = [str(t_dates[i].date()) if hasattr(t_dates[i], "date") else str(t_dates[i]) for i in tick_positions]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=8)

        coverage = np.mean((t_true >= t_lb) & (t_true <= t_ub))
        ax.set_title(f"{t} — Friday Close Predictions (coverage: {coverage:.1%})")
        ax.set_ylabel("Price ($)")
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        path = save_dir / f"{t}_predictions.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        logger.info(f"Saved plot: {path}")

    # Combined summary plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for idx, t in enumerate(unique_tickers[:6]):
        ax = axes[idx // 3, idx % 3]
        mask = np.array([tk == t for tk in tickers])
        if not np.any(mask):
            continue
        t_true = y_true[mask]
        t_lb = lower_bounds[mask]
        t_ub = t_lb + range_width

        ax.fill_between(range(np.sum(mask)), t_lb, t_ub, alpha=0.3, color="blue")
        ax.plot(range(np.sum(mask)), t_true, "k.-", markersize=2)
        coverage = np.mean((t_true >= t_lb) & (t_true <= t_ub))
        ax.set_title(f"{t} ({coverage:.0%})")
        ax.grid(True, alpha=0.3)

    fig.suptitle("Friday Close Range Predictions — Test Set", fontsize=14)
    fig.tight_layout()
    path = save_dir / "summary.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved summary plot: {path}")
