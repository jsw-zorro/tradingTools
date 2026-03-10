"""Split conformal prediction for calibrated lower bounds.

All conformal math is done in RETURN SPACE (percentage) so the bounds
scale naturally with stock price.  At inference the percentage bounds
are converted back to dollar prices.
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from config import CONFORMAL_ALPHA, CONFORMAL_PATH, RANGE_WIDTH

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------

def fit_conformal(
    y_true_return: np.ndarray,
    y_pred_return: np.ndarray,
    samples: list[dict],
    alpha: float = CONFORMAL_ALPHA,
) -> dict:
    """Fit conformal prediction on calibration set in RETURN SPACE.

    Parameters
    ----------
    y_true_return : actual friday returns  (friday_close / current_price - 1)
    y_pred_return : predicted friday returns
    samples       : raw sample dicts (for per-ticker stats)
    alpha         : lower-tail probability (e.g. 0.05 for 5 % violation)

    Returns dict persisted to disk.
    """
    residuals = y_true_return - y_pred_return  # signed, in return space

    # Locally-weighted: scale by recent return volatility so calm stocks
    # get tighter bounds and volatile stocks get wider bounds.
    vol_scales = _return_vol_scales(samples)
    scaled_residuals = residuals / vol_scales

    quantile_value = float(np.quantile(scaled_residuals, alpha))

    logger.info(
        f"Conformal fit: {len(residuals)} cal samples, "
        f"alpha={alpha}, quantile={quantile_value:.4f}"
    )
    logger.info(
        f"  Return residual stats: mean={np.mean(residuals):.4f}, "
        f"std={np.std(residuals):.4f}, min={np.min(residuals):.4f}, "
        f"max={np.max(residuals):.4f}"
    )

    return {
        "scaled_residuals": sorted(scaled_residuals.tolist()),
        "quantile": quantile_value,
        "alpha": alpha,
    }


def _return_vol_scales(samples: list[dict]) -> np.ndarray:
    """Per-sample volatility scale in return space."""
    scales = []
    for s in samples:
        close = s["window"]["Close"].values
        if len(close) >= 6:
            rv = np.std(np.diff(np.log(close[-6:])))
        else:
            rv = 0.01
        scales.append(max(rv, 0.001))
    return np.array(scales)


# ---------------------------------------------------------------------------
# Save / Load
# ---------------------------------------------------------------------------

def save_conformal(conformal_data: dict, path: Path = CONFORMAL_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(conformal_data, f, indent=2)
    logger.info(f"Saved conformal data to {path}")


def load_conformal(path: Path = CONFORMAL_PATH) -> dict:
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def compute_vol_scale(recent_window: pd.DataFrame) -> float:
    """Compute local volatility scale from recent price window."""
    close = recent_window["Close"].values
    rv = np.std(np.diff(np.log(close[-6:]))) if len(close) >= 6 else 0.01
    return max(rv, 0.001)


def predict_lower_bound(
    y_pred_return: float,
    recent_window: pd.DataFrame,
    conformal_data: dict,
) -> float:
    """Conformal lower bound in RETURN SPACE.

    lower_return = predicted_return + quantile * vol_scale
    (quantile < 0, so this subtracts)
    """
    quantile = conformal_data["quantile"]
    vol_scale = compute_vol_scale(recent_window)
    return y_pred_return + quantile * vol_scale


def predict_multi_level_bounds(
    y_pred_return: float,
    current_price: float,
    recent_window: pd.DataFrame,
    conformal_data: dict,
    alphas: list[float],
) -> list[dict]:
    """Compute conformal lower bounds at multiple alpha (risk) levels.

    Returns list of dicts with keys: alpha, lower_return, lower_price.
    """
    vol_scale = compute_vol_scale(recent_window)
    scaled_residuals = np.array(conformal_data["scaled_residuals"])
    results = []
    for alpha in alphas:
        quantile_val = float(np.quantile(scaled_residuals, alpha))
        lower_return = y_pred_return + quantile_val * vol_scale
        lower_price = current_price * (1 + lower_return)
        results.append({
            "alpha": alpha,
            "lower_return": lower_return,
            "lower_price": lower_price,
        })
    return results


def estimate_expected_loss(
    strike: float,
    y_pred_return: float,
    current_price: float,
    recent_window: pd.DataFrame,
    conformal_data: dict,
) -> dict:
    """Estimate put-selling expected loss using the full residual distribution.

    Uses empirical distribution of scaled residuals to simulate possible
    Friday closes and compute expected loss if assigned.

    Returns dict with prob_itm, expected_loss_per_share, expected_loss_per_contract.
    """
    vol_scale = compute_vol_scale(recent_window)
    scaled_residuals = np.array(conformal_data["scaled_residuals"])

    # Simulate actual returns from residual distribution
    actual_returns = y_pred_return + scaled_residuals * vol_scale
    strike_return = strike / current_price - 1

    # Loss per share when assigned: max(strike - friday_close, 0) / current_price = max(strike_return - actual_return, 0)
    return_losses = np.maximum(strike_return - actual_returns, 0)
    prob_itm = float(np.mean(actual_returns < strike_return))
    expected_loss_per_share = float(current_price * np.mean(return_losses))
    expected_loss_per_contract = expected_loss_per_share * 100

    return {
        "prob_itm": prob_itm,
        "expected_loss_per_share": expected_loss_per_share,
        "expected_loss_per_contract": expected_loss_per_contract,
    }


def predict_range(
    y_pred_return: float,
    current_price: float,
    recent_window: pd.DataFrame,
    conformal_data: dict,
    range_width: float = RANGE_WIDTH,
) -> tuple[float, float]:
    """Predict [lower_bound, lower_bound + range_width] in DOLLAR SPACE.

    lower_bound_$ = current_price * (1 + lower_return)
    upper_bound_$ = lower_bound_$ + range_width
    """
    lb_return = predict_lower_bound(y_pred_return, recent_window, conformal_data)
    lb_price = current_price * (1 + lb_return)
    return (lb_price, lb_price + range_width)
