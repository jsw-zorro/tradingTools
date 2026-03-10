"""Inference script: ticker -> put-selling risk ladder for Friday close."""

import argparse
import logging
from datetime import timedelta

import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor

from config import (
    AG_MODEL_COMPACT_DIR,
    AG_MODEL_DIR,
    PREFERRED_MODEL,
    PUT_RISK_LABELS,
    PUT_RISK_LEVELS,
    RANGE_WIDTH,
    TARGET_TICKERS,
)
from conformal import (
    estimate_expected_loss,
    load_conformal,
    predict_multi_level_bounds,
    predict_range,
)
from data import fetch_recent_data
from features import compute_features_for_sample

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _load_model_and_data(ticker: str) -> tuple:
    """Load model, conformal data, and fetch recent market data.

    Returns (predictor, conformal_data, window, context, current_price,
             upcoming_friday, last_window_date, cutoff_dow, cal_days_to_friday).
    """
    model_dir = AG_MODEL_COMPACT_DIR if AG_MODEL_COMPACT_DIR.exists() else AG_MODEL_DIR
    predictor = TabularPredictor.load(str(model_dir))
    conformal_data = load_conformal()

    recent = fetch_recent_data(ticker)
    window = recent["window"]
    current_price = float(window["Close"].values[-1])

    now = pd.Timestamp.now()
    today_dow = now.dayofweek
    if today_dow <= 4:
        days_until_friday = 4 - today_dow
    else:
        days_until_friday = 4 + (7 - today_dow)
    if days_until_friday == 0 and now.hour >= 16:
        days_until_friday = 7
    upcoming_friday = (now + timedelta(days=days_until_friday)).normalize()

    last_window_date = window.index[-1]
    cutoff_dow = min(last_window_date.dayofweek, 3)
    cal_days_to_friday = (upcoming_friday - last_window_date).days

    return (
        predictor, conformal_data, window, recent["context"],
        current_price, upcoming_friday, last_window_date,
        cutoff_dow, cal_days_to_friday,
    )


def _predict_return(
    predictor, ticker, upcoming_friday, cutoff_dow,
    cal_days_to_friday, window, context,
) -> float:
    """Build features and predict return."""
    sample = {
        "ticker": ticker,
        "friday_date": upcoming_friday,
        "cutoff_dow": cutoff_dow,
        "days_to_friday": cal_days_to_friday,
        "window": window,
        "context": context,
    }
    features = compute_features_for_sample(sample)
    X = pd.DataFrame([features]).fillna(0.0)
    model_name = PREFERRED_MODEL if PREFERRED_MODEL in predictor.model_names() else None
    return float(predictor.predict(X, model=model_name).values[0])


def predict_friday_range(ticker: str) -> dict:
    """Predict this Friday's close range for a given ticker.

    Model predicts the *return* (friday_close / current_price - 1),
    then we convert back to dollar prices.
    """
    (predictor, conformal_data, window, context,
     current_price, upcoming_friday, last_window_date,
     cutoff_dow, cal_days_to_friday) = _load_model_and_data(ticker)

    predicted_return = _predict_return(
        predictor, ticker, upcoming_friday, cutoff_dow,
        cal_days_to_friday, window, context,
    )
    predicted_close = current_price * (1 + predicted_return)

    lower_bound, upper_bound = predict_range(
        y_pred_return=predicted_return,
        current_price=current_price,
        recent_window=window,
        conformal_data=conformal_data,
        range_width=RANGE_WIDTH,
    )

    dow_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    return {
        "ticker": ticker,
        "friday_date": upcoming_friday,
        "predicted_return": predicted_return,
        "predicted_close": predicted_close,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "current_price": current_price,
        "safe_put_strike": int(lower_bound / 5) * 5,
        "data_through": last_window_date,
        "cutoff_day": dow_names[cutoff_dow],
        "days_to_friday": cal_days_to_friday,
    }


def predict_put_ladder(ticker: str, fetch_premiums: bool = True) -> dict:
    """Predict put-selling risk ladder for this Friday.

    For each risk level (alpha), computes:
    - Conformal lower bound -> strike price
    - P(ITM) and expected loss from residual distribution
    - Live option premium (if fetch_premiums=True)
    - Expected value per contract

    Returns dict with prediction info and ladder rows.
    """
    from options import fetch_put_premiums, get_friday_expiry, round_to_strike

    (predictor, conformal_data, window, context,
     current_price, upcoming_friday, last_window_date,
     cutoff_dow, cal_days_to_friday) = _load_model_and_data(ticker)

    predicted_return = _predict_return(
        predictor, ticker, upcoming_friday, cutoff_dow,
        cal_days_to_friday, window, context,
    )
    predicted_close = current_price * (1 + predicted_return)

    # Multi-level conformal bounds
    bounds = predict_multi_level_bounds(
        y_pred_return=predicted_return,
        current_price=current_price,
        recent_window=window,
        conformal_data=conformal_data,
        alphas=PUT_RISK_LEVELS,
    )

    # Build ladder rows
    ladder = []
    strikes = []
    for bound, label in zip(bounds, PUT_RISK_LABELS):
        strike = round_to_strike(bound["lower_price"])
        strikes.append(strike)

        loss = estimate_expected_loss(
            strike=strike,
            y_pred_return=predicted_return,
            current_price=current_price,
            recent_window=window,
            conformal_data=conformal_data,
        )

        ladder.append({
            "label": label,
            "alpha": bound["alpha"],
            "lower_price": bound["lower_price"],
            "strike": strike,
            "prob_itm": loss["prob_itm"],
            "expected_loss_per_contract": loss["expected_loss_per_contract"],
            "bid": None,
            "ev_per_contract": None,
        })

    # Fetch live premiums
    if fetch_premiums:
        expiry = get_friday_expiry(ticker, upcoming_friday)
        if expiry:
            premiums = fetch_put_premiums(ticker, expiry, strikes)
            for row in ladder:
                prem = premiums.get(row["strike"])
                if prem:
                    row["bid"] = prem["bid"]
                    row["ev_per_contract"] = (
                        prem["bid"] * 100 - row["expected_loss_per_contract"]
                    )

    # Find recommendation: highest positive EV
    best = None
    for row in ladder:
        if row["ev_per_contract"] is not None and row["ev_per_contract"] > 0:
            if best is None or row["ev_per_contract"] > best["ev_per_contract"]:
                best = row

    dow_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    return {
        "ticker": ticker,
        "friday_date": upcoming_friday,
        "predicted_return": predicted_return,
        "predicted_close": predicted_close,
        "current_price": current_price,
        "data_through": last_window_date,
        "cutoff_day": dow_names[cutoff_dow],
        "days_to_friday": cal_days_to_friday,
        "ladder": ladder,
        "recommendation": best,
    }


def _print_put_ladder(result: dict) -> None:
    """Print the put-selling risk ladder."""
    ticker = result["ticker"]
    friday = result["friday_date"].date()
    current = result["current_price"]
    pred_close = result["predicted_close"]
    pred_ret = result["predicted_return"]
    data_through = result["data_through"].date()
    cutoff = result["cutoff_day"]

    print()
    print("=" * 78)
    print(f"  {ticker} Put-Selling Analysis for Friday {friday}")
    print("=" * 78)
    print(f"  Current price:    ${current:.2f}")
    print(f"  Predicted close:  ${pred_close:.2f} ({pred_ret:+.2%})")
    print(f"  Data through:     {data_through} ({cutoff})")
    print()
    print("  RISK LADDER")
    print("  " + "-" * 74)

    has_premiums = any(r["bid"] is not None for r in result["ladder"])
    if has_premiums:
        print(f"  {'Risk Level':<24} {'Strike':>8} {'Bid':>7} {'P(ITM)':>8} {'E[Loss]':>9} {'EV':>10}")
    else:
        print(f"  {'Risk Level':<24} {'Strike':>8} {'P(ITM)':>8} {'E[Loss]':>9}")
    print("  " + "-" * 74)

    rec = result["recommendation"]
    for row in result["ladder"]:
        marker = " <<<"  if rec and row["strike"] == rec["strike"] else ""

        if has_premiums:
            bid_str = f"${row['bid']:.2f}" if row["bid"] is not None else "  n/a"
            ev_str = f"${row['ev_per_contract']:.0f}" if row["ev_per_contract"] is not None else "    n/a"
            print(
                f"  {row['label']:<24} ${row['strike']:>6.0f} {bid_str:>7} "
                f"{row['prob_itm']:>7.1%} ${row['expected_loss_per_contract']:>8.0f} "
                f"{ev_str:>10}{marker}"
            )
        else:
            print(
                f"  {row['label']:<24} ${row['strike']:>6.0f} "
                f"{row['prob_itm']:>7.1%} ${row['expected_loss_per_contract']:>8.0f}{marker}"
            )

    print("  " + "-" * 74)

    if rec:
        print()
        bid_str = f" @ ${rec['bid']:.2f} (${rec['bid'] * 100:.0f}/contract)" if rec["bid"] else ""
        print(f"  >>> RECOMMENDED: Sell {ticker} ${rec['strike']:.0f} put{bid_str}, EV=${rec['ev_per_contract']:.0f}")
    elif has_premiums:
        print()
        print("  >>> No strike has positive expected value at current premiums.")

    print("=" * 78)
    print()


def _print_range(result: dict) -> None:
    """Print the legacy range prediction."""
    print()
    print("=" * 60)
    print(f"  {result['ticker']} Friday {result['friday_date'].date()} Prediction")
    print("=" * 60)
    print(f"  Data through:     {result['data_through'].date()} ({result['cutoff_day']})")
    print(f"  Days to Friday:   {result['days_to_friday']}")
    print(f"  Predicted return: {result['predicted_return']:+.2%}")
    print(f"  Predicted range:  ${result['lower_bound']:.2f} - ${result['upper_bound']:.2f}")
    print(f"  Lower bound:      ${result['lower_bound']:.2f}")
    print(f"  Point prediction: ${result['predicted_close']:.2f}")
    print(f"  Current price:    ${result['current_price']:.2f}")
    print(f"  Safe put strike:  ${result['safe_put_strike']} or below")
    print("=" * 60)
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Friday close predictor for put-selling")
    parser.add_argument(
        "--ticker", "-t",
        required=True,
        help=f"Stock ticker (trained on: {', '.join(TARGET_TICKERS)})",
    )
    parser.add_argument(
        "--no-premiums",
        action="store_true",
        help="Skip fetching live option premiums (offline mode)",
    )
    parser.add_argument(
        "--range",
        action="store_true",
        help="Show legacy $10 range prediction instead of put ladder",
    )
    args = parser.parse_args()

    ticker = args.ticker.upper()
    if ticker not in TARGET_TICKERS:
        logger.warning(
            f"{ticker} was not in training set ({TARGET_TICKERS}). "
            f"Prediction may be less reliable."
        )

    if args.range:
        result = predict_friday_range(ticker)
        _print_range(result)
    else:
        result = predict_put_ladder(ticker, fetch_premiums=not args.no_premiums)
        _print_put_ladder(result)


if __name__ == "__main__":
    main()
