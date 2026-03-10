"""Inference script: ticker -> $10 predicted range for Friday close."""

import argparse
import logging
from datetime import timedelta

import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor

from config import AG_MODEL_DIR, RANGE_WIDTH, TARGET_TICKERS
from conformal import load_conformal, predict_range
from data import fetch_recent_data
from features import compute_features_for_sample

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def predict_friday_range(ticker: str) -> dict:
    """Predict this Friday's close range for a given ticker.

    Model predicts the *return* (friday_close / current_price - 1),
    then we convert back to dollar prices.
    """
    # Load model and conformal data
    predictor = TabularPredictor.load(str(AG_MODEL_DIR))
    conformal_data = load_conformal()

    # Fetch recent data
    recent = fetch_recent_data(ticker)
    window = recent["window"]
    current_price = float(window["Close"].values[-1])

    # Compute upcoming Friday and cutoff info
    now = pd.Timestamp.now()
    today_dow = now.dayofweek  # 0=Mon … 4=Fri
    if today_dow <= 4:
        days_until_friday = 4 - today_dow
    else:
        days_until_friday = 4 + (7 - today_dow)
    if days_until_friday == 0 and now.hour >= 16:
        days_until_friday = 7
    upcoming_friday = (now + timedelta(days=days_until_friday)).normalize()

    cutoff_dow = min(today_dow, 3)
    last_window_date = window.index[-1]
    cal_days_to_friday = (upcoming_friday - last_window_date).days

    # Build features
    sample = {
        "ticker": ticker,
        "friday_date": upcoming_friday,
        "cutoff_dow": cutoff_dow,
        "days_to_friday": cal_days_to_friday,
        "window": window,
        "context": recent["context"],
    }
    features = compute_features_for_sample(sample)
    X = pd.DataFrame([features])
    X = X.fillna(0.0)

    # Predict return, convert to dollar price
    predicted_return = float(predictor.predict(X).values[0])
    predicted_close = current_price * (1 + predicted_return)

    # Conformal range (return-space conformal → dollar prices)
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict Friday close range")
    parser.add_argument(
        "--ticker", "-t",
        required=True,
        help=f"Stock ticker (trained on: {', '.join(TARGET_TICKERS)})",
    )
    args = parser.parse_args()

    ticker = args.ticker.upper()
    if ticker not in TARGET_TICKERS:
        logger.warning(
            f"{ticker} was not in training set ({TARGET_TICKERS}). "
            f"Prediction may be less reliable."
        )

    result = predict_friday_range(ticker)

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


if __name__ == "__main__":
    main()
