"""Main training script for Friday close range predictor.

End-to-end pipeline: fetch data -> build features -> train AutoGluon
(predicting friday *return* not absolute price) -> calibrate conformal
-> evaluate on test set -> evaluate on NFLX OOS -> save artifacts.

Run 4 improvements:
- good_quality preset (1-layer stacking, less overfitting vs best_quality's 3-layer)
- Feature pruning: remove redundant features detected in Run 3
- NFLX out-of-sample evaluation (unseen ticker)
- Percentage MAE reporting for cross-stock comparison
"""

import logging
import shutil
import sys
import time

import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor

from config import ARTIFACTS_DIR, AG_MODEL_DIR, OOS_TICKERS
from conformal import fit_conformal, predict_range, save_conformal
from data import build_samples, fetch_all_data, walk_forward_split
from evaluate import compute_metrics, plot_predictions, print_metrics
from features import build_feature_dataframe

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Features to drop (identified as redundant/harmful in Run 3):
# - roc_5d, roc_10d: exact duplicates of return_5d, return_10d (AutoGluon detected)
# - sma_5_20_cross: binary feature, unstable with small sample
# - obv_5d_change: complex volume feature, negative importance
# - dist_sma_5, dist_sma_10: correlated with returns, negative importance
FEATURES_TO_DROP = [
    "roc_5d", "roc_10d",
    "sma_5_20_cross",
    "obv_5d_change",
    "dist_sma_5", "dist_sma_10",
]


def _drop_features(X: pd.DataFrame) -> pd.DataFrame:
    """Drop known-bad features."""
    to_drop = [f for f in FEATURES_TO_DROP if f in X.columns]
    if to_drop:
        logger.info(f"  Dropping {len(to_drop)} features: {to_drop}")
        X = X.drop(columns=to_drop)
    return X


def _build_oos_samples(data: dict[str, pd.DataFrame]) -> list[dict]:
    """Build samples for OOS tickers (same logic as build_samples but for OOS only)."""
    from config import CUTOFF_DOWS, LOOKBACK_DAYS
    from datetime import timedelta

    samples = []
    spy_df = data.get("SPY")
    vix_df = data.get("^VIX")

    for ticker in OOS_TICKERS:
        df = data.get(ticker)
        if df is None:
            logger.warning(f"No data for OOS ticker {ticker}")
            continue
        fridays = df.index[df.index.dayofweek == 4]

        for friday in fridays:
            week_start = friday - timedelta(days=friday.dayofweek)

            for cutoff_dow in CUTOFF_DOWS:
                target_calendar_day = week_start + timedelta(days=cutoff_dow)
                available = df.index[df.index <= target_calendar_day]
                if len(available) < LOOKBACK_DAYS:
                    continue
                cutoff_date = available[-1]
                cutoff_idx = df.index.get_loc(cutoff_date)
                if cutoff_idx < LOOKBACK_DAYS - 1:
                    continue
                window_start_idx = cutoff_idx - LOOKBACK_DAYS + 1
                window = df.iloc[window_start_idx : cutoff_idx + 1].copy()

                context = {}
                for ctx_ticker, ctx_df in [("SPY", spy_df), ("VIX", vix_df)]:
                    if ctx_df is None:
                        continue
                    ctx_window = ctx_df.loc[
                        (ctx_df.index >= window.index[0])
                        & (ctx_df.index <= window.index[-1])
                    ]
                    context[ctx_ticker] = ctx_window

                days_to_friday = (friday - cutoff_date).days

                samples.append({
                    "ticker": ticker,
                    "friday_date": friday,
                    "cutoff_dow": cutoff_dow,
                    "days_to_friday": days_to_friday,
                    "target": float(df.loc[friday, "Close"]),
                    "window": window,
                    "context": context,
                })

    samples.sort(key=lambda s: (s["friday_date"], s["cutoff_dow"]))
    logger.info(f"Built {len(samples)} OOS samples for {OOS_TICKERS}")
    return samples


def main() -> None:
    t0 = time.time()

    # --- 1. Fetch data ---
    logger.info("=" * 60)
    logger.info("STEP 1: Fetching data (including OOS tickers)")
    logger.info("=" * 60)
    data = fetch_all_data()
    for ticker, df in data.items():
        logger.info(f"  {ticker}: {len(df)} rows ({df.index[0].date()} to {df.index[-1].date()})")

    # --- 2. Build samples ---
    logger.info("=" * 60)
    logger.info("STEP 2: Building samples")
    logger.info("=" * 60)
    samples = build_samples(data)
    train_samples, cal_samples, test_samples = walk_forward_split(samples)

    # Build OOS samples (for NFLX evaluation)
    oos_samples = _build_oos_samples(data)

    # --- 3. Feature engineering ---
    logger.info("=" * 60)
    logger.info("STEP 3: Feature engineering")
    logger.info("=" * 60)
    X_train, y_train, bp_train = build_feature_dataframe(train_samples)
    X_cal, y_cal, bp_cal = build_feature_dataframe(cal_samples)
    X_test, y_test, bp_test = build_feature_dataframe(test_samples)

    # OOS features
    if oos_samples:
        X_oos, y_oos, bp_oos = build_feature_dataframe(oos_samples)
    else:
        X_oos, y_oos, bp_oos = pd.DataFrame(), pd.Series(dtype=float), pd.Series(dtype=float)

    # Feature pruning
    X_train = _drop_features(X_train)
    X_cal = _drop_features(X_cal)
    X_test = _drop_features(X_test)
    if len(X_oos) > 0:
        X_oos = _drop_features(X_oos)

    logger.info(f"  Train: {X_train.shape}, Cal: {X_cal.shape}, Test: {X_test.shape}")
    if len(X_oos) > 0:
        logger.info(f"  OOS (NFLX): {X_oos.shape}")
    logger.info(f"  Features ({X_train.shape[1]}): {list(X_train.columns)}")
    logger.info(f"  Target (friday_return) train stats: "
                f"mean={y_train.mean():.4f}, std={y_train.std():.4f}, "
                f"min={y_train.min():.4f}, max={y_train.max():.4f}")

    # --- 4. Train AutoGluon ---
    logger.info("=" * 60)
    logger.info("STEP 4: Training AutoGluon (good_quality, target=return)")
    logger.info("=" * 60)
    logger.info("  Using good_quality preset (1-layer stacking, less overfitting)")

    train_df = X_train.copy()
    train_df["friday_return"] = y_train.values

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    if AG_MODEL_DIR.exists():
        shutil.rmtree(AG_MODEL_DIR)

    predictor = TabularPredictor(
        label="friday_return",
        path=str(AG_MODEL_DIR),
        eval_metric="mean_absolute_error",
        problem_type="regression",
    )

    predictor.fit(
        train_data=train_df,
        presets="good_quality",
        time_limit=3600,
        num_cpus=48,
    )

    # Leaderboard
    logger.info("\nAutoGluon Leaderboard:")
    leaderboard = predictor.leaderboard(silent=True)
    print(leaderboard.to_string())

    # Feature importance
    logger.info("\nFeature Importance (top 20):")
    try:
        importance = predictor.feature_importance(
            pd.concat([X_train, y_train], axis=1).rename(
                columns={y_train.name: "friday_return"}
            ),
            num_shuffle_sets=3,
        )
        print(importance.head(20).to_string())
    except Exception as e:
        logger.warning(f"Feature importance failed: {e}")

    # --- 5. Calibrate conformal prediction ---
    logger.info("=" * 60)
    logger.info("STEP 5: Conformal calibration (return-space)")
    logger.info("=" * 60)

    cal_pred_return = predictor.predict(X_cal).values

    conformal_data = fit_conformal(
        y_true_return=y_cal.values,
        y_pred_return=cal_pred_return,
        samples=cal_samples,
    )
    save_conformal(conformal_data)

    # --- 5b. Check train set fit (for overfitting diagnosis) ---
    logger.info("=" * 60)
    logger.info("STEP 5b: Train set fit (overfitting check)")
    logger.info("=" * 60)
    train_pred_return = predictor.predict(X_train).values
    train_return_mae = np.mean(np.abs(train_pred_return - y_train.values))
    train_pred_price = bp_train.values * (1 + train_pred_return)
    train_true_price = bp_train.values * (1 + y_train.values)
    train_dollar_mae = np.mean(np.abs(train_pred_price - train_true_price))
    train_pct_mae = np.mean(np.abs(train_pred_return - y_train.values) / np.maximum(np.abs(y_train.values), 0.001)) * 100
    logger.info(f"  Train return MAE: {train_return_mae:.4f} ({train_return_mae*100:.2f}%)")
    logger.info(f"  Train dollar MAE: ${train_dollar_mae:.2f}")

    # --- 6. Evaluate on test set ---
    logger.info("=" * 60)
    logger.info("STEP 6: Test set evaluation")
    logger.info("=" * 60)

    test_pred_return = predictor.predict(X_test).values
    test_pred_price = bp_test.values * (1 + test_pred_return)
    test_true_price = bp_test.values * (1 + y_test.values)
    test_tickers = [s["ticker"] for s in test_samples[: len(y_test)]]
    test_dates = [s["friday_date"] for s in test_samples[: len(y_test)]]

    # Return-space MAE
    return_mae = np.mean(np.abs(test_pred_return - y_test.values))
    logger.info(f"  Return-space MAE: {return_mae:.4f} ({return_mae*100:.2f}%)")
    logger.info(f"  Train vs Test return MAE gap: {train_return_mae:.4f} vs {return_mae:.4f} "
                f"(ratio: {return_mae/max(train_return_mae, 1e-6):.2f}x)")

    # Dollar-space MAE
    dollar_mae = np.mean(np.abs(test_pred_price - test_true_price))
    logger.info(f"  Dollar-space MAE: ${dollar_mae:.2f}")

    # Per-ticker percentage MAE
    logger.info("\n  Per-ticker return MAE:")
    for t in sorted(set(test_tickers)):
        mask = np.array([tk == t for tk in test_tickers])
        t_return_mae = np.mean(np.abs(test_pred_return[mask] - y_test.values[mask]))
        t_dollar_mae = np.mean(np.abs(test_pred_price[mask] - test_true_price[mask]))
        logger.info(f"    {t}: return MAE={t_return_mae*100:.2f}%, dollar MAE=${t_dollar_mae:.2f}")

    # Compute lower bounds
    lower_bounds = np.array([
        predict_range(
            y_pred_return=test_pred_return[i],
            current_price=bp_test.values[i],
            recent_window=test_samples[i]["window"],
            conformal_data=conformal_data,
        )[0]
        for i in range(len(test_pred_return))
    ])

    metrics = compute_metrics(
        y_true=test_true_price,
        lower_bounds=lower_bounds,
        y_pred=test_pred_price,
        tickers=test_tickers,
    )
    print_metrics(metrics)

    # --- 7. NFLX Out-of-Sample Evaluation ---
    if len(X_oos) > 0:
        logger.info("=" * 60)
        logger.info("STEP 7: NFLX Out-of-Sample Evaluation")
        logger.info("=" * 60)

        oos_pred_return = predictor.predict(X_oos).values
        oos_pred_price = bp_oos.values * (1 + oos_pred_return)
        oos_true_price = bp_oos.values * (1 + y_oos.values)
        oos_tickers = [s["ticker"] for s in oos_samples[: len(y_oos)]]
        oos_dates = [s["friday_date"] for s in oos_samples[: len(y_oos)]]

        oos_return_mae = np.mean(np.abs(oos_pred_return - y_oos.values))
        oos_dollar_mae = np.mean(np.abs(oos_pred_price - oos_true_price))
        logger.info(f"  NFLX return MAE: {oos_return_mae:.4f} ({oos_return_mae*100:.2f}%)")
        logger.info(f"  NFLX dollar MAE: ${oos_dollar_mae:.2f}")
        logger.info(f"  NFLX RMSE: ${np.sqrt(np.mean((oos_pred_price - oos_true_price)**2)):.2f}")

        # Conformal bounds for NFLX
        oos_lower_bounds = np.array([
            predict_range(
                y_pred_return=oos_pred_return[i],
                current_price=bp_oos.values[i],
                recent_window=oos_samples[i]["window"],
                conformal_data=conformal_data,
            )[0]
            for i in range(len(oos_pred_return))
        ])

        oos_metrics = compute_metrics(
            y_true=oos_true_price,
            lower_bounds=oos_lower_bounds,
            y_pred=oos_pred_price,
            tickers=oos_tickers,
        )
        print("\n--- NFLX Out-of-Sample Results ---")
        print_metrics(oos_metrics)

        # Plot NFLX
        plot_predictions(
            y_true=oos_true_price,
            lower_bounds=oos_lower_bounds,
            y_pred=oos_pred_price,
            dates=oos_dates,
            tickers=oos_tickers,
            save_dir=ARTIFACTS_DIR / "plots" / "oos",
        )
    else:
        logger.warning("No OOS samples available")

    # --- 8. Generate plots ---
    logger.info("=" * 60)
    logger.info("STEP 8: Generating plots")
    logger.info("=" * 60)

    plot_predictions(
        y_true=test_true_price,
        lower_bounds=lower_bounds,
        y_pred=test_pred_price,
        dates=test_dates,
        tickers=test_tickers,
    )

    # --- Summary ---
    elapsed = time.time() - t0
    logger.info("=" * 60)
    logger.info(f"DONE in {elapsed / 60:.1f} minutes")
    logger.info(f"Model saved to: {AG_MODEL_DIR}")
    logger.info(f"Conformal data saved to: {ARTIFACTS_DIR / 'conformal.json'}")
    logger.info(f"Plots saved to: {ARTIFACTS_DIR / 'plots'}")
    logger.info("=" * 60)

    overall = metrics["overall"]
    logger.info(f"\n=== COMPARISON: Run 3 vs Run 4 ===")
    logger.info(f"  Train return MAE: {train_return_mae*100:.2f}%")
    logger.info(f"  Test return MAE:  {return_mae*100:.2f}%")
    logger.info(f"  Overfit ratio:    {return_mae/max(train_return_mae, 1e-6):.2f}x")
    logger.info(f"  Test dollar MAE:  ${overall['mae']:.2f}")
    logger.info(f"  Coverage:         {overall['coverage']:.1%}")
    logger.info(f"  Violation rate:   {overall['lower_violation_rate']:.1%}")
    if len(X_oos) > 0:
        logger.info(f"  NFLX OOS MAE:     ${oos_dollar_mae:.2f}")
        logger.info(f"  NFLX OOS return:  {oos_return_mae*100:.2f}%")

    if overall["coverage"] < 0.85:
        logger.warning(f"Coverage {overall['coverage']:.1%} is below 85% target!")
    if overall["lower_violation_rate"] > 0.05:
        logger.warning(
            f"Lower violation rate {overall['lower_violation_rate']:.1%} "
            f"is above 5% target!"
        )


if __name__ == "__main__":
    main()
