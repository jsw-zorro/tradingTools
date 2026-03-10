"""Configuration for Friday close range predictor."""

from pathlib import Path

# --- Paths ---
PROJECT_DIR = Path(__file__).parent
ARTIFACTS_DIR = PROJECT_DIR / "artifacts"
AG_MODEL_DIR = ARTIFACTS_DIR / "ag_model"
AG_MODEL_COMPACT_DIR = ARTIFACTS_DIR / "ag_model_compact"
CONFORMAL_PATH = ARTIFACTS_DIR / "conformal.json"

# Best individual model (outperforms full ensemble at 1/10th the size)
PREFERRED_MODEL = "RandomForestMSE_BAG_L1_FULL"

# --- Tickers ---
TARGET_TICKERS = [
    "GOOGL", "TSLA", "NVDA", "AAPL", "BABA", "META",  # original 6
    "AMZN", "MSFT", "AMD",                              # added in Run 5
    "NFLX", "COST", "JPM", "V", "DIS", "PYPL",         # added in Run 6
]
OOS_TICKERS = ["SQ", "UBER"]  # Out-of-sample test only (not trained on)
CONTEXT_TICKERS = ["SPY", "^VIX"]
ALL_TICKERS = TARGET_TICKERS + OOS_TICKERS + CONTEXT_TICKERS

# --- Data ---
DATA_START = "2019-01-01"
DATA_END = "2025-03-01"
LOOKBACK_DAYS = 30  # Trading days of history per sample
CUTOFF_DOWS = [0, 1, 2, 3]  # Mon, Tue, Wed, Thu — generate a sample for each cutoff day

# --- Walk-forward splits (by fraction of sorted samples) ---
TRAIN_FRAC = 0.60
CAL_FRAC = 0.20  # calibration
TEST_FRAC = 0.20

# --- AutoGluon ---
AG_PRESETS = "best_quality"
AG_EVAL_METRIC = "mean_absolute_error"
AG_TIME_LIMIT = 3600  # seconds

# --- Conformal prediction ---
CONFORMAL_COVERAGE = 0.90  # target coverage
CONFORMAL_ALPHA = 0.03     # one-sided lower quantile — tighter than 0.05 for better violation control
RANGE_WIDTH = 10.0         # $10 range

# --- Put-selling risk levels ---
PUT_RISK_LEVELS = [0.05, 0.10, 0.15, 0.20]
PUT_RISK_LABELS = ["Conservative (5%)", "Moderate (10%)", "Aggressive (15%)", "Very Aggressive (20%)"]

# --- Inference ---
PREDICTION_LOOKBACK = 30   # trading days to fetch for live prediction
