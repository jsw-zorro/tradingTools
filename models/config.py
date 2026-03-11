"""Configuration for Friday close range predictor."""

from pathlib import Path

# --- Paths ---
PROJECT_DIR = Path(__file__).parent
ARTIFACTS_DIR = PROJECT_DIR / "artifacts"
AG_MODEL_DIR = ARTIFACTS_DIR / "ag_model"
AG_MODEL_COMPACT_DIR = ARTIFACTS_DIR / "ag_model_compact"
CONFORMAL_PATH = ARTIFACTS_DIR / "conformal.json"

# Best individual model for compact deployment
PREFERRED_MODEL = "LightGBMXT_BAG_L1_FULL"

# --- GICS Sectors ---
SECTORS = [
    "Technology", "Consumer_Discretionary", "Communication_Services",
    "Health_Care", "Financials", "Consumer_Staples", "Industrials",
    "Energy", "Utilities", "Real_Estate", "Materials",
]

# --- Tickers (100 training + 5 OOS) ---
TARGET_TICKERS = [
    # Information Technology (21)
    "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AVGO", "AMD", "ADBE",
    "CRM", "INTC", "CSCO", "ORCL", "TXN", "QCOM", "AMAT", "MU",
    "LRCX", "KLAC", "NOW", "PANW", "CDNS",
    # Consumer Discretionary (12)
    "AMZN", "TSLA", "HD", "MCD", "NKE", "LOW", "TJX", "SBUX",
    "BKNG", "ROST", "CMG", "BABA",
    # Communication Services (5)
    "DIS", "NFLX", "CMCSA", "T", "VZ",
    # Health Care (11)
    "UNH", "JNJ", "LLY", "PFE", "ABT", "MRK", "TMO", "AMGN",
    "ISRG", "GILD", "DHR",
    # Financials (13)
    "JPM", "V", "MA", "BAC", "GS", "MS", "BLK", "AXP",
    "SPGI", "C", "PYPL", "ICE", "SCHW",
    # Consumer Staples (8)
    "PG", "KO", "PEP", "COST", "WMT", "PM", "CL", "MDLZ",
    # Industrials (11)
    "CAT", "GE", "HON", "UNP", "RTX", "LMT", "BA", "DE",
    "WM", "FDX", "ETN",
    # Energy (6)
    "XOM", "CVX", "COP", "SLB", "EOG", "OXY",
    # Utilities (4)
    "NEE", "DUK", "SO", "AEP",
    # Real Estate (4)
    "PLD", "AMT", "CCI", "EQIX",
    # Materials (5)
    "LIN", "APD", "SHW", "FCX", "NEM",
]
OOS_TICKERS = ["UBER", "PLTR", "ABNB", "SPOT", "COIN"]
CONTEXT_TICKERS = ["SPY", "^VIX"]
ALL_TICKERS = TARGET_TICKERS + OOS_TICKERS + CONTEXT_TICKERS

# Ticker -> GICS sector mapping (for sector one-hot features)
_SECTOR_TICKERS = {
    "Technology": [
        "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AVGO", "AMD", "ADBE",
        "CRM", "INTC", "CSCO", "ORCL", "TXN", "QCOM", "AMAT", "MU",
        "LRCX", "KLAC", "NOW", "PANW", "CDNS",
        "PLTR",  # OOS
    ],
    "Consumer_Discretionary": [
        "AMZN", "TSLA", "HD", "MCD", "NKE", "LOW", "TJX", "SBUX",
        "BKNG", "ROST", "CMG", "BABA",
        "UBER", "ABNB",  # OOS
    ],
    "Communication_Services": [
        "DIS", "NFLX", "CMCSA", "T", "VZ",
        "SPOT",  # OOS
    ],
    "Health_Care": [
        "UNH", "JNJ", "LLY", "PFE", "ABT", "MRK", "TMO", "AMGN",
        "ISRG", "GILD", "DHR",
    ],
    "Financials": [
        "JPM", "V", "MA", "BAC", "GS", "MS", "BLK", "AXP",
        "SPGI", "C", "PYPL", "ICE", "SCHW",
        "COIN",  # OOS
    ],
    "Consumer_Staples": [
        "PG", "KO", "PEP", "COST", "WMT", "PM", "CL", "MDLZ",
    ],
    "Industrials": [
        "CAT", "GE", "HON", "UNP", "RTX", "LMT", "BA", "DE",
        "WM", "FDX", "ETN",
    ],
    "Energy": ["XOM", "CVX", "COP", "SLB", "EOG", "OXY"],
    "Utilities": ["NEE", "DUK", "SO", "AEP"],
    "Real_Estate": ["PLD", "AMT", "CCI", "EQIX"],
    "Materials": ["LIN", "APD", "SHW", "FCX", "NEM"],
}
TICKER_SECTORS = {}
for _sector, _tickers in _SECTOR_TICKERS.items():
    for _t in _tickers:
        TICKER_SECTORS[_t] = _sector

# --- Data ---
DATA_START = "2019-01-01"
DATA_END = "2026-03-01"
LOOKBACK_DAYS = 60  # Trading days of history per sample
CUTOFF_DOWS = [0, 1, 2, 3]  # Mon, Tue, Wed, Thu — generate a sample for each cutoff day

# --- Walk-forward splits (by fraction of sorted samples) ---
TRAIN_FRAC = 0.60
CAL_FRAC = 0.20  # calibration
TEST_FRAC = 0.20

# --- AutoGluon ---
AG_PRESETS = "good_quality"
AG_EVAL_METRIC = "mean_absolute_error"
AG_TIME_LIMIT = 10800  # 3 hours — needed for ~130k samples

# --- Conformal prediction ---
CONFORMAL_COVERAGE = 0.90  # target coverage
CONFORMAL_ALPHA = 0.03     # one-sided lower quantile — tighter than 0.05 for better violation control
RANGE_WIDTH = 10.0         # $10 range

# --- Put-selling risk levels ---
PUT_RISK_LEVELS = [0.05, 0.10, 0.15, 0.20]
PUT_RISK_LABELS = ["Conservative (5%)", "Moderate (10%)", "Aggressive (15%)", "Very Aggressive (20%)"]

# --- Inference ---
PREDICTION_LOOKBACK = 60   # trading days to fetch for live prediction
