# StrategyLab

Extensible trading strategy backtesting and monitoring platform. Backtest strategies against historical data, run parameter sweeps, monitor live markets, and receive actionable email alerts for manual execution on Robinhood.

Strategies are self-contained plugins. The first included strategy detects VIX spikes and buys UVXY puts to profit from volatility mean reversion.

## Quick Start

```bash
# Install
pip install -e ".[dev]"

# List available strategies
strategylab list-strategies

# Run a backtest with default parameters
strategylab backtest -s vix_uvxy_put

# Run a parameter sweep
strategylab sweep -s vix_uvxy_put

# One-shot live check
strategylab check-now -s vix_uvxy_put

# Start continuous monitoring
strategylab monitor
```

## Project Structure

```
tradingTools/
├── config/
│   ├── settings.yaml                 # Global settings (email, monitoring, backtest)
│   └── strategies/
│       └── vix_uvxy_put.yaml         # Per-strategy params + sweep grid
├── src/strategylab/
│   ├── core/                         # Abstract framework
│   │   ├── base_strategy.py          # BaseStrategy ABC
│   │   ├── models.py                 # Signal, Position, ExitResult, TradeRecord
│   │   └── registry.py              # Plugin registry (@register decorator)
│   ├── data/                         # Shared data layer
│   │   ├── fetcher.py               # yfinance wrappers
│   │   ├── cache.py                 # Local parquet caching
│   │   ├── options.py               # Black-Scholes pricing
│   │   ├── options_chain.py         # Unified pricing (QC data or BS fallback)
│   │   └── qc_fetcher.py           # QuantConnect API client
│   ├── strategies/                   # Strategy plugins
│   │   └── vix_uvxy_put/           # VIX spike -> UVXY put strategy
│   ├── backtest/                     # Strategy-agnostic backtester
│   │   ├── engine.py               # Core backtest loop
│   │   ├── param_sweep.py          # Grid search with multiprocessing
│   │   ├── metrics.py              # Sharpe, drawdown, win rate, etc.
│   │   └── report.py              # HTML/PNG reports
│   ├── monitor/                     # Live market monitoring
│   │   ├── watcher.py              # Polling loop
│   │   ├── alert.py                # Email sender
│   │   └── scheduler.py           # Market-hours scheduling
│   └── cli.py                      # CLI entry point
├── tests/
├── Dockerfile
├── docker-compose.yml
└── output/                          # Reports and data cache (gitignored)
```

## Configuration

Copy `.env.example` to `.env` and fill in your credentials:

```bash
cp .env.example .env
```

### Email Alerts

Set up a Gmail app password for alert delivery:

```
ALERT_EMAIL_SENDER=your-email@gmail.com
ALERT_EMAIL_PASSWORD=your-app-password
```

Update recipients in `config/settings.yaml`:

```yaml
email:
  recipients: [your-email@example.com]
```

Test the configuration:

```bash
strategylab test-email --to your-email@example.com
```

### QuantConnect (Real Options Data)

By default, backtesting uses Black-Scholes estimation for options pricing. If you have a QuantConnect account, you can use real historical options data instead.

```
QC_USER_ID=your-user-id
QC_API_TOKEN=your-api-token
QC_ORGANIZATION_ID=your-organization-id
```

Download options data via the API:

```bash
strategylab download-options -u UVXY --start 2020-01-01 --end 2025-01-01
```

Or import from a CSV (e.g. exported from a QC Research notebook):

```bash
strategylab import-options -f uvxy_options.csv -u UVXY
```

The CSV should have columns: `date`, `expiry`, `strike`, `right` (put/call), `close`. Optional: `volume`, `open_interest`, `implied_volatility`. Column names can be remapped with `--date-col`, `--strike-col`, etc.

Check what data is available:

```bash
strategylab qc-status
```

Once options data is cached, all backtests automatically use real prices with transparent fallback to Black-Scholes where gaps exist. Position metadata tracks `pricing_source` so you know which pricing method was used.

## CLI Reference

| Command | Description |
|---------|-------------|
| `strategylab list-strategies` | Show all registered strategies |
| `strategylab backtest -s <name>` | Run backtest with default params |
| `strategylab sweep -s <name>` | Parameter sweep with ranked results |
| `strategylab monitor` | Continuous monitoring (all enabled strategies) |
| `strategylab check-now -s <name>` | One-shot live check |
| `strategylab test-email --to <addr>` | Verify email config |
| `strategylab download-options` | Download options data from QuantConnect |
| `strategylab import-options` | Import options data from CSV |
| `strategylab qc-status` | Check QC connection and cached data |

Common flags: `--start YYYY-MM-DD`, `--end YYYY-MM-DD`, `--no-cache`, `-v` (verbose).

## VIX/UVXY Put Strategy

**Thesis:** After VIX spikes, volatility tends to mean-revert. UVXY (2x leveraged VIX futures ETF) decays structurally due to contango. Buying puts after spikes captures both effects.

**Signal detection:**
- VIX closes above an absolute threshold (default 30), OR
- VIX single-day % change exceeds threshold (default 20%)
- Configurable AND/OR mode with cooldown between signals

**Position construction:**
- Buy UVXY puts at ATM or OTM strike
- Target DTE ~45 days
- Position sized as % of portfolio (default 5%)

**Exit criteria** (first triggered wins):
1. Profit target (default 100%)
2. Stop loss (default 50%)
3. Max hold days (default 30)
4. VIX drops below floor (default 18)
5. Option expiration

All parameters are configurable in `config/strategies/vix_uvxy_put.yaml` and sweepable.

## Backtesting

The backtest engine is strategy-agnostic. For each trading day it:
1. Checks exit conditions on open positions
2. Detects new signals
3. Constructs positions if capacity allows
4. Marks-to-market the equity curve

Parameter sweeps generate all combinations from the strategy's param grid, run them in parallel with `multiprocessing`, and rank by composite score (Sharpe 40%, win rate 20%, profit factor 20%, trade count 10%, max drawdown 10%).

Reports are saved to `output/reports/` as HTML with embedded charts.

## Adding a New Strategy

1. Create `src/strategylab/strategies/my_strategy/`:
   - `__init__.py` importing the strategy class
   - `strategy.py` with `@register` class implementing `BaseStrategy`
   - `signals.py`, `position.py`, `exits.py` as needed

2. Create `config/strategies/my_strategy.yaml` with `default_params` and `sweep` sections.

3. Add the strategy name to `enabled_strategies` in `config/settings.yaml` for monitoring.

No changes to core, backtest, or monitor code needed.

```python
from strategylab.core import BaseStrategy, register, Signal, Position, ExitResult

@register
class MyStrategy(BaseStrategy):
    name = "my_strategy"
    description = "Description here"
    required_tickers = ["SPY", "QQQ"]

    def detect_signals(self, data, params): ...
    def construct_position(self, signal, data, params, portfolio_value): ...
    def check_exit(self, position, current_date, data, params): ...
    def format_alert(self, signal, recommendation): ...
    def get_param_grid(self): ...
    def get_default_params(self): ...
```

## Friday Close Predictor — Put-Selling Tool (`models/`)

ML pipeline that predicts Friday close prices and generates a put-selling risk ladder with live option premiums and expected value calculations. Trained on 100 S&P 500 stocks across all 11 GICS sectors. Uses conformal prediction to produce calibrated bounds at multiple risk levels.

### How It Works

1. **Features:** 40 features from 60-day OHLCV windows — momentum (MACD, RSI), volatility (Garman-Klass, ATR, Bollinger), returns (1d–50d), VIX/SPY context, calendar signals, GICS sector encoding
2. **Model:** AutoGluon-Tabular ensemble (LightGBMXT, LightGBM, XGBoost, CatBoost, NeuralNetTorch) with 8-fold bagging
3. **Target:** Friday return (friday_close / current_price - 1), not absolute price — patterns transfer across stocks of any price level
4. **Calibration:** Split conformal prediction in return-space with volatility scaling. Produces lower bounds at any risk level (5%, 10%, 15%, 20%)
5. **Put ladder:** For each risk level, rounds to a standard option strike, fetches live bid/ask from yfinance, and computes expected value per contract
6. **Data cutoff:** Uses Mon/Tue/Wed/Thu data to predict that week's Friday close (4 samples per Friday per stock)

### Results (Run 8, 100 stocks, test period ~2024-2026)

| Metric | Value |
|--------|-------|
| Return MAE | 2.43% |
| Dollar MAE | $6.84 |
| Lower bound violation rate | 3.6% (target: ≤5%) |
| Training stocks | 100 (across 11 GICS sectors) |
| Training samples | 139,700 |
| Out-of-sample (5 unseen tickers) | 3.83% return MAE |
| Compact model size | 98 MB |

Low-volatility stocks: DUK 1.38%, SO 1.30%, JNJ 1.37%. High-volatility stocks: TSLA 4.94%, MU 5.17%, INTC 5.03%.

### Quick Start — Predict a Put to Sell

```bash
# Install deps (works on macOS M1)
pip install "autogluon.tabular[all]" yfinance pandas numpy scikit-learn matplotlib

# Show put-selling risk ladder with live premiums
cd models/
python predict.py -t META

# Without live option prices (offline mode)
python predict.py -t META --no-premiums

# Legacy $10 range output
python predict.py -t AAPL --range
```

Example output:
```
==============================================================================
  META Put-Selling Analysis for Friday 2026-03-14
==============================================================================
  Current price:    $658.00
  Predicted close:  $661.29 (+0.50%)
  Data through:     2026-03-10 (Tuesday)

  RISK LADDER
  --------------------------------------------------------------------------
  Risk Level              Strike     Bid   P(ITM)     E[Loss]         EV
  --------------------------------------------------------------------------
  Conservative (5%)         $630   $0.45     5.0%        $89       -$44
  Moderate (10%)            $645   $1.85    10.0%       $154        $31
  Aggressive (15%)          $650   $2.90    15.0%       $218        $72 <<<
  Very Aggressive (20%)     $655   $4.10    20.0%       $277       $133
  --------------------------------------------------------------------------

  >>> RECOMMENDED: Sell META $650 put @ $2.90 ($290/contract), EV=$72
==============================================================================
```

The tool recommends the strike with the highest positive expected value (EV = premium collected − expected loss from assignment).

### Training

Training requires Docker (AutoGluon has heavy native deps). Uses 48 CPUs, takes ~2.5 hours with 100 stocks.

```bash
# Build Docker image (one-time)
docker build -t friday-pred models/

# Train (volume-mount the models/ directory)
docker run --rm -v $(pwd)/models:/app/models friday-pred python models/train.py

# Run inference inside Docker
docker run --rm -v $(pwd)/models:/app/models friday-pred python models/predict.py -t AAPL
```

Artifacts saved to `models/artifacts/`:
- `ag_model/` — Full AutoGluon ensemble (~1.2 GB, gitignored)
- `ag_model_compact/` — Top 5 L1 models (98 MB, pushed to GitHub)
- `conformal.json` — Calibration residuals (27,940 samples)
- `plots/` — Per-stock prediction charts (gitignored)

### Trained Tickers (100)

**Technology (21):** AAPL, MSFT, NVDA, GOOGL, META, AVGO, AMD, ADBE, CRM, INTC, CSCO, ORCL, TXN, QCOM, AMAT, MU, LRCX, KLAC, NOW, PANW, CDNS

**Consumer Discretionary (12):** AMZN, TSLA, HD, MCD, NKE, LOW, TJX, SBUX, BKNG, ROST, CMG, BABA

**Financials (13):** JPM, V, MA, BAC, GS, MS, BLK, AXP, SPGI, C, PYPL, ICE, SCHW

**Health Care (11):** UNH, JNJ, LLY, PFE, ABT, MRK, TMO, AMGN, ISRG, GILD, DHR

**Industrials (11):** CAT, GE, HON, UNP, RTX, LMT, BA, DE, WM, FDX, ETN

**Consumer Staples (8):** PG, KO, PEP, COST, WMT, PM, CL, MDLZ

**Energy (6):** XOM, CVX, COP, SLB, EOG, OXY

**Communication Services (5):** DIS, NFLX, CMCSA, T, VZ

**Materials (5):** LIN, APD, SHW, FCX, NEM

**Utilities (4):** NEE, DUK, SO, AEP

**Real Estate (4):** PLD, AMT, CCI, EQIX

The model can also predict untrained tickers (with reduced accuracy) — it uses price-normalized features and sector encoding, not ticker-specific patterns.

### Project Structure

```
models/
├── predict.py        # Inference: ticker → put-selling risk ladder
├── train.py          # End-to-end training pipeline
├── data.py           # yfinance fetch + sample construction
├── features.py       # 40 features from 60-day OHLCV windows
├── conformal.py      # Split conformal prediction (multi-level bounds)
├── evaluate.py       # Coverage, sharpness, MAE metrics + plots
├── options.py        # Live option chain via yfinance
├── config.py         # 100 tickers, sectors, hyperparams
├── Dockerfile        # Training environment (Python 3.11 + AutoGluon)
└── artifacts/        # Saved model + calibration
    ├── ag_model_compact/  # 98 MB (on GitHub)
    └── conformal.json     # Calibration data (on GitHub)
```

## Docker

```bash
# Start monitoring
docker compose up monitor

# Run a backtest
docker compose run --rm backtest

# Run a sweep
docker compose --profile sweep run --rm sweep
```

## Testing

```bash
pip install -e ".[dev]"
pytest tests/ -v
```
