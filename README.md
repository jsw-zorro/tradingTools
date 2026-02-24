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
