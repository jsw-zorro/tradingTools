# StrategyLab

## What This Is

An extensible trading strategy backtesting and monitoring platform. Strategies are self-contained plugins that register with a core framework. The system backtests against historical data, runs parameter sweeps, monitors live markets, and sends email alerts for manual Robinhood execution.

## Tech Stack

- Python 3.12, installed via mise at `~/.local/share/mise/installs/python/3.12.11/`
- Package: `strategylab` (editable install from `src/`)
- Key deps: yfinance, pandas, numpy, scipy, matplotlib, jinja2, click, rich, schedule, pyyaml, requests
- Dev deps: pytest, pytest-cov

## Project Layout

```
src/strategylab/
├── core/           # BaseStrategy ABC, models (Signal/Position/ExitResult/TradeRecord), plugin registry
├── data/           # yfinance fetcher, parquet cache, Black-Scholes, QuantConnect API, unified OptionsChain
├── strategies/     # Plugin directory - auto-discovered via __init__.py walk_packages
│   └── vix_uvxy_put/   # First strategy: VIX spike -> UVXY put
├── backtest/       # Strategy-agnostic engine, param_sweep (multiprocessing), metrics, HTML reports
├── monitor/        # Live watcher polling loop, email alerts, market-hours scheduler
└── cli.py          # Click CLI entry point
config/             # YAML configs: settings.yaml (global), strategies/*.yaml (per-strategy)
tests/              # pytest tests (42 currently, all passing)
output/             # gitignored: reports/ and data_cache/
```

## Architecture

- **Plugin system**: Strategies implement `BaseStrategy` ABC and use `@register` decorator. `strategies/__init__.py` auto-imports all submodules.
- **Options pricing**: `OptionsChain` class in `data/options_chain.py` is the unified interface. Tries QuantConnect real data first, falls back to Black-Scholes. Position metadata tracks `pricing_source`.
- **Config**: `config.py` loads YAML. Global settings in `config/settings.yaml`, per-strategy in `config/strategies/<name>.yaml`. Secrets via env vars (see `.env.example`).

## Common Commands

```bash
# Install
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# CLI
strategylab list-strategies
strategylab backtest -s vix_uvxy_put
strategylab sweep -s vix_uvxy_put
strategylab check-now -s vix_uvxy_put
strategylab monitor
strategylab qc-status
strategylab download-options -u UVXY --start 2020-01-01 --end 2025-01-01
strategylab import-options -f data.csv -u UVXY
```

## Code Conventions

- Type hints everywhere (Python 3.11+ syntax: `list[str]`, `dict[str, X]`, `X | None`)
- Dataclasses for models, not Pydantic
- `pd.Timestamp` for all dates
- Strategy params passed as plain `dict`, not typed config objects
- Logging via stdlib `logging`, not print
- Tests use synthetic data, no network calls

## Adding a Strategy

1. Create `src/strategylab/strategies/<name>/` with `__init__.py`, `strategy.py` (has `@register` class), plus `signals.py`, `position.py`, `exits.py` as needed
2. Create `config/strategies/<name>.yaml` with `default_params` and `sweep` sections
3. Add to `enabled_strategies` list in `config/settings.yaml` for monitoring
4. No changes to core, backtest, or monitor code

## Key Files to Know

- `core/base_strategy.py` - The contract all strategies implement
- `core/registry.py` - `@register` decorator and strategy lookup
- `data/options_chain.py` - Where QC data meets Black-Scholes (the pricing abstraction)
- `backtest/engine.py` - The daily backtest loop
- `backtest/param_sweep.py` - Grid search with multiprocessing.Pool
- `cli.py` - All CLI commands

## Git

- Remote: `git@github.com:jsw-zorro/tradingTools.git`
- Single branch: `main`
- `output/` is gitignored (reports, data cache, parquet files)
