"""Grid search over strategy parameter space."""

import itertools
import logging
from multiprocessing import Pool, cpu_count

import pandas as pd

from strategylab.backtest.engine import BacktestResult, run_backtest
from strategylab.backtest.metrics import calculate_metrics
from strategylab.core.base_strategy import BaseStrategy

logger = logging.getLogger(__name__)


def generate_param_grid(grid_spec: dict[str, list]) -> list[dict]:
    """Generate all parameter combinations from a grid specification."""
    keys = list(grid_spec.keys())
    values = list(grid_spec.values())
    combos = list(itertools.product(*values))
    return [dict(zip(keys, combo)) for combo in combos]


def _run_single(args: tuple) -> dict | None:
    """Run a single backtest - designed for multiprocessing."""
    strategy_cls, data, base_params, sweep_params, settings = args
    try:
        strategy = strategy_cls()
        merged_params = {**base_params, **sweep_params}
        result = run_backtest(
            strategy,
            data,
            merged_params,
            initial_capital=settings.get("initial_capital", 100_000),
            commission_per_contract=settings.get("commission_per_contract", 0.65),
            slippage_pct=settings.get("slippage_pct", 1.0),
        )
        metrics = calculate_metrics(result)
        metrics["params"] = merged_params
        return metrics
    except Exception as e:
        logger.warning("Sweep iteration failed: %s", e)
        return None


def run_sweep(
    strategy: BaseStrategy,
    data: dict[str, pd.DataFrame],
    param_grid: dict[str, list] | None = None,
    max_workers: int | None = None,
    settings: dict | None = None,
) -> pd.DataFrame:
    """Run parameter sweep across the grid, returning ranked results.

    Args:
        strategy: The strategy to sweep.
        data: Market data dict.
        param_grid: Override grid (defaults to strategy.get_param_grid()).
        max_workers: Parallel workers (defaults to cpu_count - 1).
        settings: Backtest settings (initial_capital, etc.)

    Returns:
        DataFrame of results sorted by composite_score descending.
    """
    if param_grid is None:
        param_grid = strategy.get_param_grid()
    if settings is None:
        settings = {}

    base_params = strategy.get_default_params()
    combos = generate_param_grid(param_grid)
    total = len(combos)
    logger.info("Running sweep: %d parameter combinations", total)

    if max_workers is None:
        max_workers = max(1, cpu_count() - 1)

    strategy_cls = type(strategy)
    args_list = [
        (strategy_cls, data, base_params, combo, settings)
        for combo in combos
    ]

    results = []
    # Use multiprocessing for large sweeps, sequential for small ones
    if total > 10 and max_workers > 1:
        with Pool(max_workers) as pool:
            for i, result in enumerate(pool.imap_unordered(_run_single, args_list)):
                if result is not None:
                    results.append(result)
                if (i + 1) % 100 == 0:
                    logger.info("Completed %d/%d combinations", i + 1, total)
    else:
        for i, args in enumerate(args_list):
            result = _run_single(args)
            if result is not None:
                results.append(result)

    if not results:
        logger.warning("No successful sweep results")
        return pd.DataFrame()

    df = pd.DataFrame(results)
    df = df.sort_values("composite_score", ascending=False).reset_index(drop=True)
    logger.info("Sweep complete. Top composite score: %.2f", df.iloc[0]["composite_score"])
    return df
