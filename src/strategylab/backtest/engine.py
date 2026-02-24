"""Core backtesting loop - works with any BaseStrategy."""

import logging

import pandas as pd

from strategylab.core.base_strategy import BaseStrategy
from strategylab.core.models import ExitResult, Position, TradeRecord

logger = logging.getLogger(__name__)


class BacktestResult:
    """Container for backtest output."""

    def __init__(
        self,
        strategy_name: str,
        params: dict,
        trades: list[TradeRecord],
        equity_curve: pd.Series,
        initial_capital: float,
    ):
        self.strategy_name = strategy_name
        self.params = params
        self.trades = trades
        self.equity_curve = equity_curve
        self.initial_capital = initial_capital
        self.final_equity = equity_curve.iloc[-1] if len(equity_curve) > 0 else initial_capital

    @property
    def total_return_pct(self) -> float:
        return ((self.final_equity - self.initial_capital) / self.initial_capital) * 100

    @property
    def num_trades(self) -> int:
        return len(self.trades)


def run_backtest(
    strategy: BaseStrategy,
    data: dict[str, pd.DataFrame],
    params: dict,
    initial_capital: float = 100_000,
    commission_per_contract: float = 0.65,
    slippage_pct: float = 1.0,
) -> BacktestResult:
    """Run a backtest for a strategy with given parameters.

    Args:
        strategy: Any BaseStrategy implementation.
        data: {ticker: OHLCV DataFrame} for all required tickers.
        params: Strategy parameters dict.
        initial_capital: Starting portfolio value.
        commission_per_contract: Commission per options contract.
        slippage_pct: Slippage as percentage of entry/exit price.

    Returns:
        BacktestResult with trades and equity curve.
    """
    # Pre-compute all signals
    signals = strategy.detect_signals(data, params)
    signal_dates = {s.date: s for s in signals}

    # Get the union of all trading dates
    all_dates = _get_trading_dates(data)
    if all_dates.empty:
        return BacktestResult(strategy.name, params, [], pd.Series(dtype=float), initial_capital)

    open_positions: list[Position] = []
    completed_trades: list[TradeRecord] = []
    cash = initial_capital
    equity_values = []
    equity_dates = []
    max_open = params.get("max_open_positions", 3)

    for current_date in all_dates:
        # 1. Check exits on all open positions
        positions_to_remove = []
        for pos in open_positions:
            exit_result = strategy.check_exit(pos, current_date, data, params)
            if exit_result is not None:
                # Apply slippage and commission
                exit_result = _apply_costs(
                    exit_result, pos, commission_per_contract, slippage_pct
                )
                completed_trades.append(TradeRecord(position=pos, exit=exit_result))
                cash += pos.cost_basis + exit_result.pnl
                positions_to_remove.append(pos)

        for pos in positions_to_remove:
            open_positions.remove(pos)

        # 2. Check for signals on this date
        if current_date in signal_dates and len(open_positions) < max_open:
            signal = signal_dates[current_date]
            position = strategy.construct_position(signal, data, params, cash)

            if position is not None:
                # Apply entry slippage and commission
                entry_commission = commission_per_contract * position.quantity
                slip_cost = position.cost_basis * (slippage_pct / 100)
                adjusted_cost = position.cost_basis + entry_commission + slip_cost
                position.cost_basis = adjusted_cost

                if adjusted_cost <= cash:
                    cash -= adjusted_cost
                    open_positions.append(position)

        # 3. Mark-to-market: estimate current value of open positions
        open_value = _mark_to_market(open_positions, current_date, data, params, strategy)
        equity_values.append(cash + open_value)
        equity_dates.append(current_date)

    equity_curve = pd.Series(equity_values, index=pd.DatetimeIndex(equity_dates))

    return BacktestResult(
        strategy_name=strategy.name,
        params=params,
        trades=completed_trades,
        equity_curve=equity_curve,
        initial_capital=initial_capital,
    )


def _get_trading_dates(data: dict[str, pd.DataFrame]) -> pd.DatetimeIndex:
    """Get sorted union of all trading dates across tickers."""
    all_dates = set()
    for df in data.values():
        if not df.empty:
            all_dates.update(df.index)
    if not all_dates:
        return pd.DatetimeIndex([])
    return pd.DatetimeIndex(sorted(all_dates))


def _mark_to_market(
    positions: list[Position],
    current_date: pd.Timestamp,
    data: dict[str, pd.DataFrame],
    params: dict,
    strategy: BaseStrategy,
) -> float:
    """Estimate current market value of all open positions."""
    total = 0.0
    for pos in positions:
        # Use check_exit's internal pricing to get current value estimate
        # by attempting an exit check (won't actually exit)
        exit_result = strategy.check_exit(pos, current_date, data, params)
        if exit_result is not None:
            # If exit conditions met, use exit price
            total += exit_result.exit_price * pos.quantity * 100
        else:
            # Position still open - estimate from cost basis + any unrealized P&L
            # Simple approach: re-price using Black-Scholes
            total += _estimate_position_value(pos, current_date, data, params)
    return total


def _estimate_position_value(
    position: Position,
    current_date: pd.Timestamp,
    data: dict[str, pd.DataFrame],
    params: dict,
) -> float:
    """Estimate current value of an open position."""
    from strategylab.data.options_chain import get_options_chain

    if position.instrument == "UVXY_PUT":
        uvxy_data = data.get("UVXY")
        if uvxy_data is not None and current_date in uvxy_data.index:
            current_price = float(uvxy_data.loc[current_date, "Close"])
            strike = position.metadata.get("strike", current_price)
            dte = position.metadata.get("dte", 45)
            days_held = (current_date - position.entry_date).days
            remaining_dte = max(dte - days_held, 0)
            iv = position.metadata.get("iv_estimate", 0.8)

            expiry = None
            expiry_str = position.metadata.get("expiry")
            if expiry_str:
                try:
                    expiry = pd.Timestamp(expiry_str)
                except (ValueError, TypeError):
                    pass

            chain = get_options_chain("UVXY")
            option_price = chain.get_put_price(
                date=current_date,
                strike=strike,
                expiry=expiry,
                underlying_price=current_price,
                iv=iv,
                dte=remaining_dte,
            )
            return option_price * position.quantity * 100

    # Fallback: return cost basis (conservative)
    return position.cost_basis


def _apply_costs(
    exit_result: ExitResult,
    position: Position,
    commission: float,
    slippage_pct: float,
) -> ExitResult:
    """Adjust exit result for commission and slippage."""
    exit_commission = commission * position.quantity
    gross_value = exit_result.exit_price * position.quantity * 100
    slip_cost = gross_value * (slippage_pct / 100)
    net_pnl = gross_value - position.cost_basis - exit_commission - slip_cost
    net_pnl_pct = (net_pnl / position.cost_basis) * 100 if position.cost_basis > 0 else 0

    return ExitResult(
        exit_date=exit_result.exit_date,
        exit_reason=exit_result.exit_reason,
        exit_price=exit_result.exit_price,
        pnl=net_pnl,
        pnl_pct=net_pnl_pct,
        days_held=exit_result.days_held,
    )
