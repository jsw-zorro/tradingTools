"""Click CLI for StrategyLab."""

import logging
import sys

import click
from rich.console import Console
from rich.table import Table

console = Console()


def _setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _ensure_strategies_loaded():
    """Import strategies package to trigger @register decorators."""
    import strategylab.strategies  # noqa: F401


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging")
def cli(verbose):
    """StrategyLab - Trading strategy backtesting and monitoring platform."""
    _setup_logging(verbose)


@cli.command("list-strategies")
def list_strategies():
    """Show all registered strategies."""
    _ensure_strategies_loaded()
    from strategylab.core.registry import get_all_strategies

    strategies = get_all_strategies()

    table = Table(title="Registered Strategies")
    table.add_column("Name", style="cyan")
    table.add_column("Description", style="white")
    table.add_column("Tickers", style="yellow")

    for name, strategy in strategies.items():
        table.add_row(name, strategy.description, ", ".join(strategy.required_tickers))

    console.print(table)


@cli.command()
@click.option("--strategy", "-s", required=True, help="Strategy name")
@click.option("--start", default=None, help="Start date (YYYY-MM-DD)")
@click.option("--end", default=None, help="End date (YYYY-MM-DD)")
@click.option("--capital", default=100000, type=float, help="Initial capital")
@click.option("--no-cache", is_flag=True, help="Skip data cache")
def backtest(strategy, start, end, capital, no_cache):
    """Run a single backtest with default parameters."""
    _ensure_strategies_loaded()
    from strategylab.backtest.engine import run_backtest
    from strategylab.backtest.metrics import calculate_metrics
    from strategylab.backtest.report import generate_report
    from strategylab.config import load_settings
    from strategylab.core.registry import get_strategy
    from strategylab.data.fetcher import fetch_multiple

    strat = get_strategy(strategy)
    params = strat.get_default_params()
    settings = load_settings().get("backtest", {})

    console.print(f"[bold]Running backtest for [cyan]{strategy}[/cyan]...[/bold]")
    data = fetch_multiple(strat.required_tickers, start=start, end=end, use_cache=not no_cache)

    result = run_backtest(
        strat, data, params,
        initial_capital=capital or settings.get("initial_capital", 100000),
        commission_per_contract=settings.get("commission_per_contract", 0.65),
        slippage_pct=settings.get("slippage_pct", 1.0),
    )
    metrics = calculate_metrics(result)

    # Print summary
    table = Table(title=f"Backtest Results: {strategy}")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="white")

    for key in ["total_return_pct", "num_trades", "win_rate", "sharpe_ratio",
                 "max_drawdown_pct", "profit_factor", "avg_pnl", "avg_days_held"]:
        val = metrics.get(key, 0)
        formatted = f"{val:.2f}" if isinstance(val, float) else str(val)
        table.add_row(key, formatted)

    console.print(table)

    # Generate report
    report_path = generate_report(result)
    console.print(f"\n[green]Report saved to {report_path}[/green]")


@cli.command()
@click.option("--strategy", "-s", required=True, help="Strategy name")
@click.option("--start", default=None, help="Start date (YYYY-MM-DD)")
@click.option("--end", default=None, help="End date (YYYY-MM-DD)")
@click.option("--max-workers", default=None, type=int, help="Parallel workers")
@click.option("--no-cache", is_flag=True, help="Skip data cache")
def sweep(strategy, start, end, max_workers, no_cache):
    """Run parameter sweep for a strategy."""
    _ensure_strategies_loaded()
    from strategylab.backtest.param_sweep import run_sweep
    from strategylab.backtest.report import generate_sweep_report
    from strategylab.config import load_settings
    from strategylab.core.registry import get_strategy
    from strategylab.data.fetcher import fetch_multiple

    strat = get_strategy(strategy)
    settings = load_settings().get("backtest", {})

    console.print(f"[bold]Running parameter sweep for [cyan]{strategy}[/cyan]...[/bold]")
    data = fetch_multiple(strat.required_tickers, start=start, end=end, use_cache=not no_cache)

    results_df = run_sweep(strat, data, max_workers=max_workers, settings=settings)

    if results_df.empty:
        console.print("[red]No results from sweep[/red]")
        return

    # Show top 5
    console.print("\n[bold]Top 5 Parameter Sets:[/bold]")
    top5 = results_df.head(5)
    table = Table()
    table.add_column("#", style="cyan")
    table.add_column("Score", style="yellow")
    table.add_column("Return %", style="white")
    table.add_column("Win Rate", style="white")
    table.add_column("Sharpe", style="white")
    table.add_column("Trades", style="white")

    for i, (_, row) in enumerate(top5.iterrows()):
        table.add_row(
            str(i + 1),
            f"{row.get('composite_score', 0):.2f}",
            f"{row.get('total_return_pct', 0):.1f}%",
            f"{row.get('win_rate', 0):.1f}%",
            f"{row.get('sharpe_ratio', 0):.2f}",
            str(row.get("num_trades", 0)),
        )

    console.print(table)

    report_path = generate_sweep_report(results_df, strategy)
    console.print(f"\n[green]Sweep report saved to {report_path}[/green]")


@cli.command()
@click.option("--market-hours/--anytime", default=True, help="Only run during market hours")
def monitor(market_hours):
    """Start live monitoring for all enabled strategies."""
    _ensure_strategies_loaded()
    from strategylab.monitor.watcher import run_monitor

    console.print("[bold]Starting StrategyLab monitor...[/bold]")
    run_monitor(market_hours_only=market_hours)


@cli.command("check-now")
@click.option("--strategy", "-s", required=True, help="Strategy name")
def check_now(strategy):
    """Run a one-shot check for a strategy."""
    _ensure_strategies_loaded()
    from strategylab.monitor.watcher import run_check

    console.print(f"[bold]Running check for [cyan]{strategy}[/cyan]...[/bold]")
    alerts = run_check(strategy)

    if alerts:
        for alert in alerts:
            console.print(f"[yellow]Alert sent: {alert['subject']}[/yellow]")
    else:
        console.print("[green]No signals detected[/green]")


@cli.command("test-email")
@click.option("--to", required=True, help="Recipient email address")
def test_email(to):
    """Send a test email to verify configuration."""
    from strategylab.monitor.alert import send_test_email

    console.print(f"Sending test email to {to}...")
    success = send_test_email(to)

    if success:
        console.print("[green]Test email sent successfully![/green]")
    else:
        console.print("[red]Failed to send test email. Check credentials and config.[/red]")
        sys.exit(1)


@cli.command("download-options")
@click.option("--underlying", "-u", default="UVXY", help="Underlying ticker")
@click.option("--start", required=True, help="Start date (YYYY-MM-DD)")
@click.option("--end", required=True, help="End date (YYYY-MM-DD)")
@click.option("--resolution", default="daily", type=click.Choice(["daily", "hour", "minute"]))
@click.option("--option-type", default="put", type=click.Choice(["put", "call", "both"]))
def download_options(underlying, start, end, resolution, option_type):
    """Download historical options chain data from QuantConnect."""
    from strategylab.data.qc_fetcher import QCClient, download_options_chain

    try:
        client = QCClient()
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        sys.exit(1)

    console.print(f"Authenticating with QuantConnect...")
    if not client.authenticate():
        console.print("[red]Authentication failed. Check QC_USER_ID and QC_API_TOKEN.[/red]")
        sys.exit(1)
    console.print("[green]Authenticated.[/green]")

    console.print(f"Downloading {underlying} {option_type} options ({start} to {end}, {resolution})...")
    df = download_options_chain(
        underlying=underlying,
        start=start,
        end=end,
        resolution=resolution,
        option_type=option_type,
        client=client,
    )

    if df.empty:
        console.print("[yellow]No options data returned.[/yellow]")
    else:
        console.print(f"[green]Downloaded {len(df)} option records.[/green]")
        console.print(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")
        console.print(f"  Strikes: {df['strike'].nunique()} unique")
        console.print(f"  Expiries: {df['expiry'].nunique()} unique")


@cli.command("import-options")
@click.option("--file", "-f", "csv_file", required=True, type=click.Path(exists=True),
              help="Path to CSV file with options chain data")
@click.option("--underlying", "-u", default="UVXY", help="Underlying ticker")
@click.option("--date-col", default="date", help="Column name for trade date")
@click.option("--expiry-col", default="expiry", help="Column name for expiry date")
@click.option("--strike-col", default="strike", help="Column name for strike price")
@click.option("--right-col", default="right", help="Column name for option right (put/call)")
@click.option("--close-col", default="close", help="Column name for close price")
def import_options(csv_file, underlying, date_col, expiry_col, strike_col, right_col, close_col):
    """Import options chain data from a CSV file (e.g. exported from QC Research)."""
    from strategylab.data.qc_fetcher import import_options_csv

    console.print(f"Importing options data from {csv_file}...")
    df = import_options_csv(
        csv_path=csv_file,
        underlying=underlying,
        date_col=date_col,
        expiry_col=expiry_col,
        strike_col=strike_col,
        right_col=right_col,
        close_col=close_col,
    )
    console.print(f"[green]Imported {len(df)} records for {underlying}.[/green]")
    console.print(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    console.print(f"  Strikes: {df['strike'].nunique()} unique")
    console.print(f"  Expiries: {df['expiry'].nunique()} unique")


@cli.command("qc-status")
def qc_status():
    """Check QuantConnect connection and available options data."""
    from strategylab.data.options_chain import get_options_chain
    from strategylab.data.qc_fetcher import QCClient

    # Check credentials
    try:
        client = QCClient()
        console.print("Authenticating with QuantConnect...")
        if client.authenticate():
            console.print("[green]QC API: Connected[/green]")
        else:
            console.print("[red]QC API: Authentication failed[/red]")
    except ValueError:
        console.print("[yellow]QC API: Not configured (set QC_USER_ID + QC_API_TOKEN)[/yellow]")

    # Check cached data
    table = Table(title="Cached Options Data")
    table.add_column("Underlying", style="cyan")
    table.add_column("Has Data", style="white")
    table.add_column("Records", style="white")
    table.add_column("Date Range", style="white")
    table.add_column("Source", style="yellow")

    for ticker in ["UVXY"]:
        chain = get_options_chain(ticker)
        if chain.has_real_data:
            df = chain._chain_data
            table.add_row(
                ticker,
                "[green]Yes[/green]",
                str(len(df)),
                f"{df['date'].min().date()} to {df['date'].max().date()}",
                "imported" if "imported" in str(chain._chain_data) else "api",
            )
        else:
            table.add_row(ticker, "[dim]No[/dim]", "-", "-", "black_scholes fallback")

    console.print(table)


if __name__ == "__main__":
    cli()
