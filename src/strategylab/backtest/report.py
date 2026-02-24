"""HTML/PNG report generation with matplotlib."""

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from jinja2 import Template

from strategylab.backtest.engine import BacktestResult
from strategylab.backtest.metrics import calculate_metrics

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent.parent / "output" / "reports"


def generate_report(
    result: BacktestResult,
    output_dir: Path | None = None,
    prefix: str = "",
) -> Path:
    """Generate a full HTML report with embedded charts for a single backtest."""
    if output_dir is None:
        output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = calculate_metrics(result)
    tag = prefix or result.strategy_name
    report_dir = output_dir / tag
    report_dir.mkdir(parents=True, exist_ok=True)

    # Generate charts
    equity_path = _plot_equity_curve(result, report_dir)
    pnl_path = _plot_pnl_histogram(result, report_dir)
    trade_path = _plot_trade_scatter(result, report_dir)

    # Generate HTML
    html_content = _render_html(metrics, result, equity_path, pnl_path, trade_path)
    html_path = report_dir / "report.html"
    html_path.write_text(html_content)

    logger.info("Report generated at %s", html_path)
    return html_path


def generate_sweep_report(
    sweep_df: pd.DataFrame,
    strategy_name: str,
    output_dir: Path | None = None,
) -> Path:
    """Generate a report for parameter sweep results."""
    if output_dir is None:
        output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    report_dir = output_dir / f"{strategy_name}_sweep"
    report_dir.mkdir(parents=True, exist_ok=True)

    # Top 20 table
    top20 = sweep_df.head(20)

    # Heatmap for top parameter pairs
    heatmap_path = _plot_param_heatmap(sweep_df, report_dir)

    # Score distribution
    dist_path = _plot_score_distribution(sweep_df, report_dir)

    html_content = _render_sweep_html(top20, strategy_name, heatmap_path, dist_path)
    html_path = report_dir / "sweep_report.html"
    html_path.write_text(html_content)

    logger.info("Sweep report generated at %s", html_path)
    return html_path


def _plot_equity_curve(result: BacktestResult, output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(12, 5))
    result.equity_curve.plot(ax=ax, linewidth=1.5)
    ax.set_title(f"Equity Curve - {result.strategy_name}")
    ax.set_ylabel("Portfolio Value ($)")
    ax.set_xlabel("Date")
    ax.axhline(y=result.initial_capital, color="gray", linestyle="--", alpha=0.5)
    ax.grid(True, alpha=0.3)
    path = output_dir / "equity_curve.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_pnl_histogram(result: BacktestResult, output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(10, 5))
    if result.trades:
        pnls = [t.exit.pnl for t in result.trades]
        colors = ["green" if p > 0 else "red" for p in pnls]
        ax.bar(range(len(pnls)), pnls, color=colors, alpha=0.7)
        ax.set_title("Trade P&L")
        ax.set_ylabel("P&L ($)")
        ax.set_xlabel("Trade #")
        ax.axhline(y=0, color="black", linewidth=0.5)
    else:
        ax.text(0.5, 0.5, "No trades", ha="center", va="center", transform=ax.transAxes)
    ax.grid(True, alpha=0.3)
    path = output_dir / "pnl_histogram.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_trade_scatter(result: BacktestResult, output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(12, 5))
    if result.trades:
        dates = [t.position.entry_date for t in result.trades]
        pnl_pcts = [t.exit.pnl_pct for t in result.trades]
        colors = ["green" if p > 0 else "red" for p in pnl_pcts]
        ax.scatter(dates, pnl_pcts, c=colors, alpha=0.7, s=50)
        ax.set_title("Trade Returns Over Time")
        ax.set_ylabel("Return (%)")
        ax.set_xlabel("Entry Date")
        ax.axhline(y=0, color="black", linewidth=0.5)
    else:
        ax.text(0.5, 0.5, "No trades", ha="center", va="center", transform=ax.transAxes)
    ax.grid(True, alpha=0.3)
    path = output_dir / "trade_scatter.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_param_heatmap(sweep_df: pd.DataFrame, output_dir: Path) -> Path | None:
    """Plot a heatmap of the two most impactful parameters."""
    fig, ax = plt.subplots(figsize=(10, 8))

    if "params" in sweep_df.columns and len(sweep_df) > 0:
        # Extract params into columns
        params_df = pd.json_normalize(sweep_df["params"])
        numeric_cols = params_df.select_dtypes(include=[np.number]).columns.tolist()

        if len(numeric_cols) >= 2:
            col1, col2 = numeric_cols[0], numeric_cols[1]
            pivot = sweep_df.copy()
            pivot[col1] = params_df[col1]
            pivot[col2] = params_df[col2]
            heatmap_data = pivot.groupby([col1, col2])["composite_score"].mean().unstack()
            im = ax.imshow(heatmap_data.values, cmap="RdYlGn", aspect="auto")
            ax.set_xticks(range(len(heatmap_data.columns)))
            ax.set_xticklabels(heatmap_data.columns, rotation=45)
            ax.set_yticks(range(len(heatmap_data.index)))
            ax.set_yticklabels(heatmap_data.index)
            ax.set_xlabel(col2)
            ax.set_ylabel(col1)
            ax.set_title(f"Composite Score: {col1} vs {col2}")
            fig.colorbar(im)
        else:
            ax.text(0.5, 0.5, "Insufficient numeric params", ha="center", va="center", transform=ax.transAxes)
    else:
        ax.text(0.5, 0.5, "No sweep data", ha="center", va="center", transform=ax.transAxes)

    path = output_dir / "param_heatmap.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_score_distribution(sweep_df: pd.DataFrame, output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(10, 5))
    if "composite_score" in sweep_df.columns:
        ax.hist(sweep_df["composite_score"], bins=30, color="steelblue", alpha=0.7, edgecolor="black")
        ax.set_title("Composite Score Distribution")
        ax.set_xlabel("Score")
        ax.set_ylabel("Count")
    ax.grid(True, alpha=0.3)
    path = output_dir / "score_distribution.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _render_html(metrics: dict, result: BacktestResult, equity_path: Path, pnl_path: Path, trade_path: Path) -> str:
    template = Template("""<!DOCTYPE html>
<html><head><title>StrategyLab Backtest Report - {{ strategy }}</title>
<style>
body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
.container { max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
h1 { color: #333; border-bottom: 2px solid #4CAF50; padding-bottom: 10px; }
h2 { color: #555; }
table { border-collapse: collapse; width: 100%; margin: 20px 0; }
th, td { border: 1px solid #ddd; padding: 10px; text-align: left; }
th { background: #4CAF50; color: white; }
tr:nth-child(even) { background: #f9f9f9; }
img { max-width: 100%; margin: 10px 0; border: 1px solid #ddd; border-radius: 4px; }
.metric-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px; margin: 20px 0; }
.metric-card { background: #f9f9f9; padding: 15px; border-radius: 6px; text-align: center; }
.metric-card .value { font-size: 24px; font-weight: bold; color: #333; }
.metric-card .label { font-size: 12px; color: #777; text-transform: uppercase; }
.positive { color: #4CAF50; }
.negative { color: #f44336; }
</style></head><body>
<div class="container">
<h1>Backtest Report: {{ strategy }}</h1>
<div class="metric-grid">
    <div class="metric-card"><div class="value {{ 'positive' if total_return > 0 else 'negative' }}">${{ "%.0f"|format(final_equity) }}</div><div class="label">Final Equity</div></div>
    <div class="metric-card"><div class="value {{ 'positive' if total_return > 0 else 'negative' }}">{{ "%.1f"|format(total_return) }}%</div><div class="label">Total Return</div></div>
    <div class="metric-card"><div class="value">{{ num_trades }}</div><div class="label">Total Trades</div></div>
    <div class="metric-card"><div class="value">{{ "%.1f"|format(win_rate) }}%</div><div class="label">Win Rate</div></div>
    <div class="metric-card"><div class="value">{{ "%.2f"|format(sharpe) }}</div><div class="label">Sharpe Ratio</div></div>
    <div class="metric-card"><div class="value {{ 'negative' }}">{{ "%.1f"|format(max_dd) }}%</div><div class="label">Max Drawdown</div></div>
    <div class="metric-card"><div class="value">{{ "%.2f"|format(profit_factor) }}</div><div class="label">Profit Factor</div></div>
    <div class="metric-card"><div class="value">${{ "%.0f"|format(avg_pnl) }}</div><div class="label">Avg P&L</div></div>
    <div class="metric-card"><div class="value">{{ "%.1f"|format(avg_days) }}d</div><div class="label">Avg Hold</div></div>
</div>
<h2>Equity Curve</h2>
<img src="equity_curve.png">
<h2>Trade P&L</h2>
<img src="pnl_histogram.png">
<h2>Trade Returns Over Time</h2>
<img src="trade_scatter.png">
{% if exit_reasons %}
<h2>Exit Reasons</h2>
<table><tr><th>Reason</th><th>Count</th></tr>
{% for reason, count in exit_reasons.items() %}
<tr><td>{{ reason }}</td><td>{{ count }}</td></tr>
{% endfor %}</table>
{% endif %}
</div></body></html>""")

    return template.render(
        strategy=result.strategy_name,
        final_equity=metrics["final_equity"],
        total_return=metrics["total_return_pct"],
        num_trades=metrics["num_trades"],
        win_rate=metrics["win_rate"],
        sharpe=metrics["sharpe_ratio"],
        max_dd=metrics["max_drawdown_pct"],
        profit_factor=metrics["profit_factor"],
        avg_pnl=metrics["avg_pnl"],
        avg_days=metrics["avg_days_held"],
        exit_reasons=metrics.get("exit_reasons", {}),
    )


def _render_sweep_html(top20: pd.DataFrame, strategy_name: str, heatmap_path: Path | None, dist_path: Path) -> str:
    display_cols = [c for c in top20.columns if c not in ("params", "exit_reasons")]

    rows_html = ""
    for _, row in top20.iterrows():
        cells = "".join(f"<td>{row.get(c, '')}</td>" if not isinstance(row.get(c), float) else f"<td>{row[c]:.2f}</td>" for c in display_cols)
        rows_html += f"<tr>{cells}</tr>"

    headers = "".join(f"<th>{c}</th>" for c in display_cols)

    return f"""<!DOCTYPE html>
<html><head><title>StrategyLab Sweep Report - {strategy_name}</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
.container {{ max-width: 1400px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
h1 {{ color: #333; border-bottom: 2px solid #2196F3; padding-bottom: 10px; }}
h2 {{ color: #555; }}
table {{ border-collapse: collapse; width: 100%; margin: 20px 0; font-size: 13px; }}
th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
th {{ background: #2196F3; color: white; }}
tr:nth-child(even) {{ background: #f9f9f9; }}
img {{ max-width: 100%; margin: 10px 0; border: 1px solid #ddd; border-radius: 4px; }}
</style></head><body>
<div class="container">
<h1>Parameter Sweep Report: {strategy_name}</h1>
<p>Total combinations tested: check logs for count</p>
<h2>Top 20 Parameter Sets</h2>
<div style="overflow-x:auto"><table><tr>{headers}</tr>{rows_html}</table></div>
<h2>Score Distribution</h2>
<img src="score_distribution.png">
{"<h2>Parameter Heatmap</h2><img src='param_heatmap.png'>" if heatmap_path else ""}
</div></body></html>"""
