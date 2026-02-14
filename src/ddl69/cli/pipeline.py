"""CLI: Run real data pipeline"""

from __future__ import annotations

import logging
from typing import Optional

import typer
import pandas as pd
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from ddl69.core.real_pipeline import RealDataPipeline

app = typer.Typer(help="DDL-69 Real Data Pipeline")
console = Console()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@app.command()
def run_inference(
    symbols: list[str] = typer.Option(
        ["SPY"],
        "--symbols",
        "-s",
        help="Symbols to process (can repeat)",
    ),
    start_date: Optional[str] = typer.Option(
        None,
        "--start",
        help="Start date (YYYY-MM-DD), default: 1 year ago",
    ),
    end_date: Optional[str] = typer.Option(
        None,
        "--end",
        help="End date (YYYY-MM-DD), default: today",
    ),
    train_split: float = typer.Option(
        0.7,
        "--split",
        help="Train/test split (0.0-1.0)",
    ),
    artifact_root: Optional[str] = typer.Option(
        None,
        "--artifacts",
        help="Root directory for parquet files",
    ),
) -> None:
    """Run end-to-end pipeline: Load -> Train -> Predict -> Cache."""
    console.print(Panel.fit(
        "[bold cyan]DDL-69 Real Data Pipeline[/bold cyan]\n"
        f"Symbols: {', '.join(symbols)}\n"
        f"Train/Test Split: {train_split:.1%}",
        border_style="cyan",
    ))

    # Initialize pipeline
    pipeline = RealDataPipeline(artifact_root=artifact_root)

    # Run
    results = pipeline.run(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        train_split=train_split,
    )

    # Display results
    _display_results(results)

    # Save summary
    _save_summary(results)


@app.command()
def load_data(
    symbol: str = typer.Option("SPY", "--symbol", "-s"),
    start_date: Optional[str] = typer.Option(None, "--start"),
    end_date: Optional[str] = typer.Option(None, "--end"),
    save: bool = typer.Option(True, "--save/--no-save"),
) -> None:
    """Load raw market data from Parquet/Polygon/Alpaca/Yahoo."""
    console.print(f"[cyan]Loading {symbol}...{start_date} to {end_date}[/cyan]")

    from ddl69.data.loaders import DataLoader

    loader = DataLoader()
    df = loader.load(symbol, start_date, end_date)

    console.print(f"[green]Loaded {len(df)} bars[/green]")
    console.print(df.head(10))

    if save:
        path = loader.save_parquet(df, symbol)
        console.print(f"[green]Saved to {path}[/green]")


@app.command()
def predict(
    symbol: str = typer.Option("SPY", "--symbol", "-s"),
    artifact_root: Optional[str] = typer.Option(None, "--artifacts"),
) -> None:
    """Quick prediction for latest bar."""
    from ddl69.data.loaders import DataLoader

    console.print(f"[cyan]Predicting {symbol}...[/cyan]")

    loader = DataLoader(artifact_root=artifact_root)
    df = loader.load(symbol)

    pipeline = RealDataPipeline(artifact_root=artifact_root)
    df = pipeline._preprocess(df)
    df = pipeline._add_indicators(df, symbol)
    df = pipeline._create_labels(df)

    # Load latest model
    try:
        import pickle

        model_path = Path(artifact_root or ".artifacts") / "models" / f"{symbol}_ensemble.pkl"
        if model_path.exists():
            with open(model_path, "rb") as f:
                pipeline.ensemble = pickle.load(f)
        else:
            console.print("[yellow]No trained model found, running full pipeline first...[/yellow]")
            return

    except Exception as e:
        console.print(f"[red]Error loading model: {e}[/red]")
        return

    pred = pipeline._predict(df.iloc[-1:].copy(), symbol)
    _display_prediction(pred)


def _display_results(results: dict) -> None:
    """Display results in rich tables."""
    table = Table(title="Pipeline Results", show_header=True, header_style="bold cyan")
    table.add_column("Symbol", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Bars", style="yellow")
    table.add_column("Accuracy", style="magenta")
    table.add_column("Sharpe", style="blue")
    table.add_column("Signal", style="green")

    for symbol, result in results.items():
        if result["status"] == "error":
            table.add_row(
                symbol,
                "[red]ERROR[/red]",
                "-",
                "-",
                "-",
                f"[red]{result['error']}[/red]",
            )
        else:
            metrics = result.get("metrics", {})
            pred = result.get("latest_predictions", {})

            signal_style = (
                "green" if pred.get("signal") == "BUY" else
                "red" if pred.get("signal") == "SELL" else
                "yellow"
            )

            table.add_row(
                symbol,
                "[green]SUCCESS[/green]",
                str(result.get("bars_processed", 0)),
                f"{metrics.get('test_accuracy', 0):.3f}",
                f"{metrics.get('sharpe_ratio', 0):.3f}",
                f"[{signal_style}]{pred.get('signal', 'N/A')}[/{signal_style}]",
            )

    console.print(table)

    # Risk metrics
    for symbol, result in results.items():
        if result["status"] == "success":
            risk = result.get("risk_metrics", {})
            if "error" not in risk:
                risk_table = Table(
                    title=f"{symbol} Risk Metrics",
                    show_header=True,
                    header_style="bold magenta",
                )
                risk_table.add_column("Metric", style="magenta")
                risk_table.add_column("Value", style="cyan")

                for key, val in risk.items():
                    risk_table.add_row(key, str(val))

                console.print(risk_table)


def _display_prediction(pred: dict) -> None:
    """Display single prediction."""
    if "error" in pred:
        console.print(f"[red]Error: {pred['error']}[/red]")
        return

    color = "green" if pred["signal"] == "BUY" else "red" if pred["signal"] == "SELL" else "yellow"

    info = f"""
[bold cyan]{pred['symbol']} @ {pred['timestamp']}[/bold cyan]

[{color}]Signal: {pred['signal']}[/{color}]
Price: ${pred['current_price']:.2f}
Buy Probability: {pred['buy_probability']:.1%}
Confidence: {pred['confidence']:.1%}
"""
    console.print(Panel(info, border_style=color))


def _save_summary(results: dict) -> None:
    """Save results to CSV."""
    summary_data = []

    for symbol, result in results.items():
        if result["status"] == "success":
            metrics = result.get("metrics", {})
            pred = result.get("latest_predictions", {})
            risk = result.get("risk_metrics", {})

            summary_data.append({
                "symbol": symbol,
                "bars": result.get("bars_processed", 0),
                "accuracy": metrics.get("test_accuracy", 0),
                "auc": metrics.get("test_auc", 0),
                "sharpe": metrics.get("sharpe_ratio", 0),
                "signal": pred.get("signal", "N/A"),
                "buy_prob": pred.get("buy_probability", 0),
                "confidence": pred.get("confidence", 0),
                "price": pred.get("current_price", 0),
                "var_95": risk.get("var_95", 0),
                "cvar_95": risk.get("cvar_95", 0),
            })

    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_path = Path(".artifacts") / f"summary_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(summary_path, index=False)
        console.print(f"[green]Summary saved to {summary_path}[/green]")


if __name__ == "__main__":
    app()
