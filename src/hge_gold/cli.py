from __future__ import annotations

from pathlib import Path

import typer

from .data import generate_sample_market_data
from .io import write_csv
from .pipeline import run_pipeline

app = typer.Typer(help="HGE-Hybrid Gold leakage-safe research pipeline")


@app.command()
def run(config: Path = Path("configs/pipeline.yaml"), source_csv: Path | None = None) -> None:
    """Execute phases 1-11 in offline/simulation mode."""
    result = run_pipeline(config, source_csv)
    typer.echo(f"Execution report: {result['report']}")


@app.command("generate-sample")
def generate_sample(
    output: Path = Path("data/sample/gold_ohlcv_sample.csv"), rows: int = 1400
) -> None:
    """Create deterministic non-confidential OHLCV sample data."""
    write_csv(output, generate_sample_market_data(rows))
    typer.echo(str(output))


if __name__ == "__main__":
    app()
