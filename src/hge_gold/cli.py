from __future__ import annotations

from pathlib import Path

import typer

from .pipeline import run_thesis_pipeline

app = typer.Typer(help="Thesis-grade leakage-safe gold forecasting pipeline V2")


@app.command()
def run(
    config: Path = Path("configs/thesis.yaml"),
    source_csv: Path | None = None,
) -> None:
    """Run the thesis research pipeline."""
    outputs = run_thesis_pipeline(config, source_csv)
    for name, path in outputs.items():
        typer.echo(f"{name}: {path}")


if __name__ == "__main__":
    app()
