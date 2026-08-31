"""Command line entry point.

`pyproject.toml` declares `ringfence = "ringfence.cli:app"`, so this module has
to exist for the installed console script to work at all. It is a thin wrapper
over the scripts in `scripts/` -- those stay runnable directly, because that is
how the numbers in ARCHITECTURE.md were produced and a reviewer should be able
to reproduce them without learning a CLI.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import typer
from rich.console import Console

from ringfence import config

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Ringfence - abuse-ring detection and chargeback drafting. Defense-only.",
)
data_app = typer.Typer(no_args_is_help=True, help="Fetch and prepare the dataset.")
app.add_typer(data_app, name="data")

console = Console()
SCRIPTS = config.ROOT / "scripts"


def _run(script: str, *args: str) -> None:
    """Invoke a script with the current interpreter, surfacing its exit code."""
    path = SCRIPTS / script
    if not path.exists():
        console.print(f"[red]missing script: {path}[/red]")
        raise typer.Exit(1)
    result = subprocess.run([sys.executable, str(path), *args], check=False)
    if result.returncode != 0:
        raise typer.Exit(result.returncode)


@data_app.command("download")
def data_download() -> None:
    """Download IEEE-CIS from Kaggle into data/raw/."""
    _run("download_data.py")


@data_app.command("prepare")
def data_prepare(
    tag: str = typer.Option("main", help="Name for this prepared dataset."),
    label_lag_days: int = typer.Option(30, help="Chargeback reporting lag to assume."),
    max_shared_clients: int = typer.Option(20, help="Hub cap for linking attributes."),
    sample: int = typer.Option(0, help="Debug: use only the first N rows."),
) -> None:
    """Build the causal features and write train/test parquet."""
    args = [
        "--tag", tag,
        "--label-lag-days", str(label_lag_days),
        "--max-shared-clients", str(max_shared_clients),
    ]
    if sample:
        args += ["--sample", str(sample)]
    _run("prepare_data.py", *args)


@app.command()
def train(
    tag: str = typer.Option("main", help="Name for this run's results."),
    data_tag: str = typer.Option(None, help="Prepared dataset to train on."),
    rounds: int = typer.Option(3000, help="Max boosting rounds."),
    no_ring_features: bool = typer.Option(False, help="Ablation: drop ring features."),
) -> None:
    """Train the detector and price it in rupees."""
    args = ["--tag", tag, "--rounds", str(rounds)]
    if data_tag:
        args += ["--data-tag", data_tag]
    if no_ring_features:
        args.append("--no-ring-features")
    _run("run_experiment.py", *args)


@app.command()
def ablate(
    tag: str = typer.Option("main", help="Prepared dataset to use."),
    force: bool = typer.Option(
        False, help="Recompute every variant instead of reusing cached results."
    ),
) -> None:
    """Measure what the honesty constraints cost, by deliberately breaking them.

    Reuses any variant whose results json already exists, so a repeat run is
    instant. Pass --force to prepare and train all three from scratch (~6 min).
    """
    args = ["--tag", tag]
    if force:
        args.append("--force")
    _run("leakage_ablation.py", *args)


@app.command()
def agent(
    tag: str = typer.Option("main", help="Dataset and model tag."),
    offline: bool = typer.Option(False, help="Skip API calls; print prompts only."),
    rings: int = typer.Option(2, help="How many top rings to write up."),
) -> None:
    """Run the case-file and dispute agents against real scored rings."""
    args = ["--tag", tag, "--rings", str(rings)]
    if offline:
        args.append("--offline")
    _run("demo_agent.py", *args)


@app.command()
def site(
    rings: int = typer.Option(5, help="How many rings to include in the queue."),
    with_casefiles: bool = typer.Option(
        False, help="Regenerate ring case files (costs an API call each)."
    ),
) -> None:
    """Rebuild docs/data.json for the GitHub Pages report from reports/."""
    args = ["--rings", str(rings)]
    if with_casefiles:
        args.append("--with-casefiles")
    _run("build_site.py", *args)


@app.command()
def status() -> None:
    """Show what data, models and results are present."""
    def mark(p: Path) -> str:
        return f"[green]ok[/green]   {p.name}" if p.exists() else f"[yellow]--[/yellow]   {p.name}"

    console.print("\n[bold]raw data[/bold]")
    for f in ("train_transaction.csv", "train_identity.csv"):
        console.print("  " + mark(config.RAW / f))
    console.print("\n[bold]prepared[/bold]")
    for f in sorted(config.PROCESSED.glob("*.parquet")):
        console.print(f"  [green]ok[/green]   {f.name}  ({f.stat().st_size/1e6:.0f} MB)")
    console.print("\n[bold]models[/bold]")
    for f in sorted((config.ROOT / "models").glob("*.txt")):
        console.print(f"  [green]ok[/green]   {f.name}")
    console.print("\n[bold]results[/bold]")
    for f in sorted(config.REPORTS.glob("results_*.json")):
        console.print(f"  [green]ok[/green]   {f.name}")
    console.print()


if __name__ == "__main__":
    app()
