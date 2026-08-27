"""Measure what the honesty constraints cost, by deliberately breaking them.

The project claims three guarantees: temporal splitting, causal ring features,
and a chargeback reporting lag. Each one LOWERS the reported score. That is easy
to assert and worth nothing unless the gap is measured -- so this script breaks
each guarantee in turn and reports how much the number inflates.

The inflated figures are what a submission would show if it made the standard
mistakes. They are not results; they are the size of the lie avoided.

Variants
--------
  honest        causal ring features, 30-day chargeback reporting lag  (shipped)
  lag0          pretend a chargeback is known the instant it happens
  whole_frame   compute ring aggregates over the entire dataset at once,
                so a transaction's features include its ring's future

Each variant is prepared and trained as a separate process, because preparation
and training together exceed 8GB.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console
from rich.table import Table

from ringfence import config

console = Console()
HERE = Path(__file__).resolve().parent

VARIANTS = {
    "honest": {
        "prepare": [],
        "label": "causal features, 30d reporting lag (shipped)",
    },
    "lag0": {
        "prepare": ["--label-lag-days", "0"],
        "label": "chargebacks known instantly (impossible)",
    },
    "whole_frame": {
        "prepare": ["--leaky"],
        "label": "ring aggregates over the whole dataset (classic leak)",
    },
}


def run(script: str, *args: str) -> None:
    result = subprocess.run([sys.executable, str(HERE / script), *args], check=False)
    if result.returncode != 0:
        raise SystemExit(f"{script} failed for args {args}")


def ensure(variant: str, base_tag: str, force: bool) -> str:
    """Prepare and train one variant if its results are not already present."""
    tag = base_tag if variant == "honest" else f"{base_tag}_{variant}"
    results = config.REPORTS / f"results_{tag}.json"
    if results.exists() and not force:
        console.log(f"[dim]{variant}: reusing {results.name}[/dim]")
        return tag

    parquet = config.PROCESSED / f"train_{tag}.parquet"
    if not parquet.exists() or force:
        console.rule(f"[bold]preparing {variant}")
        run("prepare_data.py", "--tag", tag, *VARIANTS[variant]["prepare"])

    console.rule(f"[bold]training {variant}")
    run("run_experiment.py", "--tag", tag, "--data-tag", tag)
    return tag


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="main")
    ap.add_argument("--force", action="store_true", help="recompute even if cached")
    args = ap.parse_args()

    tags = {v: ensure(v, args.tag, args.force) for v in VARIANTS}
    loaded = {
        v: json.loads((config.REPORTS / f"results_{t}.json").read_text())
        for v, t in tags.items()
    }
    honest = loaded["honest"]

    table = Table(title="What each honesty constraint costs", title_style="bold")
    table.add_column("variant")
    table.add_column("what it assumes")
    table.add_column("PR-AUC", justify="right")
    table.add_column("vs honest", justify="right")
    table.add_column("Rs saved", justify="right")

    for v, res in loaded.items():
        pr = res["headline"]["pr_auc"]
        delta = pr - honest["headline"]["pr_auc"]
        pct = 100 * delta / honest["headline"]["pr_auc"]
        saving = res["optimal"]["net_saving_inr"]
        table.add_row(
            v,
            VARIANTS[v]["label"],
            f"{pr:.4f}",
            "-" if v == "honest" else f"{pct:+.1f}%",
            f"{saving:,.0f}",
        )
    console.print()
    console.print(table)

    console.print(
        "\n[bold]Reading this table.[/bold] A variant scoring above `honest` is "
        "not better -- it is the amount by which a submission making that "
        "standard mistake would overstate itself. A variant scoring BELOW "
        "honest means the shortcut also cost it something real, and the "
        "comparison is confounded rather than clean."
    )

    out = config.REPORTS / f"leakage_ablation_{args.tag}.json"
    out.write_text(json.dumps(
        {
            v: {
                "label": VARIANTS[v]["label"],
                "pr_auc": loaded[v]["headline"]["pr_auc"],
                "roc_auc": loaded[v]["headline"]["roc_auc"],
                "net_saving_inr": loaded[v]["optimal"]["net_saving_inr"],
                "precision_at_100": loaded[v]["precision_at_k"][0]["precision@k"],
            }
            for v in VARIANTS
        },
        indent=2,
    ))
    console.print(f"[green]wrote {out}[/green]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
