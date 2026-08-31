"""Assemble docs/data.json for the GitHub Pages report.

The site is static, but nothing on it is hand-written: every number is read out
of reports/ and models/, so republishing after a re-run cannot drift from the
experiment. That property is the whole point -- a results page that can silently
disagree with its own repository is worse than no page.

  python scripts/build_site.py                 # metrics only (no API cost)
  python scripts/build_site.py --with-casefiles # also regenerate ring case files
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console

from ringfence import config

console = Console()
DOCS = config.ROOT / "docs"
DOCS.mkdir(exist_ok=True)


def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=config.ROOT, check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, OSError):
        return "unknown"


def load_json(name: str) -> dict | None:
    p = config.REPORTS / name
    return json.loads(p.read_text()) if p.exists() else None


def cost_curve(tag: str) -> list[dict]:
    """The operating points, thinned to keep the payload small but faithful.

    Keep every point where the decision changes materially: we downsample the
    saturated tails and keep full resolution through the interesting band.
    """
    p = config.REPORTS / f"cost_curve_{tag}.csv"
    df = pd.read_csv(p)
    keep = df[
        (df["flag_rate"] <= 0.25) | (df.index % 4 == 0)
    ].copy()
    cols = ["threshold", "flag_rate", "tp", "fp", "fn",
            "precision", "recall", "net_saving_inr", "fp_friction_inr",
            "fraud_loss_inr"]
    out = keep[cols].round(6).to_dict(orient="records")
    console.log(f"cost curve: {len(df)} points -> {len(out)} kept")
    return out


def ring_queue(tag: str, n: int, with_casefiles: bool) -> list[dict]:
    from scripts.demo_agent import load_scored, top_rings

    df = load_scored(tag)
    facts = top_rings(df, n)
    rows = []
    for f in facts:
        entry = f.model_dump()
        if with_casefiles:
            from ringfence.agent.casefile import write_case_file
            console.log(f"generating case file for ring {f.ring_id}")
            entry["case_file"] = write_case_file(f).model_dump()
        rows.append(entry)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="main")
    ap.add_argument("--rings", type=int, default=5)
    ap.add_argument("--with-casefiles", action="store_true")
    ap.add_argument(
        "--mirror", type=str, default=None,
        help="Also copy the built page to this directory (e.g. a portfolio "
             "repo). All assets travel together -- index.html, data.json and "
             "architecture.svg are fetched relative to the page, so copying "
             "only the HTML yields a broken report.",
    )
    args = ap.parse_args()

    # Regenerate the architecture diagram so it cannot fall out of step.
    subprocess.run([sys.executable, str(Path(__file__).parent / "make_architecture.py")],
                   check=True, capture_output=True)
    console.log("regenerated architecture.svg")

    main_res = load_json(f"results_{args.tag}.json")
    if main_res is None:
        raise SystemExit(f"reports/results_{args.tag}.json missing; run the experiment first")
    noring = load_json("results_noring.json")
    ablation = load_json(f"leakage_ablation_{args.tag}.json")

    existing = DOCS / "data.json"
    prior = json.loads(existing.read_text()) if existing.exists() else {}

    rings = (
        ring_queue(args.tag, args.rings, args.with_casefiles)
        if args.with_casefiles or "rings" not in prior
        else prior["rings"]
    )
    if not args.with_casefiles and "rings" in prior:
        console.log("reusing cached case files (pass --with-casefiles to regenerate)")

    payload = {
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        "commit": git_commit(),
        "dataset": {
            "name": "IEEE-CIS Fraud Detection",
            "url": "https://www.kaggle.com/competitions/ieee-fraud-detection/data",
            "n_total": 590540,
            "n_train": main_res["n_train"],
            "n_test": main_res["n_test"],
            "prevalence": main_res["headline"]["prevalence"],
        },
        "headline": main_res["headline"],
        "optimal": main_res["optimal"],
        "baseline_loss_inr": main_res["baseline_loss_inr"],
        "cost_model": main_res["cost_model"],
        "precision_at_k": main_res["precision_at_k"],
        "recall_at_fp_budget": main_res["recall_at_fp_budget"],
        "top_features": main_res["top_features"][:15],
        "ring_feature_gain_pct": main_res["ring_feature_gain_pct"],
        "ring_diagnostics": main_res["ring_diagnostics"]["rings"],
        "noring": {
            "headline": noring["headline"],
            "optimal": noring["optimal"],
            "precision_at_k": noring["precision_at_k"],
            "recall_at_fp_budget": noring["recall_at_fp_budget"],
            "best_iteration": noring["best_iteration"],
        } if noring else None,
        "best_iteration": main_res["best_iteration"],
        "ablation": ablation,
        "cost_curve": cost_curve(args.tag),
        "rings": rings,
    }

    out = DOCS / "data.json"
    out.write_text(json.dumps(payload, indent=1))
    console.log(f"wrote {out} ({out.stat().st_size/1024:.0f} KB)")

    if args.mirror:
        dest = Path(args.mirror).expanduser()
        dest.mkdir(parents=True, exist_ok=True)
        # Every asset the page needs at runtime. Missing one leaves a live
        # site half-broken, so this list must track what index.html fetches.
        for name in ("index.html", "data.json", "architecture.svg"):
            src = DOCS / name
            if not src.exists():
                console.log(f"[yellow]skipped {name}: not in docs/[/yellow]")
                continue
            shutil.copy2(src, dest / name)
            console.log(f"mirrored {name} -> {dest / name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
