"""Full run on real IEEE-CIS: prepare -> train -> price -> report.

Writes reports/results.json and reports/figures/*.png, and prints the numbers
that go in the pitch. Every claim in ARCHITECTURE.md should be reproducible by
running this file.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console

from ringfence import config
from ringfence.eval.metrics import CostModel, summary_report
from ringfence.model.train import feature_importance, train

console = Console()


def jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, pd.DataFrame):
        return o.to_dict(orient="records")
    if isinstance(o, pd.Series):
        return o.to_dict()
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    return o


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=3000)
    ap.add_argument("--tag", type=str, default="main")
    ap.add_argument("--data-tag", type=str, default=None,
                    help="parquet tag from prepare_data.py (defaults to --tag)")
    ap.add_argument("--no-ring-features", action="store_true",
                    help="ablation: drop every ring/entity feature")
    ap.add_argument("--keep-all-v", action="store_true",
                    help="skip V-block reduction (needs more memory)")
    args = ap.parse_args()
    data_tag = args.data_tag or args.tag

    t0 = time.time()
    console.rule("[bold]load prepared")
    meta = json.loads((config.PROCESSED / f"meta_{data_tag}.json").read_text())
    feature_names = meta["feature_names"]

    # Drop the redundant V columns chosen by VBlockReducer (fitted on train).
    vsel_path = config.PROCESSED / f"vselect_{data_tag}.json"
    if vsel_path.exists() and not args.keep_all_v:
        drop_v = set(json.loads(vsel_path.read_text())["drop"])
        feature_names = [f for f in feature_names if f not in drop_v]
        console.log(f"V-block reduction: {len(drop_v)} columns dropped, "
                    f"{len(feature_names)} features remain")

    # Read only the columns we need; the parquet holds extras for evaluation.
    # dict.fromkeys de-duplicates while preserving order: TransactionAmt is
    # both a model feature and needed for the cost model, and asking parquet for
    # it twice yields a duplicated column that breaks every X[col] lookup.
    need = list(dict.fromkeys(
        feature_names + [config.TARGET, config.TIME_COL, "TransactionAmt"]
    ))
    train_df = pd.read_parquet(config.PROCESSED / f"train_{data_tag}.parquet", columns=need)
    test_df = pd.read_parquet(config.PROCESSED / f"test_{data_tag}.parquet", columns=need)
    console.log(f"train {train_df.shape}  test {test_df.shape}")

    if args.no_ring_features:
        dropped = [f for f in feature_names
                   if f.startswith(("ring_", "client_", "entity_"))]
        feature_names = [f for f in feature_names if f not in dropped]
        console.log(f"ABLATION: dropped {len(dropped)} ring/entity features")

    t_tr = train_df[config.TIME_COL].copy()
    y_tr = train_df[config.TARGET].copy()
    y_te = test_df[config.TARGET].copy()
    amounts = test_df["TransactionAmt"].to_numpy()

    X_tr = train_df[feature_names]
    X_te = test_df[feature_names]
    del train_df
    gc.collect()

    console.rule("[bold]train")
    model = train(X_tr, y_tr, t_tr, num_boost_round=args.rounds)
    del X_tr
    gc.collect()
    scores = model.predict(X_te)
    del X_te, test_df
    gc.collect()

    console.rule("[bold]evaluate")
    rep = summary_report(y_te.to_numpy(), scores, amounts, CostModel())

    h = rep["headline"]
    best = rep["optimal"]
    console.print(f"\n[bold]Held-out test: {h['n']:,} transactions, "
                  f"{h['prevalence']:.3%} fraud[/bold]")
    console.print(f"  PR-AUC   {h['pr_auc']:.4f}   ({h['pr_auc_lift_over_baseline']:.1f}x baseline)")
    console.print(f"  ROC-AUC  {h['roc_auc']:.4f}   <- the flattering one")
    console.print("\n[bold]precision@k[/bold]")
    console.print(rep["precision_at_k"].to_string(index=False))
    console.print("\n[bold]recall at fixed false-positive budget[/bold]")
    console.print(rep["recall_at_fp_budget"].to_string(index=False))
    console.print("\n[bold]cost-optimal operating point[/bold]")
    console.print(f"  threshold {best['threshold']:.4f}  flag rate {best['flag_rate']:.3%}")
    console.print(f"  precision {best['precision']:.3f}  recall {best['recall']:.3f}")
    console.print(f"  net saving  Rs {best['net_saving_inr']:,.0f}")
    console.print(f"  vs baseline loss Rs {rep['baseline_loss_inr']:,.0f} "
                  f"({100*best['net_saving_inr']/rep['baseline_loss_inr']:.1f}% recovered)")

    imp = feature_importance(model, top=50)
    ring_gain = float(imp.loc[imp["is_ring_feature"], "gain_pct"].sum())
    console.print(f"\n[bold]ring/entity features = {ring_gain:.1f}% of top-50 gain[/bold]")
    console.print(imp.head(20)[["feature", "gain_pct", "is_ring_feature"]].to_string(index=False))

    out = {
        "tag": args.tag,
        "args": vars(args),
        "runtime_seconds": round(time.time() - t0, 1),
        "n_train": int(meta["n_train"]),
        "n_test": int(meta["n_test"]),
        "headline": h,
        "precision_at_k": jsonable(rep["precision_at_k"]),
        "recall_at_fp_budget": jsonable(rep["recall_at_fp_budget"]),
        "optimal": jsonable(best),
        "baseline_loss_inr": rep["baseline_loss_inr"],
        "cost_model": rep["cost_model"],
        "ring_feature_gain_pct": ring_gain,
        "top_features": jsonable(imp.head(30)),
        "ring_diagnostics": jsonable(meta["diagnostics"]),
        "valid_pr_auc": model.valid_pr_auc,
        "best_iteration": model.best_iteration,
    }
    path = config.REPORTS / f"results_{args.tag}.json"
    path.write_text(json.dumps(out, indent=2, default=str))
    console.print(f"\n[green]wrote {path}[/green]  ({out['runtime_seconds']}s total)")

    rep["cost_curve"].to_csv(config.REPORTS / f"cost_curve_{args.tag}.csv", index=False)
    model.save(config.ROOT / "models" / f"lgbm_{args.tag}.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
