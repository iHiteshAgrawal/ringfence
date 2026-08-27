"""Stage 1: raw CSV -> feature-complete train/test parquet.

Split out from training for a concrete reason: on a 16GB-or-less machine, doing
both in one process peaks too high. Preparation holds the full 590k x 434 frame
(~1GB) plus its feature-engineered copy; training then wants the matrix plus
LightGBM's binned Dataset. Run together they exceeded 8GB and the process was
killed mid-training with no traceback.

Two processes means preparation's memory is returned to the OS before training
starts. It also makes iteration far faster: prepare once, train many times.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console

from ringfence import config
from ringfence.data.load import load_raw
from ringfence.features.pipeline import prepare, to_matrix

console = Console()

# Columns kept alongside the features because evaluation needs them.
KEEP_EXTRA = [config.TARGET, config.TIME_COL, "TransactionAmt", "ring_id", "client_id"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label-lag-days", type=int, default=30)
    ap.add_argument("--max-shared-clients", type=int, default=20)
    ap.add_argument("--max-ring-clients", type=int, default=50)
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--tag", type=str, default="main")
    args = ap.parse_args()

    df = load_raw("train")
    if args.sample:
        df = df.sort_values(config.TIME_COL).head(args.sample).copy()
        console.log(f"SAMPLED to {len(df):,} rows")

    data = prepare(
        df,
        max_shared_clients=args.max_shared_clients,
        label_lag_days=args.label_lag_days,
    )
    del df
    gc.collect()

    meta = {
        "feature_names": data.feature_names,
        "diagnostics": data.diagnostics,
        "n_train": len(data.train),
        "n_test": len(data.test),
        "args": vars(args),
    }

    for name in ("train", "test"):
        part = getattr(data, name)
        X = to_matrix(part, data.feature_names)
        for col in KEEP_EXTRA:
            if col in part.columns:
                X[col] = part[col].to_numpy()
        out = config.PROCESSED / f"{name}_{args.tag}.parquet"
        X.to_parquet(out, index=False)
        console.log(f"wrote {out.name}: {X.shape[0]:,} x {X.shape[1]} "
                    f"({out.stat().st_size/1e6:.0f} MB)")
        del X, part
        gc.collect()

    meta_path = config.PROCESSED / f"meta_{args.tag}.json"
    meta_path.write_text(json.dumps(meta, indent=2, default=str))
    console.log(f"wrote {meta_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
