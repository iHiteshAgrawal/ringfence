"""Materialise the merged, split dataset as parquet so later stages are fast."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ringfence import config
from ringfence.data.load import add_time_features, load_raw, temporal_split


def main() -> int:
    df = load_raw("train")
    df = add_time_features(df)
    train, test = temporal_split(df)
    for name, part in (("train", train), ("test", test)):
        out = config.PROCESSED / f"{name}.parquet"
        part.to_parquet(out, index=False)
        print(f"wrote {out}  ({out.stat().st_size/1e6:.0f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
