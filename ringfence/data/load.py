"""Load IEEE-CIS into memory on an 8GB machine, and split it honestly.

Two things here are load-bearing for the whole project:

1. `read_ieee_csv` builds an explicit dtype map from a sample before reading the
   full file. A naive `pd.read_csv` on train_transaction.csv infers float64 for
   all 380 numeric columns and needs ~1.9GB for the frame alone; float32 plus
   categorical objects brings it to ~500MB.

2. `temporal_split` is the reason any metric in this repo can be believed. See
   the docstring for why it is not a random split and why there is an embargo.
"""
from __future__ import annotations

import pandas as pd
from rich.console import Console

from ringfence import config

console = Console()


def _dtype_map(path, nrows: int = 20_000) -> dict:
    """Infer a compact dtype for every column from a head sample."""
    sample = pd.read_csv(path, nrows=nrows)
    dtypes = {}
    for col in sample.columns:
        s = sample[col]
        if col in (config.ID_COL, config.TARGET, config.TIME_COL):
            continue  # leave these at pandas defaults; they must stay exact
        if pd.api.types.is_float_dtype(s):
            dtypes[col] = "float32"
        elif pd.api.types.is_integer_dtype(s):
            dtypes[col] = "float32"  # NaNs appear later in the file for many int-looking cols
        else:
            dtypes[col] = "object"
    return dtypes


def read_ieee_csv(path) -> pd.DataFrame:
    """Read one IEEE-CIS csv with a memory-conscious dtype map."""
    dtypes = _dtype_map(path)
    df = pd.read_csv(path, dtype=dtypes)
    # Convert low-cardinality object columns to category after the fact: doing
    # it during read is not possible without knowing the full category set.
    for col in df.columns:
        if df[col].dtype == "object":
            nunique = df[col].nunique(dropna=True)
            if nunique > 0 and nunique / max(len(df), 1) < 0.5:
                df[col] = df[col].astype("category")
    return df


def load_raw(kind: str = "train") -> pd.DataFrame:
    """Merge the transaction and identity tables for `train` or `test`.

    Identity is a LEFT join: only ~24% of transactions carry device/identity
    rows, and the absence of an identity record is itself signal, so we keep
    the nulls rather than dropping to the inner join.
    """
    tx_path = config.RAW / f"{kind}_transaction.csv"
    id_path = config.RAW / f"{kind}_identity.csv"
    if not tx_path.exists():
        raise FileNotFoundError(
            f"{tx_path} not found. Run `python scripts/download_data.py` first."
        )

    console.log(f"reading {tx_path.name}")
    tx = read_ieee_csv(tx_path)
    console.log(f"  {len(tx):,} rows x {tx.shape[1]} cols, {tx.memory_usage(deep=True).sum()/1e6:.0f} MB")

    console.log(f"reading {id_path.name}")
    idf = read_ieee_csv(id_path)
    # The test identity file uses id-01 style names; train uses id_01. Normalise.
    idf.columns = [c.replace("-", "_") for c in idf.columns]

    df = tx.merge(idf, on=config.ID_COL, how="left")
    console.log(f"  merged -> {len(df):,} x {df.shape[1]}, {df.memory_usage(deep=True).sum()/1e6:.0f} MB")
    return df


def temporal_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by time, with an embargo. Never randomly.

    Why not a random split: fraud is a moving target and the entity graph is
    dense. A random split puts transactions from the SAME abuse ring on both
    sides of the boundary, so the model can memorise a ring in train and be
    rewarded for recognising it in test. That inflates every metric and is the
    single most common way fraud results are quietly wrong.

    Why an embargo: a chargeback is not reported the instant it happens. In
    production, at scoring time t you do not yet know the labels for the last
    several weeks, because those disputes have not been filed. Training right up
    to the test boundary uses recency that would not exist in deployment. We
    drop `EMBARGO_SECONDS` of training data adjacent to the boundary to model
    that reporting lag.

    Returns (train, test).
    """
    t = df[config.TIME_COL]
    cutoff = t.quantile(config.TRAIN_TIME_FRACTION)
    train = df[t <= cutoff - config.EMBARGO_SECONDS]
    test = df[t > cutoff]

    console.log(
        f"temporal split @ TransactionDT={cutoff:,.0f} "
        f"(embargo {config.EMBARGO_SECONDS/86400:.0f}d)"
    )
    for name, part in (("train", train), ("test", test)):
        rate = part[config.TARGET].mean() if config.TARGET in part else float("nan")
        console.log(f"  {name}: {len(part):,} rows, fraud rate {rate:.4%}")
    dropped = len(df) - len(train) - len(test)
    console.log(f"  embargoed away: {dropped:,} rows")
    return train.copy(), test.copy()


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """TransactionDT is a timedelta in seconds from an unknown epoch.

    We do NOT know the true start date, so absolute calendar features are
    unavailable. Relative day/hour-of-day are still valid and are known to carry
    signal (fraud clusters in specific hours).
    """
    df = df.copy()
    dt = df[config.TIME_COL]
    df["_day"] = (dt // 86400).astype("int32")
    df["_hour"] = ((dt // 3600) % 24).astype("int8")
    df["_dow"] = ((dt // 86400) % 7).astype("int8")
    return df
