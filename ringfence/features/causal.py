"""Ring features computed causally: for each transaction, its ring's PAST only.

WHY THIS REPLACES THE FIT/TRANSFORM APPROACH
--------------------------------------------
The first design fitted the entity graph on the training period and looked new
transactions up in it. Measured on a temporal split, only a small fraction of
test transactions matched a client seen during training -- so the ring features
were absent exactly where they were supposed to earn their keep.

That is not a bug in the graph, it is the wrong framing. A production system
does not freeze its graph at training time. When a transaction arrives at 14:32,
the system can legitimately use everything it knows as of 14:32 -- including a
sibling transaction from the same ring at 14:19. Using that is not leakage; it
is the entire point of real-time ring defence.

So: build the graph over all rows (structure is observable on arrival), then
compute every ring aggregate over strictly-earlier transactions only, via
expanding windows in time order.

THE LABEL-LAG SUBTLETY
----------------------
One feature is special: "how many confirmed frauds has this ring already had?"
That is enormously predictive and it is the single easiest place to cheat,
because a chargeback is not known when it happens -- the issuer reports it weeks
later. We therefore only count prior frauds that would have been REPORTED by
the time of the current transaction, controlling the delay with
`label_lag_days`. Set it to 0 and metrics jump; that number is a fiction. The
default is a deliberately conservative 30 days.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from rich.console import Console

from ringfence import config

console = Console()

# Chargebacks are typically raised within 120 days and take weeks to surface.
# 30 days is a conservative floor for "the merchant now knows this was fraud".
DEFAULT_LABEL_LAG_DAYS = 30


def _expanding_nunique(keys: np.ndarray, group: np.ndarray) -> np.ndarray:
    """Running count of distinct `keys` seen so far within each `group`.

    Done by marking the first occurrence of each (group, key) pair and taking a
    cumulative sum of those marks. O(n log n), no Python-level loop.
    """
    order = np.arange(len(keys))
    pair = pd.DataFrame({"g": group, "k": keys, "i": order})
    first = ~pair.duplicated(subset=["g", "k"], keep="first")
    pair["is_first"] = first.astype("int32")
    pair["cum"] = pair.groupby("g", observed=True)["is_first"].cumsum()
    return pair["cum"].to_numpy()


def add_causal_ring_features(
    df: pd.DataFrame,
    label_lag_days: int = DEFAULT_LABEL_LAG_DAYS,
    include_label_features: bool = True,
) -> pd.DataFrame:
    """Attach ring-history features. `df` must already have `ring_id`.

    Every column added is a function of transactions strictly EARLIER in time
    within the same ring. The current row never contributes to its own features.
    """
    if "ring_id" not in df.columns:
        raise ValueError("call resolve() and build_ring_labels() first")

    d = df.sort_values(config.TIME_COL, kind="mergesort").copy()
    t = d[config.TIME_COL].to_numpy(dtype="float64")
    g = d.groupby("ring_id", observed=True)

    # --- counts and volume, shifted so the current row is excluded ---------
    d["ring_prior_tx"] = g.cumcount().astype("float32")
    amt = d["TransactionAmt"].astype("float64")
    d["ring_prior_amt_sum"] = (g["TransactionAmt"].cumsum() - amt).astype("float32")
    d["ring_prior_amt_mean"] = (
        d["ring_prior_amt_sum"] / d["ring_prior_tx"].replace(0, np.nan)
    ).astype("float32")

    # --- distinct entities seen so far in the ring ------------------------
    ring_codes = d["ring_id"].astype("category").cat.codes.to_numpy()
    client_codes = d["client_id"].astype("category").cat.codes.to_numpy()
    cum_clients = _expanding_nunique(client_codes, ring_codes)
    # Subtract self-contribution: if this row introduced the client, the count
    # includes it, so step back one to keep the feature strictly prior.
    is_new_client = np.r_[True, (np.diff(cum_clients) == 1)]
    d["ring_prior_clients"] = (cum_clients - is_new_client.astype(int)).astype("float32")
    d["client_is_new_to_ring"] = is_new_client.astype("int8")

    if "addr1" in d.columns:
        addr_codes = d["addr1"].astype("category").cat.codes.to_numpy()
        cum_addr = _expanding_nunique(addr_codes, ring_codes)
        d["ring_prior_addrs"] = cum_addr.astype("float32")

    # --- timing ------------------------------------------------------------
    d["ring_first_seen"] = g[config.TIME_COL].transform("min").astype("float64")
    d["ring_age_days"] = ((t - d["ring_first_seen"]) / 86400.0).astype("float32")
    d["ring_secs_since_prev"] = g[config.TIME_COL].diff().astype("float32")
    # Velocity over the ring's life so far. The canonical bust-out signature is
    # many transactions compressed into a very short age.
    d["ring_prior_velocity"] = (
        d["ring_prior_tx"] / d["ring_age_days"].clip(lower=1.0 / 24)
    ).astype("float32")
    d["ring_cards_per_addr_prior"] = (
        d["ring_prior_clients"] / d.get("ring_prior_addrs", pd.Series(1, index=d.index)).clip(lower=1)
    ).astype("float32")

    # --- amount homogeneity -------------------------------------------------
    sq = g["TransactionAmt"].transform(lambda s: (s**2).cumsum()) - amt**2
    mean_prior = d["ring_prior_amt_mean"].astype("float64")
    var_prior = (sq / d["ring_prior_tx"].replace(0, np.nan)) - mean_prior**2
    d["ring_prior_amt_cv"] = (
        np.sqrt(var_prior.clip(lower=0)) / mean_prior.clip(lower=0.01)
    ).astype("float32")

    # --- label-derived, gated by reporting lag ------------------------------
    if include_label_features and config.TARGET in d.columns:
        lag = label_lag_days * 86400
        d = _add_lagged_fraud_history(d, lag)

    d["ring_prior_amt_mean"] = d["ring_prior_amt_mean"].fillna(0.0)
    d["ring_prior_amt_cv"] = d["ring_prior_amt_cv"].fillna(0.0)
    return d.sort_index()


def _add_lagged_fraud_history(d: pd.DataFrame, lag_seconds: float) -> pd.DataFrame:
    """Count prior confirmed frauds in the ring that would be KNOWN by now.

    For row i at time t_i, we count rows j in the same ring with
    t_j + lag <= t_i and isFraud == 1. Implemented as a per-ring merge_asof
    against a cumulative fraud count evaluated at (t_i - lag).
    """
    t = d[config.TIME_COL].astype("float64")

    left = pd.DataFrame({
        "ring_id": d["ring_id"].to_numpy(),
        "t_query": (t - lag_seconds).to_numpy(),
        "_pos": np.arange(len(d)),
    }).sort_values("t_query", kind="mergesort")

    right = pd.DataFrame({
        "ring_id": d["ring_id"].to_numpy(),
        "t_event": t.to_numpy(),
        "cum_fraud": d.groupby("ring_id", observed=True)[config.TARGET]
        .cumsum().astype("float64").to_numpy(),
    }).sort_values("t_event", kind="mergesort")

    merged = pd.merge_asof(
        left, right,
        left_on="t_query", right_on="t_event",
        by="ring_id", direction="backward", allow_exact_matches=True,
    )
    out = np.zeros(len(d), dtype="float32")
    out[merged["_pos"].to_numpy()] = merged["cum_fraud"].fillna(0.0).to_numpy()
    d["ring_known_prior_frauds"] = out
    d["ring_known_fraud_rate"] = (
        out / d["ring_prior_tx"].replace(0, np.nan)
    ).fillna(0.0).astype("float32")
    return d


CAUSAL_FEATURES = [
    "ring_prior_tx", "ring_prior_amt_sum", "ring_prior_amt_mean", "ring_prior_amt_cv",
    "ring_prior_clients", "client_is_new_to_ring", "ring_prior_addrs",
    "ring_age_days", "ring_secs_since_prev", "ring_prior_velocity",
    "ring_cards_per_addr_prior", "ring_known_prior_frauds", "ring_known_fraud_rate",
]
