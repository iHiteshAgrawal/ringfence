"""The one path from raw transactions to a trained, priced detector.

Order matters and is enforced here so no experiment can accidentally reorder it:

    resolve            recover client entities from card fingerprint + anchor
    assign_rings       causal union-find replay; ring as known on arrival
    causal features    ring history, strictly prior, with label-reporting lag
    temporal split     with embargo
    matrix             drop identifiers/target/time, cast categoricals

The split happens AFTER feature construction on purpose: the causal features are
already guaranteed past-only (see tests/test_causal.py and tests/test_streaming.py),
so computing them over the full frame is safe and lets a test-period transaction
use its own period's earlier transactions -- which is exactly what production does.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
from rich.console import Console

from ringfence import config
from ringfence.data.load import add_time_features, temporal_split
from ringfence.entity.graph import ring_features as whole_frame_ring_features
from ringfence.entity.resolve import resolve
from ringfence.entity.streaming import assign_rings_causally
from ringfence.features.causal import DEFAULT_LABEL_LAG_DAYS, add_causal_ring_features

console = Console()

# Never fed to the model: identifiers, the target, and the raw time axis (it
# trends, and the test period is strictly later, so the model would learn
# "this is the late period" rather than anything about fraud).
DROP_ALWAYS = {
    config.ID_COL,
    config.TARGET,
    config.TIME_COL,
    "client_id",
    "ring_id",
    "_card_anchor",
    "_ring_truth",
    "_day",
}


@dataclass
class PreparedData:
    train: pd.DataFrame
    test: pd.DataFrame
    feature_names: list[str] = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)

    def matrices(self):
        X_tr = to_matrix(self.train, self.feature_names)
        X_te = to_matrix(self.test, self.feature_names)
        return X_tr, self.train[config.TARGET], X_te, self.test[config.TARGET]


def to_matrix(df: pd.DataFrame, feature_names: list[str] | None = None) -> pd.DataFrame:
    # ring_first_seen is an absolute time. Like TransactionDT it trends, and
    # the test period is strictly later, so keeping it would let the model learn
    # "this is the late period" instead of anything about fraud.
    drop = [
        c for c in df.columns
        if c in DROP_ALWAYS or c.startswith("_ring") or c == "ring_first_seen"
    ]
    X = df.drop(columns=drop, errors="ignore")
    for col in X.columns:
        dt = X[col].dtype
        if dt == "object" or isinstance(dt, pd.CategoricalDtype) or pd.api.types.is_string_dtype(dt):
            X[col] = X[col].astype("category")
    if feature_names:
        for c in feature_names:
            if c not in X.columns:
                X[c] = pd.NA
        X = X[feature_names]
    return X


def prepare(
    df: pd.DataFrame,
    max_shared_clients: int = 20,
    label_lag_days: int = DEFAULT_LABEL_LAG_DAYS,
    include_label_features: bool = True,
    leaky_whole_frame: bool = False,
) -> PreparedData:
    """Raw merged frame -> split, feature-complete train/test.

    `leaky_whole_frame=True` deliberately BREAKS the causal guarantee: ring
    aggregates are computed over the entire dataset at once, so a transaction's
    features include what its ring did afterwards. This is the standard mistake
    and it exists here only so `scripts/leakage_ablation.py` can measure how
    much it inflates the results. Never use it for a reported number.
    """
    console.log("resolving client entities")
    df = resolve(add_time_features(df))

    console.log("assigning rings causally")
    ring, ring_diag = assign_rings_causally(df, max_shared_clients=max_shared_clients)
    df["ring_id"] = ring

    if leaky_whole_frame:
        # A FAIR test of the leak has to keep the feature SET the same and vary
        # only its causality. An earlier version simply swapped in whole-frame
        # aggregates, which also dropped the lag-gated fraud-history feature --
        # so the leaky variant scored WORSE and proved nothing except that it
        # had lost the 6th most important feature.
        console.log("[bold red]LEAKY MODE[/bold red]: whole-frame ring aggregates")
        df = df.join(whole_frame_ring_features(df), on="ring_id")
        # The classic mistake in full: ring fraud rate over the entire dataset,
        # counting the current transaction and everything after it.
        g = df.groupby("ring_id", observed=True)[config.TARGET]
        df["ring_known_prior_frauds"] = g.transform("sum").astype("float32")
        df["ring_known_fraud_rate"] = g.transform("mean").astype("float32")
    else:
        console.log(f"building causal ring features (label lag {label_lag_days}d)")
        df = add_causal_ring_features(
            df,
            label_lag_days=label_lag_days,
            include_label_features=include_label_features,
        )

    train, test = temporal_split(df)
    feature_names = list(to_matrix(train).columns)
    console.log(f"{len(feature_names)} features")

    return PreparedData(
        train=train,
        test=test,
        feature_names=feature_names,
        diagnostics={"rings": ring_diag, "label_lag_days": label_lag_days},
    )
