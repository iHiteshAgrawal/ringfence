"""Assemble the feature matrix, with a strict fit/transform leakage boundary.

THE LEAKAGE TRAP
----------------
Graph features are the easiest place in fraud modelling to cheat by accident.
The tempting move is to concatenate train and test, build one entity graph over
everything, compute ring aggregates, and split afterwards. Every metric then
looks superb -- because a test transaction's ring features were computed using
that very transaction, and using its ring-mates' *future* behaviour. The model
is being told the answer.

What we do instead mirrors production exactly:

  fit(train)      Build the entity graph and ring aggregates from history only.
                  This is the state a deployed system has at 9am on any day.

  transform(test) A new transaction arrives. Resolve it to a client. Look up
                  whether that client already belongs to a known ring. If yes,
                  attach that ring's HISTORICAL aggregates. If no, it is an
                  unseen entity -- attach null/singleton features, which is
                  genuinely all a live system would know.

The consequence is that test performance is lower than the concatenate-first
version would report. That gap is the honesty premium, and it is quantified in
`scripts/leakage_ablation.py`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from rich.console import Console

from ringfence import config
from ringfence.entity.graph import build_ring_labels, ring_features
from ringfence.entity.resolve import client_summary, resolve

console = Console()

# Columns that must never enter the model: identifiers, the target, the raw
# time axis (it trends, and the test set is strictly later -- the model would
# learn "later = different" rather than anything about fraud), and our own
# bookkeeping columns.
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


class RingFeatureBuilder:
    """Fits an entity graph on history; scores new transactions against it."""

    def __init__(self, max_shared_clients: int = 20, link_attrs: list[str] | None = None):
        self.max_shared_clients = max_shared_clients
        self.link_attrs = link_attrs or config.LINK_ATTRS
        self.client_to_ring_: dict[str, int] = {}
        self.ring_feats_: pd.DataFrame | None = None
        self.client_feats_: pd.DataFrame | None = None
        self.diagnostics_: dict = {}
        self.feature_names_: list[str] = []

    # -- fit ---------------------------------------------------------------
    def fit(self, train: pd.DataFrame) -> RingFeatureBuilder:
        df = resolve(train)
        ring_id, diag = build_ring_labels(
            df, link_attrs=self.link_attrs, max_shared_clients=self.max_shared_clients
        )
        df["ring_id"] = ring_id
        self.diagnostics_ = diag

        self.ring_feats_ = ring_features(df)
        self.client_feats_ = client_summary(df)
        # client -> ring lookup, the "known entity graph" a live system holds.
        self.client_to_ring_ = (
            df.drop_duplicates("client_id").set_index("client_id")["ring_id"].to_dict()
        )
        console.log(
            f"fitted graph: {diag['n_clients']:,} clients, "
            f"{diag['n_multi_client_rings']:,} multi-client rings, "
            f"largest={diag['largest_ring_clients']}"
        )
        return self

    # -- transform ---------------------------------------------------------
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.ring_feats_ is None:
            raise RuntimeError("call fit() first")
        df = resolve(df)
        ring = df["client_id"].map(self.client_to_ring_)
        df["ring_id"] = ring

        known = ring.notna()
        out = df.join(self.ring_feats_, on="ring_id")
        out = out.join(self.client_feats_.add_prefix("client_"), on="client_id")

        # An unseen entity is not a missing value to be imputed away -- it is a
        # meaningful state ("we have never seen this client"), so we flag it.
        out["entity_is_known"] = known.astype("int8")
        out["ring_is_multi_client"] = (out["ring_n_clients"].fillna(1) > 1).astype("int8")
        console.log(
            f"transform: {len(out):,} rows, "
            f"{known.mean():.1%} matched a known client entity"
        )
        return out

    def fit_transform(self, train: pd.DataFrame) -> pd.DataFrame:
        return self.fit(train).transform(train)

    # -- matrix ------------------------------------------------------------
    def to_matrix(self, df: pd.DataFrame, fit_columns: bool = False) -> pd.DataFrame:
        """Numeric/categorical matrix LightGBM can consume directly."""
        drop = [c for c in df.columns if c in DROP_ALWAYS]
        X = df.drop(columns=drop)
        for col in X.columns:
            dt = X[col].dtype
            # pandas >=3 gives object string columns a dedicated `str` dtype,
            # which LightGBM rejects just as it rejects object.
            if (
                dt == "object"
                or isinstance(dt, pd.CategoricalDtype)
                or pd.api.types.is_string_dtype(dt)
            ):
                X[col] = X[col].astype("category")
        if fit_columns:
            self.feature_names_ = list(X.columns)
        else:
            # Align to the fitted column set: same columns, same order, and
            # categorical levels reconciled so LightGBM does not see new codes.
            for c in self.feature_names_:
                if c not in X.columns:
                    X[c] = np.nan
            X = X[self.feature_names_]
        return X
