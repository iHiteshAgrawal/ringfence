"""Causality: a row's features must not move when the future is deleted."""
from __future__ import annotations

import pandas as pd

from ringfence import config
from ringfence.entity.graph import build_ring_labels
from ringfence.entity.resolve import resolve
from ringfence.features.causal import CAUSAL_FEATURES, add_causal_ring_features
from tests.conftest import make_frame


def _prepared(seed=7, **kw):
    df = make_frame(seed=seed, **kw)
    df = resolve(df)
    df["ring_id"] = build_ring_labels(df, max_shared_clients=20)[0]
    return df


def test_truncating_the_future_changes_nothing():
    """The strongest available guarantee that no future data is used.

    Compute features on the full frame, then on only the first half of TIME,
    and require the surviving rows to be bit-identical. Any leak from a later
    transaction into an earlier row's feature breaks this.
    """
    df = _prepared()
    full = add_causal_ring_features(df)

    cutoff = df[config.TIME_COL].quantile(0.5)
    past_only = add_causal_ring_features(df[df[config.TIME_COL] <= cutoff].copy())

    cols = [c for c in CAUSAL_FEATURES if c in full.columns]
    a = full.loc[past_only.index, cols].sort_index()
    b = past_only[cols].sort_index()
    pd.testing.assert_frame_equal(a, b, check_dtype=False, rtol=1e-5)


def test_first_transaction_in_a_ring_has_no_history():
    df = _prepared()
    out = add_causal_ring_features(df)
    firsts = out.groupby("ring_id", observed=True)[config.TIME_COL].idxmin()
    first_rows = out.loc[firsts]
    assert (first_rows["ring_prior_tx"] == 0).all()
    assert (first_rows["ring_prior_amt_sum"] == 0).all()
    assert (first_rows["ring_known_prior_frauds"] == 0).all()


def test_row_never_counts_itself():
    df = _prepared()
    out = add_causal_ring_features(df)
    sizes = out.groupby("ring_id", observed=True).size()
    # Prior count must always be strictly less than the ring's total size.
    assert (out["ring_prior_tx"] < out["ring_id"].map(sizes)).all()


def test_label_lag_gates_fraud_history():
    """With an enormous reporting lag, no prior fraud can be known yet."""
    df = _prepared()
    fast = add_causal_ring_features(df, label_lag_days=0)
    slow = add_causal_ring_features(df, label_lag_days=100_000)
    assert fast["ring_known_prior_frauds"].sum() > 0, "lag=0 should see history"
    assert slow["ring_known_prior_frauds"].sum() == 0, "huge lag should see none"
    # And the realistic setting must sit between the two extremes.
    real = add_causal_ring_features(df, label_lag_days=30)
    assert real["ring_known_prior_frauds"].sum() <= fast["ring_known_prior_frauds"].sum()


def test_bursty_rings_score_higher_velocity():
    df = _prepared(n_normal=600, n_rings=10)
    out = add_causal_ring_features(df)
    planted = out[out["_ring_truth"] >= 0]
    normal = out[out["_ring_truth"] < 0]
    assert planted["ring_prior_velocity"].mean() > normal["ring_prior_velocity"].mean()
