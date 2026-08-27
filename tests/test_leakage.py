"""Guards against the concatenate-first mistake that inflates graph-feature results."""
from __future__ import annotations

import pandas as pd

from ringfence import config
from ringfence.data.load import temporal_split
from ringfence.features.build import RingFeatureBuilder
from tests.conftest import make_frame


def test_unseen_entities_are_marked_not_imputed():
    """A client absent from history must be flagged, not silently given features."""
    df = make_frame(seed=3)
    train = df.iloc[:600].copy()
    future = df.iloc[600:].copy()

    b = RingFeatureBuilder().fit(train)
    out = b.transform(future)

    unknown = out["entity_is_known"] == 0
    assert unknown.any(), "fixture should contain some unseen clients"
    # Unseen rows must carry NO ring aggregates -- otherwise something leaked.
    assert out.loc[unknown, "ring_n_tx"].isna().all()
    assert out.loc[unknown, "ring_velocity"].isna().all()


def test_transform_never_uses_future_rows():
    """Ring aggregates for a known entity must match history, not history+future.

    If the builder recomputed aggregates over the frame it is transforming, a
    ring's tx count would grow when we transform a larger frame. It must not.
    """
    df = make_frame(seed=4)
    train = df.iloc[:600].copy()
    small = df.iloc[600:650].copy()
    large = df.iloc[600:].copy()

    b = RingFeatureBuilder().fit(train)
    out_small = b.transform(small)
    out_large = b.transform(large).loc[small.index]

    known = (out_small["entity_is_known"] == 1) & (out_large["entity_is_known"] == 1)
    assert known.any()
    pd.testing.assert_series_equal(
        out_small.loc[known, "ring_n_tx"],
        out_large.loc[known, "ring_n_tx"],
        check_names=False,
    )


def test_target_and_time_never_reach_the_matrix():
    df = make_frame(seed=5)
    b = RingFeatureBuilder()
    out = b.fit_transform(df)
    X = b.to_matrix(out, fit_columns=True)
    for banned in (config.TARGET, config.TIME_COL, config.ID_COL, "client_id", "ring_id"):
        assert banned not in X.columns, f"{banned} leaked into the feature matrix"


def test_temporal_split_leaves_an_embargo_gap():
    df = make_frame(seed=6)
    train, test = temporal_split(df)
    assert len(train) and len(test)
    gap = test[config.TIME_COL].min() - train[config.TIME_COL].max()
    assert gap >= config.EMBARGO_SECONDS, "embargo was not applied"
