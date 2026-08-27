"""Guards on the shipped feature path: nothing forbidden reaches the model."""
from __future__ import annotations

from ringfence import config
from ringfence.data.load import temporal_split
from ringfence.features.pipeline import prepare, to_matrix
from tests.conftest import make_frame


def test_target_and_time_never_reach_the_matrix():
    data = prepare(make_frame(seed=5))
    X = to_matrix(data.train, data.feature_names)
    for banned in (config.TARGET, config.TIME_COL, config.ID_COL, "client_id", "ring_id"):
        assert banned not in X.columns, f"{banned} leaked into the feature matrix"


def test_absolute_timestamps_are_excluded():
    """ring_first_seen trends with time and would separate periods, not fraud."""
    data = prepare(make_frame(seed=6))
    assert "ring_first_seen" not in data.feature_names
    assert not any(c.startswith("_ring") for c in data.feature_names)


def test_train_and_test_share_the_same_feature_set():
    data = prepare(make_frame(seed=7))
    X_tr = to_matrix(data.train, data.feature_names)
    X_te = to_matrix(data.test, data.feature_names)
    assert list(X_tr.columns) == list(X_te.columns)


def test_temporal_split_leaves_an_embargo_gap():
    df = make_frame(seed=8)
    train, test = temporal_split(df)
    assert len(train) and len(test)
    gap = test[config.TIME_COL].min() - train[config.TIME_COL].max()
    assert gap >= config.EMBARGO_SECONDS, "embargo was not applied"


def test_leaky_mode_actually_differs_from_the_shipped_path():
    """The ablation is only meaningful if the leaky variant really is leaky.

    Whole-frame ring fraud rate must see the ring's entire label history,
    including the current row, so it cannot match the lag-gated causal version.
    """
    df = make_frame(seed=9)
    honest = prepare(df)
    leaky = prepare(df, leaky_whole_frame=True)
    h = honest.train["ring_known_fraud_rate"].sum()
    lk = leaky.train["ring_known_fraud_rate"].sum()
    assert lk > h, "leaky mode should see strictly more label history"
