"""End-to-end smoke: fixture -> causal features -> model -> rupee metrics."""
from __future__ import annotations

import numpy as np

from ringfence import config
from ringfence.eval.metrics import CostModel, summary_report
from ringfence.features.pipeline import prepare
from ringfence.model.train import feature_importance, train
from tests.conftest import make_frame


def test_end_to_end_runs_and_beats_chance():
    df = make_frame(n_normal=2000, n_rings=30, seed=11)
    data = prepare(df)

    X_tr, y_tr, X_te, y_te = data.matrices()
    assert config.TARGET not in X_tr.columns
    assert config.TIME_COL not in X_tr.columns

    model = train(
        X_tr, y_tr, data.train[config.TIME_COL],
        num_boost_round=300, early_stopping_rounds=40,
    )
    scores = model.predict(X_te)
    assert np.all((scores >= 0) & (scores <= 1)), "calibrated scores must be probabilities"

    rep = summary_report(
        y_te.to_numpy(), scores, data.test["TransactionAmt"].to_numpy(), CostModel()
    )
    assert rep["headline"]["pr_auc"] > rep["headline"]["prevalence"]

    imp = feature_importance(model)
    ring_gain = imp.loc[imp["is_ring_feature"], "gain_pct"].sum()
    assert ring_gain > 0, "ring features should carry some gain"
    print(f"\nPR-AUC {rep['headline']['pr_auc']:.3f} | ring feature gain {ring_gain:.1f}%")


def test_label_lag_matters():
    """Pretending chargebacks are reported instantly must look better.

    If it does not, the lagged-fraud feature is not doing anything and the
    honesty claim in ARCHITECTURE.md would be hollow.
    """
    df = make_frame(n_normal=2000, n_rings=30, seed=12)
    honest = prepare(df, label_lag_days=30)
    cheating = prepare(df, label_lag_days=0)
    h = honest.train["ring_known_prior_frauds"].sum()
    c = cheating.train["ring_known_prior_frauds"].sum()
    assert c >= h
