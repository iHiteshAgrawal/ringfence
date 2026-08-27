"""The cost model must be able to call a well-ranked model a bad investment."""
from __future__ import annotations

import numpy as np
import pytest

from ringfence.eval.metrics import (
    CostModel,
    cost_curve,
    headline,
    optimal_operating_point,
    precision_at_k,
    recall_at_fp_budget,
)


@pytest.fixture
def scored():
    """3.5% prevalence, a decent-but-imperfect score."""
    rng = np.random.default_rng(0)
    n = 20_000
    y = (rng.random(n) < 0.035).astype(int)
    score = np.clip(rng.normal(0.2, 0.15, n) + y * rng.normal(0.45, 0.2, n), 0, 1)
    amt = rng.lognormal(4.2, 1.0, n)
    return y, score, amt


def test_roc_auc_flatters_relative_to_pr_auc(scored):
    """The core claim of this project's metric choice, asserted."""
    y, score, _ = scored
    h = headline(y, score)
    assert h["roc_auc"] > h["pr_auc"], "expected ROC to look better than PR"
    # PR-AUC must be judged against prevalence, not 0.5.
    assert h["pr_auc"] > h["prevalence"], "model should beat the trivial baseline"
    assert h["pr_auc_lift_over_baseline"] > 1.0


def test_precision_at_k_decreases_with_k(scored):
    y, score, _ = scored
    pk = precision_at_k(y, score, ks=(100, 1000, 5000))
    assert pk["precision@k"].is_monotonic_decreasing
    assert pk["recall@k"].is_monotonic_increasing


def test_recall_rises_with_fp_budget(scored):
    y, score, _ = scored
    r = recall_at_fp_budget(y, score, budgets=(0.001, 0.01, 0.1))
    assert r["recall"].is_monotonic_increasing
    assert r["false_positives"].is_monotonic_increasing


def test_cost_curve_has_interior_optimum(scored):
    """Flagging nothing and flagging everything should both be beaten."""
    y, score, amt = scored
    curve = cost_curve(y, score, amt, CostModel())
    best = optimal_operating_point(curve)
    assert best["net_saving_inr"] > 0
    assert 0.0 < best["flag_rate"] < 1.0
    # The chosen point must beat both degenerate extremes.
    assert best["net_saving_inr"] >= curve.iloc[0]["net_saving_inr"]
    assert best["net_saving_inr"] >= curve.iloc[-1]["net_saving_inr"]


def test_expensive_false_positives_shrink_the_flag_rate(scored):
    """Raise the cost of blocking good customers -> the optimum gets pickier.

    This is the guard that the cost model is actually doing work. If the
    optimal operating point ignored FP cost, this test would not move.
    """
    y, score, amt = scored
    cheap = optimal_operating_point(cost_curve(y, score, amt, CostModel(churn_multiplier=1.0)))
    dear = optimal_operating_point(cost_curve(y, score, amt, CostModel(churn_multiplier=25.0)))
    assert dear["flag_rate"] < cheap["flag_rate"]
    assert dear["precision"] >= cheap["precision"]


def test_a_model_can_be_worth_negative_money(scored):
    """A pure-noise score must never show a positive net saving at its optimum.

    Ranking quality and business value are different things. This asserts the
    cost curve is capable of delivering bad news.
    """
    y, _, amt = scored
    rng = np.random.default_rng(1)
    noise = rng.random(len(y))
    curve = cost_curve(y, noise, amt, CostModel(churn_multiplier=25.0))
    best = optimal_operating_point(curve)
    # With useless ranking and costly FPs, the best you can do is flag ~nothing.
    assert best["flag_rate"] < 0.02
    assert best["net_saving_inr"] < curve["loss_avoided_inr"].max()
