"""Calibration must fix probabilities without touching the ranking."""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import average_precision_score

from ringfence.model.train import PlattCalibrator


@pytest.fixture
def raw_scores():
    rng = np.random.default_rng(0)
    n = 20_000
    y = (rng.random(n) < 0.035).astype(int)
    # Over-confident scores, as gradient boosting produces.
    raw = np.clip(rng.beta(1.2, 20, n) + y * rng.beta(2, 6, n), 1e-6, 1 - 1e-6)
    return y, raw


def test_ranking_is_exactly_preserved(raw_scores):
    """The whole reason we use a sigmoid instead of isotonic."""
    y, raw = raw_scores
    cal = PlattCalibrator().fit(raw, y).predict(raw)
    assert np.array_equal(np.argsort(raw), np.argsort(cal))
    assert average_precision_score(y, cal) == pytest.approx(
        average_precision_score(y, raw), rel=1e-9
    )


def test_no_score_collapse(raw_scores):
    """Isotonic crushed 7,769 distinct scores to 48. A sigmoid must not."""
    y, raw = raw_scores
    cal = PlattCalibrator().fit(raw, y).predict(raw)
    assert len(np.unique(cal)) >= 0.99 * len(np.unique(raw))
    assert (cal >= 1.0).sum() == 0, "no score may saturate at exactly 1.0"


def test_probabilities_move_toward_the_base_rate(raw_scores):
    """Calibration should make the mean predicted probability match reality."""
    y, raw = raw_scores
    cal = PlattCalibrator().fit(raw, y).predict(raw)
    assert abs(cal.mean() - y.mean()) < abs(raw.mean() - y.mean())


def test_inverted_model_is_refused():
    """A worse-than-chance model must raise, not ship an inverted score."""
    rng = np.random.default_rng(1)
    y = (rng.random(5000) < 0.2).astype(int)
    inverted = 1.0 - (y * 0.6 + rng.random(5000) * 0.3)
    with pytest.raises(ValueError, match="worse than chance"):
        PlattCalibrator().fit(inverted, y)
