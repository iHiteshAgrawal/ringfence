"""LightGBM detector with time-aware validation and probability calibration.

Two choices worth defending:

1. NO resampling and NO scale_pos_weight. The usual reflex at 3.5% prevalence
   is to rebalance, but that distorts the predicted probabilities -- and this
   system spends its probabilities in a rupee cost model, where a score of 0.9
   must actually mean "90% likely fraud". We keep the natural prior and move
   the DECISION THRESHOLD instead, which is the statistically correct knob.

2. Isotonic calibration on a held-out time slice. Gradient boosting is
   systematically over-confident. Uncalibrated scores make the cost curve pick
   the wrong threshold, so calibration is not cosmetic here -- it changes the
   money.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from rich.console import Console
from sklearn.calibration import IsotonicRegression
from sklearn.metrics import average_precision_score

from ringfence import config

console = Console()

DEFAULT_PARAMS: dict = {
    "objective": "binary",
    "metric": "average_precision",
    "learning_rate": 0.05,
    "num_leaves": 96,
    "min_data_in_leaf": 100,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 5.0,
    "max_cat_threshold": 64,
    "cat_smooth": 20.0,
    "verbosity": -1,
    "seed": config.RANDOM_SEED,
    "num_threads": 0,
}


@dataclass
class TrainedModel:
    booster: lgb.Booster
    calibrator: IsotonicRegression | None
    feature_names: list[str]
    best_iteration: int
    valid_pr_auc: float
    params: dict = field(default_factory=dict)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        raw = self.booster.predict(X, num_iteration=self.best_iteration)
        if self.calibrator is not None:
            return self.calibrator.predict(raw)
        return raw

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.booster.save_model(str(path), num_iteration=self.best_iteration)
        console.log(f"saved booster -> {path}")


def time_aware_valid_split(
    X: pd.DataFrame, y: pd.Series, t: pd.Series, valid_fraction: float = 0.2
):
    """Carve the validation set off the END of training time, never at random.

    Early stopping on a random validation split is the second-most-common way
    fraud results go wrong: the model gets to peek at the same period it is
    being scored on. Validation must sit strictly after training, like test.
    """
    cutoff = t.quantile(1 - valid_fraction)
    tr, va = t <= cutoff, t > cutoff
    console.log(
        f"valid split @ {cutoff:,.0f}: train={tr.sum():,} valid={va.sum():,} "
        f"(fraud {y[tr].mean():.3%} / {y[va].mean():.3%})"
    )
    return X[tr], y[tr], X[va], y[va]


def train(
    X: pd.DataFrame,
    y: pd.Series,
    t: pd.Series,
    params: dict | None = None,
    num_boost_round: int = 3000,
    early_stopping_rounds: int = 100,
    calibrate: bool = True,
) -> TrainedModel:
    params = {**DEFAULT_PARAMS, **(params or {})}
    X_tr, y_tr, X_va, y_va = time_aware_valid_split(X, y, t)

    cat_cols = [c for c in X.columns if isinstance(X[c].dtype, pd.CategoricalDtype)]
    dtrain = lgb.Dataset(X_tr, y_tr, categorical_feature=cat_cols, free_raw_data=False)
    dvalid = lgb.Dataset(X_va, y_va, reference=dtrain, free_raw_data=False)

    console.log(f"training on {X_tr.shape[0]:,} x {X_tr.shape[1]} ({len(cat_cols)} categorical)")
    booster = lgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        valid_sets=[dvalid],
        valid_names=["valid"],
        callbacks=[
            lgb.early_stopping(early_stopping_rounds, verbose=False),
            lgb.log_evaluation(period=200),
        ],
    )

    raw_va = booster.predict(X_va, num_iteration=booster.best_iteration)
    pr_auc = float(average_precision_score(y_va, raw_va))
    console.log(f"valid PR-AUC {pr_auc:.4f} @ iter {booster.best_iteration}")

    calibrator = None
    if calibrate:
        calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        calibrator.fit(raw_va, y_va)
        console.log("fitted isotonic calibrator on the validation slice")

    return TrainedModel(
        booster=booster,
        calibrator=calibrator,
        feature_names=list(X.columns),
        best_iteration=int(booster.best_iteration),
        valid_pr_auc=pr_auc,
        params=params,
    )


def feature_importance(model: TrainedModel, top: int = 40) -> pd.DataFrame:
    """Gain-based importance. Used to show ring features are doing real work."""
    gain = model.booster.feature_importance(importance_type="gain", iteration=model.best_iteration)
    split = model.booster.feature_importance(importance_type="split", iteration=model.best_iteration)
    df = pd.DataFrame({
        "feature": model.booster.feature_name(),
        "gain": gain,
        "split": split,
    })
    df["gain_pct"] = 100 * df["gain"] / df["gain"].sum()
    df["is_ring_feature"] = df["feature"].str.startswith(("ring_", "client_", "entity_"))
    return df.sort_values("gain", ascending=False).head(top).reset_index(drop=True)
