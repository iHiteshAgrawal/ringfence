"""Honest metrics for a 3.5%-prevalence problem, denominated in rupees.

Why this file exists at all: the standard reflex is to report AUC-ROC. Amazon's
own published Fraud Dataset Benchmark leads with it. At 3.5% prevalence it is
close to uninformative -- the false-positive rate is divided by a huge negative
count, so a model can move tens of thousands of legitimate customers into the
review queue and barely dent the number.

Everything below is chosen so that a bad model looks bad:

  PR-AUC              -- baseline is prevalence (0.035), not 0.5.
  precision@k         -- what an analyst team with finite capacity actually sees.
  recall @ FP budget  -- caps the customer harm, then asks what you caught.
  cost curve          -- converts the confusion matrix into rupees and *picks*
                         the threshold, instead of defaulting to 0.5.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)

from ringfence import config


@dataclass
class CostModel:
    """Rupee consequences of each cell of the confusion matrix.

    All assumptions are explicit and overridable. A reviewer who disagrees with
    a number can change it and re-run; they cannot be misled by it.
    """

    chargeback_fee_inr: float = config.DEFAULT_COSTS["chargeback_fee_inr"]
    fraud_loss_rate: float = config.DEFAULT_COSTS["fraud_loss_rate"]
    gross_margin_rate: float = config.DEFAULT_COSTS["gross_margin_rate"]
    churn_multiplier: float = config.DEFAULT_COSTS["churn_multiplier"]
    usd_inr: float = config.DEFAULT_COSTS["usd_inr"]
    review_cost_inr: float = 0.0  # per flagged transaction, if a human looks

    def false_negative_cost(self, amount_inr: np.ndarray) -> np.ndarray:
        """Fraud we let through: goods gone, amount reversed, network fee paid."""
        return amount_inr * self.fraud_loss_rate + self.chargeback_fee_inr

    def false_positive_cost(self, amount_inr: np.ndarray) -> np.ndarray:
        """Good customer we blocked: lost margin, scaled for the relationship.

        churn_multiplier > 1 encodes that a wrongly declined customer does not
        simply cost you this order's margin -- a meaningful share never returns.
        """
        return amount_inr * self.gross_margin_rate * self.churn_multiplier

    def to_dict(self) -> dict:
        return {
            "chargeback_fee_inr": self.chargeback_fee_inr,
            "fraud_loss_rate": self.fraud_loss_rate,
            "gross_margin_rate": self.gross_margin_rate,
            "churn_multiplier": self.churn_multiplier,
            "usd_inr": self.usd_inr,
            "review_cost_inr": self.review_cost_inr,
        }


def to_inr(amount_usd: np.ndarray, cost: CostModel) -> np.ndarray:
    return np.asarray(amount_usd, dtype=float) * cost.usd_inr


def headline(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    """Threshold-free summary. Includes AUC-ROC only so we can show it misleads."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    prevalence = float(y_true.mean())
    ap = float(average_precision_score(y_true, y_score))
    return {
        "n": len(y_true),
        "prevalence": prevalence,
        "pr_auc": ap,
        "pr_auc_lift_over_baseline": ap / prevalence if prevalence else float("nan"),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
    }


def precision_at_k(y_true: np.ndarray, y_score: np.ndarray, ks=(100, 500, 1000, 5000)) -> pd.DataFrame:
    """What a review team of capacity k actually experiences.

    An analyst can only work a fixed queue per day. This is the metric that
    decides whether the tool gets adopted or muted.
    """
    y_true = np.asarray(y_true)
    order = np.argsort(-np.asarray(y_score))
    yt = y_true[order]
    total_pos = yt.sum()
    rows = []
    for k in ks:
        k = min(k, len(yt))
        hits = int(yt[:k].sum())
        rows.append({
            "k": k,
            "precision@k": hits / k,
            "recall@k": hits / total_pos if total_pos else float("nan"),
            "frauds_caught": hits,
        })
    return pd.DataFrame(rows)


def recall_at_fp_budget(
    y_true: np.ndarray, y_score: np.ndarray, budgets=(0.001, 0.005, 0.01, 0.05)
) -> pd.DataFrame:
    """Fix the customer harm you will tolerate, then report what you caught.

    This is the framing a risk owner actually uses: "I will accept declining
    0.5% of good customers -- how much fraud does that buy me?"
    """
    y_true = np.asarray(y_true).astype(bool)
    y_score = np.asarray(y_score)
    neg_scores = np.sort(y_score[~y_true])[::-1]
    n_neg = len(neg_scores)
    n_pos = int(y_true.sum())
    rows = []
    for b in budgets:
        idx = max(int(np.floor(b * n_neg)) - 1, 0)
        thr = neg_scores[idx] if n_neg else np.inf
        flagged = y_score >= thr
        tp = int((flagged & y_true).sum())
        fp = int((flagged & ~y_true).sum())
        rows.append({
            "fp_budget": b,
            "threshold": float(thr),
            "recall": tp / n_pos if n_pos else float("nan"),
            "precision": tp / (tp + fp) if (tp + fp) else float("nan"),
            "false_positives": fp,
        })
    return pd.DataFrame(rows)


def cost_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    amount_usd: np.ndarray,
    cost: CostModel | None = None,
    n_thresholds: int = 200,
) -> pd.DataFrame:
    """Sweep the threshold and price every operating point in rupees.

    The baseline is "approve everything", which is what a merchant with no model
    does. Net saving is baseline loss minus the loss the model leaves behind.
    A model can have a fine PR-AUC and still be worth negative money if it
    blocks high-value good customers to catch cheap fraud -- this surfaces that.
    """
    cost = cost or CostModel()
    y_true = np.asarray(y_true).astype(bool)
    y_score = np.asarray(y_score)
    amt = to_inr(amount_usd, cost)

    fn_cost_all = cost.false_negative_cost(amt)
    fp_cost_all = cost.false_positive_cost(amt)
    baseline_loss = float(fn_cost_all[y_true].sum())

    qs = np.linspace(0, 1, n_thresholds)
    thresholds = np.unique(np.quantile(y_score, qs))
    rows = []
    for thr in thresholds:
        flagged = y_score >= thr
        tp = flagged & y_true
        fp = flagged & ~y_true
        fn = ~flagged & y_true

        caught = float(fn_cost_all[tp].sum())         # loss avoided
        missed = float(fn_cost_all[fn].sum())         # loss still taken
        friction = float(fp_cost_all[fp].sum())       # harm to good customers
        review = cost.review_cost_inr * int(flagged.sum())
        model_loss = missed + friction + review

        n_tp, n_fp, n_fn = int(tp.sum()), int(fp.sum()), int(fn.sum())
        rows.append({
            "threshold": float(thr),
            "flag_rate": float(flagged.mean()),
            "tp": n_tp, "fp": n_fp, "fn": n_fn,
            "precision": n_tp / (n_tp + n_fp) if (n_tp + n_fp) else float("nan"),
            "recall": n_tp / (n_tp + n_fn) if (n_tp + n_fn) else float("nan"),
            "loss_avoided_inr": caught,
            "fraud_loss_inr": missed,
            "fp_friction_inr": friction,
            "review_cost_inr": review,
            "total_loss_inr": model_loss,
            "net_saving_inr": baseline_loss - model_loss,
        })
    out = pd.DataFrame(rows)
    out.attrs["baseline_loss_inr"] = baseline_loss
    return out


def optimal_operating_point(curve: pd.DataFrame) -> pd.Series:
    """The threshold that maximises rupees saved. This is the one we ship."""
    return curve.loc[curve["net_saving_inr"].idxmax()]


def summary_report(
    y_true: np.ndarray,
    y_score: np.ndarray,
    amount_usd: np.ndarray,
    cost: CostModel | None = None,
) -> dict:
    """Everything a reviewer needs, in one object."""
    cost = cost or CostModel()
    curve = cost_curve(y_true, y_score, amount_usd, cost)
    best = optimal_operating_point(curve)
    return {
        "headline": headline(y_true, y_score),
        "precision_at_k": precision_at_k(y_true, y_score),
        "recall_at_fp_budget": recall_at_fp_budget(y_true, y_score),
        "cost_curve": curve,
        "optimal": best,
        "baseline_loss_inr": curve.attrs["baseline_loss_inr"],
        "cost_model": cost.to_dict(),
    }
