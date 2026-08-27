"""Prune the redundant V-block. Fitted on training data only.

IEEE-CIS ships 339 anonymised `V` columns. Vesta generated them in blocks of
engineered variants over the same underlying quantities, so within a block they
are near-duplicates: the same signal counted slightly differently. Keeping all
339 costs memory and training time, and buys almost nothing -- a boosted tree
splits on one member of a correlated group and the rest sit unused.

The reduction below is the standard one for this dataset, made explicit:

  1. Group V columns by their missingness pattern. Vesta's blocks share a NaN
     structure, so identical null counts is a reliable block signature.
  2. Inside each block, cluster by correlation and keep ONE representative per
     cluster -- the column with the most distinct values, which retains the most
     resolution.

Fitted on train only, so the choice of survivors cannot see the test period.
"""
from __future__ import annotations

import pandas as pd
from rich.console import Console

console = Console()


class VBlockReducer:
    """Selects a representative subset of the V columns."""

    def __init__(self, corr_threshold: float = 0.75, sample_rows: int = 100_000):
        self.corr_threshold = corr_threshold
        self.sample_rows = sample_rows
        self.keep_: list[str] = []
        self.dropped_: list[str] = []

    def fit(self, train: pd.DataFrame) -> VBlockReducer:
        v_cols = [c for c in train.columns if c.startswith("V") and c[1:].isdigit()]
        if not v_cols:
            return self
        sample = train[v_cols]
        if len(sample) > self.sample_rows:
            sample = sample.iloc[: self.sample_rows]

        # Block signature: how many nulls the column has.
        nulls = sample.isna().sum()
        blocks: dict[int, list[str]] = {}
        for col, n in nulls.items():
            blocks.setdefault(int(n), []).append(col)

        keep: list[str] = []
        for _, cols in sorted(blocks.items()):
            if len(cols) == 1:
                keep.append(cols[0])
                continue
            block = sample[cols].astype("float32")
            corr = block.corr().abs()
            # Greedy: walk columns by descending resolution, keep a column only
            # if it is not already well represented by one we kept.
            resolution = block.nunique().sort_values(ascending=False)
            chosen: list[str] = []
            for col in resolution.index:
                if all(corr.loc[col, c] < self.corr_threshold for c in chosen):
                    chosen.append(col)
            keep.extend(chosen)

        self.keep_ = sorted(keep, key=lambda c: int(c[1:]))
        self.dropped_ = [c for c in v_cols if c not in set(self.keep_)]
        console.log(
            f"V-block reduction: kept {len(self.keep_)} of {len(v_cols)} "
            f"({len(blocks)} blocks, corr>{self.corr_threshold} pruned)"
        )
        return self

    def columns_to_drop(self) -> list[str]:
        return list(self.dropped_)
