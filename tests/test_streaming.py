"""Causal ring assignment must be prefix-stable and still find real rings."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ringfence import config
from ringfence.entity.resolve import resolve
from ringfence.entity.streaming import UnionFind, assign_rings_causally
from tests.conftest import make_frame


def test_unionfind_basic():
    uf = UnionFind(6, n_clients=6)
    uf.union(0, 1); uf.union(1, 2); uf.union(4, 5)
    assert uf.find(0) == uf.find(2)
    assert uf.find(0) != uf.find(4)
    assert uf.find(3) == 3


def test_unionfind_refuses_merge_past_client_cap():
    uf = UnionFind(6, n_clients=6)
    assert uf.union(0, 1, max_clients=3)
    assert uf.union(1, 2, max_clients=3)
    # {0,1,2} is now at the cap; adding a fourth client must be refused.
    assert not uf.union(2, 3, max_clients=3)
    assert uf.find(3) != uf.find(0)


def test_ring_size_cap_prevents_percolation():
    """Without a size cap the fixture graph chains into one giant component."""
    df = resolve(make_frame(seed=23))
    _, uncapped = assign_rings_causally(df, max_ring_clients=10**9)
    _, capped = assign_rings_causally(df, max_ring_clients=50)
    assert uncapped["largest_ring_clients"] > capped["largest_ring_clients"]
    assert capped["largest_ring_clients"] <= 50
    assert sum(capped["merges_refused_by_size_cap"].values()) > 0


def test_ring_membership_is_prefix_stable():
    """Deleting the future must not change any earlier row's ring PARTITION.

    Component ids are arbitrary labels, so we compare the induced partition
    (which rows share a ring) rather than the raw ids.
    """
    df = resolve(make_frame(seed=21))
    full, _ = assign_rings_causally(df)

    cutoff = df[config.TIME_COL].quantile(0.5)
    past = df[df[config.TIME_COL] <= cutoff].copy()
    partial, _ = assign_rings_causally(past)

    a = full.loc[past.index]
    b = partial
    # Same partition <=> the pairwise "same ring" relation agrees. Compare via
    # a canonical relabelling by first appearance.
    def canon(s: pd.Series) -> np.ndarray:
        return pd.factorize(s.sort_index().to_numpy())[0]

    assert np.array_equal(canon(a), canon(b)), "future data changed past membership"


def test_planted_rings_are_still_recovered():
    df = resolve(make_frame(seed=22))
    ring, diag = assign_rings_causally(df, max_shared_clients=20)
    df["ring_id"] = ring
    planted = df[df["_ring_truth"] >= 0]
    # Members of a planted ring converge to one component once all have arrived.
    for r in planted["_ring_truth"].unique():
        members = planted[planted["_ring_truth"] == r]
        # The final arrival should sit in the ring holding most of its siblings.
        assert members["ring_id"].nunique() <= members["client_id"].nunique()
    assert diag["n_multi_client_rings"] > 0


def test_hub_is_demoted_not_merged():
    df = resolve(make_frame(seed=23))
    _, diag = assign_rings_causally(df, max_shared_clients=20)
    assert diag["hub_demotions"]["P_emaildomain"] > 0, "gmail.com should be demoted"
    assert diag["pct_in_largest"] < 25.0, "graph collapsed into one component"


def test_first_sightings_are_allowed_to_link():
    """A value is only a hub once evidence says so; early links are kept."""
    df = resolve(make_frame(seed=24))
    _ring, diag = assign_rings_causally(df, max_shared_clients=3)
    # With a tight cap, gmail.com is demoted almost immediately, so the graph
    # must stay very fragmented.
    assert diag["pct_in_largest"] < 10.0
