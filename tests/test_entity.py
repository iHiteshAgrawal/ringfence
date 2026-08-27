"""Ring detection must recover planted rings and must not collapse on hubs."""
from __future__ import annotations

from ringfence.entity.graph import build_ring_labels, ring_features, sweep_selectivity
from ringfence.entity.resolve import resolve


def test_client_id_separates_cards(frame):
    df = resolve(frame)
    # Each planted ring has `ring_size` distinct cards; they must not collapse
    # into one client just because they share an address.
    ring0 = df[df["_ring_truth"] == 0]
    assert ring0["client_id"].nunique() == 6


def test_planted_rings_are_recovered(frame):
    df = resolve(frame)
    ring_id, _diag = build_ring_labels(df, max_shared_clients=20)
    df["ring_id"] = ring_id

    # Every planted ring's members must land in exactly one component together.
    for r in df.loc[df["_ring_truth"] >= 0, "_ring_truth"].unique():
        members = df[df["_ring_truth"] == r]
        assert members["ring_id"].nunique() == 1, f"planted ring {r} was split"

    # Distinct planted rings must not be merged with each other.
    truth_to_ring = df[df["_ring_truth"] >= 0].groupby("_ring_truth")["ring_id"].first()
    assert truth_to_ring.nunique() == len(truth_to_ring), "planted rings merged"


def test_hub_attribute_does_not_collapse_graph(frame):
    """gmail.com touches most clients; it must not fuse everything into one ring."""
    df = resolve(frame)
    _, diag = build_ring_labels(df, max_shared_clients=20)
    # No single component may swallow a large share of clients.
    assert diag["largest_ring_clients"] / diag["n_clients"] < 0.25
    # And the hub must actually have been rejected.
    assert diag["attrs"]["P_emaildomain"]["hub_values_dropped"] >= 1


def test_ring_features_flag_planted_rings(frame):
    df = resolve(frame)
    df["ring_id"] = build_ring_labels(df, max_shared_clients=20)[0]
    feats = ring_features(df)
    df = df.join(feats, on="ring_id")

    planted = df[df["_ring_truth"] >= 0]
    normal = df[df["_ring_truth"] < 0]
    # Planted rings are multi-client and bursty by construction.
    assert planted["ring_n_clients"].mean() > normal["ring_n_clients"].mean()
    assert planted["ring_velocity"].mean() > normal["ring_velocity"].mean()


def test_looser_cap_links_more(frame):
    df = resolve(frame)
    sweep = sweep_selectivity(df, caps=[2, 20, 100000])
    # Monotone: raising the cap can only add edges, never remove them.
    assert sweep["clients_linked"].is_monotonic_increasing
    # And an unbounded cap should visibly degenerate via the gmail.com hub.
    assert sweep.iloc[-1]["pct_in_largest"] > sweep.iloc[0]["pct_in_largest"]
