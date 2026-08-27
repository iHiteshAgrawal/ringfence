"""Link clients into rings, and score the rings.

The central engineering problem here is the *hub*. If you naively join clients
that share an email domain, `gmail.com` links roughly 40% of the dataset into a
single connected component and the graph tells you nothing. Every real-world
entity graph has this failure mode, and handling it is most of the work.

We solve it with a selectivity cap: an attribute value may only create edges if
it is shared by at most `max_shared_clients` clients. `gmail.com` is discarded
as a hub; a specific rare billing address shared by 6 different cards is exactly
the signal we want. The cap is a tunable, and its effect is measured rather
than assumed -- see `sweep_selectivity`.

Graph is built with scipy.sparse.csgraph rather than networkx: at ~500k client
nodes networkx's Python-object-per-node model needs several GB, while the
sparse-matrix representation needs tens of MB.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import connected_components

from ringfence import config


def build_ring_labels(
    df: pd.DataFrame,
    link_attrs: list[str] | None = None,
    max_shared_clients: int = 20,
    min_shared_clients: int = 2,
) -> tuple[pd.Series, dict]:
    """Assign every client to a ring (connected component).

    Returns (ring_id_per_row, diagnostics).

    A client that shares no selective attribute with anyone else becomes its own
    singleton ring. That is the common case and it is correct: most clients are
    not in a ring.
    """
    link_attrs = link_attrs or config.LINK_ATTRS
    clients = df["client_id"].astype("category")
    client_codes = clients.cat.codes.to_numpy()
    n_clients = len(clients.cat.categories)

    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    node_offset = n_clients
    diagnostics: dict = {"attrs": {}, "max_shared_clients": max_shared_clients}

    for attr in link_attrs:
        if attr not in df.columns:
            continue
        vals = df[attr].astype("string")
        mask = vals.notna()
        if not mask.any():
            continue

        # How many DISTINCT clients does each attribute value touch?
        pair = pd.DataFrame(
            {"v": vals[mask].to_numpy(), "c": client_codes[mask.to_numpy()]}
        ).drop_duplicates()
        counts = pair.groupby("v")["c"].size()

        # Keep only selective values: shared by at least 2 clients (otherwise no
        # edge is created anyway) and at most the cap (otherwise it is a hub).
        keep = counts[(counts >= min_shared_clients) & (counts <= max_shared_clients)].index
        sel = pair[pair["v"].isin(set(keep))]

        diagnostics["attrs"][attr] = {
            "distinct_values": int(counts.size),
            "hub_values_dropped": int((counts > max_shared_clients).sum()),
            "linking_values_kept": len(keep),
            "client_attr_edges": len(sel),
        }
        if sel.empty:
            continue

        # Bipartite edge: client node <-> attribute-value node.
        val_codes = pd.Categorical(sel["v"]).codes.astype(np.int64)
        rows.append(sel["c"].to_numpy().astype(np.int64))
        cols.append(val_codes + node_offset)
        node_offset += int(val_codes.max()) + 1

    n_nodes = node_offset
    if rows:
        r = np.concatenate(rows)
        c = np.concatenate(cols)
        data = np.ones(len(r), dtype=np.int8)
        adj = sparse.coo_matrix((data, (r, c)), shape=(n_nodes, n_nodes)).tocsr()
    else:
        adj = sparse.csr_matrix((n_nodes, n_nodes), dtype=np.int8)

    _n_comp, labels = connected_components(adj, directed=False)
    client_ring = labels[:n_clients]
    ring_per_row = pd.Series(client_ring[client_codes], index=df.index, name="ring_id")

    sizes = pd.Series(client_ring).value_counts()
    multi = sizes[sizes > 1]
    diagnostics.update({
        "n_clients": int(n_clients),
        "n_rings": len(sizes),
        "n_multi_client_rings": len(multi),
        "clients_in_multi_rings": int(multi.sum()),
        "largest_ring_clients": int(sizes.max()) if len(sizes) else 0,
    })
    return ring_per_row, diagnostics


def ring_features(df: pd.DataFrame) -> pd.DataFrame:
    """Per-ring aggregates, then broadcast back to rows.

    Every feature here is computable at scoring time from the transaction's own
    ring, using only data at or before that transaction -- with the exception
    noted in ARCHITECTURE.md under "leakage boundary".
    """
    g = df.groupby("ring_id", observed=True)
    feats = pd.DataFrame({
        "ring_n_tx": g.size(),
        "ring_n_clients": g["client_id"].nunique(),
        "ring_total_amt": g["TransactionAmt"].sum(),
        "ring_mean_amt": g["TransactionAmt"].mean(),
        "ring_std_amt": g["TransactionAmt"].std().fillna(0.0),
        "ring_first_dt": g[config.TIME_COL].min(),
        "ring_last_dt": g[config.TIME_COL].max(),
        "ring_n_addr": g["addr1"].nunique(),
        "ring_n_pemail": g["P_emaildomain"].nunique(),
    })
    feats["ring_span_days"] = (feats["ring_last_dt"] - feats["ring_first_dt"]) / 86400.0
    # Velocity: transactions per day across the ring. A ring that fires 40
    # transactions in two days looks nothing like a family sharing an address.
    feats["ring_velocity"] = feats["ring_n_tx"] / feats["ring_span_days"].clip(lower=1.0)
    # Fan-out: how many distinct cards per address. High fan-out on a single
    # address is the canonical bust-out signature.
    feats["ring_cards_per_addr"] = feats["ring_n_clients"] / feats["ring_n_addr"].clip(lower=1)
    # Amount homogeneity: scripted rings often reuse near-identical amounts.
    feats["ring_amt_cv"] = feats["ring_std_amt"] / feats["ring_mean_amt"].clip(lower=0.01)
    feats = feats.drop(columns=["ring_first_dt", "ring_last_dt"])
    return feats


def sweep_selectivity(
    df: pd.DataFrame, caps: list[int], link_attrs: list[str] | None = None
) -> pd.DataFrame:
    """Measure how the cap changes graph shape. Guards against the hub collapse.

    We want the largest ring to stay small relative to the population. If the
    largest component swallows a big fraction of clients, the cap is too loose
    and the graph has degenerated.
    """
    out = []
    for cap in caps:
        _, diag = build_ring_labels(df, link_attrs=link_attrs, max_shared_clients=cap)
        out.append({
            "cap": cap,
            "n_rings": diag["n_rings"],
            "multi_client_rings": diag["n_multi_client_rings"],
            "clients_linked": diag["clients_in_multi_rings"],
            "pct_clients_linked": 100 * diag["clients_in_multi_rings"] / diag["n_clients"],
            "largest_ring": diag["largest_ring_clients"],
            "pct_in_largest": 100 * diag["largest_ring_clients"] / diag["n_clients"],
        })
    return pd.DataFrame(out)
