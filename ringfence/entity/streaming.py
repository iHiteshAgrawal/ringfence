"""Causal ring assignment: a transaction's ring reflects only the past.

WHY THIS EXISTS
---------------
`graph.build_ring_labels` runs connected-components over the whole frame at
once. Its aggregates can still be made causal, but its *membership* cannot:
if transaction B links to A only through a path that runs via C, and C happens
later, then B is placed in A's ring using information that did not exist when B
arrived. Second-order, but it is still leakage, and on a graph problem it is
exactly the kind that quietly inflates results.

The fix is to replay history. Process transactions in time order through a
union-find, adding each transaction's edges as it arrives and recording the
component id it belongs to *at that instant*. Total cost is near-linear thanks
to path compression, so replaying 590k transactions is a matter of seconds
rather than the 180 full graph rebuilds a day-by-day loop would need.

Hub suppression is causal too. `graph.py` decides a value is a hub using its
global client count -- a (mild) use of the future. Here we keep a RUNNING count
of distinct clients per attribute value and stop creating edges once it crosses
the cap. That is what a live system would do: gmail.com looks selective for its
first few sightings and is demoted to a hub as evidence accumulates. Merges
already made are not undone, which is also what a live system would do.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from rich.console import Console

from ringfence import config

console = Console()


class UnionFind:
    """Union-find with path halving, union by size, and a client-count cap.

    `clients[root]` tracks how many CLIENT nodes a component holds, separately
    from `size`, which counts every node including attribute-value nodes.
    """

    __slots__ = ("clients", "parent", "size")

    def __init__(self, n: int, n_clients: int):
        self.parent = np.arange(n, dtype=np.int64)
        self.size = np.ones(n, dtype=np.int64)
        self.clients = np.zeros(n, dtype=np.int64)
        self.clients[:n_clients] = 1

    def find(self, x: int) -> int:
        p = self.parent
        while p[x] != x:
            p[x] = p[p[x]]  # path halving
            x = p[x]
        return int(x)

    def union(self, a: int, b: int, max_clients: int | None = None) -> bool:
        """Merge a and b. Returns False if the merge was refused by the cap."""
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return True
        if max_clients is not None and self.clients[ra] + self.clients[rb] > max_clients:
            return False
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]
        self.clients[ra] += self.clients[rb]
        return True


def assign_rings_causally(
    df: pd.DataFrame,
    link_attrs: list[str] | None = None,
    max_shared_clients: int = 20,
    max_ring_clients: int = 50,
) -> tuple[pd.Series, dict]:
    """Replay transactions in time order, returning each row's ring at arrival.

    Requires `client_id` (from resolve()). Returns (ring_id_per_row, diagnostics).

    `max_ring_clients` bounds component growth. Without it the graph percolates:
    enough individually-legitimate shared attributes chain together until one
    giant component swallows a third of the population, and "ring membership"
    stops meaning anything. Capping is also what operations needs -- a 300-card
    component is not something an analyst can action, so a ring that large is
    useless even when it is real. Merges are refused once the cap is reached;
    existing merges stand.
    """
    link_attrs = [a for a in (link_attrs or config.LINK_ATTRS) if a in df.columns]

    order = np.argsort(df[config.TIME_COL].to_numpy(), kind="mergesort")
    clients = df["client_id"].astype("category")
    client_codes = clients.cat.codes.to_numpy()
    n_clients = len(clients.cat.categories)

    # Node space: [0, n_clients) are clients; each attribute value gets a node
    # appended after that, allocated lazily as values are first seen.
    attr_value_node: dict[tuple[int, str], int] = {}
    # Running set-size of distinct clients per attribute value, for hub demotion.
    attr_client_count: dict[tuple[int, str], set] = {}

    max_nodes = n_clients + sum(
        int(df[a].astype("string").nunique(dropna=True)) for a in link_attrs
    )
    uf = UnionFind(max_nodes, n_clients)
    next_node = n_clients

    attr_arrays = {
        ai: df[a].astype("string").to_numpy()
        for ai, a in enumerate(link_attrs)
    }

    ring_of_row = np.empty(len(df), dtype=np.int64)
    hub_demotions = {a: 0 for a in link_attrs}
    edges_made = {a: 0 for a in link_attrs}
    refused = {a: 0 for a in link_attrs}

    for i in order:
        c = int(client_codes[i])
        for ai, attr in enumerate(link_attrs):
            v = attr_arrays[ai][i]
            if v is None or v is pd.NA or (isinstance(v, float) and np.isnan(v)):
                continue
            key = (ai, v)
            seen = attr_client_count.get(key)
            if seen is None:
                seen = set()
                attr_client_count[key] = seen
            if len(seen) > max_shared_clients:
                continue  # already demoted to a hub; make no further edges
            was_new = c not in seen
            seen.add(c)
            if len(seen) > max_shared_clients:
                hub_demotions[attr] += 1
                continue
            node = attr_value_node.get(key)
            if node is None:
                node = next_node
                attr_value_node[key] = node
                next_node += 1
            if uf.union(c, node, max_clients=max_ring_clients):
                if was_new:
                    edges_made[attr] += 1
            else:
                refused[attr] += 1
        ring_of_row[i] = uf.find(c)

    ring = pd.Series(ring_of_row, index=df.index, name="ring_id")

    # Diagnostics use the FINAL state, purely for reporting.
    final_ring = np.array([uf.find(c) for c in range(n_clients)])
    sizes = pd.Series(final_ring).value_counts()
    multi = sizes[sizes > 1]
    diag = {
        "n_clients": int(n_clients),
        "n_rings_final": len(sizes),
        "n_multi_client_rings": len(multi),
        "clients_in_multi_rings": int(multi.sum()),
        "largest_ring_clients": int(sizes.max()) if len(sizes) else 0,
        "pct_in_largest": float(100 * sizes.max() / n_clients) if len(sizes) else 0.0,
        "hub_demotions": hub_demotions,
        "edges_made": edges_made,
        "merges_refused_by_size_cap": refused,
        "max_shared_clients": max_shared_clients,
        "max_ring_clients": max_ring_clients,
    }
    console.log(
        f"causal rings: {diag['n_multi_client_rings']:,} multi-client rings, "
        f"largest={diag['largest_ring_clients']} ({diag['pct_in_largest']:.2f}% of clients)"
    )
    return ring, diag
