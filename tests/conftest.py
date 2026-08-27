"""A small IEEE-CIS-shaped fixture whose ONLY fraud signal is ring structure.

This fixture is deliberately hard to fit. Earlier versions leaked the label
through the card id range and then through the address range, and the smoke
test happily reported PR-AUC 1.0 at iteration 1 -- proving nothing except that
the model could read a giveaway column.

The design rule now: every scalar column is drawn from the SAME distribution
for fraud and non-fraud. What differs is only relational -- ring members share
a billing address with several other cards and transact in a burst. A model can
only do well here by using the ring features, which is exactly the property the
end-to-end test is meant to check.

Label noise is included so perfect separation is impossible.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ringfence import config

EMAILS = ["gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "anonymous.com"]
DEVICES = [None, "Windows", "iOS Device", "MacOS", "Trident/7.0"]
ADDR_POOL = np.arange(100, 500, dtype=float)   # one shared pool for everyone


def make_frame(
    n_normal: int = 800,
    n_rings: int = 5,
    ring_size: int = 6,
    tx_per_ring_client: int = 4,
    label_noise: float = 0.02,
    seed: int = 0,
) -> pd.DataFrame:
    """Normal traffic plus planted rings, separable only by relational structure."""
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    card_pool = rng.permutation(900_000) + 10_000
    cards = iter(card_pool)

    def new_identity(card: int, addr: float) -> dict:
        """Attributes that belong to a CARD, not to a transaction.

        card2/card3/card5/addr are part of the client fingerprint, so they must
        be constant across a card's transactions. Randomising them per row
        shatters one card into many clients and silently destroys the entity
        resolution the whole project depends on.
        """
        return {
            "card1": int(card),
            "card2": float(rng.integers(100, 600)),
            "card3": 150.0,
            "card5": float(rng.choice([226.0, 224.0, 166.0])),
            "addr1": float(addr),
            "addr2": 87.0,
            "P_emaildomain": str(rng.choice(EMAILS)),
            "R_emaildomain": None,
            "DeviceInfo": rng.choice(DEVICES),
        }

    def base_row(ident: dict, day: int, start_day: int, amt: float) -> dict:
        return {
            **ident,
            # Same amount distribution for everyone: amount is not a tell.
            "TransactionAmt": float(np.round(amt, 2)),
            config.TIME_COL: day * 86400 + int(rng.integers(0, 86400)),
            "D1": float(day - start_day),
        }

    # --- normal traffic: own card, own address, spread over time ----------
    for _ in range(n_normal):
        start_day = int(rng.integers(0, 150))
        ident = new_identity(next(cards), rng.choice(ADDR_POOL))
        for _ in range(int(rng.integers(2, 6))):
            day = start_day + int(rng.integers(0, 60))
            r = base_row(ident, day, start_day, rng.lognormal(4.0, 0.9))
            r["isFraud"] = int(rng.random() < label_noise)
            r["_ring_truth"] = -1
            rows.append(r)

    # --- planted rings: many cards, ONE shared address, tight burst -------
    for ring_idx in range(n_rings):
        shared_addr = rng.choice(ADDR_POOL)   # drawn from the same pool
        start_day = int(rng.integers(0, 150))
        for _ in range(ring_size):
            ident = new_identity(next(cards), shared_addr)
            card_start = start_day - int(rng.integers(0, 3))
            for _ in range(tx_per_ring_client):
                day = start_day + int(rng.integers(0, 3))   # burst
                r = base_row(ident, day, card_start, rng.lognormal(4.0, 0.9))
                r["isFraud"] = int(rng.random() > label_noise)
                r["_ring_truth"] = ring_idx
                rows.append(r)

    df = pd.DataFrame(rows).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    df[config.ID_COL] = np.arange(len(df))
    return df


@pytest.fixture
def frame() -> pd.DataFrame:
    return make_frame()
