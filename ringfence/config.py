"""Central paths and constants.

Everything that a reviewer might want to change lives here, so the rest of the
codebase never hardcodes a split point or a cost assumption.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"

for _p in (RAW, INTERIM, PROCESSED, REPORTS, FIGURES):
    _p.mkdir(parents=True, exist_ok=True)

# --- Dataset -------------------------------------------------------------
KAGGLE_COMPETITION = "ieee-fraud-detection"
TARGET = "isFraud"
TIME_COL = "TransactionDT"
ID_COL = "TransactionID"

# IEEE-CIS ships a labelled train set and an unlabelled test set (labels were
# held by Kaggle). So our "held-out" set must be carved out of train, by time.
# Fraction of the TIME axis used for training. The remainder is held out.
TRAIN_TIME_FRACTION = 0.80

# A gap between train and test in seconds. Chargebacks are reported with a lag,
# so in production a label for a transaction at time t is not known until much
# later. Training right up to the test boundary leaks that unavailable-in-
# practice recency. 7 days is a conservative, documented choice.
EMBARGO_SECONDS = 7 * 24 * 3600

# --- Entity resolution ---------------------------------------------------
# Columns joined to form a stable "client" fingerprint. IEEE-CIS has no user id;
# the community-standard reconstruction is card + address + a day-zero anchor
# derived from D1 (days since the card began transacting).
CLIENT_KEYS = ["card1", "card2", "card3", "card5", "addr1", "addr2"]

# Attributes that link distinct clients into a ring. Each becomes an edge type.
LINK_ATTRS = ["card1", "addr1", "P_emaildomain", "R_emaildomain", "DeviceInfo"]

# --- Cost model (INR) ----------------------------------------------------
# Documented in ARCHITECTURE.md. These drive threshold selection, so they are
# assumptions a reviewer must be able to see and challenge.
DEFAULT_COSTS = {
    # Cost of letting a fraudulent transaction through: the chargeback. The
    # merchant loses the goods AND the amount AND pays a network fee.
    "chargeback_fee_inr": 1500.0,
    # Fraction of transaction value lost when a fraud charges back.
    "fraud_loss_rate": 1.0,
    # Cost of blocking a legitimate customer: lost margin on this order, plus
    # some probability that the customer never comes back.
    "gross_margin_rate": 0.25,
    "churn_multiplier": 2.0,
    # USD->INR, since IEEE-CIS TransactionAmt is USD.
    "usd_inr": 88.0,
}

RANDOM_SEED = 42
