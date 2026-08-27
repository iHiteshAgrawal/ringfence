# Ringfence

**Abuse-ring detection and chargeback dispute drafting for card-not-present fraud.**

Built for the Razorpay AI Buildathon, Track 02 — AI Risk Manager.

> **Defense-only.** This repository detects and responds to fraud. It contains no
> offense-capable code: no synthetic-identity generation, no evasion testing, no
> carding utilities. See [Scope and safety](#scope-and-safety).

## The problem

A merchant does not lose money to fraudsters one at a time. They lose it to
*rings*: one operator running many cards through a handful of shared addresses,
email domains and devices. Score transactions in isolation and you catch the
tail of a ring after it has already cost you. Score the ring and you can stop
the rest of it.

Then the chargebacks arrive anyway for what got through — and contesting them is
manual, deadline-bound work that most merchants simply eat.

Ringfence does both halves:

1. **Detect** — resolve transactions into client entities, link clients into
   rings by shared attributes, and score at the ring level.
2. **Respond** — for disputes that do land, draft a Razorpay
   [contest-a-dispute](https://razorpay.com/docs/api/disputes/contest/) payload
   with the evidence bundle assembled and a written summary.

## Why the numbers here can be believed

Fraud results are easy to fake by accident. Three deliberate choices:

- **Real labels.** [IEEE-CIS](https://www.kaggle.com/competitions/ieee-fraud-detection/data)
  (590,540 real card-not-present transactions, 3.5% fraud rate). Its ground truth
  *is* reported chargebacks — the exact loss class this track is about.
- **Temporal split with an embargo.** Never random. A random split scatters one
  ring across both sides of the boundary and lets the model memorise it. We also
  drop 7 days adjacent to the cutoff, because in production chargeback labels
  arrive with a reporting lag you would not have at scoring time.
- **Cost-weighted metrics, not AUC-ROC.** At 3.5% prevalence AUC-ROC is
  dominated by the majority class and hides false-positive cost. We report
  PR-AUC, precision@k at realistic review capacity, recall at a fixed FP budget,
  and a rupee cost curve that *selects* the threshold.

## Status

Under active development. See [ARCHITECTURE.md](ARCHITECTURE.md).

## Quickstart

```bash
uv venv --python 3.11 && uv pip install -e .
python scripts/download_data.py    # needs a Kaggle token, see the script docstring
python scripts/build_dataset.py
```

## Scope and safety

The Buildathon brief for this track states: *"Strictly defense-only: anything
offense-capable is disqualified."* This project takes that seriously.

- All data is the public IEEE-CIS research dataset. No real cardholder data.
- The model scores risk and explains itself. It does not generate identities,
  probe defences, or test evasion.
- The dispute agent drafts evidence for a *merchant contesting a chargeback* —
  it never fabricates evidence, and it is wired to `action: "draft"`, never
  `"submit"`, so a human always signs off.

## License

MIT
