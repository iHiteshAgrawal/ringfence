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

## Results

On **118,108 held-out transactions**, from a period strictly later than all training data:

| | With rings | Without | Change |
|---|---:|---:|---:|
| PR-AUC | **0.5986** | 0.5027 | +19.1% |
| Recall @ 0.5% FP budget | 0.456 | 0.362 | +26.0% |
| precision@100 | 0.980 | 1.000 | −2.0% |
| Money saved | **₹18,476,797** | ₹12,835,072 | **+44.0%** |

Against a ₹59,770,222 baseline loss, the cost-optimal setting recovers **30.9%**
while flagging 2.51% of traffic. The ring layer is worth ₹5.6M of that.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full design, the leakage
boundary, and an honest limitations section.

## Quickstart

```bash
uv venv --python 3.11 && uv pip install -e .
python scripts/download_data.py                  # needs Kaggle credentials
python scripts/prepare_data.py  --tag main
python scripts/run_experiment.py --tag main
pytest                                           # 56 tests
```

The agent layer runs against real rings from the held-out set:

```bash
python scripts/demo_agent.py --offline           # no API key needed
python scripts/demo_agent.py                     # needs OPENROUTER_API_KEY
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
