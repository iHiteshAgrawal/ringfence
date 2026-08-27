# Ringfence — Architecture

**Razorpay AI Buildathon, Track 02 (AI Risk Manager).**
Abuse-ring detection for card-not-present fraud, plus a chargeback dispute
drafter. Defense-only.

---

## 1. The thesis

Fraudsters do not operate one card at a time. A batch of stolen card numbers
gets run systematically: a short window, a handful of delivery addresses, one
device, similar order sizes. Scored individually, each transaction is
unremarkable. Scored as a group, the operator is obvious.

Ringfence therefore scores **the group**. Catching one member tells you
something actionable about the rest, *before* they cost anything — where
row-by-row scoring only ever catches the tail of a ring.

The second half addresses what gets through: when a chargeback lands anyway,
an agent drafts the evidence bundle to contest it through Razorpay's
[dispute API](https://razorpay.com/docs/api/disputes/contest/).

---

## 2. Data

[IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection/data)
— 590,540 real card-not-present e-commerce transactions, 394 features,
**3.5% fraud rate**.

Chosen for one reason above all: **its labels are reported chargebacks.** The
official labelling logic marks a transaction fraudulent when a chargeback is
reported on the card, and propagates that to later transactions sharing a user
account, email or billing address. So the ground truth is the exact loss class
this track names — and its construction is itself ring-shaped, which
independently supports the modelling approach.

Synthetic alternatives (PaySim, Sparkov) were rejected: measuring precision and
recall against invented labels answers nothing.

### Splits

| Split | Rows | Fraud rate |
|---|---:|---:|
| Train | 453,779 | 3.52% |
| *Embargo (discarded)* | *18,653* | — |
| Held-out test | 118,108 | 3.44% |

Kaggle's own test set is unlabelled, so the held-out set is carved from train
**by time** at the 80th percentile of `TransactionDT`.

---

## 3. The leakage boundary

This is the core of the design. Graph features on fraud data are the easiest
place in applied ML to cheat by accident, and three separate mechanisms had to
be closed. Each has a test that fails if it regresses.

### 3.1 Temporal split, never random

A random split scatters members of one ring across both sides of the boundary.
The model memorises a ring in training and is rewarded for "recognising" it at
test time. Splitting by time reproduces deployment: train on the past, score
the future.

### 3.2 Causal ring membership

Aggregates being past-only is not sufficient. If connected components are
computed over the whole frame, a transaction's *membership* can be decided by a
transaction that had not happened yet — B joins A's ring only via a path through
a later C.

Fixed by replaying history through a **union-find** in time order
([`entity/streaming.py`](ringfence/entity/streaming.py)): each transaction adds
its edges on arrival and records the component it belongs to *at that instant*.
Near-linear via path halving, so a full replay is seconds rather than the 180
graph rebuilds a day-by-day loop would need.

Hub suppression is causal too. Deciding `gmail.com` is a hub from its *global*
count is a mild use of the future; instead a running count of distinct clients
per attribute value is kept, and edge creation stops once it crosses the cap.
A value looks selective for its first few sightings and is demoted as evidence
accumulates — exactly what a live system would do.

> **Test:** `test_ring_membership_is_prefix_stable` — deleting all future data
> must not change the ring partition of any earlier row.

### 3.3 Label reporting lag

`ring_known_prior_frauds` — how many confirmed frauds a ring already has — is
enormously predictive, and the single easiest place to cheat. A chargeback is
**not known when it happens**; the issuer reports it weeks later.

So only frauds that would already have been *reported* are counted, gated by
`label_lag_days` (default **30**, a conservative floor). Set it to 0 and the
metrics improve; that improvement is fiction, and the parameter makes it
measurable instead of invisible.

> **Test:** `test_label_lag_gates_fraud_history`, and
> `test_truncating_the_future_changes_nothing` for the aggregates generally.

### 3.4 Excluded features

`TransactionDT` and `ring_first_seen` are dropped. Both are absolute times that
trend, and since the test period is strictly later, keeping them lets the model
learn *"this is the late period"* rather than anything about fraud.

---

## 4. Pipeline

### 4.1 Entity resolution — [`entity/resolve.py`](ringfence/entity/resolve.py)

IEEE-CIS ships no user id. Identity is reconstructed from
`card1/card2/card3/card5 + addr1/addr2`, plus a **card anchor**.

`D1` is documented as *days since the card began transacting*, so
`day_of_transaction − D1` is the card's **start day** — constant across every
transaction that card ever makes. Two rows agreeing on both fingerprint and
start day are almost certainly the same physical card; agreeing on the
fingerprint alone is a collision. This upgrades a lossy fingerprint into a
usable `client_id`.

Result: **222,477 distinct clients** across 590,540 transactions.

### 4.2 Ring assignment — [`entity/streaming.py`](ringfence/entity/streaming.py)

Clients are linked by shared `card1`, `addr1`, `P_emaildomain`,
`R_emaildomain`, `DeviceInfo`, subject to two caps:

- **`max_shared_clients = 20`** (selectivity). A value shared by more than 20
  clients is a hub and stops creating edges. Demotions observed: `card1` 1,298,
  `DeviceInfo` 232, `addr1` 76, `P_emaildomain` 57.
- **`max_ring_clients = 50`** (size). Without it the graph *percolates*: enough
  individually-legitimate shared attributes chain together until one component
  swallows a third of the population and membership stops meaning anything.
  It is also the operational bound — a 300-card component is not something an
  analyst can action, so a ring that large is useless even when real.
  14,142 merges refused on `card1` alone.

Result on the full dataset: **7,865 multi-client rings**, containing
**61,729 clients** (27.7% of all clients); largest ring 50 clients (at cap).

### 4.3 Causal features — [`features/causal.py`](ringfence/features/causal.py)

For each transaction, computed over its ring's **strictly earlier**
transactions: prior transaction count, prior amount sum/mean/CV, prior distinct
clients and addresses, ring age, seconds since previous, velocity,
cards-per-address, and the lag-gated fraud history.

### 4.4 Feature selection — [`features/select.py`](ringfence/features/select.py)

IEEE-CIS's 339 anonymised `V` columns are engineered variants over the same
underlying quantities. They are grouped by missingness pattern (Vesta's blocks
share a NaN signature), then correlation-clustered within each block keeping one
representative — the column with the most distinct values.

**339 → 137 kept.** Fitted on train only, so survivor choice cannot see the test
period. Total feature count 446 → 244.

### 4.5 Model — [`model/train.py`](ringfence/model/train.py)

LightGBM, binary objective, `average_precision` metric, early stopping on a
**time-aware** validation split (last 20% of the training period — validating on
a random slice is the second-most-common way fraud results go wrong).

Two deliberate choices:

- **No resampling, no `scale_pos_weight`.** The reflex at 3.5% prevalence is to
  rebalance, but that distorts predicted probabilities — and this system
  *spends* its probabilities in a rupee cost model, where 0.9 must mean 90%.
  The natural prior is kept and the **decision threshold** is moved instead,
  which is the statistically correct knob.
- **Sigmoid (Platt) calibration, not isotonic.** Isotonic was tried first and
  was wrong. Fitted on a validation slice with few positives it collapses a
  continuous score into a step function: measured here, **7,769 distinct scores
  became 48**, hundreds tied at exactly 1.0. Ties are fatal for precision@k
  (which orders arbitrarily inside a tie) and **PR-AUC fell 0.250 → 0.143, a
  43% loss, purely from calibration.** A sigmoid is strictly monotonic, so
  ranking is preserved exactly while probabilities are still corrected.

> **Test:** `test_ranking_is_exactly_preserved` asserts
> `argsort(raw) == argsort(calibrated)`; the trainer re-asserts it on real data
> every run, and refuses a negative calibration slope rather than shipping an
> inverted score.

---

## 5. Evaluation — [`eval/metrics.py`](ringfence/eval/metrics.py)

### 5.1 Why not AUC-ROC

At 3.4% prevalence AUC-ROC is dominated by the negative class: the
false-positive rate is divided by an enormous denominator, so a model can move
tens of thousands of legitimate customers into review and barely dent it.
Amazon's own [Fraud Dataset Benchmark](https://github.com/amazon-science/fraud-dataset-benchmark)
still leads with it.

This model scores **ROC-AUC 0.9307** and **PR-AUC 0.5986** on identical
predictions. The gap is the argument.

### 5.2 What is reported instead

- **PR-AUC**, judged against prevalence (0.0344) rather than 0.5.
- **precision@k** — what a review team of finite capacity actually experiences.
- **recall at a fixed false-positive budget** — the framing a risk owner uses:
  *"I will accept declining 0.5% of good customers; what does that buy me?"*
- **A rupee cost curve** that selects the threshold.

### 5.3 Cost model

All assumptions are explicit in [`config.py`](ringfence/config.py) and
overridable, so a reviewer who disagrees can change a number and re-run.

| Assumption | Value |
|---|---|
| Chargeback penalty fee | ₹1,500 |
| Fraud loss rate | 100% of transaction value |
| Gross margin (for FP cost) | 25% |
| Churn multiplier on a blocked good customer | 2.0× |
| USD→INR (IEEE-CIS amounts are USD) | 88 |

- **False negative** = `amount × loss_rate + chargeback_fee`
- **False positive** = `amount × margin × churn_multiplier`

The baseline is "approve everything" — what a merchant with no model does. The
curve sweeps 200 thresholds and reports net saving against that baseline.

> **Test:** `test_a_model_can_be_worth_negative_money` asserts a pure-noise
> score never shows a positive net saving at its optimum. An instrument that can
> only report success is not measuring anything.

---

## 6. Results

Held out: **118,108 transactions**, 3.441% fraud, strictly later than all
training data.

| Metric | Value |
|---|---:|
| PR-AUC | **0.5986** (17.4× the 0.0344 baseline) |
| ROC-AUC | 0.9307 |
| precision@100 | 0.980 |
| precision@1,000 | 0.905 |
| precision@5,000 | 0.494 |

**Recall at a fixed false-positive budget**

| FP budget | Threshold | Recall | Precision | False positives |
|---:|---:|---:|---:|---:|
| 0.1% | 0.869 | 0.261 | 0.903 | 114 |
| 0.5% | 0.525 | 0.456 | 0.765 | 570 |
| 1.0% | 0.348 | 0.524 | 0.651 | 1,140 |
| 5.0% | 0.090 | 0.711 | 0.336 | 5,702 |

**Cost-optimal operating point** — threshold 0.400, flagging 2.51% of traffic:
precision 0.693, recall 0.506 (2,056 TP / 912 FP / 2,008 FN).
**₹18,476,797 saved against a ₹59,770,222 baseline loss — 30.9% recovered.**

### 6.1 Ablation: does the ring layer earn its place?

Identical pipeline, every ring/entity feature removed.

| Measure | With rings | Without | Change |
|---|---:|---:|---:|
| PR-AUC | 0.5986 | 0.5027 | **+19.1%** |
| Recall @ 0.5% FP | 0.456 | 0.362 | **+26.0%** |
| precision@1,000 | 0.905 | 0.881 | +2.7% |
| precision@100 | 0.980 | 1.000 | **−2.0%** |
| Net saving | ₹18,476,797 | ₹12,835,072 | **+44.0%** |

The ring layer is worth **₹5,641,725 more saved, a 44% improvement**. It is
reported alongside the one measure where it does slightly worse: at the very top
of the ranking both models are saturated, and the ring model gives up two
transactions in the top hundred.

The ablation model also needed **735 boosting iterations to the ring model's
233** — without ring structure it works considerably harder for a worse result.

---

## 7. Honest limitations

1. **Ring features are 7.9% of model gain**, not a majority. IEEE-CIS's `C*` and
   `D*` columns already encode substantial velocity and count signal. The
   ablation shows the ring layer adds real and large value; it does not
   dominate, and this document does not claim it does.
2. **`ring_known_fraud_rate` is the 6th most important feature overall** (3.76%
   gain). It is the lag-gated one — which is precisely why gating it honestly
   matters. Ungated it would score better and mean less.
3. **27.7% of clients fall in a multi-client ring.** For the remaining 72%
   Ringfence contributes little beyond conventional features. This is a
   detector for organised abuse, not for lone-actor fraud.
4. **The cost model is parameterised, not measured.** Chargeback fees and churn
   multipliers vary by merchant category. The numbers are defaults; the
   *method* is the contribution, and the framework accepts a merchant's own.
5. **IEEE-CIS is US e-commerce, ~2018.** Indian COD/RTO dynamics differ.
   Amounts are converted at a flat rate rather than modelled.
6. **`max_ring_clients` is a bound on ambition, not a discovery.** Rings larger
   than 50 clients get fragmented. This is defensible operationally but means
   very large organised networks are seen only in pieces.

---

## 8. Reproducing

```bash
uv venv --python 3.11 && uv pip install -e .
python scripts/download_data.py     # needs Kaggle credentials
python scripts/prepare_data.py  --tag main
python scripts/run_experiment.py --tag main
python scripts/run_experiment.py --tag noring --data-tag main --no-ring-features
python scripts/demo_agent.py --offline   # agent layer, no API key needed
python scripts/demo_agent.py             # needs OPENROUTER_API_KEY in .env
pytest                                   # 56 tests
```

Preparation and training are separate processes deliberately: run together they
peak above 8GB and get OOM-killed mid-training with no traceback. Split, the
full pipeline trains in **32 seconds**.

Results land in `reports/results_<tag>.json`.

---

## 9. The dispute agent

For chargebacks that land anyway. Two components, both drafting only.

### 9.1 Triage — [`agent/triage.py`](ringfence/agent/triage.py)

**No LLM, and no fabricated win model.** There is no public dataset of dispute
*outcomes*, so training a win-probability model would mean inventing labels —
exactly what the rest of this project exists to avoid.

Instead the detector's own score is inverted. Conditioned on a chargeback
having been filed, it answers a different question:

| Fraud score | Reading | Action |
|---|---|---|
| **High** | The card really was stolen; the cardholder is a victim telling the truth | **Accept.** Unwinnable, and fighting a real victim's claim is wrong |
| **Low** | The transaction looked legitimate on every signal, yet is being disputed — the friendly-fraud signature | **Contest** |

So a model validated on real labels does the work, and the ethical constraint
falls out of the same signal: `test_genuine_fraud_is_never_contested` asserts
the system will not fight a genuine victim.

Win probability is a **saturating** curve, not a sum:

```
p = base + (ceiling - base) · lift / (lift + K)
```

An earlier additive version returned `p_win = 0.85` for friendly fraud and
`0.28` against a genuinely stolen card — both far outside anything observed in
card-not-present representment. Bounds are now 0.10–0.62 (friendly) and
0.02–0.15 (genuine), matching the 20–40% band commonly reported.
`test_win_probability_stays_inside_plausible_bounds` guards the regression.

The contest/accept decision is then plain expected value against stated
constants (₹400 staff effort, ₹250 lost-representment fee), all in
[`triage.py`](ringfence/agent/triage.py) and overridable.

### 9.2 Model selection — [`agent/llm.py`](ringfence/agent/llm.py)

Both agents call **OpenRouter**, so the model behind either is an environment
variable rather than a code change.

Selected by surveying the catalogue (417 models, 332 supporting JSON-schema
structured output) against three requirements: structured output, reasoning
support, and price at the volume each job actually runs.

**`google/gemini-3.7-flash`** ($0.38 / $1.88 per Mtok) for both. It is the
newest Gemini generation on the platform *and* priced below the older
3.6-flash ($0.75 / $3.75) and 3.5-flash ($1.50 / $9.00) — newer and cheaper, so
there is no tradeoff to weigh.

Considered and rejected for the drafter: the pro tier
(`gemini-3.1-pro-preview`, $2.00 / $12.00). It is an older generation *and* a
preview, and stability matters more than tier for the component whose output
reaches a bank.

**Measured cost:** one case file is 818 input / 635 output tokens =
**₹0.13**. A thousand case files cost ₹132. These are reasoning models, so
output tokens run well above what the prompt length suggests — budget on
measured usage, not the nominal rate.

`RINGFENCE_DRAFTER_MODEL` and `RINGFENCE_CASEFILE_MODEL` override either.
`google/gemini-2.5-flash-lite` ($0.10 / $0.40) is a reasonable downgrade for
case files if ring volume grows.

### 9.3 Drafting — [`agent/drafter.py`](ringfence/agent/drafter.py)

The model does what arithmetic cannot: read the merchant's document inventory,
judge which document answers *this* reason code, and write the ≤1000-character
issuer summary.

**Safety property.** Every document id in the payload must exist in the
inventory. A hallucinated id is not a formatting bug — it is a fabricated
evidence claim sent to a bank. The prompt asks for this; `_validate_provenance`
then *enforces* it, because a prompt is not a guarantee. Unrecognised ids are
dropped and surfaced loudly in the operator narrative.
`test_hallucinated_document_ids_are_rejected` is the security test for this
component.

`action` is a `Literal["draft"]` — submitting is not reachable from this code
path, and constructing a payload with `action="submit"` fails validation.

### 9.4 Analyst case files — [`agent/casefile.py`](ringfence/agent/casefile.py)

A ranked list of transaction ids is not a product; the reviewer still has to
reconstruct what happened. The case file states what links the ring, what the
model reacted to, a graded action (allow / step_up / hold / block), and —
mandatory — the **most plausible innocent explanation**. Shared addresses have
legitimate causes: families, shared housing, offices, package forwarding. An
analyst who only ever reads the prosecution case stops thinking.

Run against the real held-out set, the top-ranked ring is:

> **Ring 76963** — 4 cards, 5 transactions, ₹220,000, over 9 hours.
> Amount CV **0.000** (near-identical amounts). Linked by `DeviceInfo`
> `Z965 Build/NMF26V`, shared by all 4 cards; `P_emaildomain` shared by 3.
> Peak model score 0.998.

One Android device, four cards, identical amounts, one evening. Nothing in that
example is invented — `scripts/demo_agent.py` reproduces it.

The generated case file graded this **hold**, not block — correct, because the
ring has no confirmed prior fraud — at medium confidence, and offered as the
innocent reading: *"an individual or authorized buyer making split payments or
settling multiple identical fixed invoices on behalf of family members or
colleagues from a single shared mobile device."* That is a real possibility, and
an analyst who never sees it stops thinking.

## 10. Scope and safety

The track brief: *"Strictly defense-only: anything offense-capable is
disqualified."*

- Public research data only; no live cardholder data.
- The system scores risk and explains its reasoning. It does not generate
  identities, probe defences, or test evasion.
- No component ranks or optimises for evading a fraud control.
- The dispute agent contests chargebacks on a merchant's behalf using genuine
  documents, and drafts only.
