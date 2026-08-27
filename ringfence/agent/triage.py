"""Decide whether a chargeback is worth contesting. No LLM involved.

THE CENTRAL IDEA
----------------
There is no public dataset of dispute *outcomes* -- nobody publishes which
chargebacks the merchant won. So this module does NOT pretend to have a learned
win-probability model. Training one on invented labels would be exactly the kind
of dishonesty the rest of this project exists to avoid.

Instead it uses a signal we genuinely have, and inverts it.

Ringfence's detector was trained to answer "is this transaction fraudulent?".
Now condition on the fact that a chargeback has already been filed, and the same
score answers a different and more useful question:

  HIGH fraud score + chargeback  ->  the card really was stolen. The cardholder
                                     is a victim telling the truth. Contesting
                                     is both futile and wrong.

  LOW fraud score + chargeback   ->  the transaction looked entirely legitimate
                                     on every signal we have, yet someone is
                                     disputing it. That is the signature of
                                     FRIENDLY FRAUD -- the buyer did make the
                                     purchase and has forgotten, or is lying.
                                     This is the contestable case.

So the detector doubles as a friendly-fraud classifier at no extra cost, and the
"should we fight this" decision rests on a model that was honestly validated on
real labels, rather than on a fabricated one.

Everything below is arithmetic over stated assumptions. A reviewer who disagrees
with a constant can change it and re-run; they cannot be misled by it.
"""
from __future__ import annotations

from dataclasses import dataclass

from ringfence.agent.models import (
    Dispute,
    EvidenceDocument,
    Recommendation,
    TriageDecision,
)

# --- Assumptions ---------------------------------------------------------
# These are industry-plausible defaults, NOT measured on this dataset. They are
# collected here so the whole decision can be re-derived under different beliefs.

#: Score below which a disputed transaction is treated as likely friendly fraud.
FRIENDLY_FRAUD_SCORE_MAX = 0.30
#: Score above which we consider the card genuinely compromised and never fight.
GENUINE_FRAUD_SCORE_MIN = 0.70

# Win probability is modelled as a saturating curve, not a sum. Evidence has
# strongly diminishing returns: the first proof of delivery moves the needle
# enormously, the fourth supporting document barely at all. An additive model
# produces absurdities -- an early version of this file returned p_win = 0.85
# for a friendly-fraud case and 0.28 against a genuinely stolen card, both far
# outside anything observed in card-not-present representment.
#
#     p = base + (ceiling - base) * lift / (lift + K)
#
#: No-evidence floor and best-case ceiling when the pattern looks like friendly
#: fraud. Reported representment win rates in CNP retail cluster in the 20-40%
#: band, with well-documented cases reaching roughly 60%.
BASE_P_WIN_FRIENDLY = 0.10
CEILING_P_WIN_FRIENDLY = 0.62
#: Against a genuinely compromised card the issuer holds a cardholder
#: statement, and essentially no documentary evidence overcomes it.
BASE_P_WIN_GENUINE = 0.02
CEILING_P_WIN_GENUINE = 0.15
#: Half-saturation constant: the lift at which half the available headroom is
#: captured. 0.5 means ~two strong documents get you most of the way.
EVIDENCE_HALF_SATURATION = 0.5

#: Multiplicative contribution of each evidence slot we can fill.
EVIDENCE_WEIGHTS = {
    "shipping_proof": 0.22,
    "customer_communication": 0.18,
    "proof_of_service": 0.15,
    "access_activity_log": 0.14,
    "billing_proof": 0.10,
    "term_and_conditions": 0.05,
    "refund_cancellation_policy": 0.05,
    "explanation_letter": 0.03,
    "refund_confirmation": 0.20,
    "cancellation_proof": 0.12,
}
STRENGTH_FACTOR = {"strong": 1.0, "moderate": 0.6, "weak": 0.25}

#: Staff time to assemble and file a representment.
CONTEST_EFFORT_COST_INR = 400.0
#: Some networks levy a fee when a representment is filed and lost.
LOST_REPRESENTMENT_FEE_INR = 250.0

#: Below this many days remaining, hand to a human rather than auto-drafting.
URGENT_DAYS = 2

#: Reason codes where documentary evidence is structurally strong or weak.
REASON_CODE_ADJUSTMENT = {
    "4855": +0.10,  # goods/services not provided -> proof of delivery wins
    "13.1": +0.10,  # merchandise not received
    "4837": -0.15,  # no cardholder authorisation -> hardest to rebut
    "10.4": -0.15,  # other fraud, card absent
    "4853": +0.05,  # not as described -> policy + comms help
}


@dataclass
class TriageConfig:
    friendly_fraud_score_max: float = FRIENDLY_FRAUD_SCORE_MAX
    genuine_fraud_score_min: float = GENUINE_FRAUD_SCORE_MIN
    contest_effort_cost_inr: float = CONTEST_EFFORT_COST_INR
    lost_representment_fee_inr: float = LOST_REPRESENTMENT_FEE_INR
    urgent_days: int = URGENT_DAYS


def estimate_p_win(
    dispute: Dispute, evidence: list[EvidenceDocument], cfg: TriageConfig | None = None
) -> tuple[float, list[str]]:
    """Estimate the probability of winning a representment. Returns (p, notes)."""
    cfg = cfg or TriageConfig()
    notes: list[str] = []

    friendly = dispute.fraud_score <= cfg.friendly_fraud_score_max
    genuine = dispute.fraud_score >= cfg.genuine_fraud_score_min

    if genuine:
        base, ceiling = BASE_P_WIN_GENUINE, CEILING_P_WIN_GENUINE
        notes.append(
            f"fraud score {dispute.fraud_score:.2f} indicates a genuinely "
            "compromised card; the cardholder's claim is probably true"
        )
    elif friendly:
        base, ceiling = BASE_P_WIN_FRIENDLY, CEILING_P_WIN_FRIENDLY
        notes.append(
            f"fraud score {dispute.fraud_score:.2f} is low: the transaction "
            "looked legitimate on every signal, which is the friendly-fraud pattern"
        )
    else:
        # Interpolate the endpoints through the ambiguous middle band.
        span = cfg.genuine_fraud_score_min - cfg.friendly_fraud_score_max
        frac = (dispute.fraud_score - cfg.friendly_fraud_score_max) / span
        base = BASE_P_WIN_FRIENDLY + frac * (BASE_P_WIN_GENUINE - BASE_P_WIN_FRIENDLY)
        ceiling = CEILING_P_WIN_FRIENDLY + frac * (
            CEILING_P_WIN_GENUINE - CEILING_P_WIN_FRIENDLY
        )
        notes.append(f"fraud score {dispute.fraud_score:.2f} is ambiguous")

    # Evidence: each filled slot adds, scaled by how well it covers the dispute.
    best_by_slot: dict[str, float] = {}
    for doc in evidence:
        w = EVIDENCE_WEIGHTS.get(doc.slot, 0.0) * STRENGTH_FACTOR[doc.strength]
        best_by_slot[doc.slot] = max(best_by_slot.get(doc.slot, 0.0), w)
    evidence_lift = sum(best_by_slot.values())
    if evidence_lift:
        notes.append(
            f"{len(best_by_slot)} evidence slot(s) fillable, lift +{evidence_lift:.2f}"
        )
    else:
        notes.append("no usable evidence on file")

    adj = REASON_CODE_ADJUSTMENT.get(dispute.reason_code, 0.0)
    if adj:
        notes.append(f"reason code {dispute.reason_code} adjustment {adj:+.2f}")

    # A customer disputing repeatedly is itself an argument to the issuer.
    if dispute.customer_prior_disputes >= 2:
        adj += 0.08
        notes.append(
            f"customer has {dispute.customer_prior_disputes} prior disputes: "
            "a pattern worth putting in front of the issuer"
        )

    # A ring with confirmed frauds behind it makes friendly fraud unlikely.
    if dispute.ring_known_frauds:
        adj -= 0.20
        notes.append(
            f"transaction sits in a ring with {dispute.ring_known_frauds} "
            "confirmed fraud(s); organised abuse, not a confused customer"
        )

    # Saturating combination: adjustments move the lift, never the ceiling.
    lift = max(0.0, evidence_lift + adj)
    saturation = lift / (lift + EVIDENCE_HALF_SATURATION) if lift > 0 else 0.0
    p = base + (ceiling - base) * saturation
    return max(0.0, min(ceiling, p)), notes


def triage(
    dispute: Dispute,
    evidence: list[EvidenceDocument],
    cfg: TriageConfig | None = None,
) -> TriageDecision:
    """Recommend contest / accept / escalate, with the arithmetic shown."""
    cfg = cfg or TriageConfig()
    p_win, notes = estimate_p_win(dispute, evidence, cfg)
    amount = dispute.amount_inr

    # Accepting means refunding: the amount is simply lost.
    ev_accept = -amount
    # Contesting: win and keep it, lose and pay the amount plus a filing fee.
    ev_contest = (
        p_win * 0.0
        - (1 - p_win) * (amount + cfg.lost_representment_fee_inr)
        - cfg.contest_effort_cost_inr
    )

    available = sorted({d.slot for d in evidence})
    missing = [s for s in ("shipping_proof", "customer_communication", "billing_proof")
               if s not in available]

    if dispute.respond_by_days <= cfg.urgent_days:
        rec = Recommendation.ESCALATE
        notes.append(
            f"only {dispute.respond_by_days} day(s) to respond: a human should "
            "decide rather than wait on a drafting queue"
        )
    elif not evidence:
        rec = Recommendation.ACCEPT
        notes.append("nothing to file; contesting with no evidence cannot win")
    elif dispute.fraud_score >= cfg.genuine_fraud_score_min:
        rec = Recommendation.ACCEPT
        notes.append(
            "declining to contest: the card appears genuinely compromised, and "
            "fighting a real victim's claim is both unwinnable and wrong"
        )
    elif ev_contest > ev_accept:
        rec = Recommendation.CONTEST
        notes.append(
            f"contesting is worth Rs {ev_contest - ev_accept:,.0f} more than accepting"
        )
    else:
        rec = Recommendation.ACCEPT
        notes.append(
            f"expected recovery Rs {p_win * amount:,.0f} does not cover the "
            f"Rs {cfg.contest_effort_cost_inr:,.0f} of effort and filing risk"
        )

    return TriageDecision(
        recommendation=rec,
        p_win=round(p_win, 4),
        expected_value_contest_inr=round(ev_contest, 2),
        expected_value_accept_inr=round(ev_accept, 2),
        likely_friendly_fraud=dispute.fraud_score <= cfg.friendly_fraud_score_max,
        rationale=notes,
        evidence_available=available,
        evidence_missing=missing,
    )
