"""The contest/accept decision must be economically coherent and ethically safe."""
from __future__ import annotations

from ringfence.agent.models import Dispute, EvidenceDocument, Recommendation
from ringfence.agent.triage import (
    CEILING_P_WIN_FRIENDLY,
    CEILING_P_WIN_GENUINE,
    estimate_p_win,
    triage,
)


def dispute(fraud_score=0.1, amount=40000.0, reason="4855", days=7, **kw):
    return Dispute(
        id="disp_x", payment_id="pay_x", amount_inr=amount,
        reason_code=reason, reason_description="test", respond_by_days=days,
        fraud_score=fraud_score, **kw,
    )


def docs(*specs):
    return [
        EvidenceDocument(document_id=f"doc_{i}", slot=slot, description="d", strength=st)
        for i, (slot, st) in enumerate(specs)
    ]


STRONG = docs(("shipping_proof", "strong"), ("customer_communication", "strong"))


def test_genuine_fraud_is_never_contested():
    """The ethical constraint: do not fight a real victim's claim.

    A genuinely compromised card means the cardholder is telling the truth.
    Contesting is unwinnable, and pursuing it anyway would make this a tool for
    harassing fraud victims.
    """
    d = dispute(fraud_score=0.95, reason="4837")
    t = triage(d, STRONG)
    assert t.recommendation is Recommendation.ACCEPT
    assert not t.likely_friendly_fraud


def test_friendly_fraud_with_evidence_is_contested():
    t = triage(dispute(fraud_score=0.05), STRONG)
    assert t.recommendation is Recommendation.CONTEST
    assert t.likely_friendly_fraud
    assert t.expected_value_contest_inr > t.expected_value_accept_inr


def test_no_evidence_is_never_contested():
    """Filing an empty representment cannot win and costs money."""
    t = triage(dispute(fraud_score=0.05), [])
    assert t.recommendation is Recommendation.ACCEPT


def test_urgent_deadline_escalates_to_a_human():
    t = triage(dispute(fraud_score=0.05, days=1), STRONG)
    assert t.recommendation is Recommendation.ESCALATE


def test_win_probability_stays_inside_plausible_bounds():
    """A saturating curve, not an additive one. Guards the bug this had.

    Piling on evidence must never drive the estimate to implausible certainty.
    """
    many = docs(*[(s, "strong") for s in (
        "shipping_proof", "customer_communication", "proof_of_service",
        "access_activity_log", "billing_proof", "term_and_conditions",
        "refund_cancellation_policy", "explanation_letter",
    )])
    p_friendly, _ = estimate_p_win(dispute(fraud_score=0.02), many)
    assert p_friendly <= CEILING_P_WIN_FRIENDLY
    assert p_friendly < 0.7, "no evidence pile should imply near-certainty"

    p_genuine, _ = estimate_p_win(dispute(fraud_score=0.99, reason="4837"), many)
    assert p_genuine <= CEILING_P_WIN_GENUINE


def test_more_evidence_never_lowers_the_estimate():
    d = dispute(fraud_score=0.05)
    p_few, _ = estimate_p_win(d, docs(("shipping_proof", "strong")))
    p_more, _ = estimate_p_win(
        d, docs(("shipping_proof", "strong"), ("customer_communication", "strong"))
    )
    assert p_more >= p_few


def test_stronger_documents_beat_weaker_ones():
    d = dispute(fraud_score=0.05)
    p_weak, _ = estimate_p_win(d, docs(("shipping_proof", "weak")))
    p_strong, _ = estimate_p_win(d, docs(("shipping_proof", "strong")))
    assert p_strong > p_weak


def test_small_amounts_are_not_worth_the_effort():
    """Recovering Rs 300 cannot justify Rs 400 of staff time."""
    t = triage(dispute(fraud_score=0.05, amount=300.0), STRONG)
    assert t.recommendation is Recommendation.ACCEPT


def test_large_amounts_are_worth_contesting():
    t = triage(dispute(fraud_score=0.05, amount=200000.0), STRONG)
    assert t.recommendation is Recommendation.CONTEST


def test_ring_membership_suppresses_the_friendly_fraud_read():
    """Organised abuse is not a confused customer."""
    plain = estimate_p_win(dispute(fraud_score=0.05), STRONG)[0]
    ringed = estimate_p_win(
        dispute(fraud_score=0.05, ring_id=7, ring_size=12, ring_known_frauds=4), STRONG
    )[0]
    assert ringed < plain


def test_repeat_disputer_raises_the_estimate():
    plain = estimate_p_win(dispute(fraud_score=0.05), STRONG)[0]
    repeat = estimate_p_win(dispute(fraud_score=0.05, customer_prior_disputes=3), STRONG)[0]
    assert repeat > plain


def test_rationale_is_always_populated():
    """Every decision must be explainable to an operator."""
    for score in (0.02, 0.5, 0.98):
        t = triage(dispute(fraud_score=score), STRONG)
        assert t.rationale
        assert all(isinstance(r, str) and r for r in t.rationale)
