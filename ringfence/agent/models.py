"""Types for the dispute side, mapped onto Razorpay's dispute API.

Field names deliberately mirror
https://razorpay.com/docs/api/disputes/contest/ so a payload built here can be
handed to the real endpoint without translation.
"""
from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

# The evidence slots Razorpay's contest endpoint accepts. Each takes a list of
# document ids previously uploaded through the Documents API.
EVIDENCE_SLOTS = (
    "shipping_proof",
    "billing_proof",
    "cancellation_proof",
    "customer_communication",
    "proof_of_service",
    "explanation_letter",
    "refund_confirmation",
    "access_activity_log",
    "refund_cancellation_policy",
    "term_and_conditions",
)


class DisputePhase(str, Enum):
    """Razorpay's five escalating phases."""

    FRAUD = "fraud"
    RETRIEVAL = "retrieval"
    CHARGEBACK = "chargeback"
    PRE_ARBITRATION = "pre_arbitration"
    ARBITRATION = "arbitration"


class DisputeStatus(str, Enum):
    OPEN = "open"
    UNDER_REVIEW = "under_review"
    WON = "won"
    LOST = "lost"
    CLOSED = "closed"


class EvidenceDocument(BaseModel):
    """A document the merchant already holds. Never invented by the agent."""

    document_id: str
    slot: str = Field(description="Which Razorpay evidence slot this fits")
    description: str
    # Whether the document actually covers this dispute. A shipping label with
    # no delivery confirmation is weaker than a signed proof of delivery, and
    # the agent must be able to tell the difference.
    strength: Literal["strong", "moderate", "weak"] = "moderate"


class Dispute(BaseModel):
    """A chargeback as it arrives from Razorpay, plus what we know locally."""

    id: str
    payment_id: str
    amount_inr: float
    phase: DisputePhase = DisputePhase.CHARGEBACK
    status: DisputeStatus = DisputeStatus.OPEN
    reason_code: str
    reason_description: str
    respond_by_days: int = Field(description="Days left to submit evidence")

    # What Ringfence's detector thought of the original transaction. This is
    # the pivot of the whole triage -- see triage.py.
    fraud_score: float = Field(ge=0.0, le=1.0)
    ring_id: int | None = None
    ring_size: int | None = None
    ring_known_frauds: int | None = None

    # Customer history, which bears on whether a dispute is plausible.
    customer_prior_orders: int = 0
    customer_prior_disputes: int = 0


class Recommendation(str, Enum):
    CONTEST = "contest"
    ACCEPT = "accept"
    ESCALATE = "escalate_to_human"


class TriageDecision(BaseModel):
    """The economics, computed without an LLM so it can be audited and tested."""

    recommendation: Recommendation
    p_win: float
    expected_value_contest_inr: float
    expected_value_accept_inr: float
    likely_friendly_fraud: bool
    rationale: list[str]
    evidence_available: list[str]
    evidence_missing: list[str]


class ContestPayload(BaseModel):
    """Exactly the body Razorpay's contest endpoint expects.

    `action` is fixed to "draft". Submitting is a human decision.
    """

    amount: int = Field(description="Contested amount in paise")
    summary: str = Field(max_length=1000)
    action: Literal["draft"] = "draft"
    shipping_proof: list[str] = []
    billing_proof: list[str] = []
    cancellation_proof: list[str] = []
    customer_communication: list[str] = []
    proof_of_service: list[str] = []
    explanation_letter: list[str] = []
    refund_confirmation: list[str] = []
    access_activity_log: list[str] = []
    refund_cancellation_policy: list[str] = []
    term_and_conditions: list[str] = []

    def document_ids(self) -> list[str]:
        ids: list[str] = []
        for slot in EVIDENCE_SLOTS:
            ids.extend(getattr(self, slot))
        return ids


class DraftedContest(BaseModel):
    """What the agent returns: the payload plus its reasoning."""

    payload: ContestPayload
    triage: TriageDecision
    narrative: str = Field(description="Plain-English account for the operator")
