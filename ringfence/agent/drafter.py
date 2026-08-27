"""Draft a Razorpay contest payload with Claude. Draft only, never submit.

DIVISION OF LABOUR
------------------
The economics are NOT delegated to the model. `triage.py` decides contest vs
accept by arithmetic that can be unit-tested and audited. The model is used for
the part arithmetic cannot do: reading a merchant's document inventory, judging
which document actually answers this issuer's reason code, and writing the
<=1000 character summary a human reviewer will read.

SAFETY PROPERTY
---------------
Every document id in the drafted payload MUST come from the evidence inventory
that was passed in. A hallucinated document id is not a formatting bug -- it is
a fabricated evidence claim submitted to a bank. The prompt asks for this, and
then `_validate_provenance` enforces it, because a prompt is not a guarantee.
Anything unrecognised is dropped and recorded, never silently forwarded.
"""
from __future__ import annotations

from openai import OpenAI
from pydantic import BaseModel, Field

from ringfence.agent.llm import DRAFTER_MODEL, structured
from ringfence.agent.models import (
    EVIDENCE_SLOTS,
    ContestPayload,
    Dispute,
    DraftedContest,
    EvidenceDocument,
    Recommendation,
    TriageDecision,
)
from ringfence.agent.triage import TriageConfig, triage


class _SlotAssignment(BaseModel):
    """One evidence slot the model wants to fill."""

    slot: str = Field(description="One of Razorpay's evidence slot names")
    document_ids: list[str] = Field(description="Ids taken ONLY from the inventory")
    why: str = Field(description="Why this document answers this reason code")


class _DraftResponse(BaseModel):
    summary: str = Field(description="<=1000 characters, addressed to the issuer")
    assignments: list[_SlotAssignment]
    narrative: str = Field(description="Plain-English note for the merchant's operator")
    omitted: list[str] = Field(
        default_factory=list,
        description="Document ids deliberately left out, and why",
    )


SYSTEM = """You prepare chargeback representment packets for a merchant on Razorpay.

You are given one dispute, the merchant's document inventory, and a triage
decision that has ALREADY been made by an audited economic model. Do not
re-litigate that decision.

Your job is narrow and specific:

1. Map documents to Razorpay evidence slots. Choose the document that actually
   rebuts THIS reason code, not everything available. A packet of four
   well-chosen documents beats eleven loosely relevant ones; issuers review
   these under time pressure.
2. Write the `summary` for the issuer: at most 1000 characters, factual, and
   organised as claim then evidence. No adjectives, no pleading, no speculation
   about the cardholder's motives. State what was bought, that it was
   delivered or rendered, and which attached document proves it.
3. Write a short `narrative` for the merchant's own operator explaining the
   packet in plain English.

Absolute constraints:

- Use ONLY document ids that appear in the inventory you were given. Never
  invent, guess, or extrapolate an id. If the evidence needed does not exist,
  say so in `narrative` and leave the slot empty.
- Never assert a fact the documents do not support. You are assembling genuine
  evidence, not constructing an argument.
- Valid slot names are exactly: {slots}

If the strongest available document is weak, say that plainly in `narrative`
rather than overstating the packet's strength."""


def _render_inventory(evidence: list[EvidenceDocument]) -> str:
    if not evidence:
        return "(the merchant holds no documents for this order)"
    return "\n".join(
        f"- id={d.document_id} | fits_slot={d.slot} | strength={d.strength} | {d.description}"
        for d in evidence
    )


def _render_dispute(d: Dispute, t: TriageDecision) -> str:
    ring = (
        f"\nRing: id={d.ring_id}, {d.ring_size} linked cards, "
        f"{d.ring_known_frauds} confirmed prior frauds"
        if d.ring_id is not None
        else "\nRing: this card is not linked to any known ring"
    )
    return f"""Dispute {d.id} on payment {d.payment_id}
Amount: Rs {d.amount_inr:,.2f}
Phase: {d.phase.value} | Status: {d.status.value}
Reason code {d.reason_code}: {d.reason_description}
Days left to respond: {d.respond_by_days}
Customer history: {d.customer_prior_orders} prior orders, {d.customer_prior_disputes} prior disputes{ring}

Triage (already decided, do not revisit):
  recommendation: {t.recommendation.value}
  estimated win probability: {t.p_win:.1%}
  likely friendly fraud: {t.likely_friendly_fraud}
  reasoning:
""" + "\n".join(f"    - {r}" for r in t.rationale)


def _validate_provenance(
    assignments: list[_SlotAssignment], evidence: list[EvidenceDocument]
) -> tuple[dict[str, list[str]], list[str]]:
    """Keep only ids that genuinely exist. Returns (slots, rejected_ids)."""
    known = {d.document_id for d in evidence}
    slots: dict[str, list[str]] = {s: [] for s in EVIDENCE_SLOTS}
    rejected: list[str] = []
    for a in assignments:
        if a.slot not in slots:
            rejected.extend(a.document_ids)
            continue
        for doc_id in a.document_ids:
            if doc_id in known:
                if doc_id not in slots[a.slot]:
                    slots[a.slot].append(doc_id)
            else:
                rejected.append(doc_id)
    return slots, rejected


def draft_contest(
    dispute: Dispute,
    evidence: list[EvidenceDocument],
    client: OpenAI | None = None,
    cfg: TriageConfig | None = None,
    model: str = DRAFTER_MODEL,
) -> DraftedContest:
    """Triage, then have Claude assemble the packet. Returns action='draft'."""
    decision = triage(dispute, evidence, cfg)

    if decision.recommendation is not Recommendation.CONTEST:
        return DraftedContest(
            payload=ContestPayload(amount=round(dispute.amount_inr * 100), summary=""),
            triage=decision,
            narrative=(
                f"No packet drafted: triage recommends {decision.recommendation.value}. "
                + (decision.rationale[-1] if decision.rationale else "")
            ),
        )

    parsed = structured(
        _DraftResponse,
        system=SYSTEM.format(slots=", ".join(EVIDENCE_SLOTS)),
        user=(
            f"{_render_dispute(dispute, decision)}\n\n"
            f"Merchant document inventory:\n{_render_inventory(evidence)}\n\n"
            "Assemble the representment packet."
        ),
        model=model,
        client=client,
    )

    slots, rejected = _validate_provenance(parsed.assignments, evidence)
    narrative = parsed.narrative
    if rejected:
        # Loud, not silent: a hallucinated id must be visible to the operator.
        narrative += (
            f"\n\nWARNING: {len(rejected)} document id(s) proposed by the model "
            f"were not in the merchant's inventory and were dropped: "
            f"{', '.join(sorted(set(rejected)))}."
        )

    summary = parsed.summary.strip()[:1000]
    payload = ContestPayload(
        amount=round(dispute.amount_inr * 100),
        summary=summary,
        **slots,
    )
    return DraftedContest(payload=payload, triage=decision, narrative=narrative)
