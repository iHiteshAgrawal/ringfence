"""Turn a flagged ring into something an analyst can act on in 30 seconds.

WHY THIS EXISTS
---------------
A ranked list of transaction ids is not a product. The reviewer still has to
reconstruct what happened: which cards are involved, what connects them, what
the model reacted to, and what the sensible action is. That reconstruction is
the actual bottleneck -- verification capacity, not detection capacity.

The model is given only computed facts about the ring. It does not see raw
cardholder data, and it is explicitly instructed not to invent detail. Its job
is to explain a mechanism, not to decide guilt: the recommended action is
graded (allow / step up / hold / block) and always reversible by a human.
"""
from __future__ import annotations

from openai import OpenAI
from pydantic import BaseModel, Field

from ringfence.agent.llm import CASEFILE_MODEL, structured


class RingFacts(BaseModel):
    """Computed facts about one flagged ring. Every field comes from the data."""

    ring_id: int
    n_clients: int
    n_transactions: int
    n_addresses: int
    total_amount_inr: float
    span_days: float
    velocity_per_day: float
    cards_per_address: float
    amount_cv: float = Field(description="Coefficient of variation of amounts")
    known_prior_frauds: int
    shared_attributes: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Which attribute values link the ring, e.g. {'addr1': ['12345']}",
    )
    max_fraud_score: float
    mean_fraud_score: float
    top_features: list[str] = Field(
        default_factory=list, description="Features that drove the score, ranked"
    )


class CaseFile(BaseModel):
    headline: str = Field(description="One line an analyst can triage from")
    what_links_them: str = Field(description="The concrete connection, in plain English")
    why_suspicious: str = Field(description="The behavioural pattern, tied to the numbers")
    innocent_explanation: str = Field(
        description="The most plausible legitimate reading of the same facts"
    )
    recommended_action: str = Field(description="allow | step_up | hold | block")
    confidence: str = Field(description="low | medium | high")
    what_would_change_the_call: str = Field(
        description="The specific evidence that would flip this decision"
    )


SYSTEM = """You write case files for fraud analysts reviewing flagged groups of
payment accounts on Razorpay.

You are given computed facts about one ring. Write a case file the analyst can
act on quickly.

Rules:

- Use ONLY the facts provided. Never invent a name, an address, a device, or a
  behaviour that is not in the input. If something is not given, do not mention it.
- Cite the actual numbers. "12 cards on one address in 31 hours" is useful;
  "highly suspicious velocity" is not.
- `innocent_explanation` is mandatory and must be written in good faith. Shared
  addresses have legitimate causes: families, shared student housing, offices,
  package-forwarding services, hostels. An analyst who only ever reads the
  prosecution case stops thinking. If you genuinely cannot construct one, say
  precisely which fact rules it out.
- Grade the action honestly:
    allow    - the innocent explanation is at least as likely
    step_up  - request additional verification before proceeding
    hold     - pause for manual review; do not auto-decline
    block    - only with confirmed prior fraud in the ring
- `confidence` reflects the evidence, not the severity. A large ring with a
  clean innocent explanation is low confidence.

You are supporting a human decision, not making it. Never write as though the
outcome is settled."""


def _render(facts: RingFacts) -> str:
    links = "\n".join(
        f"  {attr}: {', '.join(vals)}" for attr, vals in facts.shared_attributes.items()
    ) or "  (none recorded)"
    feats = ", ".join(facts.top_features) if facts.top_features else "(not supplied)"
    return f"""Ring {facts.ring_id}

Size and shape:
  distinct cards (clients): {facts.n_clients}
  transactions: {facts.n_transactions}
  distinct billing addresses: {facts.n_addresses}
  cards per address: {facts.cards_per_address:.2f}

Money and timing:
  total value: Rs {facts.total_amount_inr:,.2f}
  active span: {facts.span_days:.2f} days
  velocity: {facts.velocity_per_day:.2f} transactions/day
  amount variability (CV): {facts.amount_cv:.3f}  (near 0 means near-identical amounts)

History:
  confirmed prior frauds in this ring: {facts.known_prior_frauds}

Model output:
  peak fraud score: {facts.max_fraud_score:.3f}
  mean fraud score: {facts.mean_fraud_score:.3f}
  features that drove it: {feats}

Shared attributes linking these cards:
{links}"""


def write_case_file(
    facts: RingFacts,
    client: OpenAI | None = None,
    model: str = CASEFILE_MODEL,
) -> CaseFile:
    """Produce an analyst-ready case file for one ring."""
    return structured(
        CaseFile,
        system=SYSTEM,
        user=_render(facts),
        model=model,
        client=client,
        max_tokens=4000,
    )


VALID_ACTIONS = {"allow", "step_up", "hold", "block"}
VALID_CONFIDENCE = {"low", "medium", "high"}
