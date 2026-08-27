"""The drafter's safety properties, tested without spending an API call.

The provenance check is the security boundary of this whole component, so it is
tested directly rather than through the model.
"""
from __future__ import annotations

from ringfence.agent.drafter import _SlotAssignment, _validate_provenance
from ringfence.agent.models import ContestPayload, EvidenceDocument


def inventory():
    return [
        EvidenceDocument(document_id="doc_real_1", slot="shipping_proof", description="POD"),
        EvidenceDocument(document_id="doc_real_2", slot="billing_proof", description="Invoice"),
    ]


def test_real_document_ids_pass_through():
    slots, rejected = _validate_provenance(
        [_SlotAssignment(slot="shipping_proof", document_ids=["doc_real_1"], why="x")],
        inventory(),
    )
    assert slots["shipping_proof"] == ["doc_real_1"]
    assert rejected == []


def test_hallucinated_document_ids_are_rejected():
    """The single most important test in this module.

    A fabricated document id forwarded to a bank is a fabricated evidence
    claim. The prompt asks the model not to invent ids; this asserts the code
    does not rely on the prompt being obeyed.
    """
    slots, rejected = _validate_provenance(
        [_SlotAssignment(
            slot="shipping_proof",
            document_ids=["doc_real_1", "doc_INVENTED", "doc_also_fake"],
            why="x",
        )],
        inventory(),
    )
    assert slots["shipping_proof"] == ["doc_real_1"]
    assert set(rejected) == {"doc_INVENTED", "doc_also_fake"}


def test_unknown_slot_names_are_rejected():
    slots, rejected = _validate_provenance(
        [_SlotAssignment(slot="not_a_real_slot", document_ids=["doc_real_1"], why="x")],
        inventory(),
    )
    assert all(v == [] for v in slots.values())
    assert rejected == ["doc_real_1"]


def test_duplicate_ids_are_deduplicated():
    slots, _ = _validate_provenance(
        [
            _SlotAssignment(slot="shipping_proof", document_ids=["doc_real_1"], why="x"),
            _SlotAssignment(slot="shipping_proof", document_ids=["doc_real_1"], why="y"),
        ],
        inventory(),
    )
    assert slots["shipping_proof"] == ["doc_real_1"]


def test_empty_inventory_yields_empty_payload():
    slots, rejected = _validate_provenance(
        [_SlotAssignment(slot="shipping_proof", document_ids=["anything"], why="x")], []
    )
    assert all(v == [] for v in slots.values())
    assert rejected == ["anything"]


def test_payload_action_is_always_draft():
    """Submitting must never be reachable from this code path."""
    p = ContestPayload(amount=100, summary="s")
    assert p.action == "draft"
    # The field is a Literal["draft"]; anything else must fail validation.
    import pydantic
    import pytest
    with pytest.raises(pydantic.ValidationError):
        ContestPayload(amount=100, summary="s", action="submit")


def test_summary_length_is_capped():
    """Razorpay rejects summaries over 1000 characters."""
    import pydantic
    import pytest
    with pytest.raises(pydantic.ValidationError):
        ContestPayload(amount=100, summary="x" * 1001)
