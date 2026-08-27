"""Schema strictification and model configuration."""
from __future__ import annotations

from pydantic import BaseModel, Field

from ringfence.agent.llm import CASEFILE_MODEL, DRAFTER_MODEL, _strictify


class Nested(BaseModel):
    a: str
    b: int = 3


class Outer(BaseModel):
    name: str
    items: list[Nested]
    optional_note: str = "default"
    tags: list[str] = Field(default_factory=list)


def test_strictify_forces_additional_properties_false():
    """Strict structured output rejects schemas that allow extra keys."""
    s = _strictify(Outer.model_json_schema())
    assert s["additionalProperties"] is False
    for sub in s.get("$defs", {}).values():
        assert sub["additionalProperties"] is False


def test_strictify_marks_every_property_required():
    """Pydantic omits fields with defaults from `required`; strict mode wants all.

    A schema that validates locally is rejected by the API without this, which
    is a confusing failure to debug at call time.
    """
    s = _strictify(Outer.model_json_schema())
    assert set(s["required"]) == set(s["properties"].keys())
    assert "optional_note" in s["required"]
    assert "tags" in s["required"]
    nested = s["$defs"]["Nested"]
    assert set(nested["required"]) == {"a", "b"}


def test_strictify_is_idempotent():
    once = _strictify(Outer.model_json_schema())
    twice = _strictify(_strictify(Outer.model_json_schema()))
    assert once == twice


def test_models_are_gemini_and_overridable():
    """Anthropic models were removed on cost grounds; keep it that way."""
    for m in (DRAFTER_MODEL, CASEFILE_MODEL):
        assert m.startswith("google/gemini"), m
        assert "anthropic" not in m
