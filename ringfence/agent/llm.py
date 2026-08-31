"""OpenRouter client and model selection for the agent layer.

WHY OPENROUTER
--------------
One key, one endpoint, every model. Swapping the model behind either agent is
an environment variable, not a code change -- which matters here because the two
jobs have different economics and may want different models over time.

MODEL CHOICE
------------
Surveyed the OpenRouter catalogue (417 models, 332 with structured-output
support) and selected on three requirements: JSON-schema structured output,
reasoning support, and price at the volume each job actually runs at.

`google/gemini-3.7-flash` ($0.38 / $1.88 per Mtok) is the default for both.
It is the newest Gemini generation on the platform and is priced BELOW the
older 3.6-flash ($0.75 / $3.75) and 3.5-flash ($1.50 / $9.00), so there is no
tradeoff to reason about -- it is newer and cheaper.

The pro tier (`gemini-3.1-pro-preview`, $2.00 / $12.00) was considered for the
drafter, whose output reaches a bank. Rejected: it is an older generation (3.1)
AND a preview, and stability matters more than tier for that component.

Note on cost: these are reasoning models. A trivial probe prompt cost 269
output tokens, so budget on measured usage rather than the nominal rate.
"""
from __future__ import annotations

import json
import os
from typing import TypeVar

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

from ringfence import config

load_dotenv(config.ROOT / ".env")

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

#: Safety-critical, low volume: its output becomes evidence sent to a bank.
DRAFTER_MODEL = os.environ.get("RINGFENCE_DRAFTER_MODEL", "google/gemini-3.7-flash")
#: Internal analyst notes, higher volume. `google/gemini-2.5-flash-lite`
#: ($0.10/$0.40) is a reasonable downgrade if ring volume grows.
CASEFILE_MODEL = os.environ.get("RINGFENCE_CASEFILE_MODEL", "google/gemini-3.7-flash")

T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    pass


def get_client(api_key: str | None = None) -> OpenAI:
    key = api_key or os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise LLMError(
            "OPENROUTER_API_KEY is not set. Put it in .env at the repo root, "
            "or export it."
        )
    return OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=key,
        default_headers={
            "HTTP-Referer": "https://github.com/iHiteshAgrawal/ringfence",
            "X-Title": "Ringfence",
        },
    )


def _strictify(schema: dict) -> dict:
    """Make a Pydantic JSON schema acceptable to strict structured output.

    Strict mode requires every object to set additionalProperties:false and to
    list every property in `required`. Pydantic omits both for fields that have
    defaults, so a schema that validates locally is rejected by the API.
    """
    if not isinstance(schema, dict):
        return schema
    if schema.get("type") == "object" or "properties" in schema:
        schema["additionalProperties"] = False
        props = schema.get("properties", {})
        if props:
            schema["required"] = list(props.keys())
        for sub in props.values():
            _strictify(sub)
    for key in ("items", "$defs", "definitions"):
        node = schema.get(key)
        if isinstance(node, dict):
            for sub in node.values():
                _strictify(sub)
            if key == "items":
                _strictify(node)
        elif isinstance(node, list):
            for sub in node:
                _strictify(sub)
    for key in ("anyOf", "oneOf", "allOf"):
        for sub in schema.get(key, []):
            _strictify(sub)
    return schema


def structured(
    output_model: type[T],
    system: str,
    user: str,
    model: str,
    client: OpenAI | None = None,
    max_tokens: int = 8000,
) -> T:
    """One structured call. Returns a validated instance of `output_model`."""
    client = client or get_client()
    schema = _strictify(output_model.model_json_schema())

    response = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": output_model.__name__,
                "strict": True,
                "schema": schema,
            },
        },
    )
    content = response.choices[0].message.content
    if not content:
        raise LLMError(f"{model} returned empty content")
    try:
        return output_model.model_validate(json.loads(content))
    except (json.JSONDecodeError, ValueError) as exc:
        raise LLMError(f"{model} returned unparseable output: {exc}") from exc
