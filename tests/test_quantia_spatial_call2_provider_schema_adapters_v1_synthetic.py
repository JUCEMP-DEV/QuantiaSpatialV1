from __future__ import annotations

from typing import Any

from app.quantia_spatialV1.adaptive_reconstruction.call2_schema import CALL2_SCHEMA
from app.quantia_spatialV1.providers.call2_providers import (
    Call2StrictSchemaAdapter,
    GroqQwenCall2Provider,
    MistralCall2Provider,
)


def _assert_strict_objects(node: Any) -> None:
    if not isinstance(node, dict):
        return
    if node.get("type") == "object":
        props = node.get("properties", {})
        assert node.get("additionalProperties") is False
        assert set(node.get("required", [])) == set(props)
        for child in props.values():
            _assert_strict_objects(child)
    if node.get("type") == "array":
        _assert_strict_objects(node.get("items"))
    for child in node.get("anyOf", []) or []:
        _assert_strict_objects(child)


def test_call2_strict_schema_adapter_closes_every_object() -> None:
    strict = Call2StrictSchemaAdapter.adapt(CALL2_SCHEMA)
    _assert_strict_objects(strict)
    delta = strict["properties"]["deltas"]["items"]
    assert "wall_id" in delta["required"]
    assert "anyOf" in delta["properties"]["wall_id"]
    assert {branch.get("type") for branch in delta["properties"]["wall_id"]["anyOf"]} == {"string", "null"}


def test_groq_request_uses_qwen_vision_and_strict_schema() -> None:
    provider = GroqQwenCall2Provider(api_key="test-key")
    payload = provider.build_request(
        prompt="test",
        media_bytes=b"png",
        media_mime_type="image/png",
        response_json_schema=CALL2_SCHEMA,
    )
    assert payload["model"] == "qwen/qwen3.8-27b"
    assert payload["response_format"]["type"] == "json_schema"
    assert payload["response_format"]["json_schema"]["strict"] is True
    assert payload["messages"][0]["content"][1]["type"] == "image_url"
    assert payload["messages"][0]["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_mistral_request_uses_medium_3_5_vision_and_json_schema() -> None:
    provider = MistralCall2Provider(api_key="test-key")
    payload = provider.build_request(
        prompt="test",
        media_bytes=b"png",
        media_mime_type="image/png",
        response_json_schema=CALL2_SCHEMA,
    )
    assert payload["model"] == "mistral-medium-3-5"
    assert payload["response_format"]["type"] == "json_schema"
    assert payload["response_format"]["json_schema"]["strict"] is True
    assert payload["messages"][0]["content"][1]["type"] == "image_url"
    assert payload["messages"][0]["content"][1]["image_url"].startswith("data:image/png;base64,")
