from __future__ import annotations

import pytest

from app.quantia_spatialV1.stages.evidence.replay_signature import (
    ReplaySignatureMismatchError,
    ReplaySignatureValidator,
)


def _signature(*, context=None):
    return ReplaySignatureValidator.build(
        prompt="PROMPT_V1",
        schema={"type": "object", "required": ["niveles"]},
        raster_bytes=b"raster-v1",
        raster_mime_type="image/png",
        project_site_context=context,
    )


def test_replay_signature_accepts_exact_request() -> None:
    current = _signature(context={"site_width_m": 8.5})
    ReplaySignatureValidator.validate(
        replay_payload={"signature": current.as_dict(), "result": {"data": {}}},
        current=current,
    )


def test_replay_signature_accepts_legacy_hash_names_without_context() -> None:
    current = _signature()
    ReplaySignatureValidator.validate(
        replay_payload={
            "signature": {
                "prompt_sha256": current.prompt_sha256,
                "response_schema_sha256": current.schema_sha256,
                "media_sha256": current.raster_sha256,
                "media_mime_type": current.raster_mime_type,
            }
        },
        current=current,
    )


@pytest.mark.parametrize(
    "field,replacement",
    [
        ("prompt_sha256", "x"),
        ("schema_sha256", "x"),
        ("raster_sha256", "x"),
        ("raster_mime_type", "image/jpeg"),
    ],
)
def test_replay_signature_rejects_changed_request(field: str, replacement: str) -> None:
    current = _signature()
    replay = current.as_dict()
    replay[field] = replacement
    with pytest.raises(ReplaySignatureMismatchError):
        ReplaySignatureValidator.validate(
            replay_payload={"signature": replay},
            current=current,
        )


def test_replay_signature_rejects_context_change() -> None:
    current = _signature(context={"site_width_m": 8.5})
    old = _signature()
    with pytest.raises(ReplaySignatureMismatchError):
        ReplaySignatureValidator.validate(
            replay_payload={"signature": old.as_dict()},
            current=current,
        )


def test_replay_signature_rejects_unsigned_payload() -> None:
    with pytest.raises(ReplaySignatureMismatchError):
        ReplaySignatureValidator.validate(
            replay_payload={"result": {"data": {}}},
            current=_signature(),
        )
