from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any


class ReplaySignatureMismatchError(ValueError):
    """El replay no corresponde exactamente a la solicitud actual."""


@dataclass(frozen=True, slots=True)
class ReplaySignature:
    prompt_sha256: str
    schema_sha256: str
    raster_sha256: str
    raster_mime_type: str
    project_site_context_sha256: str | None
    version: str = "GEMINI_REPLAY_SIGNATURE_V1"

    def as_dict(self) -> dict[str, str | None]:
        return asdict(self)


class ReplaySignatureValidator:
    """Acepta replay solo cuando prompt, schema, raster, MIME y contexto coinciden."""

    @classmethod
    def build(
        cls,
        *,
        prompt: str,
        schema: dict[str, Any],
        raster_bytes: bytes,
        raster_mime_type: str,
        project_site_context: dict[str, Any] | None,
    ) -> ReplaySignature:
        return ReplaySignature(
            prompt_sha256=cls._sha_text(prompt),
            schema_sha256=cls._sha_json(schema),
            raster_sha256=hashlib.sha256(raster_bytes).hexdigest(),
            raster_mime_type=str(raster_mime_type).strip().lower(),
            project_site_context_sha256=(
                cls._sha_json(project_site_context)
                if project_site_context
                else None
            ),
        )

    @classmethod
    def validate(
        cls,
        *,
        replay_payload: dict[str, Any],
        current: ReplaySignature,
    ) -> None:
        signature = replay_payload.get("signature")
        if not isinstance(signature, dict):
            raise ReplaySignatureMismatchError(
                "Replay rechazado: falta signature verificable."
            )

        received = {
            "prompt_sha256": signature.get("prompt_sha256"),
            "schema_sha256": (
                signature.get("schema_sha256")
                or signature.get("response_schema_sha256")
            ),
            "raster_sha256": (
                signature.get("raster_sha256")
                or signature.get("media_sha256")
            ),
            "raster_mime_type": str(
                signature.get("raster_mime_type")
                or signature.get("media_mime_type")
                or ""
            ).strip().lower(),
            "project_site_context_sha256": signature.get(
                "project_site_context_sha256"
            ),
        }
        expected = current.as_dict()
        mismatches = [
            key
            for key in received
            if received[key] != expected[key]
        ]
        if mismatches:
            raise ReplaySignatureMismatchError(
                "Replay rechazado por firma incompatible: "
                + ", ".join(mismatches)
                + "."
            )

    @staticmethod
    def _sha_text(value: str) -> str:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    @staticmethod
    def _sha_json(value: dict[str, Any]) -> str:
        raw = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()
