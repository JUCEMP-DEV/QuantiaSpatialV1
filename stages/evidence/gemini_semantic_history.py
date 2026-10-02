from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence
from uuid import uuid4

from app.quantia_spatialV1.models.level_view import LevelView


class GeminiSemanticHistory:
    """
    Registro append-only opcional de la extracción Gemini de Fase 01.5.

    El flujo productivo no escribe dentro de tests. La persistencia solo se
    activa cuando el caller entrega ``history_path`` o configura
    ``QUANTIA_GEMINI_SEMANTIC_HISTORY_PATH``.

    La llamada legacy se registra por página completa; posteriormente la misma
    respuesta se clasifica por cada LevelView de esa página.
    """

    ENV_PATH = "QUANTIA_GEMINI_SEMANTIC_HISTORY_PATH"

    def __init__(self, *, history_path: str | Path | None = None) -> None:
        configured = str(os.getenv(self.ENV_PATH, "") or "").strip()
        if history_path is not None:
            self.history_path: Path | None = Path(history_path)
        elif configured:
            self.history_path = Path(configured)
        else:
            self.history_path = None

    @property
    def path_text(self) -> str | None:
        return str(self.history_path) if self.history_path is not None else None

    @staticmethod
    def new_page_call_id(
        *,
        source_document_id: str | None,
        source_page_number: int,
    ) -> str:
        document = str(source_document_id or "DOCUMENT").strip() or "DOCUMENT"
        suffix = uuid4().hex[:12]
        return f"{document}__PAGE_{source_page_number}__GEMINI__{suffix}"

    def append_started(
        self,
        *,
        call_id: str,
        source_document_id: str | None,
        source_page_number: int,
        level_views: Sequence[LevelView],
        prompt: str,
        schema: dict[str, Any],
        raster_bytes: bytes,
        raster_mime_type: str,
        semantic_contract_version: str,
        project_site_context: dict[str, Any] | None = None,
    ) -> None:
        self._append(
            {
                "event": "STARTED",
                "call_scope": "PAGE",
                "call_id": call_id,
                "semantic_contract_version": semantic_contract_version,
                "source_document_id": source_document_id,
                "source_page_number": source_page_number,
                "level_view_ids": [item.id for item in level_views],
                "level_names": [item.level_name for item in level_views],
                "prompt_sha256": self._sha256_text(prompt),
                "schema_sha256": self._sha256_json(schema),
                "raster_sha256": hashlib.sha256(raster_bytes).hexdigest(),
                "raster_mime_type": raster_mime_type,
                "project_site_context": self._sanitize(project_site_context),
                "project_site_context_sha256": (
                    self._sha256_json(project_site_context)
                    if project_site_context
                    else None
                ),
            }
        )

    def append_succeeded(
        self,
        *,
        call_id: str,
        source_document_id: str | None,
        source_page_number: int,
        level_views: Sequence[LevelView],
        provider_result: object | None,
        semantic_payloads: dict[str, dict[str, Any]],
        response_payload: dict[str, Any],
        replay_used: bool,
        project_site_context: dict[str, Any] | None = None,
    ) -> None:
        raw = getattr(provider_result, "raw", None) if provider_result is not None else None
        raw_dict = dict(raw) if isinstance(raw, dict) else None

        self._append(
            {
                "event": "SUCCEEDED",
                "call_scope": "PAGE",
                "call_id": call_id,
                "source_document_id": source_document_id,
                "source_page_number": source_page_number,
                "level_view_ids": [item.id for item in level_views],
                "level_names": [item.level_name for item in level_views],
                "provider": getattr(provider_result, "provider", None),
                "model": getattr(provider_result, "model", None),
                "fallback_used": bool(
                    getattr(provider_result, "fallback_used", False)
                ) if provider_result is not None else False,
                "replay_used": bool(replay_used),
                "project_site_context": self._sanitize(project_site_context),
                "response_payload": response_payload,
                "semantic_payloads_by_level_view": semantic_payloads,
                "usage_metadata": raw_dict.get("usageMetadata") if raw_dict else None,
                "provider_raw_response": self._sanitize(raw_dict),
            }
        )

    def append_failed(
        self,
        *,
        call_id: str,
        source_document_id: str | None,
        source_page_number: int,
        level_views: Sequence[LevelView],
        error: BaseException,
    ) -> None:
        self._append(
            {
                "event": "FAILED",
                "call_scope": "PAGE",
                "call_id": call_id,
                "source_document_id": source_document_id,
                "source_page_number": source_page_number,
                "level_view_ids": [item.id for item in level_views],
                "level_names": [item.level_name for item in level_views],
                "error_type": type(error).__name__,
                "error": str(error),
            }
        )

    def _append(self, payload: dict[str, Any]) -> None:
        if self.history_path is None:
            return

        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            **payload,
        }
        line = json.dumps(
            self._sanitize(record),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        with self.history_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(line)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())

    @staticmethod
    def _sha256_text(value: str) -> str:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    @staticmethod
    def _sha256_json(value: dict[str, Any]) -> str:
        raw = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    @classmethod
    def _sanitize(cls, value: Any) -> Any:
        secret_keys = {
            "api_key",
            "apikey",
            "x-goog-api-key",
            "authorization",
            "access_token",
            "refresh_token",
            "thoughtsignature",
        }

        if isinstance(value, dict):
            result: dict[str, Any] = {}
            for key, item in value.items():
                normalized = str(key).strip().lower()
                result[str(key)] = (
                    "[REDACTED]"
                    if normalized in secret_keys
                    else cls._sanitize(item)
                )
            return result
        if isinstance(value, list):
            return [cls._sanitize(item) for item in value]
        if isinstance(value, tuple):
            return [cls._sanitize(item) for item in value]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)
