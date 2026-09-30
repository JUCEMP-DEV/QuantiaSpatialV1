from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Sequence

from app.quantia_spatialV1.models.evidence import EvidenceGeometry, RawEvidence
from app.quantia_spatialV1.models.level_view import LevelView, PixelBBox
from app.quantia_spatialV1.phase_015_evidence.gemini_semantic_contract import (
    SEMANTIC_CONTRACT_VERSION,
    SEMANTIC_SOURCE_MODE,
)
from app.quantia_spatialV1.phase_015_evidence.gemini_semantic_extractor import (
    GeminiSemanticExtractor,
)
from app.quantia_spatialV1.phase_015_evidence.gemini_semantic_history import (
    GeminiSemanticHistory,
)
from app.quantia_spatialV1.providers.vision import (
    GeminiSpatialVisionProvider,
    SpatialVisionProviderError,
)


@dataclass(frozen=True, slots=True)
class GeminiEvidenceExtractionResult:
    """Compatibilidad para un único LevelView que ocupa la página completa."""

    evidence: list[RawEvidence]
    call_id: str
    semantic_payload: dict[str, Any]
    history_path: str | None


@dataclass(frozen=True, slots=True)
class GeminiPageEvidenceExtractionResult:
    """Resultado de una única extracción legacy sobre la página completa."""

    evidence_by_level_view: dict[str, list[RawEvidence]]
    call_id: str
    semantic_payloads_by_level_view: dict[str, dict[str, Any]]
    response_payload: dict[str, Any]
    history_path: str | None
    model: str | None
    fallback_used: bool
    replay_used: bool


class GeminiEvidenceAdapter:
    """
    Fase 01.5 — Gemini legacy + postprocesamiento semántico.

    Regla de integración:
        página completa -> QUANTIA_EXTRACTION_PROMPT legacy -> JSON legacy
        -> GeminiSemanticExtractor -> RawEvidence por LevelView.

    El prompt/schema legacy no se redefinen aquí. Se importan únicamente cuando
    se ejecuta una llamada real, evitando convertir el motor espacial en dueño
    de esos contratos externos.
    """

    def __init__(
        self,
        *,
        provider: GeminiSpatialVisionProvider | None = None,
        semantic_history: GeminiSemanticHistory | None = None,
        semantic_extractor: GeminiSemanticExtractor | None = None,
        prompt: str | None = None,
        response_json_schema: dict[str, Any] | None = None,
    ) -> None:
        self.provider = provider or GeminiSpatialVisionProvider()
        self.semantic_history = semantic_history or GeminiSemanticHistory()
        self.semantic_extractor = semantic_extractor or GeminiSemanticExtractor()
        self._prompt = prompt
        self._response_json_schema = response_json_schema

    def extract_page_with_trace(
        self,
        *,
        page_raster_bytes: bytes,
        page_raster_mime_type: str,
        source_document_id: str | None,
        source_page_number: int,
        level_views: Sequence[LevelView],
        replay_payload: dict[str, Any] | None = None,
    ) -> GeminiPageEvidenceExtractionResult:
        views = list(level_views)
        self._validate_page_request(
            page_raster_bytes=page_raster_bytes,
            source_page_number=source_page_number,
            level_views=views,
        )

        prompt = self._resolve_prompt()
        schema = self._resolve_schema()
        call_id = self.semantic_history.new_page_call_id(
            source_document_id=source_document_id,
            source_page_number=source_page_number,
        )

        self.semantic_history.append_started(
            call_id=call_id,
            source_document_id=source_document_id,
            source_page_number=source_page_number,
            level_views=views,
            prompt=prompt,
            schema=schema,
            raster_bytes=page_raster_bytes,
            raster_mime_type=page_raster_mime_type,
            semantic_contract_version=SEMANTIC_CONTRACT_VERSION,
        )

        provider_result: object | None = None
        replay_used = replay_payload is not None

        try:
            if replay_payload is not None:
                payload, replay_model, replay_fallback = self._payload_from_replay(
                    replay_payload
                )
            else:
                provider_result = self.provider.analyze(
                    prompt=prompt,
                    media_bytes=page_raster_bytes,
                    media_mime_type=page_raster_mime_type,
                    response_json_schema=schema,
                )
                payload = getattr(provider_result, "data", None)
                replay_model = None
                replay_fallback = False
        except SpatialVisionProviderError as exc:
            self.semantic_history.append_failed(
                call_id=call_id,
                source_document_id=source_document_id,
                source_page_number=source_page_number,
                level_views=views,
                error=exc,
            )
            raise ValueError(
                f"Gemini no pudo extraer evidencia de la página {source_page_number}."
            ) from exc
        except Exception as exc:
            self.semantic_history.append_failed(
                call_id=call_id,
                source_document_id=source_document_id,
                source_page_number=source_page_number,
                level_views=views,
                error=exc,
            )
            raise

        if not isinstance(payload, dict):
            error = ValueError(
                "Gemini no devolvió el objeto JSON del contrato legacy de extracción."
            )
            self.semantic_history.append_failed(
                call_id=call_id,
                source_document_id=source_document_id,
                source_page_number=source_page_number,
                level_views=views,
                error=error,
            )
            raise error

        semantic_payloads: dict[str, dict[str, Any]] = {}
        evidence_by_level: dict[str, list[RawEvidence]] = {}
        response_digest = self._stable_json_digest(payload)

        model = (
            str(getattr(provider_result, "model", "") or "").strip() or None
            if provider_result is not None
            else replay_model
        )
        fallback_used = (
            bool(getattr(provider_result, "fallback_used", False))
            if provider_result is not None
            else replay_fallback
        )

        allow_unique_level_fallback = len(views) == 1
        for level_view in views:
            semantic_payload = self.semantic_extractor.extract(
                payload=payload,
                level_view=level_view,
                allow_unique_level_fallback=allow_unique_level_fallback,
            )
            semantic_payloads[level_view.id] = semantic_payload
            observations = semantic_payload.get("observations") or []
            evidence_by_level[level_view.id] = [
                self._build_evidence(
                    level_view=level_view,
                    observation=observation,
                    index=index,
                    call_id=call_id,
                    response_digest=response_digest,
                    model=model,
                    fallback_used=fallback_used,
                    page_width_px=level_view.source_page_width_px,
                    page_height_px=level_view.source_page_height_px,
                )
                for index, observation in enumerate(observations, start=1)
                if isinstance(observation, dict)
            ]

        self.semantic_history.append_succeeded(
            call_id=call_id,
            source_document_id=source_document_id,
            source_page_number=source_page_number,
            level_views=views,
            provider_result=provider_result,
            semantic_payloads=semantic_payloads,
            response_payload=payload,
            replay_used=replay_used,
        )

        return GeminiPageEvidenceExtractionResult(
            evidence_by_level_view=evidence_by_level,
            call_id=call_id,
            semantic_payloads_by_level_view=semantic_payloads,
            response_payload=payload,
            history_path=self.semantic_history.path_text,
            model=model,
            fallback_used=fallback_used,
            replay_used=replay_used,
        )

    def reproject_page_result(
        self,
        *,
        source_result: GeminiPageEvidenceExtractionResult,
        level_views: Sequence[LevelView],
    ) -> GeminiPageEvidenceExtractionResult:
        """Reproyecta una respuesta Gemini ya obtenida sobre nuevos LevelView.

        No ejecuta proveedor, no consume tokens y no crea un nuevo registro de
        historial. Se reutiliza exactamente `response_payload` y únicamente se
        recalculan las geometrías locales contra el raster canónico actualizado.
        """
        views = list(level_views)
        if not views:
            raise ValueError("reproject_page_result requiere al menos un LevelView.")
        page_number = views[0].source_page_number
        for view in views:
            if view.source_page_number != page_number:
                raise ValueError("Todos los LevelView reproyectados deben pertenecer a la misma página.")

        payload = source_result.response_payload
        if not isinstance(payload, dict):
            raise ValueError("La respuesta Gemini fuente no contiene response_payload válido.")

        semantic_payloads: dict[str, dict[str, Any]] = {}
        evidence_by_level: dict[str, list[RawEvidence]] = {}
        response_digest = self._stable_json_digest(payload)
        allow_unique_level_fallback = len(views) == 1

        for level_view in views:
            semantic_payload = self.semantic_extractor.extract(
                payload=payload,
                level_view=level_view,
                allow_unique_level_fallback=allow_unique_level_fallback,
            )
            semantic_payloads[level_view.id] = semantic_payload
            observations = semantic_payload.get("observations") or []
            evidence_by_level[level_view.id] = [
                self._build_evidence(
                    level_view=level_view,
                    observation=observation,
                    index=index,
                    call_id=source_result.call_id,
                    response_digest=response_digest,
                    model=source_result.model,
                    fallback_used=source_result.fallback_used,
                    page_width_px=level_view.source_page_width_px,
                    page_height_px=level_view.source_page_height_px,
                )
                for index, observation in enumerate(observations, start=1)
                if isinstance(observation, dict)
            ]

        return GeminiPageEvidenceExtractionResult(
            evidence_by_level_view=evidence_by_level,
            call_id=source_result.call_id,
            semantic_payloads_by_level_view=semantic_payloads,
            response_payload=payload,
            history_path=source_result.history_path,
            model=source_result.model,
            fallback_used=source_result.fallback_used,
            replay_used=source_result.replay_used,
        )

    def extract_with_trace(self, *, level_view: LevelView) -> GeminiEvidenceExtractionResult:
        """
        Compatibilidad controlada: solo permite llamada directa si el LevelView
        representa exactamente la página completa. Evita repetir el error de
        ejecutar el prompt legacy sobre un recorte aislado.
        """
        bbox = level_view.source_bbox_px
        is_full_page = (
            bbox.x_min == 0
            and bbox.y_min == 0
            and bbox.x_max == level_view.source_page_width_px
            and bbox.y_max == level_view.source_page_height_px
        )
        if not is_full_page:
            raise ValueError(
                "La extracción legacy de Gemini debe ejecutarse sobre la página completa, "
                "no sobre un LevelView recortado."
            )

        page_result = self.extract_page_with_trace(
            page_raster_bytes=level_view.raster_bytes,
            page_raster_mime_type=level_view.raster_mime_type,
            source_document_id=level_view.source_document_id,
            source_page_number=level_view.source_page_number,
            level_views=[level_view],
        )
        return GeminiEvidenceExtractionResult(
            evidence=page_result.evidence_by_level_view[level_view.id],
            call_id=page_result.call_id,
            semantic_payload=page_result.semantic_payloads_by_level_view[level_view.id],
            history_path=page_result.history_path,
        )

    def extract(self, *, level_view: LevelView) -> list[RawEvidence]:
        return self.extract_with_trace(level_view=level_view).evidence

    def _build_evidence(
        self,
        *,
        level_view: LevelView,
        observation: dict[str, Any],
        index: int,
        call_id: str,
        response_digest: str,
        model: str | None,
        fallback_used: bool,
        page_width_px: int,
        page_height_px: int,
    ) -> RawEvidence:
        category = str(observation.get("category") or "OTHER").strip().upper()
        description = str(observation.get("description") or "").strip()
        visible_text = self._optional_text(observation.get("visible_text"))
        confidence = self._parse_confidence(observation.get("confidence"))

        geometry, geometry_metadata = self._build_geometry_from_page_bbox(
            bbox_normalized=observation.get("bbox_normalized"),
            level_view=level_view,
            page_width_px=page_width_px,
            page_height_px=page_height_px,
        )

        identity_payload = {
            "response_digest": response_digest,
            "level_view_id": level_view.id,
            "index": index,
            "observation": observation,
        }
        evidence_id = self._stable_id(
            level_view_id=level_view.id,
            payload=identity_payload,
        )

        metadata: dict[str, Any] = {
            "source_document_id": level_view.source_document_id,
            "source_page_number": level_view.source_page_number,
            "level_name": level_view.level_name,
            "semantic_contract_version": SEMANTIC_CONTRACT_VERSION,
            "semantic_source_mode": SEMANTIC_SOURCE_MODE,
            "semantic_call_scope": "PAGE",
            "semantic_call_id": call_id,
            "semantic_response_digest": response_digest,
            "semantic_category": category,
            "semantic_state": observation.get("state"),
            "semantic_subtype": observation.get("subtype"),
            "semantic_name": observation.get("name"),
            "description": description or None,
            "location_text": observation.get("location_text"),
            "orientation": observation.get("orientation"),
            "dimension_side": observation.get("dimension_side"),
            "reference_start": observation.get("reference_start"),
            "reference_end": observation.get("reference_end"),
            "span_type": observation.get("span_type"),
            "measurements": list(observation.get("measurements") or []),
            "properties": list(observation.get("properties") or []),
            "relations": list(observation.get("relations") or []),
            "semantic_evidence": list(observation.get("evidence") or []),
            "raw_observation": dict(observation),
            **geometry_metadata,
        }
        if model:
            metadata["gemini_model"] = model
        metadata["fallback_used"] = bool(fallback_used)

        return RawEvidence(
            id=evidence_id,
            level_view_id=level_view.id,
            source="GEMINI",
            kind="GEMINI_OBSERVATION",
            geometry=geometry,
            text=visible_text or description or None,
            confidence=confidence,
            metadata=metadata,
            confirmed=False,
        )

    @staticmethod
    def _build_geometry_from_page_bbox(
        *,
        bbox_normalized: object,
        level_view: LevelView,
        page_width_px: int,
        page_height_px: int,
    ) -> tuple[EvidenceGeometry, dict[str, Any]]:
        if not isinstance(bbox_normalized, dict):
            return EvidenceGeometry(geometry_type="NONE"), {"localization_status": "NOT_PROVIDED"}

        try:
            x_min_n = float(bbox_normalized["x_min"])
            y_min_n = float(bbox_normalized["y_min"])
            x_max_n = float(bbox_normalized["x_max"])
            y_max_n = float(bbox_normalized["y_max"])
        except (KeyError, TypeError, ValueError):
            return EvidenceGeometry(geometry_type="NONE"), {
                "localization_status": "INVALID",
                "bbox_normalized_raw": bbox_normalized,
            }

        if not (
            0.0 <= x_min_n < x_max_n <= 1.0
            and 0.0 <= y_min_n < y_max_n <= 1.0
        ):
            return EvidenceGeometry(geometry_type="NONE"), {
                "localization_status": "INVALID",
                "bbox_normalized_raw": dict(bbox_normalized),
            }

        page_bbox = PixelBBox(
            x_min=max(0, min(page_width_px - 1, int(round(x_min_n * page_width_px)))),
            y_min=max(0, min(page_height_px - 1, int(round(y_min_n * page_height_px)))),
            x_max=max(1, min(page_width_px, int(round(x_max_n * page_width_px)))),
            y_max=max(1, min(page_height_px, int(round(y_max_n * page_height_px)))),
        )
        source = level_view.source_bbox_px
        ix_min = max(page_bbox.x_min, source.x_min)
        iy_min = max(page_bbox.y_min, source.y_min)
        ix_max = min(page_bbox.x_max, source.x_max)
        iy_max = min(page_bbox.y_max, source.y_max)

        if ix_max <= ix_min or iy_max <= iy_min:
            return EvidenceGeometry(geometry_type="NONE"), {
                "localization_status": "OUTSIDE_LEVEL_VIEW",
                "bbox_normalized_page": dict(bbox_normalized),
            }

        local_bbox = PixelBBox(
            x_min=ix_min - source.x_min,
            y_min=iy_min - source.y_min,
            x_max=ix_max - source.x_min,
            y_max=iy_max - source.y_min,
        )
        return EvidenceGeometry(geometry_type="BBOX", bbox_px=local_bbox), {
            "localization_status": "LOCALIZED_FROM_PAGE",
            "bbox_normalized_page": dict(bbox_normalized),
            "bbox_page_px": page_bbox.model_dump(),
        }

    @staticmethod
    def _validate_page_request(
        *,
        page_raster_bytes: bytes,
        source_page_number: int,
        level_views: Sequence[LevelView],
    ) -> None:
        if not page_raster_bytes:
            raise ValueError("GeminiEvidenceAdapter requiere raster de página completa.")
        if source_page_number <= 0:
            raise ValueError("source_page_number debe ser mayor que cero.")
        ids: set[str] = set()
        for level_view in level_views:
            if level_view.source_page_number != source_page_number:
                raise ValueError(
                    f"LevelView {level_view.id} pertenece a otra página."
                )
            if level_view.id in ids:
                raise ValueError(f"LevelView.id duplicado: {level_view.id}.")
            ids.add(level_view.id)

    def _resolve_prompt(self) -> str:
        if self._prompt is not None:
            return self._prompt
        from app.prompts.quantia_extraction_prompt import QUANTIA_EXTRACTION_PROMPT

        return QUANTIA_EXTRACTION_PROMPT

    def _resolve_schema(self) -> dict[str, Any]:
        if self._response_json_schema is not None:
            return dict(self._response_json_schema)
        from app.schemas.gemini_extraction_transport import (
            get_gemini_extraction_transport_schema,
        )

        return get_gemini_extraction_transport_schema()

    @staticmethod
    def _payload_from_replay(
        replay_payload: dict[str, Any],
    ) -> tuple[dict[str, Any], str | None, bool]:
        if "result" in replay_payload and isinstance(replay_payload["result"], dict):
            result = replay_payload["result"]
            data = result.get("data")
            if not isinstance(data, dict):
                text = result.get("text")
                if isinstance(text, str) and text.strip():
                    parsed = json.loads(text)
                    data = parsed if isinstance(parsed, dict) else None
            if not isinstance(data, dict):
                raise ValueError("Replay legacy no contiene result.data JSON válido.")
            model = str(result.get("model") or "").strip() or None
            fallback_used = bool(result.get("fallback_used", False))
            return dict(data), model, fallback_used

        required = {"resumen", "documento", "predio", "niveles"}
        if required.issubset(replay_payload.keys()):
            return dict(replay_payload), None, False

        raise ValueError("Payload replay de Gemini legacy no reconocido.")

    @staticmethod
    def _stable_json_digest(value: dict[str, Any]) -> str:
        raw = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    @staticmethod
    def _stable_id(*, level_view_id: str, payload: object) -> str:
        raw = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
        digest = hashlib.sha256(raw).hexdigest()[:16]
        return f"{level_view_id}__GEMINI_EVIDENCE__{digest}"

    @staticmethod
    def _optional_text(value: object) -> str | None:
        text = str(value or "").strip()
        return text or None

    @staticmethod
    def _parse_confidence(value: object) -> float | None:
        if value is None or isinstance(value, bool):
            return None
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return None
        return parsed if 0.0 <= parsed <= 1.0 else None
