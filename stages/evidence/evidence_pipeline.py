from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import Any, Callable

from pydantic import BaseModel, Field

from app.quantia_spatialV1.core.models.evidence import RawEvidence
from app.quantia_spatialV1.core.models.level_view import LevelView
from app.quantia_spatialV1.stages.evidence.evidence_geometry_validator import (
    EvidenceGeometryValidator,
)
from app.quantia_spatialV1.stages.evidence.evidence_parameterizer import (
    EvidenceParameterizer,
)
from app.quantia_spatialV1.stages.evidence.ocr_evidence_adapter import OCREvidenceAdapter
from app.quantia_spatialV1.stages.evidence.opencv_evidence_adapter import (
    OpenCVEvidenceAdapter,
)
from app.quantia_spatialV1.stages.evidence.pymupdf_evidence_adapter import (
    PyMuPDFEvidenceAdapter,
)


class EvidenceSourceDiagnostic(BaseModel):
    status: str
    count: int = 0
    error: str | None = None


class EvidencePipelineDiagnostics(BaseModel):
    """Diagnóstico técnico de F01.5. No clasifica arquitectura."""

    pymupdf_count: int = 0
    opencv_count: int = 0
    ocr_count: int = 0
    gemini_count: int = 0
    total_count: int = 0
    invalid_geometry_count: int = 0
    kind_counts: dict[str, int] = Field(default_factory=dict)
    source_diagnostics: dict[str, EvidenceSourceDiagnostic] = Field(default_factory=dict)
    gemini_call_id: str | None = None
    gemini_semantic_history_path: str | None = None
    parameter_count: int = 0
    parameter_counts_by_source: dict[str, int] = Field(default_factory=dict)


class EvidencePipelineResult(BaseModel):
    """Salida canónica no destructiva de F01.5 para un LevelView."""

    level_view_id: str
    source_document_id: str | None = None
    source_page_number: int
    evidence: list[RawEvidence] = Field(default_factory=list)
    diagnostics: EvidencePipelineDiagnostics
    warnings: list[str] = Field(default_factory=list)

    @property
    def all_evidence(self) -> list[RawEvidence]:
        return list(self.evidence)

    def by_source(self, source: str) -> list[RawEvidence]:
        normalized = str(source or "").strip().upper()
        return [item for item in self.evidence if item.source == normalized]

    def by_kind(self, kind: str) -> list[RawEvidence]:
        normalized = str(kind or "").strip().upper()
        return [item for item in self.evidence if item.kind == normalized]


class EvidencePipeline:
    """
    Fase 01.5 — extracción multimodal no destructiva por LevelView.

    PyMuPDF, OpenCV y OCR se ejecutan localmente sobre el LevelView. Gemini no
    se llama desde aquí: su prompt legacy necesita la página completa, por lo
    que el orquestador ejecuta una sola llamada por página y entrega aquí las
    RawEvidence ya clasificadas para este LevelView.

    Después de reunir las fuentes, cada RawEvidence recibe `parameters[]`
    mediante EvidenceParameterizer. geometry/text/metadata permanecen intactos.

    Un fallo de cualquier fuente no elimina evidencia válida de las demás.
    """

    PDF_MIME_TYPE = "application/pdf"
    IMAGE_MIME_TYPES = {"image/png", "image/jpeg"}
    IMAGE_MIME_ALIASES = {"image/jpg": "image/jpeg"}

    def __init__(
        self,
        *,
        pymupdf_adapter: PyMuPDFEvidenceAdapter | None = None,
        opencv_adapter: OpenCVEvidenceAdapter | None = None,
        ocr_adapter: OCREvidenceAdapter | None = None,
        geometry_validator: EvidenceGeometryValidator | None = None,
        parameterizer: EvidenceParameterizer | None = None,
    ) -> None:
        self.pymupdf_adapter = pymupdf_adapter or PyMuPDFEvidenceAdapter()
        self.opencv_adapter = opencv_adapter or OpenCVEvidenceAdapter()
        self.ocr_adapter = ocr_adapter or OCREvidenceAdapter()
        self.geometry_validator = geometry_validator or EvidenceGeometryValidator()
        self.parameterizer = parameterizer or EvidenceParameterizer()

    def run(
        self,
        *,
        document_bytes: bytes,
        media_mime_type: str,
        level_view: LevelView,
        gemini_evidence: Sequence[RawEvidence] | None = None,
        gemini_status: str = "NOT_AVAILABLE",
        gemini_error: str | None = None,
        gemini_call_id: str | None = None,
        gemini_semantic_history_path: str | None = None,
    ) -> EvidencePipelineResult:
        self._validate_input(
            document_bytes=document_bytes,
            media_mime_type=media_mime_type,
            level_view=level_view,
        )

        normalized_mime = self._normalize_mime_type(media_mime_type)
        warnings: list[str] = []
        source_diagnostics: dict[str, EvidenceSourceDiagnostic] = {}

        if normalized_mime == self.PDF_MIME_TYPE:
            pymupdf_evidence = self._run_source(
                source="PYMUPDF",
                operation=lambda: self.pymupdf_adapter.extract(
                    document_bytes=document_bytes,
                    level_view=level_view,
                ),
                warnings=warnings,
                diagnostics=source_diagnostics,
            )
        else:
            pymupdf_evidence = []
            source_diagnostics["PYMUPDF"] = EvidenceSourceDiagnostic(
                status="NOT_APPLICABLE",
                count=0,
            )

        opencv_evidence = self._run_source(
            source="OPENCV",
            operation=lambda: self.opencv_adapter.extract(level_view=level_view),
            warnings=warnings,
            diagnostics=source_diagnostics,
        )

        ocr_evidence = self._run_source(
            source="OCR",
            operation=lambda: self.ocr_adapter.extract(level_view=level_view),
            warnings=warnings,
            diagnostics=source_diagnostics,
        )

        supplied_gemini = list(gemini_evidence or [])
        self._validate_supplied_gemini(level_view=level_view, evidence=supplied_gemini)
        normalized_gemini_status = str(gemini_status or "NOT_AVAILABLE").strip().upper()
        if supplied_gemini and normalized_gemini_status not in {"SUCCEEDED", "REPLAY"}:
            normalized_gemini_status = "SUCCEEDED"
        if not supplied_gemini and normalized_gemini_status == "SUCCEEDED":
            normalized_gemini_status = "SUCCEEDED"

        source_diagnostics["GEMINI"] = EvidenceSourceDiagnostic(
            status=normalized_gemini_status,
            count=len(supplied_gemini),
            error=gemini_error,
        )
        if normalized_gemini_status == "FAILED":
            warnings.append(
                "GEMINI no estuvo disponible para esta página. "
                "F01.5 conserva PyMuPDF/OpenCV/OCR y continúa sin fabricar semántica."
            )

        raw_evidence = [
            *pymupdf_evidence,
            *opencv_evidence,
            *ocr_evidence,
            *supplied_gemini,
        ]
        self._validate_evidence_identity(
            level_view=level_view,
            evidence=raw_evidence,
        )

        # Parametrización general posterior a la extracción.
        # No altera ni descarta geometry/text/metadata de ninguna fuente.
        evidence = self.parameterizer.parameterize_many(raw_evidence)
        self._validate_evidence_identity(level_view=level_view, evidence=evidence)

        invalid_geometry_count = 0
        for item in evidence:
            validation = self.geometry_validator.validate(
                evidence=item,
                level_view=level_view,
            )
            if validation.valid:
                continue
            invalid_geometry_count += 1
            warnings.extend(validation.warnings)

        kind_counter = Counter(str(item.kind) for item in evidence)
        parameter_counts_by_source = {
            source: sum(
                len(item.parameters)
                for item in evidence
                if item.source == source
            )
            for source in ("PYMUPDF", "OPENCV", "OCR", "GEMINI")
        }

        diagnostics = EvidencePipelineDiagnostics(
            pymupdf_count=len(pymupdf_evidence),
            opencv_count=len(opencv_evidence),
            ocr_count=len(ocr_evidence),
            gemini_count=len(supplied_gemini),
            total_count=len(evidence),
            invalid_geometry_count=invalid_geometry_count,
            kind_counts=dict(sorted(kind_counter.items())),
            source_diagnostics=source_diagnostics,
            gemini_call_id=gemini_call_id,
            gemini_semantic_history_path=gemini_semantic_history_path,
            parameter_count=sum(len(item.parameters) for item in evidence),
            parameter_counts_by_source=parameter_counts_by_source,
        )

        if not evidence:
            warnings.append(
                f"Fase 01.5 no produjo evidencia para LevelView {level_view.id}."
            )

        failed_sources = [
            source
            for source, diagnostic in source_diagnostics.items()
            if diagnostic.status == "FAILED"
        ]
        if failed_sources:
            warnings.append(
                "Fase 01.5 terminó con fuentes fallidas: "
                + ", ".join(sorted(failed_sources))
                + ". La evidencia de las demás fuentes fue conservada."
            )

        result = EvidencePipelineResult(
            level_view_id=level_view.id,
            source_document_id=level_view.source_document_id,
            source_page_number=level_view.source_page_number,
            evidence=evidence,
            diagnostics=diagnostics,
            warnings=self._deduplicate_warnings(warnings),
        )
        self._validate_output(result=result, level_view=level_view)
        return result

    @staticmethod
    def _run_source(
        *,
        source: str,
        operation: Callable[[], list[RawEvidence]],
        warnings: list[str],
        diagnostics: dict[str, EvidenceSourceDiagnostic],
    ) -> list[RawEvidence]:
        try:
            evidence = list(operation())
            diagnostics[source] = EvidenceSourceDiagnostic(
                status="SUCCEEDED",
                count=len(evidence),
            )
            return evidence
        except Exception as exc:
            diagnostics[source] = EvidenceSourceDiagnostic(
                status="FAILED",
                count=0,
                error=f"{type(exc).__name__}: {exc}",
            )
            warnings.append(f"{source} falló: {type(exc).__name__}: {exc}")
            return []

    def _validate_input(
        self,
        *,
        document_bytes: bytes,
        media_mime_type: str,
        level_view: LevelView,
    ) -> None:
        if not document_bytes:
            raise ValueError("EvidencePipeline requiere document_bytes.")
        normalized_mime = self._normalize_mime_type(media_mime_type)
        supported = {self.PDF_MIME_TYPE, *self.IMAGE_MIME_TYPES}
        if normalized_mime not in supported:
            raise ValueError(
                "Tipo de documento no soportado por EvidencePipeline: "
                f"{normalized_mime or 'desconocido'}."
            )
        if not level_view.id:
            raise ValueError("EvidencePipeline recibió LevelView.id vacío.")
        if not level_view.raster_bytes:
            raise ValueError("EvidencePipeline recibió un LevelView sin raster_bytes.")

    @staticmethod
    def _validate_supplied_gemini(
        *,
        level_view: LevelView,
        evidence: Sequence[RawEvidence],
    ) -> None:
        for item in evidence:
            if item.level_view_id != level_view.id:
                raise ValueError(
                    f"Evidencia Gemini {item.id} pertenece a otro LevelView."
                )
            if item.source != "GEMINI" or item.kind != "GEMINI_OBSERVATION":
                raise ValueError(
                    f"Evidencia suministrada como Gemini tiene tipo inválido: {item.id}."
                )
            if item.confirmed:
                raise ValueError(f"Evidencia Gemini {item.id} llegó confirmed=True.")

    @staticmethod
    def _validate_evidence_identity(
        *,
        level_view: LevelView,
        evidence: list[RawEvidence],
    ) -> None:
        known_ids: set[str] = set()
        for item in evidence:
            if item.level_view_id != level_view.id:
                raise ValueError(
                    f"Evidencia {item.id} pertenece a LevelView {item.level_view_id}; "
                    f"se esperaba {level_view.id}."
                )
            if item.id in known_ids:
                raise ValueError(f"Fase 01.5 produjo ID duplicado: {item.id}.")
            known_ids.add(item.id)

    @staticmethod
    def _validate_output(
        *,
        result: EvidencePipelineResult,
        level_view: LevelView,
    ) -> None:
        if result.level_view_id != level_view.id:
            raise RuntimeError("EvidencePipelineResult quedó asociado a otro LevelView.")
        if result.source_page_number != level_view.source_page_number:
            raise RuntimeError("EvidencePipelineResult quedó asociado a otra página.")
        if result.diagnostics.total_count != len(result.evidence):
            raise RuntimeError("diagnostics.total_count no coincide con evidence.")
        for item in result.evidence:
            if item.level_view_id != result.level_view_id:
                raise RuntimeError(f"Evidencia {item.id} salió asociada a otro LevelView.")
            if item.confirmed:
                raise RuntimeError(f"Evidencia automática {item.id} salió confirmed=True.")

    def _normalize_mime_type(self, value: Any) -> str:
        normalized = str(value or "").strip().lower()
        return self.IMAGE_MIME_ALIASES.get(normalized, normalized)

    @staticmethod
    def _deduplicate_warnings(warnings: list[str]) -> list[str]:
        return list(dict.fromkeys(warning for warning in warnings if warning))
