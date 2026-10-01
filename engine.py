from __future__ import annotations

from collections import defaultdict
from statistics import median
from io import BytesIO
from typing import Any

from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, Field, ValidationError

from app.quantia_spatialV1.models.level_view import LevelView
from app.quantia_spatialV1.models.project_site_context import ProjectSiteContext
from app.quantia_spatialV1.phase_01_level.gemini_level_localization_service import (
    GeminiLevelLocalizationService,
    GeminiLevelLocalizationServiceError,
)
from app.quantia_spatialV1.phase_01_level.level_identification_service import (
    LevelIdentificationResult,
    LevelIdentificationService,
)
from app.quantia_spatialV1.phase_01_level.pdf_level_identification_service import (
    PDFLevelIdentificationService,
)
from app.quantia_spatialV1.phase_01_level.pymupdf_level_source import (
    PyMuPDFLevelSource,
)
from app.quantia_spatialV1.phase_01_level.level_view_rerasterizer import (
    LevelViewRerasterizer,
)
from app.quantia_spatialV1.phase_015_evidence.evidence_pipeline import (
    EvidencePipeline,
    EvidencePipelineResult,
)
from app.quantia_spatialV1.phase_015_evidence.gemini_evidence_adapter import (
    GeminiEvidenceAdapter,
    GeminiPageEvidenceExtractionResult,
)
from app.quantia_spatialV1.phase_02_boundaries.perimeter_wall_pipeline import (
    PerimeterWallPipeline,
    PerimeterWallPipelineResult,
)
from app.quantia_spatialV1.phase_02_boundaries.perimeter_raster_reprojector import (
    PerimeterRasterReprojector,
)
from app.quantia_spatialV1.reconstruction_core.raster_density_policy import (
    RasterDensityPolicy,
)
from app.quantia_spatialV1.reconstruction_core.scale_evidence_resolver import (
    ScaleEvidenceResolver,
)
from app.quantia_spatialV1.prompts.extraction import build_level_discovery_prompt
from app.quantia_spatialV1.providers.vision import (
    GeminiSpatialVisionProvider,
    SpatialVisionProviderError,
)
from app.quantia_spatialV1.transport.gemini_extraction import (
    GeminiLevelDiscoveryResponse,
    get_gemini_level_discovery_schema,
)


class QuantiaSpatialEngineError(RuntimeError):
    """Error interno del motor espacial F01 -> F02."""


class QuantiaPhase01Result(BaseModel):
    source_document_id: str | None = None
    media_mime_type: str
    page_count: int
    page_results: list[LevelIdentificationResult] = Field(default_factory=list)
    level_views: list[LevelView] = Field(default_factory=list)
    discoveries: dict[int, GeminiLevelDiscoveryResponse] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)

    @property
    def level_count(self) -> int:
        return len(self.level_views)

    @property
    def unresolved_count(self) -> int:
        return sum(page.unresolved_count for page in self.page_results)


class QuantiaPhase02LevelResult(BaseModel):
    """Verdad cerrada hasta F02 para un LevelView."""

    level_view: LevelView
    evidence: EvidencePipelineResult
    perimeter: PerimeterWallPipelineResult


class MetricRasterLevelDiagnostic(BaseModel):
    level_view_id: str
    level_name: str | None = None
    source_page_number: int
    bootstrap_render_scale: float
    target_render_scale: float | None = None
    bootstrap_scale_state: str
    bootstrap_m_per_px: float | None = None
    bootstrap_px_per_m: float | None = None
    final_scale_state: str | None = None
    final_m_per_px: float | None = None
    final_px_per_m: float | None = None


class MetricRasterNormalizationResult(BaseModel):
    version: str = "CANONICAL_METRIC_RASTER_V2"
    enabled: bool = True
    state: str = "UNRESOLVED"
    target_geometry_px_per_m: float = 90.0
    page_render_scales: dict[int, float] = Field(default_factory=dict)
    levels: list[MetricRasterLevelDiagnostic] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class QuantiaSpatialEngineResult(BaseModel):
    """Resultado actual del motor: F01 + F01.5 + F02 sobre raster canónico."""

    phase_01: QuantiaPhase01Result
    levels: list[QuantiaPhase02LevelResult] = Field(default_factory=list)
    raster_normalization: MetricRasterNormalizationResult | None = None
    project_site_context: ProjectSiteContext | None = None
    warnings: list[str] = Field(default_factory=list)


class QuantiaSpatialEngine:
    """
    Motor interno vigente hasta Fase 02.

    Flujo:
        documento -> F01 LevelView[]
        página completa -> Gemini legacy (una llamada por página)
        LevelView -> PyMuPDF/OpenCV/OCR + semántica Gemini -> RawEvidence
        RawEvidence -> F02 PerimeterWallLayer

    F03-F06 permanecen fuera de este engine hasta ser adaptadas al nuevo
    PerimeterWallLayer. No se importan módulos legacy de boundary.
    """

    PDF_MIME_TYPE = "application/pdf"
    IMAGE_MIME_TYPES = {"image/png", "image/jpeg"}
    IMAGE_MIME_ALIASES = {"image/jpg": "image/jpeg"}

    def __init__(
        self,
        *,
        vision_provider: GeminiSpatialVisionProvider | None = None,
        pdf_source: PyMuPDFLevelSource | None = None,
        identification_service: LevelIdentificationService | None = None,
        localization_service: GeminiLevelLocalizationService | None = None,
        pdf_identification_service: PDFLevelIdentificationService | None = None,
        evidence_pipeline: EvidencePipeline | None = None,
        gemini_evidence_adapter: GeminiEvidenceAdapter | None = None,
        perimeter_pipeline: PerimeterWallPipeline | None = None,
    ) -> None:
        self.vision_provider = vision_provider or GeminiSpatialVisionProvider()
        self.pdf_source = pdf_source or PyMuPDFLevelSource()
        self.identification_service = identification_service or LevelIdentificationService()
        self.localization_service = localization_service or GeminiLevelLocalizationService(
            provider=self.vision_provider
        )
        self.pdf_identification_service = pdf_identification_service or PDFLevelIdentificationService(
            pdf_source=self.pdf_source,
            identification_service=self.identification_service,
            localization_service=self.localization_service,
        )
        self.evidence_pipeline = evidence_pipeline or EvidencePipeline()
        self.gemini_evidence_adapter = gemini_evidence_adapter or GeminiEvidenceAdapter(
            provider=self.vision_provider
        )
        self.perimeter_pipeline = perimeter_pipeline or PerimeterWallPipeline()
        self.level_view_rerasterizer = LevelViewRerasterizer(pdf_source=self.pdf_source)
        self.perimeter_raster_reprojector = PerimeterRasterReprojector()

    def run(
        self,
        *,
        document_bytes: bytes,
        media_mime_type: str,
        source_document_id: str,
        render_scale: float | None = None,
        known_level_names_by_page: dict[int, list[str]] | None = None,
        isolated_pages: set[int] | None = None,
        discovery_payloads_by_page: dict[int, dict[str, Any]] | None = None,
        localization_payloads_by_page: dict[int, dict[str, Any]] | None = None,
        gemini_extraction_payloads_by_page: dict[int, dict[str, Any]] | None = None,
        project_site_context: ProjectSiteContext | dict[str, Any] | None = None,
        enable_gemini_discovery: bool = True,
        enable_gemini_extraction: bool = True,
        enable_metric_raster_normalization: bool = True,
        target_geometry_px_per_m: float = 90.0,
    ) -> QuantiaSpatialEngineResult:
        """Ejecuta F01 -> F01.5 -> F02 con normalización raster canónica.

        Para PDF el `render_scale` recibido es bootstrap. F01 identifica/localiza
        niveles sobre ese raster; F01.5/F02 producen la evidencia mínima necesaria
        para resolver escala. Después el motor rerasteriza la página a la densidad
        arquitectónica objetivo, regenera F01.5 y REPROYECTA la verdad F02 validada
        al nuevo sistema px sin volver a detectar el perímetro. La respuesta Gemini
        también se reproyecta: no existe una segunda llamada al modelo.
        """
        source_id = str(source_document_id or "").strip()
        if not source_id:
            raise QuantiaSpatialEngineError(
                "El motor requiere source_document_id explícito."
            )
        if target_geometry_px_per_m <= 0:
            raise QuantiaSpatialEngineError(
                "target_geometry_px_per_m debe ser mayor que cero."
            )

        site_context = ProjectSiteContext.coerce(project_site_context)

        normalized_mime = self._normalize_mime_type(media_mime_type)
        phase_01_bootstrap = self.run_phase_01(
            document_bytes=document_bytes,
            media_mime_type=normalized_mime,
            source_document_id=source_id,
            render_scale=render_scale,
            known_level_names_by_page=known_level_names_by_page,
            isolated_pages=isolated_pages,
            discovery_payloads_by_page=discovery_payloads_by_page,
            localization_payloads_by_page=localization_payloads_by_page,
            enable_gemini_discovery=enable_gemini_discovery,
        )

        page_rasters = self._prepare_page_rasters(
            document_bytes=document_bytes,
            media_mime_type=normalized_mime,
            render_scale=render_scale,
        )
        gemini_by_page, gemini_failures, gemini_warnings = self._extract_page_gemini(
            phase_01=phase_01_bootstrap,
            page_rasters=page_rasters,
            source_document_id=source_id,
            replay_by_page=gemini_extraction_payloads_by_page or {},
            enable_gemini_extraction=enable_gemini_extraction,
            project_site_context=site_context,
        )

        bootstrap_levels, bootstrap_warnings = self._run_f015_f02(
            document_bytes=document_bytes,
            media_mime_type=normalized_mime,
            level_views=phase_01_bootstrap.level_views,
            gemini_by_page=gemini_by_page,
            gemini_failures=gemini_failures,
        )
        warnings = [
            *phase_01_bootstrap.warnings,
            *gemini_warnings,
            *bootstrap_warnings,
        ]

        if (
            normalized_mime != self.PDF_MIME_TYPE
            or not enable_metric_raster_normalization
            or render_scale is None
            or not phase_01_bootstrap.level_views
        ):
            normalization = MetricRasterNormalizationResult(
                enabled=bool(enable_metric_raster_normalization),
                state=(
                    "NOT_APPLICABLE"
                    if normalized_mime != self.PDF_MIME_TYPE
                    else "DISABLED"
                    if not enable_metric_raster_normalization
                    else "UNRESOLVED"
                ),
                target_geometry_px_per_m=float(target_geometry_px_per_m),
                warnings=(
                    ["La normalización métrica canónica requiere PDF y LevelView resolubles."]
                    if normalized_mime == self.PDF_MIME_TYPE
                    and enable_metric_raster_normalization
                    and not phase_01_bootstrap.level_views
                    else []
                ),
            )
            return QuantiaSpatialEngineResult(
                phase_01=phase_01_bootstrap,
                levels=bootstrap_levels,
                raster_normalization=normalization,
                project_site_context=site_context,
                warnings=self._dedupe_warnings([*warnings, *normalization.warnings]),
            )

        normalization, page_targets = self._plan_metric_raster_normalization(
            bootstrap_levels=bootstrap_levels,
            bootstrap_render_scale=float(render_scale),
            target_geometry_px_per_m=float(target_geometry_px_per_m),
        )
        warnings.extend(normalization.warnings)

        if not page_targets:
            return QuantiaSpatialEngineResult(
                phase_01=phase_01_bootstrap,
                levels=bootstrap_levels,
                raster_normalization=normalization,
                project_site_context=site_context,
                warnings=self._dedupe_warnings(warnings),
            )

        phase_01_final = self._rerasterize_phase_01(
            document_bytes=document_bytes,
            phase_01=phase_01_bootstrap,
            page_targets=page_targets,
        )
        phase_01_final = self._publish_metric_scale_to_phase_01(
            bootstrap_phase_01=phase_01_bootstrap,
            final_phase_01=phase_01_final,
            normalization=normalization,
        )
        final_gemini = self._reproject_gemini_to_phase_01(
            phase_01=phase_01_final,
            source_gemini=gemini_by_page,
        )
        final_evidence_by_id, final_warnings = self._run_f015_only(
            document_bytes=document_bytes,
            media_mime_type=normalized_mime,
            level_views=phase_01_final.level_views,
            gemini_by_page=final_gemini,
            gemini_failures=gemini_failures,
        )
        warnings.extend(final_warnings)
        final_levels, reproject_warnings = self._reproject_bootstrap_f02(
            bootstrap_levels=bootstrap_levels,
            final_level_views=phase_01_final.level_views,
            final_evidence_by_id=final_evidence_by_id,
        )
        warnings.extend(reproject_warnings)

        normalization = self._validate_metric_raster_normalization(
            normalization=normalization,
            final_levels=final_levels,
            page_targets=page_targets,
        )
        warnings.extend(normalization.warnings)

        return QuantiaSpatialEngineResult(
            phase_01=phase_01_final,
            levels=final_levels,
            raster_normalization=normalization,
            project_site_context=site_context,
            warnings=self._dedupe_warnings(warnings),
        )

    def _extract_page_gemini(
        self,
        *,
        phase_01: QuantiaPhase01Result,
        page_rasters: dict[int, tuple[bytes, str]],
        source_document_id: str,
        replay_by_page: dict[int, dict[str, Any]],
        enable_gemini_extraction: bool,
        project_site_context: ProjectSiteContext | None,
    ) -> tuple[dict[int, GeminiPageEvidenceExtractionResult], dict[int, str], list[str]]:
        views_by_page: dict[int, list[LevelView]] = defaultdict(list)
        for level_view in phase_01.level_views:
            views_by_page[level_view.source_page_number].append(level_view)

        results: dict[int, GeminiPageEvidenceExtractionResult] = {}
        failures: dict[int, str] = {}
        warnings: list[str] = []
        for page_number, views in sorted(views_by_page.items()):
            page_info = page_rasters.get(page_number)
            if page_info is None:
                raise QuantiaSpatialEngineError(
                    f"No existe raster fuente para página {page_number}."
                )
            replay_payload = replay_by_page.get(page_number)
            if not enable_gemini_extraction and replay_payload is None:
                continue
            try:
                results[page_number] = self.gemini_evidence_adapter.extract_page_with_trace(
                    page_raster_bytes=page_info[0],
                    page_raster_mime_type=page_info[1],
                    source_document_id=source_document_id,
                    source_page_number=page_number,
                    level_views=views,
                    replay_payload=replay_payload,
                    project_site_context=project_site_context,
                )
            except Exception as exc:
                message = f"{type(exc).__name__}: {exc}"
                failures[page_number] = message
                warnings.append(
                    f"Página {page_number}: Gemini extraction no disponible; "
                    "F01.5/F02 continúan con PyMuPDF/OpenCV/OCR. " + message
                )
        return results, failures, warnings

    def _run_f015_only(
        self,
        *,
        document_bytes: bytes,
        media_mime_type: str,
        level_views: list[LevelView],
        gemini_by_page: dict[int, GeminiPageEvidenceExtractionResult],
        gemini_failures: dict[int, str],
    ) -> tuple[dict[str, EvidencePipelineResult], list[str]]:
        results: dict[str, EvidencePipelineResult] = {}
        warnings: list[str] = []
        for level_view in level_views:
            page_gemini = gemini_by_page.get(level_view.source_page_number)
            if page_gemini is not None:
                gemini_evidence = page_gemini.evidence_by_level_view.get(level_view.id, [])
                gemini_status = "REPLAY" if page_gemini.replay_used else "SUCCEEDED"
                gemini_error = None
                gemini_call_id = page_gemini.call_id
                history_path = page_gemini.history_path
            else:
                gemini_evidence = []
                gemini_status = (
                    "FAILED"
                    if level_view.source_page_number in gemini_failures
                    else "NOT_AVAILABLE"
                )
                gemini_error = gemini_failures.get(level_view.source_page_number)
                gemini_call_id = None
                history_path = None

            try:
                evidence_result = self.evidence_pipeline.run(
                    document_bytes=document_bytes,
                    media_mime_type=media_mime_type,
                    level_view=level_view,
                    gemini_evidence=gemini_evidence,
                    gemini_status=gemini_status,
                    gemini_error=gemini_error,
                    gemini_call_id=gemini_call_id,
                    gemini_semantic_history_path=history_path,
                )
            except (ValueError, RuntimeError) as exc:
                raise QuantiaSpatialEngineError(
                    f"Fase 01.5 falló en LevelView {level_view.id}: {exc}"
                ) from exc

            results[level_view.id] = evidence_result
            warnings.extend(evidence_result.warnings)
        return results, warnings

    def _run_f015_f02(
        self,
        *,
        document_bytes: bytes,
        media_mime_type: str,
        level_views: list[LevelView],
        gemini_by_page: dict[int, GeminiPageEvidenceExtractionResult],
        gemini_failures: dict[int, str],
    ) -> tuple[list[QuantiaPhase02LevelResult], list[str]]:
        evidence_by_id, warnings = self._run_f015_only(
            document_bytes=document_bytes,
            media_mime_type=media_mime_type,
            level_views=level_views,
            gemini_by_page=gemini_by_page,
            gemini_failures=gemini_failures,
        )
        level_results: list[QuantiaPhase02LevelResult] = []
        for level_view in level_views:
            evidence_result = evidence_by_id[level_view.id]
            try:
                perimeter_result = self.perimeter_pipeline.run(
                    level_view=level_view,
                    evidence=evidence_result.evidence,
                )
            except (ValueError, RuntimeError) as exc:
                raise QuantiaSpatialEngineError(
                    f"Fase 02 falló en LevelView {level_view.id}: {exc}"
                ) from exc

            warnings.extend(perimeter_result.warnings)
            level_results.append(
                QuantiaPhase02LevelResult(
                    level_view=level_view,
                    evidence=evidence_result,
                    perimeter=perimeter_result,
                )
            )
        return level_results, warnings

    def _publish_metric_scale_to_phase_01(
        self,
        *,
        bootstrap_phase_01: QuantiaPhase01Result,
        final_phase_01: QuantiaPhase01Result,
        normalization: MetricRasterNormalizationResult,
    ) -> QuantiaPhase01Result:
        bootstrap_by_id = {item.id: item for item in bootstrap_phase_01.level_views}
        diagnostic_by_id = {item.level_view_id: item for item in normalization.levels}
        final_by_id: dict[str, LevelView] = {}

        for level_view in final_phase_01.level_views:
            source = bootstrap_by_id.get(level_view.id)
            diagnostic = diagnostic_by_id.get(level_view.id)
            if (
                source is None
                or diagnostic is None
                or diagnostic.bootstrap_m_per_px is None
            ):
                final_by_id[level_view.id] = level_view
                continue

            sx = float(level_view.source_page_width_px) / float(source.source_page_width_px)
            sy = float(level_view.source_page_height_px) / float(source.source_page_height_px)
            pixel_scale = (sx * sy) ** 0.5
            if pixel_scale <= 0.0:
                final_by_id[level_view.id] = level_view
                continue

            metric_scale = float(diagnostic.bootstrap_m_per_px) / pixel_scale
            final_by_id[level_view.id] = level_view.model_copy(
                deep=True,
                update={
                    "metric_scale_m_per_px": metric_scale,
                    "metric_scale_source": "CANONICAL_METRIC_RASTER_V2",
                },
            )

        page_results: list[LevelIdentificationResult] = []
        for page_result in final_phase_01.page_results:
            page_results.append(
                page_result.model_copy(
                    deep=True,
                    update={
                        "level_views": [final_by_id[item.id] for item in page_result.level_views]
                    },
                )
            )

        return final_phase_01.model_copy(
            deep=True,
            update={
                "page_results": page_results,
                "level_views": [final_by_id[item.id] for item in final_phase_01.level_views],
            },
        )

    def _reproject_bootstrap_f02(
        self,
        *,
        bootstrap_levels: list[QuantiaPhase02LevelResult],
        final_level_views: list[LevelView],
        final_evidence_by_id: dict[str, EvidencePipelineResult],
    ) -> tuple[list[QuantiaPhase02LevelResult], list[str]]:
        bootstrap_by_id = {item.level_view.id: item for item in bootstrap_levels}
        warnings: list[str] = []
        results: list[QuantiaPhase02LevelResult] = []

        for level_view in final_level_views:
            source = bootstrap_by_id.get(level_view.id)
            if source is None:
                raise QuantiaSpatialEngineError(
                    f"No existe F02 bootstrap para LevelView {level_view.id}."
                )
            evidence_result = final_evidence_by_id.get(level_view.id)
            if evidence_result is None:
                raise QuantiaSpatialEngineError(
                    f"No existe F01.5 final para LevelView {level_view.id}."
                )

            perimeter_result = self.perimeter_raster_reprojector.reproject(
                source_result=source.perimeter,
                source_level_view=source.level_view,
                target_level_view=level_view,
            )
            validation = self.perimeter_pipeline.validator.validate(
                level_view=level_view,
                resolution=perimeter_result.resolution,
                wall_layer=perimeter_result.wall_layer,
                grounding=perimeter_result.grounding,
                wall_graph=perimeter_result.wall_graph,
            )
            perimeter_result = perimeter_result.model_copy(
                deep=True,
                update={
                    "validation": validation,
                    "state": validation.state,
                    "warnings": self._dedupe_warnings(
                        [
                            *perimeter_result.warnings,
                            *validation.warnings,
                            *validation.errors,
                        ]
                    ),
                },
            )
            warnings.extend(perimeter_result.warnings)
            results.append(
                QuantiaPhase02LevelResult(
                    level_view=level_view,
                    evidence=evidence_result,
                    perimeter=perimeter_result,
                )
            )

        return results, warnings

    def _plan_metric_raster_normalization(
        self,
        *,
        bootstrap_levels: list[QuantiaPhase02LevelResult],
        bootstrap_render_scale: float,
        target_geometry_px_per_m: float,
    ) -> tuple[MetricRasterNormalizationResult, dict[int, float]]:
        resolver = ScaleEvidenceResolver()
        policy = RasterDensityPolicy(
            target_geometry_px_per_m=target_geometry_px_per_m
        )
        diagnostics: list[MetricRasterLevelDiagnostic] = []
        recommendations_by_page: dict[int, list[float]] = defaultdict(list)
        page_level_counts: dict[int, int] = defaultdict(int)
        warnings: list[str] = []

        for item in bootstrap_levels:
            level_view = item.level_view
            page_number = level_view.source_page_number
            page_level_counts[page_number] += 1
            perimeter = item.perimeter.editable_perimeter
            scale = resolver.resolve(
                level_view=level_view,
                evidence=item.evidence.evidence,
                perimeter=perimeter,
            )
            target_render_scale: float | None = None
            bootstrap_px_per_m = (
                1.0 / scale.selected_m_per_px
                if scale.selected_m_per_px is not None
                else None
            )
            if scale.is_resolved and scale.current_render_scale is not None:
                recommendation = policy.recommend_from_grounded_scale(
                    current_render_scale=scale.current_render_scale,
                    current_m_per_px=scale.selected_m_per_px,
                )
                target_render_scale = recommendation.recommended_render_scale
                recommendations_by_page[page_number].append(target_render_scale)
            else:
                warnings.append(
                    f"Página {page_number} / {level_view.level_name or level_view.id}: "
                    f"escala bootstrap {scale.state}; no puede publicarse raster normalizado."
                )

            diagnostics.append(
                MetricRasterLevelDiagnostic(
                    level_view_id=level_view.id,
                    level_name=level_view.level_name,
                    source_page_number=page_number,
                    bootstrap_render_scale=float(bootstrap_render_scale),
                    target_render_scale=target_render_scale,
                    bootstrap_scale_state=scale.state,
                    bootstrap_m_per_px=scale.selected_m_per_px,
                    bootstrap_px_per_m=bootstrap_px_per_m,
                )
            )

        page_targets: dict[int, float] = {}
        for page_number, values in sorted(recommendations_by_page.items()):
            if len(values) != page_level_counts[page_number]:
                continue
            chosen = float(median(values))
            spread = max(abs(value - chosen) / chosen for value in values) if values else 0.0
            if spread > resolver.STRONG_CONSISTENCY_TOL:
                warnings.append(
                    f"Página {page_number}: los LevelView requieren densidades incompatibles "
                    f"(dispersión relativa {spread:.4f}); no se fuerza una escala de página."
                )
                continue
            page_targets[page_number] = chosen

        if not diagnostics:
            state = "UNRESOLVED"
        elif len(page_targets) == len(page_level_counts):
            state = "PLANNED"
        elif page_targets:
            state = "PARTIAL"
        else:
            state = "UNRESOLVED"

        return (
            MetricRasterNormalizationResult(
                enabled=True,
                state=state,
                target_geometry_px_per_m=target_geometry_px_per_m,
                page_render_scales=dict(page_targets),
                levels=diagnostics,
                warnings=self._dedupe_warnings(warnings),
            ),
            page_targets,
        )

    def _rerasterize_phase_01(
        self,
        *,
        document_bytes: bytes,
        phase_01: QuantiaPhase01Result,
        page_targets: dict[int, float],
    ) -> QuantiaPhase01Result:
        page_results: list[LevelIdentificationResult] = []
        level_views: list[LevelView] = []
        for page_result in phase_01.page_results:
            target = page_targets.get(page_result.source_page_number)
            if target is None or not page_result.level_views:
                page_results.append(page_result)
                level_views.extend(page_result.level_views)
                continue
            rerasterized = self.level_view_rerasterizer.rerasterize_page(
                document_bytes=document_bytes,
                page_result=page_result,
                target_render_scale=target,
            )
            page_results.append(rerasterized.page_result)
            level_views.extend(rerasterized.level_views)

        return QuantiaPhase01Result(
            source_document_id=phase_01.source_document_id,
            media_mime_type=phase_01.media_mime_type,
            page_count=phase_01.page_count,
            page_results=page_results,
            level_views=level_views,
            discoveries=dict(phase_01.discoveries),
            warnings=list(phase_01.warnings),
        )

    def _reproject_gemini_to_phase_01(
        self,
        *,
        phase_01: QuantiaPhase01Result,
        source_gemini: dict[int, GeminiPageEvidenceExtractionResult],
    ) -> dict[int, GeminiPageEvidenceExtractionResult]:
        views_by_page: dict[int, list[LevelView]] = defaultdict(list)
        for level_view in phase_01.level_views:
            views_by_page[level_view.source_page_number].append(level_view)
        result: dict[int, GeminiPageEvidenceExtractionResult] = {}
        for page_number, source_result in source_gemini.items():
            views = views_by_page.get(page_number, [])
            if not views:
                continue
            result[page_number] = self.gemini_evidence_adapter.reproject_page_result(
                source_result=source_result,
                level_views=views,
            )
        return result

    def _validate_metric_raster_normalization(
        self,
        *,
        normalization: MetricRasterNormalizationResult,
        final_levels: list[QuantiaPhase02LevelResult],
        page_targets: dict[int, float],
    ) -> MetricRasterNormalizationResult:
        resolver = ScaleEvidenceResolver()
        diagnostics_by_id = {item.level_view_id: item for item in normalization.levels}
        warnings = list(normalization.warnings)
        validated = 0
        attempted = 0

        for item in final_levels:
            diagnostic = diagnostics_by_id.get(item.level_view.id)
            if diagnostic is None:
                continue
            if item.level_view.source_page_number not in page_targets:
                continue
            attempted += 1
            scale = resolver.resolve(
                level_view=item.level_view,
                evidence=item.evidence.evidence,
                perimeter=item.perimeter.editable_perimeter,
            )
            diagnostic.final_scale_state = scale.state
            diagnostic.final_m_per_px = scale.selected_m_per_px
            diagnostic.final_px_per_m = (
                1.0 / scale.selected_m_per_px
                if scale.selected_m_per_px is not None
                else None
            )
            if not scale.is_resolved or diagnostic.final_px_per_m is None:
                warnings.append(
                    f"{item.level_view.level_name or item.level_view.id}: "
                    "rerasterización ejecutada pero la escala final no quedó RESOLVED."
                )
                continue
            relative_error = abs(
                diagnostic.final_px_per_m - normalization.target_geometry_px_per_m
            ) / normalization.target_geometry_px_per_m
            if relative_error > resolver.STRONG_CONSISTENCY_TOL:
                warnings.append(
                    f"{item.level_view.level_name or item.level_view.id}: "
                    f"densidad final {diagnostic.final_px_per_m:.3f} px/m fuera de "
                    f"la tolerancia de {normalization.target_geometry_px_per_m:.3f} px/m."
                )
                continue
            validated += 1

        if attempted == 0:
            state = normalization.state
        elif validated == attempted and attempted == len(normalization.levels):
            state = "NORMALIZED"
        elif validated == attempted:
            state = "PARTIAL"
        elif validated:
            state = "REVIEW"
        else:
            state = "FAILED_VALIDATION"

        normalization.state = state
        normalization.warnings = self._dedupe_warnings(warnings)
        return normalization

    def _prepare_page_rasters(
        self,
        *,
        document_bytes: bytes,
        media_mime_type: str,
        render_scale: float | None,
    ) -> dict[int, tuple[bytes, str]]:
        if media_mime_type == self.PDF_MIME_TYPE:
            if render_scale is None:
                raise QuantiaSpatialEngineError(
                    "PDF requiere render_scale explícito para recuperar raster de página."
                )
            source_result = self.pdf_source.read(
                document_bytes=document_bytes,
                render_scale=render_scale,
            )
            return {
                page.page_number: (page.raster_bytes, page.raster_mime_type)
                for page in source_result.pages
            }

        if media_mime_type in self.IMAGE_MIME_TYPES:
            return {1: (self._normalize_image_to_png(document_bytes), "image/png")}

        raise QuantiaSpatialEngineError(
            f"Formato no soportado: {media_mime_type or 'desconocido'}."
        )

    # ========================================================
    # FASE 01
    # ========================================================

    def run_phase_01(
        self,
        *,
        document_bytes: bytes,
        media_mime_type: str,
        source_document_id: str | None = None,
        render_scale: float | None = None,
        known_level_names_by_page: (dict[int, list[str]] | None) = None,
        isolated_pages: (set[int] | None) = None,
        discovery_payloads_by_page: (dict[int, dict[str, Any]] | None) = None,
        localization_payloads_by_page: (dict[int, dict[str, Any]] | None) = None,
        enable_gemini_discovery: bool = True,
    ) -> QuantiaPhase01Result:
        """
        Ejecuta exclusivamente Fase 1.

        Soporta:

            PDF
            PNG
            JPEG

        Replay:

            discovery_payloads_by_page
            localization_payloads_by_page

        permite ejecutar el motor sin llamadas reales a Gemini.
        """

        if not document_bytes:
            raise QuantiaSpatialEngineError("document_bytes no puede estar vacío.")

        normalized_mime = self._normalize_mime_type(media_mime_type)

        known_by_page = known_level_names_by_page or {}

        isolated = isolated_pages or set()

        discovery_replay = discovery_payloads_by_page or {}

        localization_replay = localization_payloads_by_page or {}

        if normalized_mime == self.PDF_MIME_TYPE:
            if render_scale is None:
                raise QuantiaSpatialEngineError(
                    "Los documentos PDF requieren render_scale explícito."
                )

            return self._run_pdf_phase_01(
                document_bytes=document_bytes,
                render_scale=render_scale,
                source_document_id=(source_document_id),
                known_level_names_by_page=(known_by_page),
                isolated_pages=(isolated),
                discovery_payloads_by_page=(discovery_replay),
                localization_payloads_by_page=(localization_replay),
                enable_gemini_discovery=(enable_gemini_discovery),
            )

        if normalized_mime in self.IMAGE_MIME_TYPES:
            return self._run_image_phase_01(
                document_bytes=document_bytes,
                media_mime_type=(normalized_mime),
                source_document_id=(source_document_id),
                known_level_names_by_page=(known_by_page),
                isolated_pages=(isolated),
                discovery_payloads_by_page=(discovery_replay),
                localization_payloads_by_page=(localization_replay),
                enable_gemini_discovery=(enable_gemini_discovery),
            )

        raise QuantiaSpatialEngineError(
            f"Formato no soportado por Fase 1: {normalized_mime or 'desconocido'}."
        )

    # ========================================================
    # PDF
    # ========================================================

    def _run_pdf_phase_01(
        self,
        *,
        document_bytes: bytes,
        render_scale: float,
        source_document_id: str | None,
        known_level_names_by_page: dict[
            int,
            list[str],
        ],
        isolated_pages: set[int],
        discovery_payloads_by_page: dict[
            int,
            dict[str, Any],
        ],
        localization_payloads_by_page: dict[
            int,
            dict[str, Any],
        ],
        enable_gemini_discovery: bool,
    ) -> QuantiaPhase01Result:

        source_result = self.pdf_source.read(
            document_bytes=document_bytes,
            render_scale=render_scale,
        )

        self._validate_page_configuration(
            page_count=source_result.page_count,
            known_level_names_by_page=(known_level_names_by_page),
            isolated_pages=(isolated_pages),
            discovery_payloads_by_page=(discovery_payloads_by_page),
            localization_payloads_by_page=(localization_payloads_by_page),
        )

        resolved_known: dict[
            int,
            list[str],
        ] = {}

        discoveries: dict[
            int,
            GeminiLevelDiscoveryResponse,
        ] = {}

        warnings = list(source_result.warnings)

        # ----------------------------------------------------
        # DESCUBRIMIENTO POR PÁGINA
        # ----------------------------------------------------

        for source_page in source_result.pages:
            page_number = source_page.page_number

            explicit_names = known_level_names_by_page.get(
                page_number,
                [],
            )

            vector_names = self._vector_level_names(source_page)

            current_names = self._merge_level_names(
                explicit_names,
                vector_names,
            )

            discovery: GeminiLevelDiscoveryResponse | None = None

            replay_payload = discovery_payloads_by_page.get(page_number)

            if replay_payload is not None:
                discovery = self._validate_discovery_payload(
                    payload=(replay_payload),
                    expected_page_number=(page_number),
                )

            elif enable_gemini_discovery and self._should_discover_pdf_page(
                page_count=(source_result.page_count),
                explicit_names=(explicit_names),
                vector_names=(vector_names),
            ):
                try:
                    discovery = self._discover_levels(
                        page_number=(page_number),
                        raster_bytes=(source_page.raster_bytes),
                        raster_mime_type=(source_page.raster_mime_type),
                    )

                except QuantiaSpatialEngineError as exc:
                    warnings.append(f"Página {page_number}: {exc}")

            if discovery is not None:
                discoveries[page_number] = discovery

                current_names = self._merge_level_names(
                    current_names,
                    discovery.to_known_level_names(),
                )

                if discovery.plantas_sin_nombre > 0:
                    warnings.append(
                        f"Página {page_number}: Gemini detectó "
                        f"{discovery.plantas_sin_nombre} "
                        "planta(s) cuyo nivel no pudo "
                        "identificar sin inventar datos."
                    )

            resolved_known[page_number] = current_names

        # ----------------------------------------------------
        # IDENTIFICACIÓN / LOCALIZACIÓN
        # ----------------------------------------------------

        pdf_result = self.pdf_identification_service.identify_pdf(
            document_bytes=document_bytes,
            render_scale=render_scale,
            source_document_id=(source_document_id),
            known_level_names_by_page=(resolved_known),
            gemini_payloads_by_page=(localization_payloads_by_page),
            isolated_pages=(isolated_pages),
        )

        warnings.extend(pdf_result.warnings)

        return QuantiaPhase01Result(
            source_document_id=(source_document_id),
            media_mime_type=(self.PDF_MIME_TYPE),
            page_count=(pdf_result.page_count),
            page_results=(pdf_result.pages),
            level_views=(pdf_result.level_views),
            discoveries=(discoveries),
            warnings=(self._dedupe_warnings(warnings)),
        )

    # ========================================================
    # IMAGEN
    # ========================================================

    def _run_image_phase_01(
        self,
        *,
        document_bytes: bytes,
        media_mime_type: str,
        source_document_id: str | None,
        known_level_names_by_page: dict[
            int,
            list[str],
        ],
        isolated_pages: set[int],
        discovery_payloads_by_page: dict[
            int,
            dict[str, Any],
        ],
        localization_payloads_by_page: dict[
            int,
            dict[str, Any],
        ],
        enable_gemini_discovery: bool,
    ) -> QuantiaPhase01Result:

        self._validate_page_configuration(
            page_count=1,
            known_level_names_by_page=(known_level_names_by_page),
            isolated_pages=(isolated_pages),
            discovery_payloads_by_page=(discovery_payloads_by_page),
            localization_payloads_by_page=(localization_payloads_by_page),
        )

        raster_bytes = self._normalize_image_to_png(document_bytes)

        page_number = 1

        known_names = list(known_level_names_by_page.get(page_number, []))

        discoveries: dict[
            int,
            GeminiLevelDiscoveryResponse,
        ] = {}

        warnings: list[str] = []

        # ----------------------------------------------------
        # DISCOVERY
        # ----------------------------------------------------

        replay_discovery = discovery_payloads_by_page.get(page_number)

        discovery: GeminiLevelDiscoveryResponse | None = None

        if replay_discovery is not None:
            discovery = self._validate_discovery_payload(
                payload=(replay_discovery),
                expected_page_number=1,
            )

        elif not known_names and enable_gemini_discovery:
            try:
                discovery = self._discover_levels(
                    page_number=1,
                    raster_bytes=(raster_bytes),
                    raster_mime_type=("image/png"),
                )

            except QuantiaSpatialEngineError as exc:
                warnings.append(str(exc))

        if discovery is not None:
            discoveries[page_number] = discovery

            known_names = self._merge_level_names(
                known_names,
                discovery.to_known_level_names(),
            )

            if discovery.plantas_sin_nombre > 0:
                warnings.append(
                    "Gemini detectó "
                    f"{discovery.plantas_sin_nombre} "
                    "planta(s) cuyo nivel no pudo identificar."
                )

        # ----------------------------------------------------
        # LOCALIZACIÓN
        # ----------------------------------------------------

        isolated = page_number in isolated_pages

        localization_payload = localization_payloads_by_page.get(page_number)

        if (
            localization_payload is None
            and known_names
            and not (isolated and len(known_names) == 1)
        ):
            try:
                localization = self.localization_service.localize(
                    page_number=1,
                    expected_level_names=(known_names),
                    raster_bytes=(raster_bytes),
                    raster_mime_type=("image/png"),
                )

                localization_payload = localization.to_detector_payload()

            except GeminiLevelLocalizationServiceError as exc:
                warnings.append(f"Gemini no pudo localizar los niveles: {exc}")

        # ----------------------------------------------------
        # LEVEL VIEW
        # ----------------------------------------------------

        page_result = self.identification_service.identify_page(
            source_raster_bytes=(raster_bytes),
            source_page_number=1,
            source_document_id=(source_document_id),
            known_level_names=(known_names),
            pdf_level_markers=[],
            gemini_payload=(localization_payload),
            single_level_isolated=(isolated),
        )

        warnings.extend(page_result.warnings)

        # ----------------------------------------------------
        # PLANTA SIN NOMBRE NO AISLADA
        # ----------------------------------------------------

        if discovery is not None and discovery.has_unnamed_levels and not isolated:
            warnings.append(
                "Existen plantas sin nombre que no pueden "
                "aislarse todavía sin una referencia semántica "
                "o espacial adicional."
            )

        return QuantiaPhase01Result(
            source_document_id=(source_document_id),
            media_mime_type=(media_mime_type),
            page_count=1,
            page_results=[page_result],
            level_views=(page_result.level_views),
            discoveries=(discoveries),
            warnings=(self._dedupe_warnings(warnings)),
        )

    # ========================================================
    # GEMINI DISCOVERY
    # ========================================================

    def _discover_levels(
        self,
        *,
        page_number: int,
        raster_bytes: bytes,
        raster_mime_type: str,
    ) -> GeminiLevelDiscoveryResponse:

        prompt = build_level_discovery_prompt(page_number=(page_number))

        schema = get_gemini_level_discovery_schema()

        try:
            result = self.vision_provider.analyze(
                prompt=prompt,
                media_bytes=(raster_bytes),
                media_mime_type=(raster_mime_type),
                response_json_schema=(schema),
            )

        except SpatialVisionProviderError as exc:
            raise QuantiaSpatialEngineError(
                "Gemini no pudo descubrir los niveles de la página."
            ) from exc

        if not isinstance(
            result.data,
            dict,
        ):
            raise QuantiaSpatialEngineError(
                "Gemini no devolvió un objeto JSON "
                "durante el descubrimiento de niveles."
            )

        return self._validate_discovery_payload(
            payload=result.data,
            expected_page_number=(page_number),
        )

    # ========================================================
    # VALIDAR DISCOVERY
    # ========================================================

    @staticmethod
    def _validate_discovery_payload(
        *,
        payload: dict[str, Any],
        expected_page_number: int,
    ) -> GeminiLevelDiscoveryResponse:

        try:
            response = GeminiLevelDiscoveryResponse.model_validate(payload)

        except ValidationError as exc:
            raise QuantiaSpatialEngineError(
                "La respuesta de descubrimiento Gemini no cumple el contrato de Fase 1."
            ) from exc

        if response.pagina != expected_page_number:
            raise QuantiaSpatialEngineError(
                "El discovery Gemini corresponde "
                "a una página distinta. "
                f"Esperada={expected_page_number}, "
                f"recibida={response.pagina}."
            )

        return response

    # ========================================================
    # ¿SE NECESITA DISCOVERY EN PDF?
    # ========================================================

    @staticmethod
    def _should_discover_pdf_page(
        *,
        page_count: int,
        explicit_names: list[str],
        vector_names: list[str],
    ) -> bool:
        """
        Evita llamadas Gemini cuando la evidencia documental
        ya resuelve claramente la identificación.

        Casos donde se utiliza discovery:

        - no existe nombre conocido;
        - PDF de una sola página con un único marcador,
          porque todavía podría contener otra planta.

        Si existen varios marcadores de nivel diferentes,
        ya sabemos qué niveles deben localizarse.

        En PDF multipágina con exactamente un marcador por
        página tampoco necesitamos discovery adicional.
        """

        if explicit_names:
            return False

        if not vector_names:
            return True

        if page_count == 1 and len(vector_names) == 1:
            return True

        return False

    # ========================================================
    # MARCADORES VECTORIALES
    # ========================================================

    @staticmethod
    def _vector_level_names(
        source_page,
    ) -> list[str]:

        result: list[str] = []
        seen: set[str] = set()

        for marker in source_page.level_markers:
            key = marker.normalized_text

            if key in seen:
                continue

            seen.add(key)

            result.append(marker.text)

        return result

    # ========================================================
    # FUSIÓN DE NOMBRES
    # ========================================================

    @staticmethod
    def _merge_level_names(
        first: list[str],
        second: list[str],
    ) -> list[str]:

        result: list[str] = []
        seen: set[str] = set()

        for value in [
            *first,
            *second,
        ]:
            name = str(value or "").strip()

            if not name:
                continue

            key = " ".join(name.casefold().split())

            if key in seen:
                continue

            seen.add(key)

            result.append(name)

        return result

    # ========================================================
    # NORMALIZAR IMAGEN
    # ========================================================

    @staticmethod
    def _normalize_image_to_png(
        image_bytes: bytes,
    ) -> bytes:

        try:
            with Image.open(BytesIO(image_bytes)) as source:
                source.load()

                oriented = ImageOps.exif_transpose(source)

                normalized = oriented.convert("RGB")

                output = BytesIO()

                normalized.save(
                    output,
                    format="PNG",
                )

                normalized.close()

                if oriented is not source:
                    oriented.close()

                result = output.getvalue()

        except (
            UnidentifiedImageError,
            OSError,
            ValueError,
        ) as exc:
            raise QuantiaSpatialEngineError(
                "No fue posible interpretar la imagen de entrada."
            ) from exc

        if not result:
            raise QuantiaSpatialEngineError(
                "La normalización de imagen produjo un raster vacío."
            )

        return result

    # ========================================================
    # MIME
    # ========================================================

    @classmethod
    def _normalize_mime_type(
        cls,
        value: str,
    ) -> str:

        normalized = str(value or "").strip().lower()

        return cls.IMAGE_MIME_ALIASES.get(
            normalized,
            normalized,
        )

    # ========================================================
    # VALIDACIÓN DE PÁGINAS
    # ========================================================

    @staticmethod
    def _validate_page_configuration(
        *,
        page_count: int,
        known_level_names_by_page: dict[
            int,
            list[str],
        ],
        isolated_pages: set[int],
        discovery_payloads_by_page: dict[
            int,
            dict[str, Any],
        ],
        localization_payloads_by_page: dict[
            int,
            dict[str, Any],
        ],
    ) -> None:

        keys = {
            *known_level_names_by_page.keys(),
            *isolated_pages,
            *discovery_payloads_by_page.keys(),
            *localization_payloads_by_page.keys(),
        }

        for page_number in keys:
            if not isinstance(
                page_number,
                int,
            ):
                raise QuantiaSpatialEngineError(
                    "Las referencias de página deben ser enteros."
                )

            if page_number <= 0 or page_number > page_count:
                raise QuantiaSpatialEngineError(
                    "Se recibió configuración para "
                    "una página inexistente: "
                    f"{page_number}."
                )

    # ========================================================
    # WARNINGS
    # ========================================================

    @staticmethod
    def _dedupe_warnings(
        warnings: list[str],
    ) -> list[str]:

        result: list[str] = []
        seen: set[str] = set()

        for warning in warnings:
            value = str(warning or "").strip()

            if not value or value in seen:
                continue

            seen.add(value)

            result.append(value)

        return result


# ============================================================
# FACTORY
# ============================================================


def get_quantia_spatial_engine() -> QuantiaSpatialEngine:
    return QuantiaSpatialEngine()
