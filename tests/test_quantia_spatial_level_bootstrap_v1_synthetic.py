from __future__ import annotations

from io import BytesIO

from PIL import Image

from app.quantia_spatialV1.ai.schemas.gemini_extraction import (
    GeminiDiscoveredLevel,
    GeminiLevelDiscoveryResponse,
)
from app.quantia_spatialV1.core.models.project_site_context import ProjectSiteContext
from app.quantia_spatialV1.stages.levels.level_bootstrap_resolver import (
    LevelBootstrapResolver,
)
from app.quantia_spatialV1.stages.levels.level_identification_service import (
    LevelIdentificationResult,
    LevelIdentificationService,
)


TRACE_ID = "QSV1-CONT-20261002-A"


def _png_bytes(width: int = 320, height: int = 200) -> bytes:
    image = Image.new("RGB", (width, height), "white")
    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def _empty_identification() -> LevelIdentificationResult:
    return LevelIdentificationResult(
        source_document_id="DOC",
        source_page_number=1,
        source_page_width_px=320,
        source_page_height_px=200,
        detections=[],
        level_views=[],
        unresolved=[],
        warnings=[],
    )


def test_single_filename_candidate_enables_review_fallback_without_publishing_name() -> None:
    decision = LevelBootstrapResolver().resolve_page(
        page_number=1,
        page_count=1,
        source_file_name="PlantaBaja Miguel V.pdf",
    )

    assert decision.metadata_level_candidates == ["Planta Baja"]
    assert decision.known_level_names == []
    assert decision.fallback_full_page is True
    assert decision.fallback_level_name is None
    assert decision.state == "REVIEW"
    assert "DOCUMENT_METADATA" in decision.sources


def test_multiple_filename_candidates_require_localization_and_stay_review() -> None:
    decision = LevelBootstrapResolver().resolve_page(
        page_number=1,
        page_count=1,
        source_file_name="arquitectonico pb-pa migue-h.pdf",
    )

    assert decision.metadata_level_candidates == ["Planta Baja", "Planta Alta"]
    assert decision.known_level_names == ["Planta Baja", "Planta Alta"]
    assert decision.fallback_full_page is False
    assert decision.state == "REVIEW"
    assert decision.sources == ["DOCUMENT_METADATA"]


def test_explicit_level_wins_over_conflicting_document_metadata() -> None:
    decision = LevelBootstrapResolver().resolve_page(
        page_number=1,
        page_count=1,
        explicit_level_names=["Planta Baja"],
        source_file_name="PlantaAlta.pdf",
    )

    assert decision.known_level_names == ["Planta Baja"]
    assert decision.state == "RESOLVED"
    assert "USER_DECLARED" in decision.sources
    assert decision.warnings


def test_project_single_level_enables_controlled_full_page_fallback() -> None:
    decision = LevelBootstrapResolver().resolve_page(
        page_number=1,
        page_count=1,
        project_site_context=ProjectSiteContext(level_count=1),
        source_file_name="arquitectonico.pdf",
    )

    assert decision.known_level_names == []
    assert decision.fallback_full_page is True
    assert decision.fallback_level_name is None
    assert decision.state == "REVIEW"
    assert "PROJECT_CONTEXT" in decision.sources


def test_single_unnamed_gemini_level_enables_fallback_without_inventing_name() -> None:
    discovery = GeminiLevelDiscoveryResponse(
        pagina=1,
        niveles=[],
        plantas_sin_nombre=1,
    )
    decision = LevelBootstrapResolver().resolve_page(
        page_number=1,
        page_count=1,
        discovery=discovery,
    )

    assert decision.fallback_full_page is True
    assert decision.fallback_level_name is None
    assert decision.state == "REVIEW"
    assert "GEMINI" in decision.sources


def test_single_named_gemini_level_can_fallback_as_inferred_if_localization_fails() -> None:
    discovery = GeminiLevelDiscoveryResponse(
        pagina=1,
        niveles=[
            GeminiDiscoveredLevel(
                nombre="Planta Baja",
                confianza=0.92,
            )
        ],
        plantas_sin_nombre=0,
    )
    decision = LevelBootstrapResolver().resolve_page(
        page_number=1,
        page_count=1,
        discovery=discovery,
    )

    assert decision.known_level_names == ["Planta Baja"]
    assert decision.fallback_full_page is True
    assert decision.fallback_level_name == "Planta Baja"
    assert decision.state == "REVIEW"


def test_without_evidence_bootstrap_remains_unresolved() -> None:
    decision = LevelBootstrapResolver().resolve_page(
        page_number=1,
        page_count=1,
        source_file_name="arquitectonico.pdf",
    )

    assert decision.known_level_names == []
    assert decision.fallback_full_page is False
    assert decision.state == "UNRESOLVED"
    assert decision.sources == ["UNRESOLVED"]


def test_full_page_fallback_without_name_publishes_candidate_not_detected() -> None:
    result = LevelIdentificationService().apply_full_page_fallback(
        previous_result=_empty_identification(),
        source_raster_bytes=_png_bytes(),
        source_document_id="DOC",
        level_name=None,
        evidence_sources=["DOCUMENT_METADATA"],
        reason="single_candidate",
    )

    assert len(result.level_views) == 1
    view = result.level_views[0]
    assert view.level_name is None
    assert view.state == "CANDIDATO"
    assert view.source_bbox_px.x_min == 0
    assert view.source_bbox_px.y_min == 0
    assert view.source_bbox_px.x_max == 320
    assert view.source_bbox_px.y_max == 200
    assert view.evidence[-1].source == "LEVEL_BOOTSTRAP"


def test_full_page_fallback_with_supported_name_publishes_inferred_not_detected() -> None:
    result = LevelIdentificationService().apply_full_page_fallback(
        previous_result=_empty_identification(),
        source_raster_bytes=_png_bytes(),
        source_document_id="DOC",
        level_name="Planta Baja",
        evidence_sources=["GEMINI"],
        reason="gemini_single_named_level",
    )

    assert len(result.level_views) == 1
    view = result.level_views[0]
    assert view.level_name == "Planta Baja"
    assert view.state == "INFERIDO"
