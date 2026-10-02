from __future__ import annotations

from io import BytesIO

from PIL import Image

from app.quantia_spatialV1.core.models.evidence import EvidenceGeometry, RawEvidence
from app.quantia_spatialV1.core.models.level_view import LevelView, LevelViewTransform, PixelBBox
from app.quantia_spatialV1.core.scale.scale_evidence_resolver import ScaleEvidenceResolver


def _png(width: int = 1000, height: int = 500) -> bytes:
    image = Image.new("L", (width, height), 255)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _level() -> LevelView:
    width, height = 1000, 500
    bbox = PixelBBox(x_min=0, y_min=0, x_max=width, y_max=height)
    return LevelView(
        id="LV_SCALE_V17",
        level_name="Planta Test",
        source_document_id="DOC",
        source_page_number=1,
        source_bbox_px=bbox,
        source_page_width_px=width,
        source_page_height_px=height,
        raster_width_px=width,
        raster_height_px=height,
        raster_mime_type="image/png",
        raster_bytes=_png(width, height),
        transform=LevelViewTransform(
            offset_x_px=0,
            offset_y_px=0,
            source_page_width_px=width,
            source_page_height_px=height,
            local_width_px=width,
            local_height_px=height,
        ),
        state="DETECTADO",
        confidence=0.95,
        evidence=[],
    )


def _declared_scale(eid: str, text: str, *, render_scale: float = 1.5875) -> RawEvidence:
    return RawEvidence(
        id=eid,
        level_view_id="LV_SCALE_V17",
        source="PYMUPDF",
        kind="VECTOR_TEXT",
        geometry=EvidenceGeometry(geometry_type="NONE"),
        text=text,
        confidence=1.0,
        metadata={
            "page_to_raster_scale_x": render_scale,
            "page_to_raster_scale_y": render_scale,
        },
    )


def test_scale_resolver_contract_is_v1_7() -> None:
    assert ScaleEvidenceResolver.VERSION == "SCALE_EVIDENCE_RESOLVER_V1_7"


def test_scale_resolver_without_metric_evidence_is_unresolved() -> None:
    result = ScaleEvidenceResolver().resolve(level_view=_level(), evidence=[])
    assert result.state == "UNRESOLVED"
    assert result.selected_m_per_px is None


def test_declared_pdf_scale_resolves_through_public_contract() -> None:
    result = ScaleEvidenceResolver().resolve(
        level_view=_level(),
        evidence=[_declared_scale("S1", "ESCALA 1:50")],
    )
    assert result.state == "RESOLVED"
    assert result.selected_method == "DECLARED_SCALE_PDF"
    assert result.selected_m_per_px is not None
    assert abs(result.selected_m_per_px - (50 * 0.0254 / (72.0 * 1.5875))) < 1e-12


def test_incompatible_strong_declared_scales_block_publication() -> None:
    result = ScaleEvidenceResolver().resolve(
        level_view=_level(),
        evidence=[
            _declared_scale("S1", "ESCALA 1:50"),
            _declared_scale("S2", "ESCALA 1:100"),
        ],
    )
    assert result.state == "CONFLICT"
    assert any("incompatibles" in warning for warning in result.warnings)
