from __future__ import annotations

from io import BytesIO

from PIL import Image

from app.quantia_spatialV1.models.evidence import EvidenceGeometry, RawEvidence
from app.quantia_spatialV1.models.level_view import LevelView, LevelViewEvidence, LevelViewTransform, PixelBBox, PixelPoint
from app.quantia_spatialV1.phase_01_level.level_identification_service import LevelIdentificationService
from app.quantia_spatialV1.phase_015_evidence.gemini_semantic_extractor import GeminiSemanticExtractor
from app.quantia_spatialV1.phase_02_boundaries.perimeter_delivery import (
    EditablePerimeterModel,
    EditablePerimeterVertex,
    EditablePerimeterWall,
    PerimeterComparisonBaseline,
    PerimeterMetricSummary,
)
from app.quantia_spatialV1.phase_02_boundaries.perimeter_models import PerimeterPointPx
from app.quantia_spatialV1.reconstruction_core.project_scale_reconciler import ProjectScaleReconciler
from app.quantia_spatialV1.reconstruction_core.raster_density_policy import RasterDensityPolicy
from app.quantia_spatialV1.reconstruction_core.scale_evidence_resolver import (
    LevelScaleEvidenceResult,
    ScaleEvidenceCandidate,
    ScaleEvidenceResolver,
)
from app.quantia_spatialV1.transport.gemini_level_localization import GeminiLevelLocalizationResponse


def _png(width: int = 1000, height: int = 500) -> bytes:
    image = Image.new("L", (width, height), 255)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _level(*, evidence: list[LevelViewEvidence] | None = None) -> LevelView:
    width, height = 1000, 500
    bbox = PixelBBox(x_min=0, y_min=0, x_max=width, y_max=height)
    return LevelView(
        id="LV_TEST",
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
        evidence=list(evidence or []),
    )


def _vector_line(eid: str, x: int, *, render_scale: float = 1.0) -> RawEvidence:
    return RawEvidence(
        id=eid,
        level_view_id="LV_TEST",
        source="PYMUPDF",
        kind="VECTOR_LINE",
        geometry=EvidenceGeometry(
            geometry_type="SEGMENT",
            points=[PixelPoint(x=x, y=0), PixelPoint(x=x, y=500)],
        ),
        confidence=1.0,
        metadata={
            "page_to_raster_scale_x": render_scale,
            "page_to_raster_scale_y": render_scale,
        },
    )


def _dimension(eid: str, start: str, end: str, value_m: float) -> RawEvidence:
    return RawEvidence(
        id=eid,
        level_view_id="LV_TEST",
        source="GEMINI",
        kind="GEMINI_OBSERVATION",
        geometry=EvidenceGeometry(geometry_type="NONE"),
        text=str(value_m),
        confidence=0.95,
        metadata={
            "semantic_category": "DIMENSION",
            "orientation": "HORIZONTAL",
            "span_type": "TRAMO",
            "reference_start": start,
            "reference_end": end,
            "measurements": [{"name": "dimension", "value": value_m, "unit": "M"}],
        },
    )



def _horizontal_line(eid: str, y: int, *, render_scale: float = 1.0) -> RawEvidence:
    return RawEvidence(
        id=eid,
        level_view_id="LV_TEST",
        source="PYMUPDF",
        kind="VECTOR_LINE",
        geometry=EvidenceGeometry(
            geometry_type="SEGMENT",
            points=[PixelPoint(x=0, y=y), PixelPoint(x=1000, y=y)],
        ),
        confidence=1.0,
        metadata={
            "page_to_raster_scale_x": render_scale,
            "page_to_raster_scale_y": render_scale,
        },
    )


def _axis_label(eid: str, text: str, x: int, y: int) -> RawEvidence:
    return RawEvidence(
        id=eid,
        level_view_id="LV_TEST",
        source="PYMUPDF",
        kind="VECTOR_TEXT",
        geometry=EvidenceGeometry(
            geometry_type="BBOX",
            bbox_px=PixelBBox(x_min=x - 3, y_min=y - 3, x_max=x + 3, y_max=y + 3),
        ),
        text=text,
        confidence=1.0,
        metadata={
            "page_to_raster_scale_x": 1.0,
            "page_to_raster_scale_y": 1.0,
        },
    )


def _general_dimension(eid: str, start: str, end: str, value_m: float, *, orientation: str) -> RawEvidence:
    return RawEvidence(
        id=eid,
        level_view_id="LV_TEST",
        source="GEMINI",
        kind="GEMINI_OBSERVATION",
        geometry=EvidenceGeometry(geometry_type="NONE"),
        text=str(value_m),
        confidence=0.98,
        metadata={
            "semantic_category": "DIMENSION",
            "orientation": orientation,
            "span_type": "GENERAL",
            "reference_start": start,
            "reference_end": end,
            "measurements": [{"name": "dimension", "value": value_m, "unit": "M"}],
        },
    )

def test_raster_policy_matches_reference_factors() -> None:
    policy = RasterDensityPolicy(target_geometry_px_per_m=90.0)
    assert abs(policy.recommend_from_declared_scale(denominator=25).recommended_render_scale - 0.79375) < 1e-8
    assert abs(policy.recommend_from_declared_scale(denominator=30).recommended_render_scale - 0.9525) < 1e-8
    assert abs(policy.recommend_from_declared_scale(denominator=50).recommended_render_scale - 1.5875) < 1e-8


def test_declared_scale_pdf_becomes_metric_candidate() -> None:
    level = _level()
    text = RawEvidence(
        id="SCALE_TEXT",
        level_view_id=level.id,
        source="PYMUPDF",
        kind="VECTOR_TEXT",
        geometry=EvidenceGeometry(geometry_type="NONE"),
        text="ESCALA 1:50",
        confidence=1.0,
        metadata={"page_to_raster_scale_x": 1.5875, "page_to_raster_scale_y": 1.5875},
    )
    result = ScaleEvidenceResolver().resolve(level_view=level, evidence=[text])
    assert result.state == "RESOLVED"
    assert result.selected_m_per_px is not None
    assert abs(result.selected_m_per_px - (1.0 / 90.0)) < 1e-8
    assert result.declared_scale_denominators == [50.0]


def test_f01_gemini_scale_hint_is_preserved_in_level_view_evidence() -> None:
    payload = {
        "pagina": 1,
        "niveles": [
            {
                "nombre": "Planta Test",
                "localizado": True,
                "bbox_normalizado": {"x_min": 0.0, "y_min": 0.0, "x_max": 1.0, "y_max": 1.0},
                "confianza": 0.95,
                "escala_declarada": "1:30",
                "confianza_escala": 0.93,
            }
        ],
    }
    validated = GeminiLevelLocalizationResponse.model_validate(payload)
    result = LevelIdentificationService().identify_page(
        source_raster_bytes=_png(),
        source_page_number=1,
        source_document_id="DOC",
        known_level_names=["Planta Test"],
        gemini_payload=validated.to_detector_payload(),
        single_level_isolated=False,
    )
    assert len(result.level_views) == 1
    hints = [item for item in result.level_views[0].evidence if item.reference == "declared_scale_hint"]
    assert len(hints) == 1
    assert hints[0].text == "1:30"
    assert hints[0].confidence == 0.93


def test_old_localization_payload_without_scale_remains_compatible() -> None:
    payload = {
        "pagina": 1,
        "niveles": [
            {
                "nombre": "Planta Test",
                "localizado": True,
                "bbox_normalizado": {"x_min": 0.0, "y_min": 0.0, "x_max": 1.0, "y_max": 1.0},
                "confianza": 0.95,
            }
        ],
    }
    response = GeminiLevelLocalizationResponse.model_validate(payload)
    assert response.niveles[0].escala_declarada is None
    assert response.niveles[0].confianza_escala is None


def test_axis_dimension_fit_can_resolve_scale_without_printed_scale() -> None:
    level = _level()
    evidence = [
        _vector_line("L1", 100),
        _vector_line("L2", 300),
        _vector_line("L3", 500),
        _dimension("D1", "1", "2", 2.0),
        _dimension("D2", "2", "3", 2.0),
    ]
    result = ScaleEvidenceResolver().resolve(level_view=level, evidence=evidence)
    axis_candidates = [item for item in result.candidates if item.method == "DIMENSION_AXIS_FIT"]
    assert axis_candidates
    assert result.selected_m_per_px is not None
    assert abs(result.selected_m_per_px - 0.01) < 0.001


def test_conflicting_strong_scale_sources_are_not_silently_merged() -> None:
    level = _level()
    evidence = [
        RawEvidence(
            id="SCALE50",
            level_view_id=level.id,
            source="PYMUPDF",
            kind="VECTOR_TEXT",
            geometry=EvidenceGeometry(geometry_type="NONE"),
            text="Escala 1:50",
            confidence=1.0,
            metadata={"page_to_raster_scale_x": 1.0, "page_to_raster_scale_y": 1.0},
        ),
        _vector_line("L1", 100),
        _vector_line("L2", 300),
        _vector_line("L3", 500),
        _dimension("D1", "1", "2", 2.0),
        _dimension("D2", "2", "3", 2.0),
    ]
    result = ScaleEvidenceResolver().resolve(level_view=level, evidence=evidence)
    assert result.state == "CONFLICT"
    assert result.warnings


def test_project_reconciler_normalizes_different_local_scales() -> None:
    a = LevelScaleEvidenceResult(
        level_view_id="A",
        state="RESOLVED",
        selected_m_per_px=1.0 / 90.0,
        selected_confidence=0.95,
        selected_method="DIMENSION_AXIS_FIT",
        current_render_scale=0.79375,
    )
    b = LevelScaleEvidenceResult(
        level_view_id="B",
        state="RESOLVED",
        selected_m_per_px=1.0 / 60.0,
        selected_confidence=0.90,
        selected_method="DECLARED_SCALE_PDF",
        current_render_scale=1.0,
    )
    project = ProjectScaleReconciler().reconcile(levels=[a, b], target_px_per_m=90.0)
    assert project.state == "RESOLVED"
    assert abs(project.for_level("A").scale_factor_to_canonical - 1.0) < 1e-9
    assert abs(project.for_level("B").scale_factor_to_canonical - 1.5) < 1e-9
    assert len(project.relations) == 1


def test_project_reconciler_does_not_invent_missing_scale() -> None:
    unresolved = LevelScaleEvidenceResult(level_view_id="A", state="UNRESOLVED")
    project = ProjectScaleReconciler().reconcile(levels=[unresolved])
    assert project.state == "UNRESOLVED"
    assert project.for_level("A").local_m_per_px is None
    assert project.for_level("A").scale_factor_to_canonical == 1.0


def test_grounded_policy_preserves_target_density() -> None:
    policy = RasterDensityPolicy(target_geometry_px_per_m=90.0)
    result = policy.recommend_from_grounded_scale(current_render_scale=0.7935, current_m_per_px=1.0 / 45.0)
    assert abs(result.expected_px_per_m - 90.0) < 1e-9
    assert result.recommended_render_scale > 0.7935


def test_resolved_axis_span_fallback_resolves_without_declared_scale_or_affine_chain() -> None:
    level = _level()
    evidence = [
        _vector_line("VX1", 100),
        _vector_line("VX3", 500),
        _horizontal_line("HYA", 100),
        _horizontal_line("HYC", 300),
        _axis_label("TXT1", "1", 100, 12),
        _axis_label("TXT3", "3", 500, 12),
        _axis_label("TXTA", "A", 12, 100),
        _axis_label("TXTC", "C", 12, 300),
        _general_dimension("DG_H", "1", "3", 4.0, orientation="HORIZONTAL"),
        _general_dimension("DG_V", "A", "C", 2.0, orientation="VERTICAL"),
    ]

    result = ScaleEvidenceResolver().resolve(level_view=level, evidence=evidence)

    assert result.declared_scale_denominators == []
    assert not any(item.method == "DIMENSION_AXIS_FIT" for item in result.candidates)
    fallback = [
        item for item in result.candidates
        if item.method == "DIMENSION_RESOLVED_AXIS_SPAN"
    ]
    assert len(fallback) == 2
    assert result.state == "RESOLVED"
    assert result.selected_method == "DIMENSION_RESOLVED_AXIS_SPAN"
    assert result.selected_m_per_px is not None
    assert abs(result.selected_m_per_px - 0.01) < 1e-6


def test_single_tramo_axis_span_does_not_resolve_global_scale_alone() -> None:
    level = _level()
    evidence = [
        _vector_line("VX1", 100),
        _vector_line("VX2", 300),
        _axis_label("TXT1", "1", 100, 12),
        _axis_label("TXT2", "2", 300, 12),
        _dimension("D1", "1", "2", 2.0),
    ]

    result = ScaleEvidenceResolver().resolve(level_view=level, evidence=evidence)
    fallback = [
        item for item in result.candidates
        if item.method == "DIMENSION_RESOLVED_AXIS_SPAN"
    ]
    # La cadena puede resolverse por el método afín original. Si no lo hace,
    # el fallback aislado nunca debe tener confianza suficiente para cerrar
    # por sí solo la escala global.
    if fallback and not any(item.method == "DIMENSION_AXIS_FIT" for item in result.candidates):
        assert max(item.confidence for item in fallback) <= 0.72
        assert result.state != "RESOLVED"



def test_unique_gemini_level_fallback_recovers_semantics_for_single_level_view() -> None:
    level = _level()
    payload = {
        "niveles": [
            {
                "nombre": "Nivel Planta",
                "cotas": [
                    {
                        "texto": "7.23",
                        "valor_m": 7.23,
                        "referencia": "Planta | horizontal | eje 1→2 | tramo",
                        "estado": "DETECTADO",
                        "confianza": 0.95,
                    }
                ],
                "ejes": [],
                "espacios": [],
                "escaleras": [],
            }
        ]
    }

    semantic = GeminiSemanticExtractor().extract(
        payload=payload,
        level_view=level,
        allow_unique_level_fallback=True,
    )

    assert semantic["matched_level_name"] == "Nivel Planta"
    dimensions = [
        item for item in semantic["observations"]
        if item.get("category") == "DIMENSION"
    ]
    assert len(dimensions) == 1
    assert dimensions[0]["reference_start"] == "1"
    assert dimensions[0]["reference_end"] == "2"


def test_unique_gemini_level_fallback_is_disabled_without_single_view_policy() -> None:
    selected = GeminiSemanticExtractor._select_level(
        payload={"niveles": [{"nombre": "Nivel Planta"}]},
        level_name="Planta Baja",
        allow_unique_level_fallback=False,
    )
    assert selected is None


def test_multilevel_payload_never_uses_unique_level_fallback_without_exact_match() -> None:
    selected = GeminiSemanticExtractor._select_level(
        payload={
            "niveles": [
                {"nombre": "Nivel Genérico A"},
                {"nombre": "Nivel Genérico B"},
            ]
        },
        level_name="Planta Baja",
        allow_unique_level_fallback=True,
    )
    assert selected is None



def _rect_perimeter(
    level: LevelView,
    *,
    x_min: float,
    y_min: float,
    x_max: float,
    y_max: float,
) -> EditablePerimeterModel:
    points = [
        (x_min, y_min),
        (x_max, y_min),
        (x_max, y_max),
        (x_min, y_max),
    ]
    vertex_ids = [f"V{index}" for index in range(4)]
    vertices = [
        EditablePerimeterVertex(
            id=vertex_ids[index],
            sequence_index=index,
            point_px=PerimeterPointPx(x=x, y=y),
            incoming_wall_id=f"W{(index - 1) % 4}",
            outgoing_wall_id=f"W{index}",
        )
        for index, (x, y) in enumerate(points)
    ]
    walls = []
    for index, first in enumerate(points):
        second = points[(index + 1) % 4]
        walls.append(
            EditablePerimeterWall(
                id=f"W{index}",
                sequence_index=index,
                start_vertex_id=vertex_ids[index],
                end_vertex_id=vertex_ids[(index + 1) % 4],
                drawing_orientation=(
                    "HORIZONTAL" if first[1] == second[1] else "VERTICAL"
                ),
                drawing_side="TEST",
                length_px=((second[0] - first[0]) ** 2 + (second[1] - first[1]) ** 2) ** 0.5,
                metric_status="UNRESOLVED",
            )
        )
    return EditablePerimeterModel(
        id="P_TEST",
        level_view_id=level.id,
        source_wall_layer_id="WL_TEST",
        source_candidate_id="C_TEST",
        status="GEOMETRY_ONLY",
        vertices=vertices,
        walls=walls,
        metric_summary=PerimeterMetricSummary(
            status="UNRESOLVED",
            grounded_wall_count=0,
        ),
        comparison_baseline=PerimeterComparisonBaseline(
            available=False,
            state="UNAVAILABLE",
        ),
    )


def _observed_line(
    eid: str,
    *,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    render_scale: float = 0.7936603584094404,
) -> RawEvidence:
    return RawEvidence(
        id=eid,
        level_view_id="LV_TEST",
        source="OPENCV",
        kind="RASTER_LINE",
        geometry=EvidenceGeometry(
            geometry_type="SEGMENT",
            points=[
                PixelPoint(x=round(x1), y=round(y1)),
                PixelPoint(x=round(x2), y=round(y2)),
            ],
        ),
        confidence=1.0,
        metadata={
            "page_to_raster_scale_x": render_scale,
            "page_to_raster_scale_y": render_scale,
        },
    )


def _general_dimension_with_side(
    eid: str,
    *,
    orientation: str,
    value_m: float,
    side: str | None = None,
) -> RawEvidence:
    return RawEvidence(
        id=eid,
        level_view_id="LV_TEST",
        source="GEMINI",
        kind="GEMINI_OBSERVATION",
        geometry=EvidenceGeometry(geometry_type="NONE"),
        text=str(value_m),
        confidence=0.98,
        metadata={
            "semantic_category": "DIMENSION",
            "orientation": orientation,
            "dimension_side": side,
            "span_type": "GENERAL",
            "measurements": [
                {"name": "dimension", "value": value_m, "unit": "M"}
            ],
        },
    )


def test_dimension_reference_normalizes_miguel_v_planta_alta_sides() -> None:
    extractor = GeminiSemanticExtractor()

    top = extractor._parse_dimension_reference(
        "Planta | horizontal superior | eje 0→8 | general"
    )
    bottom = extractor._parse_dimension_reference(
        "Planta | horizontal inferior | eje 0→11 | tramo"
    )
    left = extractor._parse_dimension_reference(
        "Planta | vertical izquierda | eje J→A | general"
    )
    right = extractor._parse_dimension_reference(
        "Planta | vertical derecha | eje I→A | general"
    )

    assert top["orientation"] == "HORIZONTAL"
    assert top["dimension_side"] == "TOP"
    assert top["span_type"] == "GENERAL"
    assert bottom["orientation"] == "HORIZONTAL"
    assert bottom["dimension_side"] == "BOTTOM"
    assert left["orientation"] == "VERTICAL"
    assert left["dimension_side"] == "LEFT"
    assert right["orientation"] == "VERTICAL"
    assert right["dimension_side"] == "RIGHT"


def test_miguel_v_pb_graphic_general_spans_resolve_without_declared_scale() -> None:
    # Firma geométrica observada en la evidencia visual real de Miguel V PB:
    # general H 20.00 m ≈ 1497 px y general V 7.70 m ≈ 578 px.
    level = _level()
    perimeter = _rect_perimeter(
        level,
        x_min=128,
        y_min=276,
        x_max=1623,
        y_max=854,
    )
    evidence = [
        # Límite superior largo ajeno a la cota: debe perder contra el general
        # inferior porque sus extremos no coinciden con el perímetro.
        _observed_line("SITE_TOP", x1=53, y1=187, x2=1623, y2=187),
        _observed_line("DIM_H_GENERAL", x1=128, y1=902, x2=1625, y2=902),
        _observed_line("DIM_V_GENERAL", x1=1687, y1=276, x2=1687, y2=854),
        _general_dimension_with_side(
            "DG_H",
            orientation="HORIZONTAL",
            value_m=20.0,
        ),
        _general_dimension_with_side(
            "DG_V",
            orientation="VERTICAL",
            value_m=7.70,
        ),
    ]

    result = ScaleEvidenceResolver().resolve(
        level_view=level,
        evidence=evidence,
        perimeter=perimeter,
    )

    graphics = [
        item for item in result.candidates
        if item.method == "DIMENSION_GENERAL_GRAPHIC_SPAN"
    ]
    assert result.state == "RESOLVED"
    assert result.selected_method == "DIMENSION_GENERAL_GRAPHIC_SPAN"
    assert {item.semantic_orientation for item in graphics} == {"HORIZONTAL", "VERTICAL"}
    assert result.selected_m_per_px is not None
    assert abs(result.selected_m_per_px - 0.01334) < 0.00015

    recommendation = RasterDensityPolicy(
        target_geometry_px_per_m=90.0
    ).recommend_from_grounded_scale(
        current_render_scale=0.7936603584094404,
        current_m_per_px=result.selected_m_per_px,
    )
    assert 0.94 < recommendation.recommended_render_scale < 0.97


def test_miguel_v_pa_graphic_general_spans_resolve_three_sides() -> None:
    # Firma geométrica observada en Miguel V PA:
    # 20.80 m/1466 px, 8.50 m/601 px, 7.70 m/544 px.
    level = _level()
    perimeter = _rect_perimeter(
        level,
        x_min=129,
        y_min=512,
        x_max=1595,
        y_max=1118,
    )
    evidence = [
        _observed_line("DIM_H_TOP", x1=129, y1=487, x2=1595, y2=487),
        _observed_line("DIM_V_LEFT", x1=101, y1=519, x2=101, y2=1120),
        _observed_line("DIM_V_RIGHT", x1=1660, y1=575, x2=1660, y2=1119),
        _general_dimension_with_side(
            "DG_H_TOP",
            orientation="HORIZONTAL",
            value_m=20.80,
            side="TOP",
        ),
        _general_dimension_with_side(
            "DG_V_LEFT",
            orientation="VERTICAL",
            value_m=8.50,
            side="LEFT",
        ),
        _general_dimension_with_side(
            "DG_V_RIGHT",
            orientation="VERTICAL",
            value_m=7.70,
            side="RIGHT",
        ),
    ]

    result = ScaleEvidenceResolver().resolve(
        level_view=level,
        evidence=evidence,
        perimeter=perimeter,
    )

    graphics = [
        item for item in result.candidates
        if item.method == "DIMENSION_GENERAL_GRAPHIC_SPAN"
    ]
    assert result.state == "RESOLVED"
    assert result.selected_method == "DIMENSION_GENERAL_GRAPHIC_SPAN"
    assert len(graphics) == 3
    assert {item.graphic_side for item in graphics} == {"TOP", "LEFT", "RIGHT"}
    assert result.selected_m_per_px is not None
    assert abs(result.selected_m_per_px - 0.01416) < 0.00015

    recommendation = RasterDensityPolicy(
        target_geometry_px_per_m=90.0
    ).recommend_from_grounded_scale(
        current_render_scale=0.7936603584094404,
        current_m_per_px=result.selected_m_per_px,
    )
    assert 0.99 < recommendation.recommended_render_scale < 1.03



def test_fragmented_general_dimension_graphics_are_stitched_before_scale_resolution() -> None:
    level = _level()
    perimeter = _rect_perimeter(
        level,
        x_min=128,
        y_min=276,
        x_max=1623,
        y_max=854,
    )
    evidence = [
        # Cota H general 20 m fragmentada por ticks/exportación CAD.
        _observed_line("H1", x1=128, y1=902, x2=600, y2=902),
        _observed_line("H2", x1=606, y1=902, x2=1100, y2=902),
        _observed_line("H3", x1=1105, y1=902, x2=1625, y2=902),
        # Cota V general 7.70 m fragmentada.
        _observed_line("V1", x1=1687, y1=276, x2=1687, y2=460),
        _observed_line("V2", x1=1687, y1=465, x2=1687, y2=650),
        _observed_line("V3", x1=1687, y1=655, x2=1687, y2=854),
        _general_dimension_with_side(
            "DG_H",
            orientation="HORIZONTAL",
            value_m=20.0,
        ),
        _general_dimension_with_side(
            "DG_V",
            orientation="VERTICAL",
            value_m=7.70,
        ),
    ]

    result = ScaleEvidenceResolver().resolve(
        level_view=level,
        evidence=evidence,
        perimeter=perimeter,
    )

    graphics = [
        item for item in result.candidates
        if item.method == "DIMENSION_GENERAL_GRAPHIC_SPAN"
    ]
    assert result.state == "RESOLVED"
    assert result.selected_method == "DIMENSION_GENERAL_GRAPHIC_SPAN"
    assert {item.semantic_orientation for item in graphics} == {"HORIZONTAL", "VERTICAL"}
    assert all(len(item.evidence_ids) >= 3 for item in graphics)
    assert result.selected_m_per_px is not None
    assert abs(result.selected_m_per_px - 0.01334) < 0.0002


def test_direct_hv_graphic_consensus_cannot_be_outvoted_by_many_correlated_axis_spans() -> None:
    resolver = ScaleEvidenceResolver()
    graphics = [
        ScaleEvidenceCandidate(
            id="G_H",
            level_view_id="LV",
            method="DIMENSION_GENERAL_GRAPHIC_SPAN",
            m_per_px=0.01334,
            confidence=0.96,
            semantic_orientation="HORIZONTAL",
            graphic_side="BOTTOM",
        ),
        ScaleEvidenceCandidate(
            id="G_V",
            level_view_id="LV",
            method="DIMENSION_GENERAL_GRAPHIC_SPAN",
            m_per_px=0.01336,
            confidence=0.95,
            semantic_orientation="VERTICAL",
            graphic_side="RIGHT",
        ),
    ]
    derived = [
        ScaleEvidenceCandidate(
            id="FIT_V",
            level_view_id="LV",
            method="DIMENSION_AXIS_FIT",
            m_per_px=0.01691,
            confidence=0.99,
            semantic_orientation="VERTICAL",
        )
    ]
    derived.extend(
        ScaleEvidenceCandidate(
            id=f"SPAN_{index}",
            level_view_id="LV",
            method="DIMENSION_RESOLVED_AXIS_SPAN",
            m_per_px=0.01690 + index * 1e-7,
            confidence=0.72,
            semantic_orientation="VERTICAL",
        )
        for index in range(20)
    )

    clusters = resolver._cluster_candidates([*graphics, *derived])
    best = resolver._select_best_cluster(clusters)

    assert resolver._has_hv_graphic_crosscheck(best)
    assert all(float(item.m_per_px) < 0.014 for item in best)


def test_miguel_v_pb_real_diagnostic_prefers_cross_family_horizontal_consensus() -> None:
    """Reproduce la firma exacta observada en la corrida real V1.5.

    HORIZONTAL: gráfica GENERAL + AXIS_FIT + AXISSPAN convergen ~0.01338.
    VERTICAL: gráfica GENERAL ~0.01021 y familia de ejes ~0.0169 se contradicen.
    La orientación vertical no debe bloquear el cierre de la horizontal porque
    sus dos familias geométricas son internamente incompatibles.
    """
    resolver = ScaleEvidenceResolver()
    candidates = [
        ScaleEvidenceCandidate(
            id="G_H",
            level_view_id="LV",
            method="DIMENSION_GENERAL_GRAPHIC_SPAN",
            m_per_px=0.013360053440213761,
            confidence=0.9106341456529072,
            semantic_orientation="HORIZONTAL",
            graphic_side="BOTTOM",
        ),
        ScaleEvidenceCandidate(
            id="G_V",
            level_view_id="LV",
            method="DIMENSION_GENERAL_GRAPHIC_SPAN",
            m_per_px=0.010212201591511937,
            confidence=0.8993834854438252,
            semantic_orientation="VERTICAL",
            graphic_side="RIGHT",
        ),
        ScaleEvidenceCandidate(
            id="FIT_H",
            level_view_id="LV",
            method="DIMENSION_AXIS_FIT",
            m_per_px=0.013382198952879579,
            confidence=0.9940031344529763,
            semantic_orientation="HORIZONTAL",
        ),
        ScaleEvidenceCandidate(
            id="SPAN_H_GENERAL",
            level_view_id="LV",
            method="DIMENSION_RESOLVED_AXIS_SPAN",
            m_per_px=0.01337551119728094,
            confidence=0.99,
            semantic_orientation="HORIZONTAL",
        ),
        ScaleEvidenceCandidate(
            id="FIT_V",
            level_view_id="LV",
            method="DIMENSION_AXIS_FIT",
            m_per_px=0.0169164265129683,
            confidence=0.9884375263855849,
            semantic_orientation="VERTICAL",
        ),
        ScaleEvidenceCandidate(
            id="SPAN_V_GENERAL",
            level_view_id="LV",
            method="DIMENSION_RESOLVED_AXIS_SPAN",
            m_per_px=0.016850774146165058,
            confidence=0.99,
            semantic_orientation="VERTICAL",
        ),
    ]

    clusters = resolver._cluster_candidates(candidates)
    best = resolver._select_best_cluster(clusters)

    assert resolver._cross_family_consensus_orientation(best) == "HORIZONTAL"
    assert resolver._orientation_has_internal_geometry_conflict(
        candidates=candidates,
        orientation="VERTICAL",
    )
    assert not resolver._orientation_has_internal_geometry_conflict(
        candidates=candidates,
        orientation="HORIZONTAL",
    )
    assert 0.0132 < resolver._weighted_mean(best) < 0.0135


def test_axis_fit_and_axis_spans_count_as_one_independent_family() -> None:
    resolver = ScaleEvidenceResolver()
    cluster = [
        ScaleEvidenceCandidate(
            id="FIT_H",
            level_view_id="LV",
            method="DIMENSION_AXIS_FIT",
            m_per_px=0.01338,
            confidence=0.99,
            semantic_orientation="HORIZONTAL",
        ),
        *[
            ScaleEvidenceCandidate(
                id=f"SPAN_H_{index}",
                level_view_id="LV",
                method="DIMENSION_RESOLVED_AXIS_SPAN",
                m_per_px=0.01338,
                confidence=0.72,
                semantic_orientation="HORIZONTAL",
            )
            for index in range(20)
        ],
    ]

    score = resolver._independent_support_score(cluster)
    # Saturación de una sola familia AXIS_GEOMETRY/HORIZONTAL.
    assert abs(score - 0.99) < 1e-9


def test_real_miguel_v_pb_conflict_policy_closes_only_self_inconsistent_orientation(monkeypatch) -> None:
    from types import SimpleNamespace
    from app.quantia_spatialV1.reconstruction_core.general_dimension_graphic_span import GraphicDimensionSpan
    from app.quantia_spatialV1.reconstruction_core import scale_evidence_resolver as resolver_module

    level = _level()
    perimeter = _rect_perimeter(level, x_min=128, y_min=276, x_max=1623, y_max=854)
    evidence = [_vector_line("OBS", 100, render_scale=0.7936603584094404)]

    graphic_rows = [
        GraphicDimensionSpan(
            dimension_evidence_id="DG_H",
            semantic_orientation="HORIZONTAL",
            side="BOTTOM",
            value_m=20.0,
            span_px=1497.0,
            m_per_px=0.013360053440213761,
            line_coordinate_px=1036.658,
            line_start_px=0.0,
            line_end_px=1497.0,
            score=0.539,
            confidence=0.9106341456529072,
            geometric_evidence_ids=("GH",),
            source="PYMUPDF",
        ),
        GraphicDimensionSpan(
            dimension_evidence_id="DG_V",
            semantic_orientation="VERTICAL",
            side="RIGHT",
            value_m=7.7,
            span_px=754.0,
            m_per_px=0.010212201591511937,
            line_coordinate_px=1742.635,
            line_start_px=0.0,
            line_end_px=754.0,
            score=0.473,
            confidence=0.8993834854438252,
            geometric_evidence_ids=("GV",),
            source="PYMUPDF",
        ),
    ]
    monkeypatch.setattr(
        resolver_module.GeneralDimensionGraphicSpanResolver,
        "resolve",
        lambda self, **kwargs: graphic_rows,
    )

    def fit(orientation: str, m_per_px: float):
        return SimpleNamespace(
            state="RESOLVED",
            affine_pixels_per_semantic_unit=1.0 / m_per_px,
            matched_fraction=1.0,
            median_residual_px=0.0,
            match_tolerance_px=10.0,
            general_span_consistent=True,
            chain_dimension_evidence_ids=[],
            semantic_orientation=orientation,
        )

    axis_result = SimpleNamespace(
        state="RESOLVED",
        axes=[],
        diagnostics=SimpleNamespace(
            fits=[
                fit("HORIZONTAL", 0.013382198952879579),
                fit("VERTICAL", 0.0169164265129683),
            ]
        ),
    )
    monkeypatch.setattr(
        resolver_module.AxisGridResolver,
        "resolve",
        lambda self, **kwargs: axis_result,
    )

    result = ScaleEvidenceResolver().resolve(
        level_view=level,
        evidence=evidence,
        perimeter=perimeter,
    )

    assert result.state == "RESOLVED"
    assert result.selected_m_per_px is not None
    assert 0.0132 < result.selected_m_per_px < 0.0135
    assert any("VERTICAL" in warning and "conflicto interno" in warning for warning in result.warnings)


def test_two_internally_consistent_orientations_that_disagree_still_block(monkeypatch) -> None:
    from types import SimpleNamespace
    from app.quantia_spatialV1.reconstruction_core.general_dimension_graphic_span import GraphicDimensionSpan
    from app.quantia_spatialV1.reconstruction_core import scale_evidence_resolver as resolver_module

    level = _level()
    perimeter = _rect_perimeter(level, x_min=128, y_min=276, x_max=1623, y_max=854)
    evidence = [_vector_line("OBS", 100, render_scale=0.7936603584094404)]

    graphic_rows = [
        GraphicDimensionSpan(
            dimension_evidence_id="DG_H", semantic_orientation="HORIZONTAL", side="BOTTOM",
            value_m=20.0, span_px=1497.0, m_per_px=0.01336,
            line_coordinate_px=900.0, line_start_px=0.0, line_end_px=1497.0,
            score=0.9, confidence=0.95, geometric_evidence_ids=("GH",), source="PYMUPDF",
        ),
        GraphicDimensionSpan(
            dimension_evidence_id="DG_V", semantic_orientation="VERTICAL", side="RIGHT",
            value_m=7.7, span_px=455.6, m_per_px=0.01690,
            line_coordinate_px=1700.0, line_start_px=0.0, line_end_px=455.6,
            score=0.9, confidence=0.95, geometric_evidence_ids=("GV",), source="PYMUPDF",
        ),
    ]
    monkeypatch.setattr(
        resolver_module.GeneralDimensionGraphicSpanResolver,
        "resolve",
        lambda self, **kwargs: graphic_rows,
    )

    def fit(orientation: str, m_per_px: float):
        return SimpleNamespace(
            state="RESOLVED", affine_pixels_per_semantic_unit=1.0 / m_per_px,
            matched_fraction=1.0, median_residual_px=0.0, match_tolerance_px=10.0,
            general_span_consistent=True, chain_dimension_evidence_ids=[],
            semantic_orientation=orientation,
        )

    axis_result = SimpleNamespace(
        state="RESOLVED", axes=[],
        diagnostics=SimpleNamespace(fits=[fit("HORIZONTAL", 0.01338), fit("VERTICAL", 0.01692)]),
    )
    monkeypatch.setattr(
        resolver_module.AxisGridResolver,
        "resolve",
        lambda self, **kwargs: axis_result,
    )

    result = ScaleEvidenceResolver().resolve(level_view=level, evidence=evidence, perimeter=perimeter)
    assert result.state == "CONFLICT"
