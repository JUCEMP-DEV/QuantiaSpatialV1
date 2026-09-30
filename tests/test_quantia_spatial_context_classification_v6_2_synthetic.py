from __future__ import annotations

import math

from app.quantia_spatialV1.models.level_view import (
    LevelView,
    LevelViewTransform,
    PixelBBox,
)
from app.quantia_spatialV1.phase_02_boundaries.perimeter_delivery import (
    EditablePerimeterModel,
    EditablePerimeterVertex,
    EditablePerimeterWall,
    PerimeterComparisonBaseline,
    PerimeterMetricSummary,
)
from app.quantia_spatialV1.phase_02_boundaries.perimeter_models import PerimeterPointPx
from app.quantia_spatialV1.reconstruction_core.candidate_context_gate import CandidateContextGate
from app.quantia_spatialV1.reconstruction_core.candidate_models import WallCandidate, WallEvidenceVector
from app.quantia_spatialV1.reconstruction_core.context_models import ContextRegion, RepetitiveAxisProfile
from app.quantia_spatialV1.reconstruction_core.context_region_detector import ContextRegionDetector
from app.quantia_spatialV1.reconstruction_core.drawing_model import (
    DrawingBBox,
    DrawingLine,
    DrawingModel,
    DrawingModelDiagnostics,
    DrawingPoint,
    DrawingText,
)
from app.quantia_spatialV1.reconstruction_core.level_scale_normalizer import ProjectLevelScaleNormalizer


def _level(level_id: str, width: int, height: int) -> LevelView:
    return LevelView(
        id=level_id,
        level_name=level_id,
        source_document_id="SYNTH",
        source_page_number=1,
        source_bbox_px=PixelBBox(x_min=0, y_min=0, x_max=width, y_max=height),
        source_page_width_px=width,
        source_page_height_px=height,
        raster_width_px=width,
        raster_height_px=height,
        raster_mime_type="image/png",
        raster_bytes=b"synthetic",
        transform=LevelViewTransform(
            offset_x_px=0,
            offset_y_px=0,
            source_page_width_px=width,
            source_page_height_px=height,
            local_width_px=width,
            local_height_px=height,
        ),
        state="DETECTADO",
        confidence=1.0,
    )


def _perimeter(level_id: str, width: int, height: int, m_per_px: float | None) -> EditablePerimeterModel:
    margin = 20.0
    pts = [
        (margin, margin),
        (width - margin, margin),
        (width - margin, height - margin),
        (margin, height - margin),
    ]
    vertices = [
        EditablePerimeterVertex(
            id=f"{level_id}__V{i}",
            sequence_index=i,
            point_px=PerimeterPointPx(x=x, y=y),
            incoming_wall_id=f"{level_id}__W{(i - 1) % 4}",
            outgoing_wall_id=f"{level_id}__W{i}",
        )
        for i, (x, y) in enumerate(pts)
    ]
    walls = []
    for i, ((x1, y1), (x2, y2)) in enumerate(zip(pts, pts[1:] + pts[:1])):
        walls.append(
            EditablePerimeterWall(
                id=f"{level_id}__W{i}",
                sequence_index=i,
                start_vertex_id=f"{level_id}__V{i}",
                end_vertex_id=f"{level_id}__V{(i + 1) % 4}",
                drawing_orientation="HORIZONTAL" if abs(y2 - y1) < 1e-6 else "VERTICAL",
                drawing_side=["TOP", "RIGHT", "BOTTOM", "LEFT"][i],
                length_px=math.hypot(x2 - x1, y2 - y1),
                length_m=(math.hypot(x2 - x1, y2 - y1) * m_per_px) if m_per_px else None,
                metric_status="GROUNDED" if m_per_px else "UNRESOLVED",
            )
        )
    status = "GROUNDED" if m_per_px else "UNRESOLVED"
    return EditablePerimeterModel(
        id=f"{level_id}__P",
        level_view_id=level_id,
        source_wall_layer_id=f"{level_id}__WL",
        source_candidate_id=f"{level_id}__PC",
        status="METRIC_READY" if m_per_px else "GEOMETRY_ONLY",
        vertices=vertices,
        walls=walls,
        metric_summary=PerimeterMetricSummary(
            status=status,
            total_perimeter_ml=(sum(w.length_px for w in walls) * m_per_px) if m_per_px else None,
            grounded_wall_count=4 if m_per_px else 0,
            unresolved_wall_ids=[] if m_per_px else [w.id for w in walls],
            metric_scale_m_per_px=m_per_px,
        ),
        comparison_baseline=PerimeterComparisonBaseline(available=False, state="UNAVAILABLE"),
    )


def _line(line_id: str, x1: float, y1: float, x2: float, y2: float, *, dashed: bool = False) -> DrawingLine:
    dx, dy = x2 - x1, y2 - y1
    return DrawingLine(
        id=line_id,
        start=DrawingPoint(x=x1, y=y1),
        end=DrawingPoint(x=x2, y=y2),
        length_px=math.hypot(dx, dy),
        angle_deg=math.degrees(math.atan2(dy, dx)) % 180.0,
        stroke_width_px=1.0,
        dashed=dashed,
        evidence_ids=[line_id],
        sources=["PYMUPDF"],
        confidence=1.0,
    )


def _drawing(level_id: str, width: int, height: int, lines: list[DrawingLine], texts=None) -> DrawingModel:
    return DrawingModel(
        level_view_id=level_id,
        width_px=width,
        height_px=height,
        lines=lines,
        texts=texts or [],
        diagnostics=DrawingModelDiagnostics(
            raw_evidence_count=len(lines),
            normalized_line_count=len(lines),
            normalized_region_count=0,
            text_count=len(texts or []),
            semantic_observation_count=0,
            lsd_line_count=0,
            region_centerline_count=0,
            source_counts={"PYMUPDF": len(lines)},
        ),
    )


def _stair_lines(scale_px_per_m: float, x0: float, y0: float, prefix: str) -> list[DrawingLine]:
    tread_length = 1.00 * scale_px_per_m
    spacing = 0.25 * scale_px_per_m
    return [
        _line(f"{prefix}_{i}", x0, y0 + i * spacing, x0 + tread_length, y0 + i * spacing)
        for i in range(8)
    ]


def test_v6_2_project_scale_normalizer_interrelaciona_plantas() -> None:
    lv_a = _level("A", 1200, 800)
    lv_b = _level("B", 2400, 1600)
    p_a = _perimeter("A", 1200, 800, 0.010)
    p_b = _perimeter("B", 2400, 1600, 0.005)

    context = ProjectLevelScaleNormalizer().build(levels=[(lv_a, p_a), (lv_b, p_b)])
    a = context.for_level("A")
    b = context.for_level("B")

    assert context.state == "RESOLVED"
    assert math.isclose(a.normalized_min_dim_px, b.normalized_min_dim_px, rel_tol=1e-6)
    assert math.isclose(
        100.0 * a.scale_factor_to_canonical,
        200.0 * b.scale_factor_to_canonical,
        rel_tol=1e-6,
    )


def test_v6_2_no_infiere_escala_faltante_de_otra_planta() -> None:
    lv_a = _level("A", 1200, 800)
    lv_b = _level("B", 1200, 800)
    p_a = _perimeter("A", 1200, 800, 0.010)
    p_b = _perimeter("B", 1200, 800, None)

    context = ProjectLevelScaleNormalizer().build(levels=[(lv_a, p_a), (lv_b, p_b)])
    assert context.state == "PARTIAL"
    assert context.for_level("A").state == "RESOLVED"
    assert context.for_level("B").state == "UNRESOLVED"
    assert context.for_level("B").local_m_per_px is None


def test_v6_2_misma_escalera_mantiene_clase_con_dos_escalas() -> None:
    lv_a = _level("A", 1200, 800)
    lv_b = _level("B", 2400, 1600)
    p_a = _perimeter("A", 1200, 800, 0.010)
    p_b = _perimeter("B", 2400, 1600, 0.005)
    context = ProjectLevelScaleNormalizer().build(levels=[(lv_a, p_a), (lv_b, p_b)])

    drawing_a = _drawing("A", 1200, 800, _stair_lines(100.0, 300.0, 250.0, "A"))
    drawing_b = _drawing("B", 2400, 1600, _stair_lines(200.0, 600.0, 500.0, "B"))
    detector = ContextRegionDetector()

    regions_a = detector.detect(drawing=drawing_a, perimeter=p_a, scale_profile=context.for_level("A"))
    regions_b = detector.detect(drawing=drawing_b, perimeter=p_b, scale_profile=context.for_level("B"))

    assert any(r.region_type == "STAIR_FLIGHT_REGION" for r in regions_a)
    assert any(r.region_type == "STAIR_FLIGHT_REGION" for r in regions_b)


def test_v6_2_dimension_region_permanece_local_y_no_absorbe_muro_largo() -> None:
    width, height = 600, 400
    lv = _level("LV", width, height)
    perimeter = _perimeter("LV", width, height, 0.01)
    lines = [
        _line("DIM_H", 200, 100, 340, 100),
        _line("DIM_V1", 200, 85, 200, 115),
        _line("DIM_V2", 340, 85, 340, 115),
        _line("STRUCTURAL_LONG", 40, 130, 560, 130),
        _line("STRUCTURAL_VERTICAL", 500, 40, 500, 360),
    ]
    text = DrawingText(
        id="T1",
        text="3.50",
        bbox=DrawingBBox(x_min=255, y_min=88, x_max=285, y_max=99),
        evidence_ids=["T1"],
        sources=["PYMUPDF"],
        confidence=1.0,
    )
    drawing = _drawing("LV", width, height, lines, [text])
    context = ProjectLevelScaleNormalizer().build(levels=[(lv, perimeter)])
    regions = ContextRegionDetector().detect(
        drawing=drawing,
        perimeter=perimeter,
        scale_profile=context.for_level("LV"),
    )
    dims = [r for r in regions if r.region_type == "DIMENSION_REGION"]
    assert dims
    region = dims[0]
    assert region.metadata.get("anchor_line_id") == "DIM_H"
    assert "STRUCTURAL_LONG" not in region.member_line_ids
    area_ratio = (region.bbox.x_max - region.bbox.x_min) * (region.bbox.y_max - region.bbox.y_min) / (width * height)
    assert area_ratio < 0.18


def test_v6_2_escalera_tiene_prioridad_sobre_dimension_superpuesta() -> None:
    profile = RepetitiveAxisProfile(
        angle_deg=0.0,
        spacing_px=12.0,
        spacing_cv=0.05,
        member_count=8,
        median_length_px=100.0,
        common_span_ratio=0.95,
        length_similarity=0.95,
        track_coordinates=[100, 112, 124, 136, 148, 160, 172, 184],
        member_line_ids=[f"S{i}" for i in range(8)],
    )
    stair = ContextRegion(
        id="STAIR",
        region_type="STAIR_FLIGHT_REGION",
        bbox=DrawingBBox(x_min=80, y_min=90, x_max=220, y_max=200),
        confidence=0.80,
        quarantine_enabled=True,
        profiles=[profile],
        member_line_ids=[f"S{i}" for i in range(8)],
    )
    dimension = ContextRegion(
        id="DIM",
        region_type="DIMENSION_REGION",
        bbox=DrawingBBox(x_min=70, y_min=80, x_max=230, y_max=210),
        confidence=0.94,
        quarantine_enabled=True,
        member_line_ids=["S3"],
    )
    candidate = WallCandidate(
        id="C",
        generator="DOUBLE_FACE",
        start=DrawingPoint(x=100, y=136),
        end=DrawingPoint(x=200, y=136),
        angle_deg=0.0,
        length_px=100.0,
        thickness_px=4.0,
        face_ids=["S3"],
        evidence_ids=["S3"],
        source_names=["PYMUPDF"],
        evidence=WallEvidenceVector(
            pair_overlap=1.0,
            thickness_support=0.8,
            vector_support=1.0,
            raster_line_support=1.0,
            region_support=1.0,
            source_consensus=0.75,
            perimeter_containment=1.0,
        ),
        prior_score=0.9,
    )
    result = CandidateContextGate().apply(candidates=[candidate], regions=[dimension, stair])
    decision = result.decisions[0]
    assert decision.region_type == "STAIR_FLIGHT_REGION"
    assert decision.state == "QUARANTINE"

# ---------------------------------------------------------------------------
# Regresión funcional V6 integrada en V6.2 para mantener una sola suite activa.
# ---------------------------------------------------------------------------


def _grid_lines() -> list[DrawingLine]:
    lines: list[DrawingLine] = []
    xs = [160, 200, 240, 280, 320, 360]
    ys = [140, 180, 220]
    for idx, y in enumerate(ys):
        lines.append(_line(f"H{idx}", xs[0], y, xs[-1], y))
    for idx, x in enumerate(xs):
        lines.append(_line(f"V{idx}", x, ys[0], x, ys[-1]))
    return lines


def test_v6_2_regresion_detecta_celdas_repetidas_como_acabado() -> None:
    width, height = 600, 400
    lv = _level("GRID", width, height)
    perimeter = _perimeter("GRID", width, height, None)
    drawing = _drawing("GRID", width, height, _grid_lines())
    regions = ContextRegionDetector().detect(drawing=drawing, perimeter=perimeter)
    floor = [region for region in regions if region.region_type == "FLOOR_FINISH_GRID_REGION"]
    assert floor, [region.region_type for region in regions]
    assert max(int(region.metadata.get("cell_count", 0)) for region in floor) >= 6


def test_v6_2_regresion_detecta_hatch_diagonal() -> None:
    width, height = 600, 400
    perimeter = _perimeter("HATCH", width, height, None)
    lines = [
        _line(f"D{index}", 120 + index * 12.0, 120, 200 + index * 12.0, 200)
        for index in range(8)
    ]
    drawing = _drawing("HATCH", width, height, lines)
    regions = ContextRegionDetector().detect(drawing=drawing, perimeter=perimeter)
    assert any(region.region_type == "HATCH_FILL_REGION" for region in regions)


def test_v6_2_regresion_detecta_ejes_discontinuos_largos() -> None:
    width, height = 600, 400
    perimeter = _perimeter("AXIS", width, height, None)
    lines = [
        _line("AX0", 80, 90, 520, 90, dashed=True),
        _line("AX1", 80, 180, 520, 180, dashed=True),
        _line("AX2", 80, 270, 520, 270, dashed=True),
    ]
    drawing = _drawing("AXIS", width, height, lines)
    regions = ContextRegionDetector().detect(drawing=drawing, perimeter=perimeter)
    assert any(region.region_type == "AXIS_GRID_REGION" for region in regions)


def test_v6_2_regresion_detecta_region_de_cota_local() -> None:
    width, height = 600, 400
    perimeter = _perimeter("DIM", width, height, None)
    lines = [
        _line("DIM_H", 200, 100, 340, 100),
        _line("DIM_V1", 200, 85, 200, 115),
        _line("DIM_V2", 340, 85, 340, 115),
    ]
    text = DrawingText(
        id="TD",
        text="3.50",
        bbox=DrawingBBox(x_min=255, y_min=88, x_max=285, y_max=99),
        evidence_ids=["TD"],
        sources=["PYMUPDF"],
        confidence=1.0,
    )
    drawing = _drawing("DIM", width, height, lines, [text])
    regions = ContextRegionDetector().detect(drawing=drawing, perimeter=perimeter)
    assert any(region.region_type == "DIMENSION_REGION" for region in regions)


def test_v6_2_regresion_publica_proteccion_f02() -> None:
    width, height = 600, 400
    perimeter = _perimeter("PROTECT", width, height, None)
    drawing = _drawing("PROTECT", width, height, [])
    regions = ContextRegionDetector().detect(drawing=drawing, perimeter=perimeter)
    protected = [region for region in regions if region.region_type == "WALL_PROTECTED_REGION"]
    assert len(protected) == 4


def test_v6_2_regresion_gate_cuarentena_celda_y_protege_perimetro() -> None:
    width, height = 600, 400
    perimeter = _perimeter("GATE", width, height, None)
    drawing = _drawing("GATE", width, height, _grid_lines())
    regions = ContextRegionDetector().detect(drawing=drawing, perimeter=perimeter)

    grid_candidate = WallCandidate(
        id="GRID_CANDIDATE",
        generator="DOUBLE_FACE",
        start=DrawingPoint(x=200, y=140),
        end=DrawingPoint(x=200, y=220),
        angle_deg=90.0,
        length_px=80.0,
        thickness_px=4.0,
        face_ids=["V1"],
        evidence_ids=["V1"],
        source_names=["PYMUPDF"],
        evidence=WallEvidenceVector(
            pair_overlap=1.0,
            thickness_support=0.9,
            vector_support=1.0,
            raster_line_support=1.0,
            region_support=1.0,
            source_consensus=0.75,
            perimeter_containment=1.0,
        ),
        prior_score=0.90,
    )
    perimeter_candidate = WallCandidate(
        id="PERIMETER_CANDIDATE",
        generator="DOUBLE_FACE",
        start=DrawingPoint(x=40, y=20),
        end=DrawingPoint(x=560, y=20),
        angle_deg=0.0,
        length_px=520.0,
        thickness_px=8.0,
        face_ids=[],
        evidence_ids=[],
        source_names=["PYMUPDF"],
        evidence=WallEvidenceVector(
            pair_overlap=1.0,
            thickness_support=0.9,
            vector_support=1.0,
            raster_line_support=1.0,
            region_support=1.0,
            source_consensus=0.75,
            perimeter_containment=1.0,
        ),
        prior_score=0.95,
    )
    result = CandidateContextGate().apply(
        candidates=[grid_candidate, perimeter_candidate],
        regions=regions,
    )
    decisions = {decision.candidate_id: decision for decision in result.decisions}
    assert decisions["GRID_CANDIDATE"].state in {"QUARANTINE", "REVIEW"}
    assert decisions["PERIMETER_CANDIDATE"].state == "PROTECTED"

# ---------------------------------------------------------------------------
# V6.2: patrones de superficie segmentados + linaje estructural.
# ---------------------------------------------------------------------------


def _segmented_tile_lines(prefix: str = "TILE") -> list[DrawingLine]:
    lines: list[DrawingLine] = []
    # Dos filas regulares de segmentos independientes; deliberadamente NO forman
    # rectángulos cerrados para validar el nuevo camino de superficie V6.2.
    for row, y in enumerate((150.0, 190.0)):
        for col in range(6):
            x0 = 120.0 + col * 40.0
            lines.append(_line(f"{prefix}_H_{row}_{col}", x0, y, x0 + 30.0, y))
    return lines


def _wall_candidate(
    candidate_id: str,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    face_ids: list[str] | None = None,
    axis_support: float = 0.0,
    thickness: float = 4.0,
) -> WallCandidate:
    dx, dy = x2 - x1, y2 - y1
    return WallCandidate(
        id=candidate_id,
        generator="DOUBLE_FACE",
        start=DrawingPoint(x=x1, y=y1),
        end=DrawingPoint(x=x2, y=y2),
        angle_deg=math.degrees(math.atan2(dy, dx)) % 180.0,
        length_px=math.hypot(dx, dy),
        thickness_px=thickness,
        face_ids=face_ids or [],
        evidence_ids=face_ids or [],
        source_names=["PYMUPDF"],
        evidence=WallEvidenceVector(
            pair_overlap=1.0,
            thickness_support=0.85,
            vector_support=1.0,
            raster_line_support=1.0,
            region_support=0.95,
            source_consensus=0.75,
            perimeter_containment=1.0,
            axis_support=axis_support,
        ),
        prior_score=0.85,
    )


def test_v6_2_detecta_baldosa_segmentada_sin_celdas_cerradas() -> None:
    width, height = 600, 400
    perimeter = _perimeter("SURFACE", width, height, 0.01)
    drawing = _drawing("SURFACE", width, height, _segmented_tile_lines())
    lv = _level("SURFACE", width, height)
    scale = ProjectLevelScaleNormalizer().build(levels=[(lv, perimeter)])

    regions = ContextRegionDetector().detect(
        drawing=drawing,
        perimeter=perimeter,
        scale_profile=scale.for_level("SURFACE"),
    )
    surface = [
        region for region in regions
        if region.region_type == "FLOOR_FINISH_GRID_REGION"
        and region.metadata.get("detector") == "SURFACE_PATTERN_DETECTOR_V6_2"
    ]
    assert surface, [(region.region_type, region.metadata.get("detector")) for region in regions]
    region = surface[0]
    assert int(region.metadata.get("segment_count", 0)) >= 8
    assert region.metadata.get("physical_large_tile_support") is True
    assert float(region.metadata.get("estimated_module_along_m")) >= 0.15
    assert float(region.metadata.get("estimated_module_cross_m")) >= 0.15


def test_v6_2_patron_superficie_aislado_no_se_trata_como_muro() -> None:
    width, height = 600, 400
    perimeter = _perimeter("SURFACE_GATE", width, height, None)
    lines = _segmented_tile_lines()
    drawing = _drawing("SURFACE_GATE", width, height, lines)
    regions = ContextRegionDetector().detect(drawing=drawing, perimeter=perimeter)
    surface = [region for region in regions if region.region_type == "FLOOR_FINISH_GRID_REGION"]
    assert surface

    candidate = _wall_candidate(
        "TILE_C",
        120.0,
        150.0,
        150.0,
        150.0,
        face_ids=["TILE_H_0_0"],
        axis_support=0.10,
    )
    result = CandidateContextGate().apply(candidates=[candidate], regions=regions)
    decision = result.decisions[0]
    assert decision.region_type == "FLOOR_FINISH_GRID_REGION"
    assert decision.state in {"QUARANTINE", "REVIEW"}
    lineage = decision.metadata.get("structural_lineage") or {}
    assert float(lineage.get("score", 0.0)) < 0.64


def test_v6_2_muro_colineal_con_ancla_fuerte_sobrevive_patron_repetitivo() -> None:
    width, height = 600, 400
    perimeter = _perimeter("LINEAGE", width, height, None)
    drawing = _drawing("LINEAGE", width, height, _segmented_tile_lines())
    regions = ContextRegionDetector().detect(drawing=drawing, perimeter=perimeter)
    assert any(region.region_type == "FLOOR_FINISH_GRID_REGION" for region in regions)

    fragment = _wall_candidate(
        "STRUCTURAL_FRAGMENT",
        120.0,
        150.0,
        150.0,
        150.0,
        face_ids=["TILE_H_0_0"],
        axis_support=0.20,
    )
    anchor = _wall_candidate(
        "STRUCTURAL_ANCHOR",
        150.0,
        150.0,
        520.0,
        150.0,
        axis_support=0.88,
        thickness=8.0,
    )
    result = CandidateContextGate().apply(candidates=[fragment, anchor], regions=regions)
    decisions = {item.candidate_id: item for item in result.decisions}
    decision = decisions["STRUCTURAL_FRAGMENT"]
    assert decision.state == "ACTIVE"
    assert "STRUCTURAL_LINEAGE" in decision.reason
    lineage = decision.metadata.get("structural_lineage") or {}
    assert float(lineage.get("collinear_anchor_support", 0.0)) >= 0.78


def test_v6_2_paralelismo_sin_eje_ni_ancla_no_activa_linaje() -> None:
    candidates = [
        _wall_candidate(f"P{i}", 100.0, 100.0 + i * 20.0, 150.0, 100.0 + i * 20.0, axis_support=0.10)
        for i in range(5)
    ]
    from app.quantia_spatialV1.reconstruction_core.structural_lineage import StructuralLineageAnalyzer

    lineage = StructuralLineageAnalyzer().analyze(candidates)
    assert all(not item.strong for item in lineage.values())
    assert all(item.collinear_anchor_support < 0.58 for item in lineage.values())
