from __future__ import annotations

import math
from io import BytesIO

from PIL import Image
from shapely.geometry import box

from app.quantia_spatialV1.models.evidence import EvidenceGeometry, RawEvidence
from app.quantia_spatialV1.models.level_view import LevelView, LevelViewTransform, PixelBBox
from app.quantia_spatialV1.reconstruction_core.candidate_context_gate import CandidateContextGate
from app.quantia_spatialV1.reconstruction_core.candidate_models import WallCandidate, WallEvidenceVector
from app.quantia_spatialV1.reconstruction_core.context_models import ContextRegion, RepetitiveAxisProfile
from app.quantia_spatialV1.reconstruction_core.drawing_builder import DrawingModelBuilder
from app.quantia_spatialV1.reconstruction_core.drawing_model import (
    DrawingBBox,
    DrawingCurve,
    DrawingLine,
    DrawingModel,
    DrawingModelDiagnostics,
    DrawingPoint,
    SemanticObservation,
)
from app.quantia_spatialV1.reconstruction_core.element_context_detector import ElementContextDetector
from app.quantia_spatialV1.reconstruction_core.level_scale_normalizer import LevelScaleProfile


def _profile():
    return LevelScaleProfile(
        level_view_id="LV",
        state="RESOLVED",
        local_m_per_px=1.0 / 90.0,
        canonical_m_per_px=1.0 / 90.0,
        scale_factor_to_canonical=1.0,
        local_min_dim_px=800.0,
        normalized_min_dim_px=800.0,
        project_reference_min_dim_normalized_px=800.0,
        project_reference_area_normalized_px2=800000.0,
        source="TEST",
        confidence=1.0,
    )


def _line(ident, a, b, *, evidence_ids=None):
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    return DrawingLine(
        id=ident,
        start=DrawingPoint(x=a[0], y=a[1]),
        end=DrawingPoint(x=b[0], y=b[1]),
        length_px=math.hypot(dx, dy),
        angle_deg=math.degrees(math.atan2(dy, dx)) % 180.0,
        dashed=False,
        evidence_ids=list(evidence_ids or [ident]),
        sources=["PYMUPDF"],
    )


def _curve(x=100, y=100, r=90):
    return DrawingCurve(
        id=f"CURVE_{x}_{y}",
        control_points=[
            DrawingPoint(x=x, y=y + r),
            DrawingPoint(x=x, y=y + r * 0.55),
            DrawingPoint(x=x + r * 0.55, y=y),
            DrawingPoint(x=x + r, y=y),
        ],
        bbox=DrawingBBox(x_min=x, y_min=y, x_max=x + r, y_max=y + r),
        evidence_ids=[f"CURVE_{x}_{y}"],
        sources=["PYMUPDF"],
    )


def _semantics_for_opening(*, family: str, space_bbox: DrawingBBox, space_name="SPACE_A"):
    space = SemanticObservation(
        id="SEM_SPACE",
        family="SPACE",
        bbox=space_bbox,
        confidence=0.9,
        payload={"semantic_name": space_name},
    )
    opening = SemanticObservation(
        id=f"SEM_{family}",
        family=family,
        bbox=None,
        confidence=0.9,
        payload={
            "semantic_name": f"{space_name}_{family}_1",
            "relations": [
                {
                    "type": "LOCATED_IN",
                    "target": space_name,
                    "target_category": "SPACE",
                    "state": "DETECTADO",
                }
            ],
        },
    )
    return [space, opening]


def _drawing(*, lines=None, curves=None, semantics=None):
    return DrawingModel(
        level_view_id="LV",
        width_px=1200,
        height_px=900,
        lines=list(lines or []),
        curves=list(curves or []),
        semantic_observations=list(semantics or []),
        diagnostics=DrawingModelDiagnostics(
            raw_evidence_count=0,
            normalized_line_count=len(lines or []),
            normalized_region_count=0,
            normalized_curve_count=len(curves or []),
            text_count=0,
            semantic_observation_count=len(semantics or []),
            lsd_line_count=0,
            region_centerline_count=0,
            source_counts={},
        ),
    )


def _candidate(ident, a, b, *, face_ids, metric=0.8, evidence_ids=None):
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    return WallCandidate(
        id=ident,
        generator="DOUBLE_FACE",
        start=DrawingPoint(x=a[0], y=a[1]),
        end=DrawingPoint(x=b[0], y=b[1]),
        angle_deg=math.degrees(math.atan2(dy, dx)) % 180.0,
        length_px=math.hypot(dx, dy),
        thickness_px=12.0,
        face_ids=list(face_ids),
        evidence_ids=list(evidence_ids or face_ids),
        source_names=["PYMUPDF"],
        evidence=WallEvidenceVector(
            pair_overlap=1.0,
            thickness_support=0.9,
            vector_support=1.0,
            raster_line_support=1.0,
            region_support=0.9,
            source_consensus=0.75,
            perimeter_containment=1.0,
        ),
        prior_score=0.8,
        metadata={"metric_wall_band_support": metric},
    )


def _png(width=200, height=120):
    image = Image.new("L", (width, height), 255)
    buf = BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def _level():
    width, height = 200, 120
    bbox = PixelBBox(x_min=0, y_min=0, x_max=width, y_max=height)
    return LevelView(
        id="LV",
        level_name="Nivel",
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
        confidence=1.0,
        evidence=[],
    )


def test_v5_drawing_builder_uses_semantic_category_as_family():
    evidence = RawEvidence(
        id="SEM_DOOR",
        level_view_id="LV",
        source="GEMINI",
        kind="GEMINI_OBSERVATION",
        geometry=EvidenceGeometry(geometry_type="NONE"),
        text="Puerta interior",
        confidence=0.9,
        metadata={"semantic_category": "DOOR"},
    )
    drawing = DrawingModelBuilder().build(level_view=_level(), evidence=[evidence])
    assert len(drawing.semantic_observations) == 1
    assert drawing.semantic_observations[0].family == "DOOR"


def test_v5_semantic_space_miss_does_not_veto_strong_local_door_geometry():
    curve = _curve(x=100, y=100, r=90)
    lines = [
        _line("LEAF", (100, 100), (100, 165)),
        _line("HOST", (20, 100), (390, 100)),
    ]
    semantics = _semantics_for_opening(
        family="DOOR",
        space_bbox=DrawingBBox(x_min=700, y_min=500, x_max=1050, y_max=820),
    )
    regions = ElementContextDetector().detect(
        drawing=_drawing(lines=lines, curves=[curve], semantics=semantics),
        scale_profile=_profile(),
        architectural_polygon=box(20, 20, 1100, 850),
    )
    doors = [region for region in regions if region.region_type == "DOOR_REGION"]
    assert doors
    assert doors[0].metadata["semantic_search_support"] == 0.0
    assert doors[0].metadata["semantic_miss_geometry"] is True


def test_v5_door_curve_outside_architectural_polygon_is_rejected_even_with_shape():
    curve = _curve(x=100, y=100, r=90)
    lines = [
        _line("LEAF", (100, 100), (100, 165)),
        _line("HOST", (20, 100), (390, 100)),
    ]
    regions = ElementContextDetector().detect(
        drawing=_drawing(lines=lines, curves=[curve]),
        scale_profile=_profile(),
        architectural_polygon=box(500, 400, 1100, 850),
    )
    assert not any(region.region_type == "DOOR_REGION" for region in regions)


def test_v5_isolated_curve_without_leaf_and_host_is_not_a_door():
    regions = ElementContextDetector().detect(
        drawing=_drawing(lines=[], curves=[_curve(x=100, y=100, r=90)]),
        scale_profile=_profile(),
        architectural_polygon=box(20, 20, 1100, 850),
    )
    assert not any(region.region_type == "DOOR_REGION" for region in regions)


def test_v5_semantic_space_allows_partial_door_geometry_inside_host_space():
    curve = _curve(x=100, y=100, r=90)
    lines = [
        _line("LEAF", (100, 100), (100, 148)),
        _line("HOST", (50, 100), (310, 100)),
    ]
    semantics = _semantics_for_opening(
        family="DOOR",
        space_bbox=DrawingBBox(x_min=70, y_min=70, x_max=330, y_max=330),
    )
    regions = ElementContextDetector().detect(
        drawing=_drawing(lines=lines, curves=[curve], semantics=semantics),
        scale_profile=_profile(),
        architectural_polygon=box(20, 20, 1100, 850),
    )
    doors = [region for region in regions if region.region_type == "DOOR_REGION"]
    assert doors
    assert doors[0].metadata["semantic_search_support"] > 0.0
    assert doors[0].metadata["opening_verified"] is True


def test_v5_window_outside_semantic_host_space_is_not_hard_verified():
    frames = [
        _line("W1", (300, 300), (390, 300)),
        _line("W2", (300, 306), (390, 306)),
        _line("W3", (300, 312), (390, 312)),
        _line("W4", (300, 318), (390, 318)),
    ]
    host_left = _line("HOST_LEFT", (170, 300), (295, 300))
    host_right = _line("HOST_RIGHT", (395, 300), (570, 300))
    semantics = _semantics_for_opening(
        family="WINDOW",
        space_bbox=DrawingBBox(x_min=700, y_min=500, x_max=1050, y_max=820),
    )
    regions = ElementContextDetector().detect(
        drawing=_drawing(lines=[*frames, host_left, host_right], semantics=semantics),
        scale_profile=_profile(),
        architectural_polygon=box(50, 50, 1100, 850),
    )
    windows = [region for region in regions if region.region_type == "WINDOW_REGION"]
    assert windows
    assert all(region.metadata["semantic_search_support"] == 0.0 for region in windows)
    assert all(region.metadata["opening_verified"] is False for region in windows)
    assert all(region.quarantine_enabled is False for region in windows)


def test_v5_floor_pair_without_direct_membership_stays_review_but_carries_pair_risk():
    region = ContextRegion(
        id="FLOOR",
        region_type="FLOOR_FINISH_GRID_REGION",
        bbox=DrawingBBox(x_min=0, y_min=0, x_max=200, y_max=200),
        confidence=0.98,
        quarantine_enabled=True,
        member_line_ids=["PATTERN_A", "PATTERN_B"],
        metadata={
            "pattern_segments": [
                [20.0, 94.0, 180.0, 94.0],
                [20.0, 106.0, 180.0, 106.0],
            ]
        },
    )
    candidate = _candidate(
        "WALL",
        (20, 100),
        (180, 100),
        face_ids=["FACE_1", "FACE_2"],
        evidence_ids=["OTHER_EVIDENCE"],
        metric=0.75,
    )
    gate = CandidateContextGate()
    decision = gate._evaluate(candidate=candidate, region=region)
    assert decision is not None
    assert decision.state == "REVIEW"
    assert float(decision.metadata.get("pattern_pair_provenance", 0.0)) >= 0.70
    assert gate._context_risk(decision) >= 0.65


def test_v5_stair_profile_pair_without_real_segment_provenance_cannot_quarantine():
    region = ContextRegion(
        id="STAIR",
        region_type="STAIR_FLIGHT_REGION",
        bbox=DrawingBBox(x_min=0, y_min=0, x_max=220, y_max=220),
        confidence=0.96,
        quarantine_enabled=True,
        member_line_ids=["STAIR_A", "STAIR_B", "STAIR_C"],
        profiles=[RepetitiveAxisProfile(
            angle_deg=0.0,
            spacing_px=12.0,
            spacing_cv=0.05,
            member_count=6,
            median_length_px=160.0,
            common_span_ratio=1.0,
            length_similarity=1.0,
            track_coordinates=[82.0, 94.0, 106.0, 118.0, 130.0, 142.0],
            member_line_ids=["STAIR_A", "STAIR_B", "STAIR_C"],
        )],
        metadata={},
    )
    candidate = _candidate(
        "STAIR_WALL",
        (20, 100),
        (180, 100),
        face_ids=["FACE_1", "FACE_2"],
        metric=0.80,
    )
    gate = CandidateContextGate()
    decision = gate._evaluate(candidate=candidate, region=region)
    assert decision is not None
    assert decision.state != "QUARANTINE"
    assert float(decision.metadata.get("pattern_pair_provenance", 0.0)) == 0.0
