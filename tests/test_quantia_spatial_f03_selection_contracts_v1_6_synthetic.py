from __future__ import annotations

from app.quantia_spatialV1.reconstruction_core.candidate_context_gate import CandidateContextGate
from app.quantia_spatialV1.reconstruction_core.candidate_models import WallCandidate, WallEvidenceVector
from app.quantia_spatialV1.reconstruction_core.context_models import ContextRegion, RepetitiveAxisProfile
from app.quantia_spatialV1.reconstruction_core.drawing_model import DrawingBBox, DrawingPoint


def _candidate(
    candidate_id: str,
    *,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    thickness: float = 12.0,
    axis_support: float = 0.0,
    dashed_penalty: float = 0.0,
    face_ids: list[str] | None = None,
    metric_band_support: float = 1.0,
) -> WallCandidate:
    dx, dy = x2 - x1, y2 - y1
    angle = 0.0 if abs(dy) < 1e-9 else 90.0 if abs(dx) < 1e-9 else 45.0
    return WallCandidate(
        id=candidate_id,
        generator="DOUBLE_FACE",
        start=DrawingPoint(x=x1, y=y1),
        end=DrawingPoint(x=x2, y=y2),
        angle_deg=angle,
        length_px=(dx * dx + dy * dy) ** 0.5,
        thickness_px=thickness,
        face_ids=face_ids or [f"{candidate_id}_A", f"{candidate_id}_B"],
        evidence_ids=[],
        source_names=["LSD_LOCAL", "OPENCV"],
        evidence=WallEvidenceVector(
            pair_overlap=0.96,
            thickness_support=0.88,
            vector_support=0.0,
            raster_line_support=1.0,
            region_support=0.94,
            source_consensus=0.50,
            perimeter_containment=1.0,
            semantic_support=0.0,
            axis_support=axis_support,
            dashed_penalty=dashed_penalty,
        ),
        prior_score=0.78,
        metadata={
            "metric_thickness_support": metric_band_support,
            "metric_slenderness_support": 1.0,
            "metric_wall_band_support": metric_band_support,
        },
    )


def _axis_region() -> ContextRegion:
    return ContextRegion(
        id="AXIS",
        region_type="AXIS_GRID_REGION",
        bbox=DrawingBBox(x_min=0, y_min=0, x_max=300, y_max=300),
        confidence=0.96,
        quarantine_enabled=False,
        profiles=[],
        member_line_ids=["AXIS_1", "AXIS_2"],
        metadata={"negative_mask": False},
    )


def _floor_region(*, profiles: bool = False) -> ContextRegion:
    profile_list = []
    if profiles:
        profile_list = [
            RepetitiveAxisProfile(
                angle_deg=0.0,
                spacing_px=20.0,
                spacing_cv=0.02,
                member_count=6,
                median_length_px=180.0,
                common_span_ratio=0.95,
                length_similarity=0.95,
                track_coordinates=[70, 90, 110, 130, 150, 170],
                member_line_ids=[f"F{i}" for i in range(6)],
            )
        ]
    return ContextRegion(
        id="FLOOR",
        region_type="FLOOR_FINISH_GRID_REGION",
        bbox=DrawingBBox(x_min=40, y_min=40, x_max=260, y_max=190),
        confidence=0.96,
        quarantine_enabled=True,
        profiles=profile_list,
        member_line_ids=[f"F{i}" for i in range(6)],
        metadata={
            "pattern_segments": [
                [50.0, 70.0, 250.0, 70.0],
                [50.0, 90.0, 250.0, 90.0],
                [50.0, 110.0, 250.0, 110.0],
                [50.0, 130.0, 250.0, 130.0],
                [50.0, 150.0, 250.0, 150.0],
                [50.0, 170.0, 250.0, 170.0],
            ]
        },
    )


def _stair_region() -> ContextRegion:
    return ContextRegion(
        id="STAIR",
        region_type="STAIR_FLIGHT_REGION",
        bbox=DrawingBBox(x_min=45, y_min=45, x_max=250, y_max=205),
        confidence=0.90,
        quarantine_enabled=True,
        profiles=[
            RepetitiveAxisProfile(
                angle_deg=0.0,
                spacing_px=20.0,
                spacing_cv=0.02,
                member_count=6,
                median_length_px=180.0,
                common_span_ratio=0.95,
                length_similarity=0.95,
                track_coordinates=[70, 90, 110, 130, 150, 170],
                member_line_ids=[f"S{i}" for i in range(6)],
            )
        ],
        member_line_ids=[f"S{i}" for i in range(6)],
        metadata={
            "pattern_segments": [
                [52.0, 70.0, 230.0, 70.0],
                [52.0, 90.0, 230.0, 90.0],
                [52.0, 110.0, 230.0, 110.0],
                [52.0, 130.0, 230.0, 130.0],
                [52.0, 150.0, 230.0, 150.0],
                [52.0, 170.0, 230.0, 170.0],
            ]
        },
    )


def test_axis_bbox_does_not_quarantine_plausible_wall_centered_on_axis() -> None:
    wall = _candidate(
        "WALL_ON_AXIS",
        x1=40,
        y1=120,
        x2=260,
        y2=120,
        thickness=14.0,
        axis_support=0.99,
        metric_band_support=1.0,
    )
    decision = CandidateContextGate().apply(candidates=[wall], regions=[_axis_region()]).decisions[0]
    assert decision.state == "ACTIVE"
    assert decision.reason == "ACTIVE_AXIS_LOCATOR_NOT_NEGATIVE_MASK"
    assert decision.metadata["negative_mask"] is False
    assert decision.metadata["reference_support"] == 0.0


def test_direct_thin_axis_primitive_can_still_be_quarantined() -> None:
    axis = _candidate(
        "AXIS_PRIMITIVE",
        x1=40,
        y1=120,
        x2=260,
        y2=120,
        thickness=2.0,
        axis_support=1.0,
        dashed_penalty=0.95,
        face_ids=["AXIS_1", "OTHER"],
        metric_band_support=0.20,
    )
    decision = CandidateContextGate().apply(candidates=[axis], regions=[_axis_region()]).decisions[0]
    assert decision.state == "QUARANTINE"
    assert decision.reason == "QUARANTINE_DIRECT_AXIS_REFERENCE_PRIMITIVE"


def test_floor_containment_alone_is_review_not_quarantine() -> None:
    wall = _candidate("INTERIOR_WALL", x1=60, y1=120, x2=240, y2=120, thickness=14.0)
    region = _floor_region(profiles=False)
    decision = CandidateContextGate().apply(candidates=[wall], regions=[region]).decisions[0]
    assert decision.state == "REVIEW"
    assert decision.reason == "REVIEW_FLOOR_CONTAINMENT_WITHOUT_PATTERN_MEMBERSHIP"


def test_floor_pair_without_direct_provenance_is_review_even_if_bracketed() -> None:
    pair = _candidate("GRID_PAIR", x1=60, y1=100, x2=240, y2=100, thickness=20.0)
    region = ContextRegion(
        id="FLOOR_PAIR",
        region_type="FLOOR_FINISH_GRID_REGION",
        bbox=DrawingBBox(x_min=40, y_min=80, x_max=260, y_max=120),
        confidence=0.96,
        quarantine_enabled=True,
        profiles=[],
        member_line_ids=[],
        metadata={
            "pattern_segments": [
                [50.0, 90.0, 250.0, 90.0],
                [50.0, 110.0, 250.0, 110.0],
            ]
        },
    )
    decision = CandidateContextGate().apply(candidates=[pair], regions=[region]).decisions[0]
    assert decision.metadata["pattern_pair_support"] >= 0.90
    assert decision.metadata["pattern_pair_provenance"] >= 0.90
    assert decision.state == "REVIEW"
    assert decision.reason == "REVIEW_FLOOR_CONTAINMENT_WITHOUT_PATTERN_MEMBERSHIP"


def test_floor_direct_member_can_still_be_quarantined() -> None:
    pair = _candidate(
        "GRID_MEMBER", x1=60, y1=100, x2=240, y2=100, thickness=20.0,
        face_ids=["FLOOR_LINE", "OTHER_FACE"], metric_band_support=0.40,
    )
    region = ContextRegion(
        id="FLOOR_DIRECT",
        region_type="FLOOR_FINISH_GRID_REGION",
        bbox=DrawingBBox(x_min=40, y_min=80, x_max=260, y_max=120),
        confidence=0.96,
        quarantine_enabled=True,
        profiles=[],
        member_line_ids=["FLOOR_LINE"],
        metadata={"pattern_segments": [[50.0, 100.0, 250.0, 100.0]]},
    )
    decision = CandidateContextGate().apply(candidates=[pair], regions=[region]).decisions[0]
    assert decision.state == "QUARANTINE"
    assert decision.metadata["direct_pattern_member"] is True


def test_spatial_alignment_with_profile_is_not_enough_for_quarantine() -> None:
    # Coincide con una pista/perfil por posición, pero no usa sus primitivas ni está
    # bracketed por el patrón. Debe sobrevivir como REVIEW para el solver.
    wall = _candidate("PROFILE_ALIGNED_WALL", x1=60, y1=130, x2=240, y2=130, thickness=14.0)
    region = _floor_region(profiles=True)
    # Retiramos el segmento y=130 para que no haya pertenencia geométrica directa;
    # la coincidencia queda solo por profile coordinate / bbox.
    region = region.model_copy(update={
        "metadata": {
            "pattern_segments": [
                [50.0, 70.0, 250.0, 70.0],
                [50.0, 90.0, 250.0, 90.0],
                [50.0, 110.0, 250.0, 110.0],
                [50.0, 150.0, 250.0, 150.0],
                [50.0, 170.0, 250.0, 170.0],
            ]
        }
    })
    decision = CandidateContextGate().apply(candidates=[wall], regions=[region]).decisions[0]
    assert decision.state == "REVIEW"
    assert decision.reason.startswith("AMBIGUOUS_") or decision.reason.startswith("REVIEW_")


def test_stair_interior_crossed_by_multiple_treads_can_still_be_quarantined() -> None:
    inner = _candidate("INNER_STAIR_GRAPHIC", x1=150, y1=55, x2=150, y2=190, thickness=12.0)
    decision = CandidateContextGate().apply(candidates=[inner], regions=[_stair_region()]).decisions[0]
    assert decision.metadata["pattern_crossing_support"] >= 0.65
    assert decision.state == "QUARANTINE"


def test_stair_side_wall_where_treads_terminate_is_review() -> None:
    side_wall = _candidate("STAIR_SIDE_WALL", x1=50, y1=55, x2=50, y2=190, thickness=12.0)
    decision = CandidateContextGate().apply(candidates=[side_wall], regions=[_stair_region()]).decisions[0]
    assert decision.metadata["region_boundary_support"] >= 0.58
    assert decision.state == "REVIEW"
