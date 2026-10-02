from __future__ import annotations

from app.quantia_spatialV1.core.models.evidence import (
    EvidenceGeometry,
    RawEvidence,
)
from app.quantia_spatialV1.core.models.level_view import PixelBBox
from app.quantia_spatialV1.stages.spaces import (
    SpaceClosureEngine,
    SpaceConstraintValidator,
)
from app.quantia_spatialV1.stages.walls.adaptive.contracts import (
    AdaptiveRoutePlan,
    ModuleDecision,
    ReconstructionDiagnostics,
    SingleLineWall,
    SingleLineWallGraph,
)


def _graph() -> SingleLineWallGraph:
    walls = [
        SingleLineWall(
            id="T",
            start_px=(0, 0),
            end_px=(1000, 0),
            thickness_px=12,
            role="PERIMETER",
            confidence=0.99,
        ),
        SingleLineWall(
            id="R",
            start_px=(1000, 0),
            end_px=(1000, 600),
            thickness_px=12,
            role="PERIMETER",
            confidence=0.99,
        ),
        SingleLineWall(
            id="B",
            start_px=(1000, 600),
            end_px=(0, 600),
            thickness_px=12,
            role="PERIMETER",
            confidence=0.99,
        ),
        SingleLineWall(
            id="L",
            start_px=(0, 600),
            end_px=(0, 0),
            thickness_px=12,
            role="PERIMETER",
            confidence=0.99,
        ),
        SingleLineWall(
            id="D",
            start_px=(500, 0),
            end_px=(500, 600),
            thickness_px=10,
            role="DIVIDER",
            confidence=0.95,
        ),
    ]
    route = AdaptiveRoutePlan(
        mode="BALANCED",
        modules=[
            ModuleDecision(
                module="ROOM_TOPOLOGY",
                enabled=True,
                reason="test",
            )
        ],
        recovery_strength=0.5,
    )
    diagnostics = ReconstructionDiagnostics(
        level_view_id="LV",
        f03_seed_count=5,
        discovered_candidate_count=0,
        quarantined_candidate_count=0,
        hybrid_candidate_count=5,
        selected_wall_count=5,
        component_count=1,
        junction_count=6,
        virtual_bridge_count=0,
        interior_space_count=2,
        review_wall_count=0,
        perimeter_wall_count=4,
        divider_wall_count=1,
        seed_coverage_ratio=1,
        quarantine_ratio=0,
        review_ratio=0,
    )
    return SingleLineWallGraph(
        level_view_id="LV",
        level_name="PB",
        image_size_px=(1000, 600),
        px_per_m=100,
        walls=walls,
        logical_gaps=[],
        interior_space_count=2,
        route_plan=route,
        diagnostics=diagnostics,
    )


def _evidence() -> list[RawEvidence]:
    contour = RawEvidence(
        id="FP",
        level_view_id="LV",
        source="OPENCV",
        kind="RASTER_CONTOUR",
        geometry=EvidenceGeometry(
            geometry_type="NONE",
        ),
        confidence=0.9,
        metadata={
            "closed": True,
            "area_px2": 600000,
            "raw_contour_points_px": [
                {"x": 0, "y": 0},
                {"x": 1000, "y": 0},
                {"x": 1000, "y": 600},
                {"x": 0, "y": 600},
            ],
        },
    )
    first = RawEvidence(
        id="S1",
        level_view_id="LV",
        source="GEMINI",
        kind="GEMINI_OBSERVATION",
        geometry=EvidenceGeometry(
            geometry_type="BBOX",
            bbox_px=PixelBBox(
                x_min=80,
                y_min=80,
                x_max=420,
                y_max=520,
            ),
        ),
        confidence=0.9,
        metadata={
            "semantic_category": "SPACE",
            "semantic_name": "A",
            "properties": [
                {
                    "name": "id_proposed",
                    "value_text": "A",
                }
            ],
        },
    )
    second = RawEvidence(
        id="S2",
        level_view_id="LV",
        source="GEMINI",
        kind="GEMINI_OBSERVATION",
        geometry=EvidenceGeometry(
            geometry_type="BBOX",
            bbox_px=PixelBBox(
                x_min=580,
                y_min=80,
                x_max=920,
                y_max=520,
            ),
        ),
        confidence=0.9,
        metadata={
            "semantic_category": "SPACE",
            "semantic_name": "B",
            "properties": [
                {
                    "name": "id_proposed",
                    "value_text": "B",
                }
            ],
        },
    )
    return [contour, first, second]


def test_space_closure_builds_two_faces_without_mutating_wallgraph() -> None:
    graph = _graph()
    before = graph.model_dump(mode="json")

    result = SpaceClosureEngine().run(
        graph=graph
    )

    assert len(result.spaces) == 2
    assert (
        result.diagnostics.meaningful_space_count
        == 2
    )
    assert graph.model_dump(mode="json") == before


def test_space_constraint_matches_semantics_to_faces() -> None:
    graph = _graph()
    closure = SpaceClosureEngine().run(
        graph=graph
    )

    result = SpaceConstraintValidator().validate(
        closure=closure,
        evidence=_evidence(),
        final_graph=graph,
    )

    assert result.state == "VALID"
    assert len(result.faces) == 2
    assert all(
        face.state == "SEMANTICALLY_SEPARATED"
        for face in result.faces
    )
    assert all(
        item.status == "MATCHED"
        for item in result.assignments
    )


def test_multiple_semantics_in_one_face_is_review_not_invented_wall() -> None:
    graph = _graph().model_copy(deep=True)
    graph.walls = [
        wall
        for wall in graph.walls
        if wall.id != "D"
    ]
    graph.diagnostics.selected_wall_count = 4
    graph.diagnostics.divider_wall_count = 0
    graph.interior_space_count = 1

    closure = SpaceClosureEngine().run(
        graph=graph
    )
    result = SpaceConstraintValidator().validate(
        closure=closure,
        evidence=_evidence(),
        final_graph=graph,
    )

    assert result.state == "REVIEW"
    assert any(
        face.state
        == "UNDER_SEGMENTED_MULTIPLE_SPACES"
        for face in result.faces
    )
    assert "SPACE_UNDER_SEGMENTED" in result.warnings
