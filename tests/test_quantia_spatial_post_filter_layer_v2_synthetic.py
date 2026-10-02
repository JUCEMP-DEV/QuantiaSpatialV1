from __future__ import annotations

import copy

import numpy as np

from app.quantia_spatialV1.stages.walls.adaptive.contracts import (
    AdaptiveReconstructionRuntime,
    AdaptiveRoutePlan,
    ModuleDecision,
    PhysicalWallTrack,
    ReconstructionDiagnostics,
    SingleLineWall,
    SingleLineWallGraph,
    SpaceTopology,
)
from app.quantia_spatialV1.stages.walls.postfilter import (
    PostFilterGeometryPatternAnalyzer,
    PostReconstructionFilterEngine,
    WallHypothesisCanonicalizer,
)


def _plan() -> AdaptiveRoutePlan:
    return AdaptiveRoutePlan(
        mode="BALANCED",
        recovery_strength=0.6,
        modules=[ModuleDecision(module="WALL_CANONICALIZATION", enabled=True, reason="test")],
    )


def _diag(*, walls: int, review: int | None = None) -> ReconstructionDiagnostics:
    review = walls if review is None else review
    return ReconstructionDiagnostics(
        level_view_id="L1",
        f03_seed_count=0,
        discovered_candidate_count=walls,
        quarantined_candidate_count=0,
        hybrid_candidate_count=walls,
        selected_wall_count=walls,
        component_count=1,
        junction_count=0,
        virtual_bridge_count=0,
        interior_space_count=0,
        review_wall_count=review,
        perimeter_wall_count=0,
        divider_wall_count=walls - review,
        seed_coverage_ratio=0.0,
        quarantine_ratio=0.0,
        review_ratio=review / max(walls, 1),
    )


def _wall(
    wid: str,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    thickness: float = 3.0,
    role: str = "REVIEW",
    confidence: float = 0.45,
    f03: bool = False,
) -> SingleLineWall:
    return SingleLineWall(
        id=wid,
        start_px=start,
        end_px=end,
        thickness_px=thickness,
        role=role,
        confidence=confidence,
        source_candidate_ids=[f"C_{wid}"],
        evidence_ids=[f"E_{wid}"],
        source_names=["TEST"],
        f03_seed_protected=f03,
    )


def _track(wall: SingleLineWall, *, strong: bool = False) -> PhysicalWallTrack:
    dx = wall.end_px[0] - wall.start_px[0]
    dy = wall.end_px[1] - wall.start_px[1]
    length = float((dx * dx + dy * dy) ** 0.5)
    v = 0.80 if strong else 0.15
    return PhysicalWallTrack(
        wall.id,
        wall.start_px,
        wall.end_px,
        length,
        wall.thickness_px,
        wall.confidence,
        tuple(wall.source_candidate_ids),
        tuple(wall.evidence_ids),
        tuple(wall.source_names),
        v,
        v,
        v,
        v,
        v,
        0.0,
        0.0,
        "ACTIVE",
    )


def _runtime(walls: list[SingleLineWall], *, strong_ids: set[str] | None = None) -> AdaptiveReconstructionRuntime:
    strong_ids = strong_ids or set()
    graph = SingleLineWallGraph(
        level_view_id="L1",
        level_name="Nivel",
        image_size_px=(500, 500),
        px_per_m=100.0,
        walls=walls,
        logical_gaps=[],
        interior_space_count=0,
        route_plan=_plan(),
        diagnostics=_diag(walls=len(walls), review=sum(w.role == "REVIEW" for w in walls)),
    )
    topology = SpaceTopology(
        labels=np.zeros((500, 500), dtype=np.int32),
        exterior_labels=frozenset(),
        interior_labels=frozenset(),
        areas_px2={},
    )
    return AdaptiveReconstructionRuntime(
        graph=graph,
        wall_tracks=[_track(wall, strong=wall.id in strong_ids) for wall in walls],
        bridges=[],
        topology=topology,
        context_quarantined_count=0,
        artifacts={},
    )


def test_v2_repetitive_parallel_family_is_encapsulated_not_deleted() -> None:
    walls = [
        _wall(f"S{i}", (100, 100 + i * 12), (230, 100 + i * 12))
        for i in range(7)
    ]
    result = PostReconstructionFilterEngine().run(runtime=_runtime(walls))
    assert any(item.kind == "REPETITIVE_PARALLEL" for item in result.pattern_evidence)
    assert len(result.architectural_elements) == 7
    assert all(item.decision.preserved for item in result.architectural_elements)
    assert not result.filtered_wall_graph.walls


def test_v2_orthogonal_repetitive_grid_is_graphic_evidence() -> None:
    walls = [
        *[_wall(f"H{i}", (100, 100 + i * 15), (250, 100 + i * 15)) for i in range(6)],
        *[_wall(f"V{i}", (100 + i * 15, 100), (100 + i * 15, 250)) for i in range(6)],
    ]
    result = PostReconstructionFilterEngine().run(runtime=_runtime(walls))
    assert any(item.kind == "ORTHOGONAL_GRID" for item in result.pattern_evidence)
    assert len(result.excluded_graphics) == 12
    assert all(item.decision.preserved for item in result.excluded_graphics)


def test_v2_angular_hub_is_encapsulated_as_roof_cover_candidate() -> None:
    walls = [
        _wall("R1", (100, 100), (220, 145)),
        _wall("R2", (100, 100), (150, 220)),
        _wall("R3", (100, 100), (20, 190)),
        _wall("R4", (100, 100), (15, 70)),
    ]
    result = PostReconstructionFilterEngine().run(runtime=_runtime(walls))
    hubs = [item for item in result.pattern_evidence if item.kind == "ANGULAR_HUB"]
    assert hubs
    assert {item.decision.architectural_family_hint for item in result.architectural_elements} == {
        "ROOF_COVER_CANDIDATE"
    }


def test_v2_structural_conflict_is_unresolved_not_excluded() -> None:
    walls = [
        _wall(
            f"S{i}",
            (100, 100 + i * 12),
            (230, 100 + i * 12),
            role="DIVIDER" if i == 2 else "REVIEW",
            confidence=0.90 if i == 2 else 0.45,
        )
        for i in range(7)
    ]
    result = PostReconstructionFilterEngine().run(runtime=_runtime(walls, strong_ids={"S2"}))
    s2 = next(item for item in result.decisions if item.source_wall_id == "S2")
    assert s2.output_class == "UNRESOLVED"
    assert s2.preserved is True


def test_v2_wall_face_pair_becomes_one_centerline() -> None:
    walls = [
        _wall("A", (20, 100), (220, 100), thickness=3, role="DIVIDER", confidence=0.8),
        _wall("B", (20, 115), (220, 115), thickness=3, role="REVIEW", confidence=0.7),
    ]
    canonical, mapping = WallHypothesisCanonicalizer().canonicalize(walls=walls, px_per_m=100.0)
    assert len(canonical) == 1
    assert abs(canonical[0].start_px[1] - 107.5) < 1e-6
    assert canonical[0].thickness_px >= 15.0
    assert mapping[0].reason == "parallel_wall_faces_to_centerline"
    assert set(mapping[0].source_wall_ids) == {"A", "B"}


def test_v2_two_real_thick_parallel_walls_are_not_collapsed_as_faces() -> None:
    walls = [
        _wall("A", (20, 100), (220, 100), thickness=15, role="DIVIDER", confidence=0.8),
        _wall("B", (20, 120), (220, 120), thickness=15, role="DIVIDER", confidence=0.8),
    ]
    canonical, _ = WallHypothesisCanonicalizer().canonicalize(walls=walls, px_per_m=100.0)
    assert len(canonical) == 2


def test_v2_filter_does_not_mutate_adaptive_input_graph() -> None:
    walls = [
        _wall("A", (20, 100), (220, 100), role="DIVIDER", confidence=0.8),
        _wall("B", (20, 115), (220, 115), role="REVIEW", confidence=0.7),
    ]
    runtime = _runtime(walls, strong_ids={"A"})
    before = copy.deepcopy(runtime.graph.model_dump(mode="python"))
    PostReconstructionFilterEngine().run(runtime=runtime)
    assert runtime.graph.model_dump(mode="python") == before


def test_v2_analyzer_does_not_classify_two_parallel_walls_as_repetitive_pattern() -> None:
    walls = [
        _wall("A", (20, 100), (220, 100)),
        _wall("B", (20, 115), (220, 115)),
    ]
    patterns = PostFilterGeometryPatternAnalyzer().analyze(walls=walls, px_per_m=100.0)
    assert patterns == []
