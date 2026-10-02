from __future__ import annotations

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
    PostReconstructionFilterEngine,
    WallHypothesisCanonicalizer,
)
from app.quantia_spatialV1.stages.walls.postfilter.contracts import (
    FilterModuleDecision,
    PostFilterPlan,
)


def _route() -> AdaptiveRoutePlan:
    return AdaptiveRoutePlan(
        mode="BALANCED",
        recovery_strength=0.6,
        modules=[ModuleDecision(module="WALL_CANONICALIZATION", enabled=True, reason="test")],
    )


def _wall(
    wid: str,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    role: str = "REVIEW",
    confidence: float = 0.45,
    thickness: float = 8.0,
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
    value = 0.80 if strong else 0.10
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
        value,
        value,
        value,
        value,
        value,
        0.0,
        0.0,
        "ACTIVE",
    )


def _runtime(walls: list[SingleLineWall], *, strong_ids: set[str] | None = None) -> AdaptiveReconstructionRuntime:
    strong_ids = strong_ids or set()
    review = sum(item.role == "REVIEW" for item in walls)
    diag = ReconstructionDiagnostics(
        level_view_id="L1",
        f03_seed_count=sum(item.f03_seed_protected for item in walls),
        discovered_candidate_count=len(walls),
        quarantined_candidate_count=0,
        hybrid_candidate_count=len(walls),
        selected_wall_count=len(walls),
        component_count=1,
        junction_count=0,
        virtual_bridge_count=0,
        interior_space_count=0,
        review_wall_count=review,
        perimeter_wall_count=sum(item.role == "PERIMETER" for item in walls),
        divider_wall_count=sum(item.role == "DIVIDER" for item in walls),
        seed_coverage_ratio=0.0,
        quarantine_ratio=0.0,
        review_ratio=review / max(len(walls), 1),
    )
    graph = SingleLineWallGraph(
        level_view_id="L1",
        level_name="Nivel",
        image_size_px=(500, 500),
        px_per_m=100.0,
        walls=walls,
        logical_gaps=[],
        interior_space_count=0,
        route_plan=_route(),
        diagnostics=diag,
    )
    topology = SpaceTopology(
        labels=np.zeros((500, 500), dtype=np.int32),
        exterior_labels=frozenset(),
        interior_labels=frozenset(),
        areas_px2={},
    )
    return AdaptiveReconstructionRuntime(
        graph=graph,
        wall_tracks=[_track(item, strong=item.id in strong_ids) for item in walls],
        bridges=[],
        topology=topology,
        context_quarantined_count=0,
        artifacts={},
    )


class _TopologyOnlySelector:
    def plan(self, *, runtime, pattern_evidence=()):
        del runtime, pattern_evidence
        return PostFilterPlan(
            modules=[
                FilterModuleDecision(
                    module="TOPOLOGY_GUARD",
                    enabled=True,
                    reason="solo guard estructural para probar que el selector gobierna la ejecucion",
                )
            ]
        )

    @staticmethod
    def _count_near_coincident_pairs(walls, px_per_m):
        del walls, px_per_m
        return 0


def test_v3_selector_plan_is_operational_not_only_audit() -> None:
    walls = [_wall(f"S{i}", (100, 100 + i * 12), (230, 100 + i * 12)) for i in range(7)]
    result = PostReconstructionFilterEngine(selector=_TopologyOnlySelector()).run(runtime=_runtime(walls))
    assert any(item.kind == "REPETITIVE_PARALLEL" for item in result.pattern_evidence)
    assert not result.architectural_elements
    assert not result.excluded_graphics
    assert len(result.unresolved) == 7
    assert result.audit["selector_is_operational"] is True


def test_v3_enabled_pattern_records_responsible_module() -> None:
    walls = [_wall(f"S{i}", (100, 100 + i * 12), (230, 100 + i * 12)) for i in range(7)]
    result = PostReconstructionFilterEngine().run(runtime=_runtime(walls))
    assert result.architectural_elements
    assert all(
        "REPETITIVE_PATTERN_FILTER" in item.decision.applied_modules
        for item in result.architectural_elements
    )


def test_v3_near_coincident_role_conflict_becomes_one_centerline() -> None:
    walls = [
        _wall("A", (20, 100), (220, 100), role="PERIMETER", confidence=0.85, thickness=12),
        _wall("B", (20, 105), (220, 105), role="DIVIDER", confidence=0.82, thickness=12),
    ]
    canonical, mapping = WallHypothesisCanonicalizer().canonicalize(
        walls=walls,
        px_per_m=100.0,
        enable_duplicates=True,
        enable_near_coincident=True,
        enable_wall_faces=False,
    )
    assert len(canonical) == 1
    assert abs(canonical[0].start_px[1] - 102.5) < 1e-6
    assert mapping[0].reason == "near_coincident_centerline_bundle"
    assert set(mapping[0].source_wall_ids) == {"A", "B"}


def test_v3_near_coincident_does_not_absorb_short_unrelated_segment() -> None:
    walls = [
        _wall("LONG", (20, 100), (320, 100), role="DIVIDER", confidence=0.8, thickness=12),
        _wall("SHORT", (100, 105), (150, 105), role="REVIEW", confidence=0.5, thickness=4),
    ]
    canonical, _ = WallHypothesisCanonicalizer().canonicalize(
        walls=walls,
        px_per_m=100.0,
        enable_duplicates=False,
        enable_near_coincident=True,
        enable_wall_faces=False,
    )
    assert len(canonical) == 2


def test_v3_selector_can_disable_all_canonicalizers() -> None:
    walls = [
        _wall("A", (20, 100), (220, 100), role="DIVIDER", confidence=0.85, thickness=12),
        _wall("B", (20, 105), (220, 105), role="DIVIDER", confidence=0.82, thickness=12),
    ]
    runtime = _runtime(walls, strong_ids={"A", "B"})
    result = PostReconstructionFilterEngine(selector=_TopologyOnlySelector()).run(runtime=runtime)
    assert len(result.filtered_wall_graph.walls) == 2
    assert result.diagnostics.collapsed_hypothesis_count == 0
