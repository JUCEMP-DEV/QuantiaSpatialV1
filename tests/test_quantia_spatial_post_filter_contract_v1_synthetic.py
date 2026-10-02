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
from app.quantia_spatialV1.stages.walls.postfilter import PostReconstructionFilterEngine
from app.quantia_spatialV1.stages.walls.core.candidate_models import WallCandidate, WallEvidenceVector
from app.quantia_spatialV1.stages.walls.core.drawing_model import DrawingPoint


def _plan() -> AdaptiveRoutePlan:
    return AdaptiveRoutePlan(
        mode="BALANCED",
        recovery_strength=0.6,
        modules=[ModuleDecision(module="WALL_CANONICALIZATION", enabled=True, reason="test")],
    )


def _diag() -> ReconstructionDiagnostics:
    return ReconstructionDiagnostics(
        level_view_id="L1",
        f03_seed_count=1,
        discovered_candidate_count=3,
        quarantined_candidate_count=1,
        hybrid_candidate_count=3,
        selected_wall_count=3,
        component_count=1,
        junction_count=1,
        virtual_bridge_count=0,
        interior_space_count=1,
        review_wall_count=2,
        perimeter_wall_count=1,
        divider_wall_count=0,
        seed_coverage_ratio=1.0,
        quarantine_ratio=1/3,
        review_ratio=2/3,
    )


def _candidate(cid: str, *, state: str = "ACTIVE", region: str | None = None) -> WallCandidate:
    return WallCandidate(
        id=cid,
        generator="REGION_CENTERLINE",
        start=DrawingPoint(x=0, y=0),
        end=DrawingPoint(x=100, y=0),
        angle_deg=0,
        length_px=100,
        thickness_px=12,
        face_ids=[],
        evidence_ids=[f"E_{cid}"],
        source_names=["TEST"],
        evidence=WallEvidenceVector(
            pair_overlap=0.8,
            thickness_support=0.8,
            vector_support=0.8,
            raster_line_support=0.8,
            region_support=0.8,
            source_consensus=0.8,
            perimeter_containment=0.0,
            semantic_support=0.0,
            axis_support=0.0,
            dashed_penalty=0.0,
        ),
        prior_score=0.8,
        metadata={
            "adaptive_context_state": state,
            "adaptive_context_region_type": region,
        },
    )


def _runtime() -> AdaptiveReconstructionRuntime:
    walls = [
        SingleLineWall(
            id="PW_1", start_px=(0, 0), end_px=(100, 0), thickness_px=12,
            role="PERIMETER", confidence=0.9, source_candidate_ids=["C1"],
            evidence_ids=["E1"], source_names=["TEST"], f03_seed_protected=True,
        ),
        SingleLineWall(
            id="PW_2", start_px=(1, 2), end_px=(99, 2), thickness_px=11,
            role="REVIEW", confidence=0.7, source_candidate_ids=["C2"],
            evidence_ids=["E2"], source_names=["TEST"],
        ),
        SingleLineWall(
            id="PW_3", start_px=(10, 30), end_px=(90, 30), thickness_px=2,
            role="REVIEW", confidence=0.4, source_candidate_ids=["C3"],
            evidence_ids=["E3"], source_names=["TEST"], context_state="REVIEW",
        ),
    ]
    graph = SingleLineWallGraph(
        level_view_id="L1", level_name="Nivel", image_size_px=(200, 200), px_per_m=100,
        walls=walls, logical_gaps=[], interior_space_count=1, route_plan=_plan(), diagnostics=_diag(),
    )
    tracks = [
        PhysicalWallTrack("PW_1", (0,0),(100,0),100,12,0.9,("C1",),("E1",),("TEST",),0.8,0.8,0.8,0.8,0.8,0,0),
        PhysicalWallTrack("PW_2", (1,2),(99,2),98,11,0.7,("C2",),("E2",),("TEST",),0.8,0.8,0.8,0.8,0.8,0,0),
        PhysicalWallTrack("PW_3", (10,30),(90,30),80,2,0.4,("C3",),("E3",),("TEST",),0.1,0.1,0.4,0.2,0.2,0,0.8,"REVIEW"),
    ]
    topology = SpaceTopology(labels=np.zeros((200,200), dtype=np.int32), exterior_labels=frozenset(), interior_labels=frozenset({1}), areas_px2={1: 1000})
    return AdaptiveReconstructionRuntime(
        graph=graph, wall_tracks=tracks, bridges=[], topology=topology,
        context_quarantined_count=1,
        artifacts={
            "discovered_candidates": [
                _candidate("C1"),
                _candidate("C2"),
                _candidate("C3", state="QUARANTINE", region="DIMENSION_REGION"),
            ],
            "hybrid_candidates": [_candidate("C1"), _candidate("C2")],
            "quarantined_candidates": [_candidate("C3", state="QUARANTINE", region="DIMENSION_REGION")],
        },
    )


def test_filter_preserves_evidence_and_excludes_graphics_without_deleting() -> None:
    result = PostReconstructionFilterEngine().run(runtime=_runtime())
    assert result.audit["evidence_preserved"] is True
    assert any(item.decision.source_wall_id == "PW_3" for item in result.excluded_graphics)
    assert all(item.decision.preserved for item in result.excluded_graphics)


def test_parallel_overlapping_wall_hypotheses_become_one_centerline() -> None:
    result = PostReconstructionFilterEngine().run(runtime=_runtime())
    assert result.diagnostics.wall_class_count == 2
    assert result.diagnostics.canonical_wall_count == 1
    assert result.diagnostics.collapsed_hypothesis_count == 1
    assert len(result.filtered_wall_graph.walls) == 1
