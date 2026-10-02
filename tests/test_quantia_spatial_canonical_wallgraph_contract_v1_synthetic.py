from __future__ import annotations

import numpy as np
import pytest
import base64
import zlib

from app.quantia_spatialV1.stages.walls.adaptive.contracts import (
    AdaptiveReconstructionRuntime,
    AdaptiveRoutePlan,
    LogicalGap,
    ModuleDecision,
    PhysicalWallTrack,
    ReconstructionDiagnostics,
    SingleLineWall,
    SingleLineWallGraph,
    SpaceTopology,
)
from app.quantia_spatialV1.stages.walls.canonical import CanonicalWallGraphFinalizer
from app.quantia_spatialV1.core.models.level_view import LevelView, LevelViewTransform, PixelBBox
from app.quantia_spatialV1.stages.walls.postfilter.contracts import (
    CanonicalWallMapping,
    FilterDecision,
    PostFilterDiagnostics,
    PostFilterPlan,
    PostReconstructionFilterResult,
)


def _route() -> AdaptiveRoutePlan:
    modules = [
        ModuleDecision(module="F03_STRUCTURAL_SEED", enabled=True, reason="synthetic"),
        ModuleDecision(module="CONTEXT_NEGATIVE_MASK", enabled=False, reason="synthetic"),
        ModuleDecision(module="WALL_MASK_RECOVERY", enabled=True, reason="synthetic"),
        ModuleDecision(module="WALL_CANONICALIZATION", enabled=True, reason="synthetic"),
        ModuleDecision(module="STRICT_CONNECTIVITY", enabled=True, reason="synthetic"),
        ModuleDecision(module="ROOM_TOPOLOGY", enabled=True, reason="synthetic"),
        ModuleDecision(module="MULTIMODAL_CALL2", enabled=False, reason="synthetic"),
    ]
    return AdaptiveRoutePlan(mode="BALANCED", modules=modules, recovery_strength=0.6)


def _wall(wall_id: str, start: tuple[float, float], end: tuple[float, float]) -> SingleLineWall:
    return SingleLineWall(
        id=wall_id,
        start_px=start,
        end_px=end,
        thickness_px=10.0,
        role="REVIEW",
        confidence=0.8,
        source_candidate_ids=[f"C_{wall_id}"],
        evidence_ids=[],
        source_names=["SYNTHETIC"],
        f03_seed_protected=True,
    )


def _track(wall: SingleLineWall) -> PhysicalWallTrack:
    dx = wall.end_px[0] - wall.start_px[0]
    dy = wall.end_px[1] - wall.start_px[1]
    return PhysicalWallTrack(
        wall_id=wall.id,
        start=wall.start_px,
        end=wall.end_px,
        length_px=(dx * dx + dy * dy) ** 0.5,
        thickness_px=wall.thickness_px,
        confidence=wall.confidence,
        source_candidate_ids=tuple(wall.source_candidate_ids),
        evidence_ids=(),
        source_names=("SYNTHETIC",),
        region_support=0.9,
        thickness_support=0.9,
        vector_support=0.9,
        raster_line_support=0.9,
        source_consensus=0.9,
        axis_support=0.0,
        dashed_penalty=0.0,
    )


@pytest.fixture
def inputs():
    walls = [
        _wall("W1", (40, 40), (160, 40)),
        _wall("W2", (160, 40), (160, 160)),
        _wall("W3", (160, 160), (40, 160)),
        _wall("W4", (40, 160), (40, 40)),
    ]
    diagnostics = ReconstructionDiagnostics(
        level_view_id="LV1",
        f03_seed_count=4,
        discovered_candidate_count=4,
        quarantined_candidate_count=0,
        hybrid_candidate_count=4,
        selected_wall_count=99,  # deliberadamente obsoleto
        component_count=9,
        junction_count=0,
        virtual_bridge_count=1,
        interior_space_count=0,
        review_wall_count=99,
        perimeter_wall_count=0,
        divider_wall_count=0,
        seed_coverage_ratio=1.0,
        quarantine_ratio=0.0,
        review_ratio=1.0,
    )
    adaptive_graph = SingleLineWallGraph(
        level_view_id="LV1",
        level_name="Synthetic",
        image_size_px=(200, 200),
        px_per_m=100.0,
        walls=walls,
        logical_gaps=[
            LogicalGap(
                id="STALE",
                kind="OLD",
                start_px=(0, 0),
                end_px=(1, 1),
                gap_px=1.4,
                wall_ids=["W1", "REMOVED_WALL"],
            )
        ],
        interior_space_count=0,
        route_plan=_route(),
        diagnostics=diagnostics,
    )
    runtime = AdaptiveReconstructionRuntime(
        graph=adaptive_graph,
        wall_tracks=[_track(wall) for wall in walls],
        bridges=[],
        topology=SpaceTopology(
            labels=np.zeros((200, 200), dtype=np.int32),
            exterior_labels=frozenset(),
            interior_labels=frozenset(),
            areas_px2={},
        ),
        context_quarantined_count=0,
        artifacts={},
    )
    decisions = [
        FilterDecision(
            source_wall_id=wall.id,
            output_class="WALL",
            included_in_wallgraph=True,
            confidence=0.9,
            source_candidate_ids=wall.source_candidate_ids,
        )
        for wall in walls
    ]
    post = PostReconstructionFilterResult(
        level_view_id="LV1",
        plan=PostFilterPlan(),
        decisions=decisions,
        filtered_wall_graph=adaptive_graph,
        canonical_mapping=[
            CanonicalWallMapping(
                canonical_wall_id=wall.id,
                source_wall_ids=[wall.id],
                source_candidate_ids=wall.source_candidate_ids,
                reason="single_hypothesis",
            )
            for wall in walls
        ],
        diagnostics=PostFilterDiagnostics(
            input_wall_count=4,
            wall_class_count=4,
            architectural_element_count=0,
            excluded_graphic_count=0,
            unresolved_count=0,
            canonical_wall_count=4,
            collapsed_hypothesis_count=0,
        ),
    )
    level = LevelView(
        id="LV1",
        level_name="Synthetic",
        source_document_id="DOC",
        source_page_number=1,
        source_bbox_px=PixelBBox(x_min=0, y_min=0, x_max=200, y_max=200),
        source_page_width_px=200,
        source_page_height_px=200,
        raster_width_px=200,
        raster_height_px=200,
        raster_mime_type="image/png",
        raster_bytes=b"synthetic",
        transform=LevelViewTransform(
            offset_x_px=0,
            offset_y_px=0,
            source_page_width_px=200,
            source_page_height_px=200,
            local_width_px=200,
            local_height_px=200,
        ),
        state="DETECTADO",
    )

    return level, runtime, post


def test_finalizer_rebuilds_relations_and_contract(inputs) -> None:
    level, runtime, post = inputs
    result = CanonicalWallGraphFinalizer().finalize(
        level_view=level,
        evidence=[],
        runtime=runtime,
        post_filter=post,
    )

    assert result.integrity.valid is True
    assert result.integrity.invalid_gap_reference_count == 0
    assert result.graph.diagnostics.selected_wall_count == len(result.graph.walls) == 4
    assert result.graph.diagnostics.review_wall_count == sum(w.role == "REVIEW" for w in result.graph.walls)
    assert result.graph.diagnostics.perimeter_wall_count == sum(w.role == "PERIMETER" for w in result.graph.walls)
    assert result.graph.diagnostics.divider_wall_count == sum(w.role == "DIVIDER" for w in result.graph.walls)
    assert result.graph.diagnostics.interior_space_count == result.graph.interior_space_count
    assert all("REMOVED_WALL" not in gap.wall_ids for gap in result.graph.logical_gaps)
    assert result.audit["post_filter_wall_geometry_unchanged"] is True
    assert result.evidence_bundle.final_wall_ids == [wall.id for wall in result.graph.walls]


@pytest.mark.parametrize("kind", ["id", "geometry", "zero_length", "nan", "mapping", "partition"])
def test_finalizer_rejects_invalid_input(inputs, kind):
    level, runtime, post = inputs
    post = post.model_copy(deep=True)
    walls = post.filtered_wall_graph.walls
    if kind == "id":
        walls[1].id = walls[0].id
    elif kind == "geometry":
        walls[1].start_px, walls[1].end_px = walls[0].end_px, walls[0].start_px
    elif kind == "zero_length":
        walls[0].end_px = walls[0].start_px
    elif kind == "nan":
        walls[0].start_px = (float("nan"), 0)
    elif kind == "mapping":
        post.canonical_mapping.pop()
    else:
        post.decisions.pop()
    with pytest.raises(ValueError):
        CanonicalWallGraphFinalizer().finalize(level_view=level, evidence=[], runtime=runtime, post_filter=post)


def test_failed_integrity_is_not_published(inputs, monkeypatch):
    level, runtime, post = inputs
    finalizer = CanonicalWallGraphFinalizer()
    original = finalizer._integrity
    monkeypatch.setattr(finalizer, "_integrity", lambda graph: original(graph).model_copy(update={"valid": False}))
    with pytest.raises(ValueError, match="Invalid canonical"):
        finalizer.finalize(level_view=level, evidence=[], runtime=runtime, post_filter=post)


def test_bundle_round_trip_preserves_artifacts_and_inputs(inputs):
    from app.quantia_spatialV1.stages.walls.canonical import ReconstructionEvidenceBundle
    level, runtime, post = inputs
    mask = np.arange(400, dtype=np.uint16).reshape(20, 20)
    runtime.artifacts = {"wall_mask": mask, "logical_mask": mask.copy(),
                         "component_report": {"components": [1, 2]}, "junctions": [(1, 2)]}
    before = runtime.graph.model_dump()
    result = CanonicalWallGraphFinalizer().finalize(level_view=level, evidence=[], runtime=runtime, post_filter=post)
    bundle = ReconstructionEvidenceBundle.model_validate_json(result.evidence_bundle.model_dump_json())
    saved = bundle.preserved_artifacts["wall_mask"]
    restored = np.frombuffer(zlib.decompress(base64.b64decode(saved["data"])), dtype=saved["dtype"]).reshape(saved["shape"])
    np.testing.assert_array_equal(restored, mask)
    assert bundle.preserved_artifacts["component_report"] == {"components": [1, 2]}
    assert runtime.graph.model_dump() == before
    assert bundle.adaptive_walls == runtime.graph.model_dump(mode="json")["walls"]


def test_corrections_preserve_final_ancestry_and_seed_coverage(inputs):
    from app.quantia_spatialV1.stages.walls.adaptive.contracts import MultimodalWallReview, WallDelta
    from app.quantia_spatialV1.stages.walls.adaptive.correction_applier import WallGraphCorrectionApplier
    from app.quantia_spatialV1.stages.walls.canonical.correction_lineage import apply_with_lineage
    level, runtime, post = inputs
    review = MultimodalWallReview(level_view_id="LV1", graph_state="PARTIAL", deltas=[
        WallDelta(action="REMOVE_WALL", wall_id="W1", confidence=0.9, reason="test"),
        WallDelta(action="SPLIT_WALL", wall_id="W2", new_point_px=(160, 100), confidence=0.9, reason="test"),
        WallDelta(action="MERGE_WALLS", wall_ids=["W2__A", "W2__B"], wall_id="MERGED", confidence=0.9, reason="test"),
        WallDelta(action="ADD_WALL", start_px=(20, 10), end_px=(80, 10), confidence=0.9, reason="test"),
        WallDelta(action="ADD_WALL", start_px=(20, 20), end_px=(80, 20), confidence=0.9, reason="test"),
    ])
    applier = WallGraphCorrectionApplier()
    corrected, mapping = apply_with_lineage(applier=applier, graph=post.filtered_wall_graph, review=review,
                                            source_mapping={w.id: [w.id] for w in runtime.graph.walls})
    assert corrected == applier.apply(graph=post.filtered_wall_graph, review=review)
    assert mapping == {"MERGED": ["W2"], "W3": ["W3"], "W4": ["W4"], "CALL2_ADD_001": [], "CALL2_ADD_002": []}
    result = CanonicalWallGraphFinalizer().finalize(
        level_view=level, evidence=[], runtime=runtime, post_filter=post,
        walls_override=corrected.walls, source_mapping_override=mapping,
        correction_review=review.model_dump(mode="json"))
    assert result.graph.diagnostics.seed_coverage_ratio == 0.75
    assert result.evidence_bundle.final_source_mapping == mapping
    assert result.evidence_bundle.correction_review == review.model_dump(mode="json")
