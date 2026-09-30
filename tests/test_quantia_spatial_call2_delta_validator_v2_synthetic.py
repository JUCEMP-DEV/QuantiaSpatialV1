from __future__ import annotations

from app.quantia_spatialV1.adaptive_reconstruction import Call2DeltaValidator
from app.quantia_spatialV1.adaptive_reconstruction.contracts import (
    AdaptiveRoutePlan,
    ModuleDecision,
    MultimodalWallReview,
    ReconstructionDiagnostics,
    SingleLineWall,
    SingleLineWallGraph,
)


def _graph() -> SingleLineWallGraph:
    return SingleLineWallGraph(
        level_view_id="LV_CALL2_VALIDATOR",
        level_name="Synthetic",
        image_size_px=(1000, 800),
        px_per_m=100.0,
        walls=[
            SingleLineWall(
                id="W1",
                start_px=(100.0, 100.0),
                end_px=(500.0, 100.0),
                thickness_px=15.0,
                role="PERIMETER",
                confidence=0.90,
                f03_seed_protected=True,
            ),
            SingleLineWall(
                id="W2",
                start_px=(100.0, 300.0),
                end_px=(500.0, 300.0),
                thickness_px=15.0,
                role="DIVIDER",
                confidence=0.80,
            ),
        ],
        logical_gaps=[],
        interior_space_count=1,
        route_plan=AdaptiveRoutePlan(
            mode="BALANCED",
            modules=[ModuleDecision(module="MULTIMODAL_CALL2", enabled=True, reason="synthetic")],
            recovery_strength=0.5,
            require_multimodal_review=True,
        ),
        diagnostics=ReconstructionDiagnostics(
            level_view_id="LV_CALL2_VALIDATOR",
            f03_seed_count=1,
            discovered_candidate_count=2,
            quarantined_candidate_count=0,
            hybrid_candidate_count=2,
            selected_wall_count=2,
            component_count=1,
            junction_count=0,
            virtual_bridge_count=0,
            interior_space_count=1,
            review_wall_count=0,
            perimeter_wall_count=1,
            divider_wall_count=1,
            seed_coverage_ratio=1.0,
            quarantine_ratio=0.0,
            review_ratio=0.0,
        ),
    )


def test_call2_validator_rejects_unsafe_deltas_and_preserves_valid_ones() -> None:
    review = MultimodalWallReview.model_validate({
        "level_view_id": "LV_CALL2_VALIDATOR",
        "graph_state": "PARTIAL",
        "deltas": [
            {
                "action": "REMOVE_WALL",
                "wall_id": "W1",
                "confidence": 0.85,
                "reason": "seed-protected but not enough confidence",
            },
            {
                "action": "REMOVE_WALL",
                "wall_id": "W2",
                "confidence": 0.90,
                "reason": "valid remove",
            },
            {
                "action": "ADD_WALL",
                "wall_id": "W3",
                "start_px": [100.0, 500.0],
                "end_px": [700.0, 500.0],
                "role_hint": "DIVIDER",
                "confidence": 0.91,
                "reason": "valid add",
            },
            {
                "action": "ADD_WALL",
                "wall_id": "W_BAD",
                "start_px": [-20.0, 50.0],
                "end_px": [200.0, 50.0],
                "confidence": 0.99,
                "reason": "out of bounds",
            },
        ],
        "gap_decisions": [],
        "non_wall_architectural_regions": [
            {
                "bbox_px": [200.0, 200.0, 300.0, 350.0],
                "family_hint": "WINDOW",
                "confidence": 0.92,
                "reason": "valid region",
            }
        ],
        "unresolved_regions": [],
        "summary": "synthetic",
    })

    result = Call2DeltaValidator().validate(graph=_graph(), review=review)
    accepted = {(item.delta.action, item.delta.wall_id) for item in result.delta_items if item.accepted}
    rejected = {(item.delta.action, item.delta.wall_id) for item in result.delta_items if not item.accepted}

    assert ("REMOVE_WALL", "W2") in accepted
    assert ("ADD_WALL", "W3") in accepted
    assert ("REMOVE_WALL", "W1") in rejected
    assert ("ADD_WALL", "W_BAD") in rejected
    assert result.accepted_delta_count == 2
    assert result.rejected_delta_count == 2
    assert len(result.accepted_architectural_regions) == 1
