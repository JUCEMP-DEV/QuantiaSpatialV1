from __future__ import annotations

from app.quantia_spatialV1.stages.walls.adaptive.contracts import ReconstructionDiagnostics
from app.quantia_spatialV1.stages.walls.adaptive.decision_engine import AdaptiveModuleDecisionEngine


def _diag(**overrides) -> ReconstructionDiagnostics:
    payload = {
        "level_view_id": "LV_SYNTH",
        "f03_seed_count": 10,
        "discovered_candidate_count": 30,
        "quarantined_candidate_count": 3,
        "hybrid_candidate_count": 25,
        "selected_wall_count": 20,
        "component_count": 2,
        "junction_count": 6,
        "virtual_bridge_count": 4,
        "interior_space_count": 5,
        "review_wall_count": 2,
        "perimeter_wall_count": 8,
        "divider_wall_count": 10,
        "seed_coverage_ratio": 0.9,
        "quarantine_ratio": 0.1,
        "review_ratio": 0.1,
    }
    payload.update(overrides)
    return ReconstructionDiagnostics(**payload)


def test_adaptive_routes_are_evidence_driven_not_case_driven() -> None:
    engine = AdaptiveModuleDecisionEngine()

    seed_first = engine.plan_pre_topology(seed_count=18, discovered_count=30, quarantined_count=3)
    dense = engine.plan_pre_topology(seed_count=8, discovered_count=80, quarantined_count=20)
    no_seed = engine.plan_pre_topology(seed_count=0, discovered_count=50, quarantined_count=5)

    assert seed_first.mode == "SEED_FIRST"
    assert dense.mode == "DENSE_RECOVERY"
    assert no_seed.mode == "DENSE_RECOVERY"
    assert dense.recovery_strength > seed_first.recovery_strength


def test_post_topology_escalates_to_call2_when_geometry_is_incomplete() -> None:
    engine = AdaptiveModuleDecisionEngine()
    pre = engine.plan_pre_topology(seed_count=10, discovered_count=30, quarantined_count=3)

    healthy = engine.plan_post_topology(current=pre, diagnostics=_diag())
    incomplete = engine.plan_post_topology(
        current=pre,
        diagnostics=_diag(
            interior_space_count=0,
            component_count=12,
            review_wall_count=15,
            review_ratio=0.75,
        ),
    )

    assert healthy.require_multimodal_review is False
    assert incomplete.require_multimodal_review is True
    assert incomplete.mode == "SEED_RESCUE"
    assert incomplete.enabled("MULTIMODAL_CALL2")
