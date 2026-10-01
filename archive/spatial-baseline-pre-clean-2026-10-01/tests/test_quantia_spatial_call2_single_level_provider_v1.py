from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from app.quantia_spatialV1.adaptive_reconstruction import (
    Call2WallGraphArtifactRenderer,
    WallGraphMultimodalReviewer,
)
from app.quantia_spatialV1.process_engine import QuantiaSpatialV1ProcessEngine
from app.quantia_spatialV1.providers import build_call2_provider
from app.quantia_spatialV1.reconstruction_core import ProposalCReconstructionPipeline
from app.quantia_spatialV1.tests.call2_provider_test_utils import load_selected_case_and_level, safe_name

RUN = str(os.getenv("QUANTIA_RUN_REAL_CALL2_PROVIDER", "0") or "0").strip() == "1"
OUTPUT_ROOT = Path(__file__).resolve().parent / "output" / "call2_single_level_provider"


@pytest.mark.skipif(not RUN, reason="Requiere QUANTIA_RUN_REAL_CALL2_PROVIDER=1")
def test_call2_single_level_selected_provider_v1() -> None:
    case, selected = load_selected_case_and_level()
    provider = build_call2_provider()

    availability = provider.check_model_available()
    assert availability["available"] is True, availability

    provider_dir = OUTPUT_ROOT / safe_name(provider.provider_name) / safe_name(provider.model)
    provider_dir.mkdir(parents=True, exist_ok=True)
    history_path = provider_dir / f"{case.case_id}__{safe_name(selected.level_name)}__call2_history.jsonl"

    reviewer = WallGraphMultimodalReviewer(provider=provider, history_path=history_path)
    engine = QuantiaSpatialV1ProcessEngine(multimodal_reviewer=reviewer)
    renderer = Call2WallGraphArtifactRenderer()

    scale_builder = ProposalCReconstructionPipeline()
    scale_context = scale_builder.build_project_scale_context(
        levels=[
            (loaded.level_result.level_view, loaded.level_result.perimeter.editable_perimeter)
            for loaded in case.levels
        ]
    )
    assert scale_context.state == "RESOLVED"

    level = selected.level_result
    profile = scale_context.for_level(level.level_view.id)
    result = engine.run_level(
        level_view=level.level_view,
        perimeter=level.perimeter.editable_perimeter,
        evidence=list(level.evidence.evidence),
        scale_profile=profile,
        call2_mode="FORCE",
    )

    assert result.call2_executed is True
    assert result.call2_review is not None
    assert result.call2_validation is not None
    assert result.call2_review.level_view_id == level.level_view.id
    assert result.final_wall_graph.walls

    stem = f"{case.case_id}__{safe_name(selected.level_name)}"
    renderer.render_walls_only(
        level_view=level.level_view,
        graph=result.final_wall_graph,
        output_path=provider_dir / f"{stem}__01_call2_walls_only.png",
    )
    (provider_dir / f"{stem}__call2_review.json").write_text(
        result.call2_review.model_dump_json(indent=2), encoding="utf-8"
    )
    (provider_dir / f"{stem}__call2_validation.json").write_text(
        result.call2_validation.model_dump_json(indent=2), encoding="utf-8"
    )
    (provider_dir / f"{stem}__final_wallgraph.json").write_text(
        result.final_wall_graph.model_dump_json(indent=2), encoding="utf-8"
    )
    (provider_dir / f"{stem}__run_summary.json").write_text(
        json.dumps({
            "provider": provider.provider_name,
            "model": provider.model,
            "case": case.case_id,
            "level": selected.level_name,
            "level_view_id": level.level_view.id,
            "walls_before_call2": len(result.post_filter.filtered_wall_graph.walls),
            "walls_after_call2": len(result.final_wall_graph.walls),
            "graph_state": result.call2_review.graph_state,
            "raw_delta_count": len(result.call2_review.deltas),
            "accepted_delta_count": result.call2_validation.accepted_delta_count,
            "rejected_delta_count": result.call2_validation.rejected_delta_count,
            "architectural_regions": len(result.call2_validation.accepted_architectural_regions),
            "unresolved_regions": len(result.call2_validation.accepted_unresolved_regions),
            "summary": result.call2_review.summary,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
