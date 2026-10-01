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
from app.quantia_spatialV1.reconstruction_core import ProposalCReconstructionPipeline
from app.quantia_spatialV1.tests.quantia_case_loader import CASE_LOADERS


CASE_IDS = ("casa_viri", "miguel_h", "miguel_v")
OUTPUT_ROOT = Path(__file__).resolve().parent / "output" / "quantia_spatial_v1_call2_v2"
RUN_REAL_CALL2 = str(os.getenv("QUANTIA_RUN_REAL_CALL2", "0") or "0").strip() == "1"


def _safe(value: str) -> str:
    return "_".join(filter(None, "".join(ch.lower() if ch.isalnum() else "_" for ch in value).split("_")))


@pytest.mark.skipif(not RUN_REAL_CALL2, reason="Call 2 real requiere QUANTIA_RUN_REAL_CALL2=1")
@pytest.mark.parametrize("case_id", CASE_IDS)
def test_quantia_spatial_v1_call2_all_3_v2(case_id: str) -> None:
    """Harness integral de la llamada productiva Call 2 V2 sobre el flujo V4.

    El test no contiene lógica de reconstrucción. Ejecuta:
      Adaptive -> PostFilter V4 -> Call 2 V2 -> DeltaValidator -> CorrectionApplier.
    La única salida visual es el WallGraph final de muros.
    """

    case = CASE_LOADERS[case_id]()
    output_dir = OUTPUT_ROOT / case.case_id
    # No borrar resultados: Call 2 usa historial append-only y replay exacto
    # por hash para no consumir nuevamente una respuesta ya validada.
    output_dir.mkdir(parents=True, exist_ok=True)

    reviewer = WallGraphMultimodalReviewer(
        history_path=output_dir / f"{case.case_id}__call2_history.jsonl",
    )
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

    report: dict = {
        "case": case.case_name,
        "call2_mode": "FORCE",
        "call2_prompt_version": "CALL2_WALL_REVIEW_V2",
        "levels": [],
    }

    for loaded in case.levels:
        level = loaded.level_result
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
        assert result.call2_validation.level_view_id == level.level_view.id
        assert result.final_wall_graph.walls

        wall_ids = [wall.id for wall in result.final_wall_graph.walls]
        assert len(wall_ids) == len(set(wall_ids))
        width, height = result.final_wall_graph.image_size_px
        for wall in result.final_wall_graph.walls:
            for x, y in (wall.start_px, wall.end_px):
                assert 0.0 <= x <= width
                assert 0.0 <= y <= height

        stem = f"{case.case_id}__{_safe(loaded.level_name)}"
        renderer.render_walls_only(
            level_view=level.level_view,
            graph=result.final_wall_graph,
            output_path=output_dir / f"{stem}__01_call2_walls_only.png",
        )
        (output_dir / f"{stem}__call2_review.json").write_text(
            result.call2_review.model_dump_json(indent=2),
            encoding="utf-8",
        )
        (output_dir / f"{stem}__call2_validation.json").write_text(
            result.call2_validation.model_dump_json(indent=2),
            encoding="utf-8",
        )
        (output_dir / f"{stem}__final_wallgraph.json").write_text(
            result.final_wall_graph.model_dump_json(indent=2),
            encoding="utf-8",
        )

        validation = result.call2_validation
        report["levels"].append({
            "level": loaded.level_name,
            "level_view_id": level.level_view.id,
            "walls_before_call2": len(result.post_filter.filtered_wall_graph.walls),
            "walls_after_call2": len(result.final_wall_graph.walls),
            "graph_state": result.call2_review.graph_state,
            "raw_delta_count": len(result.call2_review.deltas),
            "accepted_delta_count": validation.accepted_delta_count,
            "rejected_delta_count": validation.rejected_delta_count,
            "raw_gap_decisions": len(result.call2_review.gap_decisions),
            "accepted_gap_decisions": sum(item.accepted for item in validation.gap_items),
            "architectural_regions": len(validation.accepted_architectural_regions),
            "unresolved_regions": len(validation.accepted_unresolved_regions),
            "summary": result.call2_review.summary,
        })

    (output_dir / f"{case.case_id}__call2_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
