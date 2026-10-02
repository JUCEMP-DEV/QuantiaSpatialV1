from __future__ import annotations

import json
import socket
from datetime import datetime
from pathlib import Path

import cv2
import pytest

from app.quantia_spatialV1.post_reconstruction_filters import PostFilterArtifactRenderer
from app.quantia_spatialV1.process_engine import QuantiaSpatialV1ProcessEngine
from app.quantia_spatialV1.reconstruction_core import ProposalCReconstructionPipeline
from app.quantia_spatialV1.tests.quantia_case_loader import CASE_LOADERS


CASE_IDS = ("casa_viri", "miguel_h", "miguel_v")
OUTPUT_ROOT = Path(__file__).resolve().parent / "output" / "canonical_integrity_v2" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def _safe(value: str) -> str:
    return "_".join(filter(None, "".join(ch.lower() if ch.isalnum() else "_" for ch in value).split("_")))


@pytest.mark.parametrize("case_id", CASE_IDS)
def test_quantia_spatial_v1_process_all_3_v1(case_id: str, monkeypatch) -> None:
    def no_network(*args, **kwargs):
        raise AssertionError("Offline reconstruction regression forbids network access")
    monkeypatch.setattr(socket.socket, "connect", no_network)
    try:
        case = CASE_LOADERS[case_id]()
    except pytest.skip.Exception as exc:
        pytest.fail(f"Required regression input missing: {exc}")
    assert len(case.levels) == 2
    baseline = json.loads((Path(__file__).parent / "data" / "postfilter_v4_geometry_baseline.json").read_text(encoding="utf-8"))
    output_dir = OUTPUT_ROOT / case.case_id
    output_dir.mkdir(parents=True, exist_ok=True)

    scale_builder = ProposalCReconstructionPipeline()
    scale_context = scale_builder.build_project_scale_context(
        levels=[(loaded.level_result.level_view, loaded.level_result.perimeter.editable_perimeter) for loaded in case.levels]
    )
    assert scale_context.state == "RESOLVED"

    engine = QuantiaSpatialV1ProcessEngine()
    renderer = PostFilterArtifactRenderer()
    report = {"case": case.case_name, "levels": []}

    for loaded in case.levels:
        level = loaded.level_result
        profile = scale_context.for_level(level.level_view.id)
        result = engine.run_level(
            level_view=level.level_view,
            perimeter=level.perimeter.editable_perimeter,
            evidence=list(level.evidence.evidence),
            scale_profile=profile,
            call2_mode="OFF",
        )
        assert result.adaptive.graph.walls
        assert result.post_filter.audit["adaptive_graph_unchanged"] is True
        assert result.post_filter.filtered_wall_graph.walls
        assert result.canonical.integrity.valid is True
        final_ids = {wall.id for wall in result.final_wall_graph.walls}
        assert len(final_ids) == len(result.final_wall_graph.walls)
        assert all(
            all(wall_id in final_ids for wall_id in gap.wall_ids)
            for gap in result.final_wall_graph.logical_gaps
        )
        assert result.final_wall_graph.diagnostics.selected_wall_count == len(result.final_wall_graph.walls)
        assert result.final_wall_graph.diagnostics.review_wall_count == sum(wall.role == "REVIEW" for wall in result.final_wall_graph.walls)
        assert result.final_wall_graph.diagnostics.perimeter_wall_count == sum(wall.role == "PERIMETER" for wall in result.final_wall_graph.walls)
        assert result.final_wall_graph.diagnostics.divider_wall_count == sum(wall.role == "DIVIDER" for wall in result.final_wall_graph.walls)
        assert result.final_wall_graph.diagnostics.interior_space_count == result.final_wall_graph.interior_space_count

        adaptive_ids = {wall.id for wall in result.adaptive.graph.walls}
        decision_ids = [decision.source_wall_id for decision in result.post_filter.decisions]
        assert len(decision_ids) == len(set(decision_ids))
        assert set(decision_ids) == adaptive_ids
        mapped_sources = [source_id for item in result.post_filter.canonical_mapping for source_id in item.source_wall_ids]
        wall_decision_ids = {decision.source_wall_id for decision in result.post_filter.decisions if decision.output_class == "WALL"}
        assert len(mapped_sources) == len(set(mapped_sources))
        assert set(mapped_sources) == wall_decision_ids

        stem = f"{case.case_id}__{_safe(loaded.level_name)}"
        geometry = [{key: wall.model_dump(mode="json")[key] for key in ("id", "start_px", "end_px", "thickness_px")}
                    for wall in result.final_wall_graph.walls]
        assert geometry == baseline[stem], "V4 wall geometry changed"
        assert result.call2_executed is False
        assert result.canonical.audit["post_filter_wall_geometry_unchanged"]
        bundle = result.canonical.evidence_bundle
        assert bundle.raw_evidence == [item.model_dump(mode="json") for item in level.evidence.evidence]
        assert bundle.adaptive_walls == [wall.model_dump(mode="json") for wall in result.adaptive.graph.walls]
        for name in ("f03_seed_candidates", "discovered_candidates", "quarantined_candidates", "hybrid_candidates", "context_decisions", "context_regions"):
            assert getattr(bundle, name) == [item.model_dump(mode="json") for item in result.adaptive.artifacts.get(name, [])]
        assert bundle.post_filter_decisions == [item.model_dump(mode="json") for item in result.post_filter.decisions]
        for name in ("architectural_elements", "excluded_graphics", "unresolved"):
            assert getattr(bundle, name) == [item.model_dump(mode="json") for item in getattr(result.post_filter, name)]
        assert set(bundle.final_source_mapping) == final_ids
        final_render = result.post_filter.model_copy(update={"filtered_wall_graph": result.final_wall_graph})
        visuals = renderer.render_wall_only(level_view=level.level_view, result=final_render)
        for name, image in visuals.items():
            cv2.imwrite(str(output_dir / f"{stem}__{name}.png"), image)
        (output_dir / f"{stem}__canonical_wallgraph.json").write_text(
            result.final_wall_graph.model_dump_json(indent=2), encoding="utf-8"
        )
        (output_dir / f"{stem}__reconstruction_evidence_bundle.json").write_text(
            result.canonical.evidence_bundle.model_dump_json(indent=2), encoding="utf-8"
        )
        (output_dir / f"{stem}__filter_audit.json").write_text(
            result.post_filter.model_dump_json(indent=2), encoding="utf-8"
        )
        report["levels"].append({
            "level": loaded.level_name,
            "adaptive_walls": len(result.adaptive.graph.walls),
            "filtered_walls": len(result.post_filter.filtered_wall_graph.walls),
            "canonical_walls": len(result.final_wall_graph.walls),
            "canonical_gaps": len(result.final_wall_graph.logical_gaps),
            "canonical_interior_spaces": result.final_wall_graph.interior_space_count,
            "canonical_integrity_valid": result.canonical.integrity.valid,
            "architectural_elements": result.post_filter.diagnostics.architectural_element_count,
            "excluded_graphics": result.post_filter.diagnostics.excluded_graphic_count,
            "unresolved": result.post_filter.diagnostics.unresolved_count,
            "collapsed_hypotheses": result.post_filter.diagnostics.collapsed_hypothesis_count,
            "pattern_groups": result.post_filter.diagnostics.pattern_group_count,
            "repetitive_patterns": result.post_filter.diagnostics.repetitive_pattern_count,
            "angular_hubs": result.post_filter.diagnostics.angular_hub_count,
            "filter_modules": [m.model_dump(mode="json") for m in result.post_filter.plan.modules],
            "call2_required": result.call2_required,
            "call2_executed": result.call2_executed,
        })

    (output_dir / f"{case.case_id}__process_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
