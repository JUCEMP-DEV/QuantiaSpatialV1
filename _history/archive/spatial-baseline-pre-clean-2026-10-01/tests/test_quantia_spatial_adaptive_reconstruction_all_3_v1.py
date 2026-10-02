from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest

from app.quantia_spatialV1.adaptive_reconstruction import (
    AdaptiveReconstructionEngine,
    WallGraphCorrectionApplier,
    WallGraphMultimodalReviewer,
)
from app.quantia_spatialV1.adaptive_reconstruction.artifact_renderer import (
    AdaptiveReconstructionArtifactRenderer,
)
from app.quantia_spatialV1.reconstruction_core import ProposalCReconstructionPipeline
from app.quantia_spatialV1.tests.quantia_case_loader import CASE_LOADERS, LoadedCase


TEST_VERSION = "ADAPTIVE_RECONSTRUCTION_ALL_3_V1"
CASE_IDS = ("casa_viri", "miguel_h", "miguel_v")
TESTS_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = TESTS_DIR / "output" / "adaptive_reconstruction_v1"
ENABLE_REAL_CALL2 = str(os.getenv("QUANTIA_ADAPTIVE_CALL2", "0")).strip().lower() in {"1", "true", "yes", "on"}
FORCE_REAL_CALL2 = str(os.getenv("QUANTIA_ADAPTIVE_CALL2_FORCE", "0")).strip().lower() in {"1", "true", "yes", "on"}


def _safe(value: str) -> str:
    text = "".join(ch.lower() if ch.isalnum() else "_" for ch in value.strip())
    while "__" in text:
        text = text.replace("__", "_")
    return text.strip("_") or "level"


def _render_corrected(level_view, graph, path: Path) -> None:
    image = cv2.imdecode(np.frombuffer(level_view.raster_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert image is not None
    canvas = np.full_like(image, 255)
    for wall in graph.walls:
        cv2.line(
            canvas,
            (int(round(wall.start_px[0])), int(round(wall.start_px[1]))),
            (int(round(wall.end_px[0])), int(round(wall.end_px[1]))),
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
    cv2.imwrite(str(path), canvas)


def _run_case(case: LoadedCase) -> None:
    output_dir = OUTPUT_ROOT / case.case_id
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pipeline = ProposalCReconstructionPipeline()
    scale_inputs = []
    for loaded in case.levels:
        perimeter = loaded.level_result.perimeter.editable_perimeter
        assert perimeter is not None
        scale_inputs.append((loaded.level_result.level_view, perimeter))
    project_scale = pipeline.build_project_scale_context(levels=scale_inputs)
    assert project_scale.state == "RESOLVED"

    engine = AdaptiveReconstructionEngine(pipeline=pipeline)
    renderer = AdaptiveReconstructionArtifactRenderer()
    reviewer = WallGraphMultimodalReviewer(
        history_path=output_dir / "adaptive_call2_history.jsonl"
    ) if ENABLE_REAL_CALL2 else None
    applier = WallGraphCorrectionApplier()

    report = {
        "test_version": TEST_VERSION,
        "case_id": case.case_id,
        "case": case.case_name,
        "policy": "NO_CASE_SPECIFIC_RULES",
        "real_call2_enabled": ENABLE_REAL_CALL2,
        "levels": [],
    }

    for loaded in case.levels:
        level_result = loaded.level_result
        perimeter = level_result.perimeter.editable_perimeter
        assert perimeter is not None
        profile = project_scale.for_level(level_result.level_view.id)
        assert profile.local_m_per_px is not None and profile.local_m_per_px > 0.0

        runtime = engine.run(
            level_view=level_result.level_view,
            perimeter=perimeter,
            evidence=list(level_result.evidence.evidence),
            scale_profile=profile,
        )
        assert runtime.graph.walls, "AdaptiveReconstructionEngine publicó WallGraph vacío."
        assert runtime.graph.route_plan.enabled("WALL_CANONICALIZATION")
        assert runtime.graph.route_plan.enabled("ROOM_TOPOLOGY")

        stem = f"{case.case_id}__{_safe(loaded.level_name)}"
        visuals = renderer.render(
            level_view=level_result.level_view,
            runtime=runtime,
            output_dir=output_dir,
            stem=stem,
        )

        graph_path = output_dir / f"{stem}__single_line_wallgraph.json"
        graph_path.write_text(
            runtime.graph.model_dump_json(indent=2),
            encoding="utf-8",
        )

        call2_payload = None
        corrected_graph_path = None
        corrected_visual_path = None
        should_call2 = FORCE_REAL_CALL2 or runtime.graph.route_plan.require_multimodal_review
        if reviewer is not None and should_call2:
            review = reviewer.review(level_view=level_result.level_view, graph=runtime.graph)
            call2_payload = review.model_dump(mode="json")
            corrected = applier.apply(graph=runtime.graph, review=review)
            corrected_graph_path = output_dir / f"{stem}__call2_corrected_wallgraph.json"
            corrected_graph_path.write_text(corrected.model_dump_json(indent=2), encoding="utf-8")
            corrected_visual_path = output_dir / f"{stem}__09_call2_corrected_wallgraph.png"
            _render_corrected(level_result.level_view, corrected, corrected_visual_path)

        level_report = {
            "level": loaded.level_name,
            "level_view_id": level_result.level_view.id,
            "route_plan": runtime.graph.route_plan.model_dump(mode="json"),
            "diagnostics": runtime.graph.diagnostics.model_dump(mode="json"),
            "single_line_wall_count": len(runtime.graph.walls),
            "logical_gap_count": len(runtime.graph.logical_gaps),
            "visuals": visuals,
            "graph_json": str(graph_path),
            "call2_executed": call2_payload is not None,
            "call2_review": call2_payload,
            "call2_corrected_graph": str(corrected_graph_path) if corrected_graph_path else None,
            "call2_corrected_visual": str(corrected_visual_path) if corrected_visual_path else None,
        }
        report["levels"].append(level_report)

        print("\n" + "=" * 100)
        print(f"ADAPTIVE RECONSTRUCTION V1 — {case.case_name} — {loaded.level_name}")
        print({
            "route_mode": runtime.graph.route_plan.mode,
            "enabled_modules": [
                item.module for item in runtime.graph.route_plan.modules if item.enabled
            ],
            "walls": len(runtime.graph.walls),
            "spaces": runtime.graph.interior_space_count,
            "review_ratio": round(runtime.graph.diagnostics.review_ratio, 3),
            "requires_call2": runtime.graph.route_plan.require_multimodal_review,
            "call2_executed": call2_payload is not None,
        })

    report_path = output_dir / f"{case.case_id}__adaptive_reconstruction_v1.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nJSON: {report_path}")


@pytest.mark.parametrize("case_id", CASE_IDS)
def test_quantia_spatial_adaptive_reconstruction_all_3_v1(case_id: str) -> None:
    _run_case(CASE_LOADERS[case_id]())


if __name__ == "__main__":
    for _case_id in CASE_IDS:
        _run_case(CASE_LOADERS[_case_id]())
