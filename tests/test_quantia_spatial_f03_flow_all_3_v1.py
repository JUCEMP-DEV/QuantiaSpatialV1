from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from app.quantia_spatialV1.reconstruction_core.reconstruction_pipeline import (
    ProposalCReconstructionPipeline,
)
from app.quantia_spatialV1.reconstruction_core.scale_evidence_resolver import (
    ScaleEvidenceResolver,
)
from app.quantia_spatialV1.tests.quantia_case_loader import CASE_LOADERS, LoadedCase


PROBE_VERSION = "F03_CANONICAL_FLOW_PROBE_V1"
CASE_IDS = ("casa_viri", "miguel_h", "miguel_v")
TARGET_PX_PER_M = 90.0
TARGET_RELATIVE_TOLERANCE = 0.05

TESTS_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = TESTS_DIR / "output"


def _safe_name(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_") or "level"


def _model_hash(value: Any) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _decode_raster(data: bytes) -> np.ndarray:
    array = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        raise AssertionError("No se pudo decodificar level_view.raster_bytes.")
    return image


def _draw_perimeter(image: np.ndarray, perimeter: Any) -> None:
    # EditablePerimeterWall referencia vértices por id; la geometría vive en
    # EditablePerimeterModel.vertices[*].point_px. El renderer debe respetar
    # ese contrato y no asumir atributos start/end inexistentes.
    vertices = {vertex.id: vertex.point_px for vertex in perimeter.vertices}
    for wall in perimeter.walls:
        start = vertices.get(wall.start_vertex_id)
        end = vertices.get(wall.end_vertex_id)
        if start is None or end is None:
            raise AssertionError(
                f"F02 visual: muro {wall.id} referencia vértices inexistentes "
                f"({wall.start_vertex_id} -> {wall.end_vertex_id})."
            )
        cv2.line(
            image,
            (int(round(start.x)), int(round(start.y))),
            (int(round(end.x)), int(round(end.y))),
            (0, 0, 255),
            3,
            cv2.LINE_AA,
        )


def _draw_selected(image: np.ndarray, full: Any) -> None:
    selected = set(full.solution.selected_candidate_ids)
    by_id = {item.id: item for item in full.candidate_graph.candidates}
    for candidate_id in sorted(selected):
        candidate = by_id.get(candidate_id)
        if candidate is None:
            continue
        cv2.line(
            image,
            (int(round(candidate.start.x)), int(round(candidate.start.y))),
            (int(round(candidate.end.x)), int(round(candidate.end.y))),
            (255, 0, 0),
            3,
            cv2.LINE_AA,
        )


def _save_visual(*, output_dir: Path, case_id: str, level_name: str, level_result: Any, full: Any) -> str:
    image = _decode_raster(level_result.level_view.raster_bytes)
    perimeter = level_result.perimeter.editable_perimeter
    if perimeter is not None:
        _draw_perimeter(image, perimeter)
    if full is not None:
        _draw_selected(image, full)

    path = output_dir / f"{case_id}__{_safe_name(level_name)}__f03_canonical_flow_probe_v1.png"
    ok = cv2.imwrite(str(path), image)
    if not ok:
        raise AssertionError(f"No se pudo escribir visual: {path}")
    return str(path)


def _scale_diagnostic(*, resolver: ScaleEvidenceResolver, level_result: Any) -> dict[str, Any]:
    perimeter = level_result.perimeter.editable_perimeter
    scale = resolver.resolve(
        level_view=level_result.level_view,
        evidence=level_result.evidence.evidence,
        perimeter=perimeter,
    )
    px_per_m = 1.0 / scale.selected_m_per_px if scale.selected_m_per_px else None
    relative_error = (
        abs(px_per_m - TARGET_PX_PER_M) / TARGET_PX_PER_M
        if px_per_m is not None
        else None
    )
    return {
        "state": scale.state,
        "selected_m_per_px": scale.selected_m_per_px,
        "selected_method": scale.selected_method,
        "selected_confidence": scale.selected_confidence,
        "px_per_m": px_per_m,
        "target_px_per_m": TARGET_PX_PER_M,
        "relative_error_to_target": relative_error,
        "warnings": list(scale.warnings),
    }


def _run_case(case: LoadedCase) -> None:
    output_dir = OUTPUT_ROOT / case.case_id
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in output_dir.glob(f"*__f03_canonical_flow_probe_v1.*"):
        path.unlink()

    resolver = ScaleEvidenceResolver()
    pipeline = ProposalCReconstructionPipeline()

    level_scale: dict[str, dict[str, Any]] = {}
    scale_inputs: list[tuple[Any, Any]] = []
    missing_perimeters: list[str] = []

    for loaded in case.levels:
        level_result = loaded.level_result
        level_scale[loaded.level_name] = _scale_diagnostic(
            resolver=resolver,
            level_result=level_result,
        )
        perimeter = level_result.perimeter.editable_perimeter
        if perimeter is None:
            missing_perimeters.append(loaded.level_name)
        else:
            scale_inputs.append((level_result.level_view, perimeter))

    project_scale = None
    if len(scale_inputs) == len(case.levels):
        project_scale = pipeline.build_project_scale_context(levels=scale_inputs)

    report: dict[str, Any] = {
        "probe_version": PROBE_VERSION,
        "case_id": case.case_id,
        "case": case.case_name,
        "gemini_calls_added": 0,
        "target_geometry_px_per_m": TARGET_PX_PER_M,
        "success_criteria": {
            "final_scale_resolved": True,
            "final_density_relative_tolerance": TARGET_RELATIVE_TOLERANCE,
            "f02_perimeter_present_all_levels": True,
            "f03_project_scale_resolved": True,
            "f03_executes_all_levels": True,
            "f02_immutable_during_f03": True,
            "visual_policy": "ONE_PNG_PER_LEVEL_VIEW",
        },
        "missing_perimeters": missing_perimeters,
        "project_scale": project_scale.model_dump(mode="json") if project_scale is not None else None,
        "levels": [],
    }

    failures: list[str] = []
    if missing_perimeters:
        failures.append("F02_MISSING_PERIMETER:" + ",".join(missing_perimeters))
    if project_scale is None:
        failures.append("F03_PROJECT_SCALE_NOT_BUILT")
    elif project_scale.state != "RESOLVED":
        failures.append(f"F03_PROJECT_SCALE_{project_scale.state}")

    for loaded in case.levels:
        level_result = loaded.level_result
        perimeter = level_result.perimeter.editable_perimeter
        scale_diag = level_scale[loaded.level_name]
        full = None
        f03_error = None
        before_hash = _model_hash(perimeter) if perimeter is not None else None
        after_hash = before_hash
        scale_profile = None

        if scale_diag["state"] != "RESOLVED":
            failures.append(f"{loaded.level_name}:FINAL_SCALE_{scale_diag['state']}")
        if (
            scale_diag["relative_error_to_target"] is None
            or scale_diag["relative_error_to_target"] > TARGET_RELATIVE_TOLERANCE
        ):
            failures.append(
                f"{loaded.level_name}:FINAL_DENSITY_OUT_OF_TOLERANCE:{scale_diag['px_per_m']}"
            )

        if perimeter is not None and project_scale is not None:
            scale_profile = project_scale.for_level(level_result.level_view.id)
            if scale_profile.state != "RESOLVED":
                failures.append(f"{loaded.level_name}:F03_SCALE_PROFILE_{scale_profile.state}")
            try:
                full = pipeline.run(
                    level_view=level_result.level_view,
                    perimeter=perimeter,
                    evidence=level_result.evidence.evidence,
                    scale_profile=scale_profile,
                )
            except Exception as exc:  # diagnóstico de flujo: registrar todo antes de fallar
                f03_error = f"{type(exc).__name__}: {exc}"
                failures.append(f"{loaded.level_name}:F03_EXECUTION_ERROR:{f03_error}")
            after_hash = _model_hash(perimeter)
            if before_hash != after_hash:
                failures.append(f"{loaded.level_name}:F02_MUTATED_BY_F03")
        else:
            failures.append(f"{loaded.level_name}:F03_NOT_EXECUTED")

        visual_file = _save_visual(
            output_dir=output_dir,
            case_id=case.case_id,
            level_name=loaded.level_name,
            level_result=level_result,
            full=full,
        )

        level_report: dict[str, Any] = {
            "level": loaded.level_name,
            "level_view_id": level_result.level_view.id,
            "raster_width_px": level_result.level_view.transform.local_width_px,
            "raster_height_px": level_result.level_view.transform.local_height_px,
            "scale": scale_diag,
            "f02": {
                "perimeter_present": perimeter is not None,
                "model_id": perimeter.id if perimeter is not None else None,
                "geometry_revision": perimeter.geometry_revision if perimeter is not None else None,
                "hash_before_f03": before_hash,
                "hash_after_f03": after_hash,
                "immutable": before_hash == after_hash if before_hash is not None else None,
            },
            "f03_scale_profile": scale_profile.model_dump(mode="json") if scale_profile is not None else None,
            "f03_executed": full is not None,
            "f03_error": f03_error,
            "visual_file": visual_file,
        }

        if full is not None:
            region_counts = Counter(region.region_type for region in full.context_gate.regions)
            level_report["f03"] = {
                "drawing_lines": full.diagnostics.drawing_line_count,
                "discovered_candidates": full.diagnostics.discovered_wall_candidate_count,
                "context_regions": full.diagnostics.context_region_count,
                "quarantine": full.diagnostics.quarantined_wall_candidate_count,
                "review": full.diagnostics.review_wall_candidate_count,
                "solver_candidates": full.diagnostics.wall_candidate_count,
                "selected": full.diagnostics.selected_wall_candidate_count,
                "perimeter_walls": full.diagnostics.perimeter_wall_count,
                "divider_walls": full.diagnostics.parametric_divider_wall_count,
                "f02_model_id_seen_by_f03": full.diagnostics.f02_model_id,
                "f02_geometry_revision_seen_by_f03": full.diagnostics.f02_geometry_revision,
                "region_types": dict(sorted(region_counts.items())),
                "warnings": list(full.warnings),
            }

        report["levels"].append(level_report)

        print("\n" + "=" * 104)
        print(f"F03 CANONICAL FLOW PROBE V1 — {case.case_name} — {loaded.level_name}")
        print({
            "scale_state": scale_diag["state"],
            "px_m": round(scale_diag["px_per_m"], 3) if scale_diag["px_per_m"] else None,
            "f02_perimeter": perimeter is not None,
            "f03_scale_state": scale_profile.state if scale_profile is not None else None,
            "f03_executed": full is not None,
            "selected": full.diagnostics.selected_wall_candidate_count if full is not None else None,
            "divider_walls": full.diagnostics.parametric_divider_wall_count if full is not None else None,
            "visual": visual_file,
        })

    report["failures"] = failures
    report["passed"] = not failures

    json_path = output_dir / f"{case.case_id}__f03_canonical_flow_probe_v1.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    pngs = sorted(output_dir.glob("*__f03_canonical_flow_probe_v1.png"))
    assert len(pngs) == len(case.levels), (
        f"{case.case_id}: debe producir exactamente 1 PNG por LevelView; "
        f"levels={len(case.levels)} pngs={len(pngs)}"
    )

    print(f"\nJSON: {json_path}")
    print(f"Visuales: {len(pngs)} = 1 por LevelView")

    assert not failures, (
        f"{case.case_id}: el flujo canónico hasta F03 no está cerrado. "
        f"failures={json.dumps(failures, ensure_ascii=False)}. "
        f"Revisar {json_path}."
    )


@pytest.mark.parametrize("case_id", CASE_IDS)
def test_f03_canonical_flow_all_3_v1(case_id: str) -> None:
    # Esta prueba queda fija. No contiene valores, nombres de ejes, escalas ni
    # umbrales específicos de ningún caso; los tres recorren exactamente el mismo flujo.
    _run_case(CASE_LOADERS[case_id]())


if __name__ == "__main__":
    for _case_id in CASE_IDS:
        _run_case(CASE_LOADERS[_case_id]())
