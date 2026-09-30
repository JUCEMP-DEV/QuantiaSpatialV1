from __future__ import annotations

import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest
from shapely.geometry import LineString

from app.quantia_spatialV1.reconstruction_core import ProposalCReconstructionPipeline
from app.quantia_spatialV1.reconstruction_core.perimeter_adapter import (
    line_is_inside_perimeter,
    perimeter_polygon,
)
from app.quantia_spatialV1.tests.quantia_case_loader import (
    CASE_LOADERS,
    LoadedCase,
    LoadedLevel,
)


TESTS_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = TESTS_DIR / "output"

RULE_VERSION = "CONTEXT_CLASSIFICATION_V6_2_STANDARDIZED_PROBE"
BASELINE = "Proposal C A1.2 + Context Classification V6.2 + Surface Pattern + Structural Lineage + Project Scale Normalization + Solver Evidence V1"


def _safe_name(value: str) -> str:
    return (
        value.strip().lower().replace(" ", "_")
        .replace("á", "a").replace("é", "e").replace("í", "i")
        .replace("ó", "o").replace("ú", "u")
    )


def _decode(data: bytes) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError("No se pudo decodificar raster LevelView.")
    return image


def _save(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"No se pudo guardar {path}")


def _reset_case_output(output_dir: Path) -> None:
    # Directorio exclusivo de esta prueba: evita que sobrevivan las 4 imágenes
    # diagnósticas de ejecuciones anteriores y garantiza 1 PNG por LevelView.
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def _draw_perimeter(image: np.ndarray, perimeter) -> None:
    vertices = {vertex.id: vertex.point_px for vertex in perimeter.vertices}
    for wall in perimeter.walls:
        a = vertices[wall.start_vertex_id]
        b = vertices[wall.end_vertex_id]
        cv2.line(
            image,
            (int(round(a.x)), int(round(a.y))),
            (int(round(b.x)), int(round(b.y))),
            (0, 0, 255),
            3,
            cv2.LINE_AA,
        )


def _draw_context_regions(image: np.ndarray, full) -> None:
    # Una sola imagen por LevelView. Las regiones se muestran como cajas finas
    # para verificar clasificación sin generar un set diagnóstico adicional.
    colors = {
        "FLOOR_FINISH_GRID_REGION": (0, 255, 255),
        "FURNITURE_MODULE_REGION": (0, 200, 255),
        "STAIR_FLIGHT_REGION": (255, 255, 0),
        "HATCH_FILL_REGION": (255, 0, 255),
        "AXIS_GRID_REGION": (255, 128, 0),
        "DIMENSION_REGION": (128, 128, 255),
        "UNKNOWN_REPETITIVE_REGION": (180, 180, 180),
    }
    for region in full.context_gate.regions:
        if region.region_type == "WALL_PROTECTED_REGION":
            continue
        color = colors.get(region.region_type, (120, 120, 120))
        p1 = (int(round(region.bbox.x_min)), int(round(region.bbox.y_min)))
        p2 = (int(round(region.bbox.x_max)), int(round(region.bbox.y_max)))
        cv2.rectangle(image, p1, p2, color, 1, cv2.LINE_AA)

    decisions = {decision.candidate_id: decision for decision in full.context_gate.decisions}
    for candidate in full.context_gate.quarantined_candidates:
        decision = decisions.get(candidate.id)
        p1 = (int(round(candidate.start.x)), int(round(candidate.start.y)))
        p2 = (int(round(candidate.end.x)), int(round(candidate.end.y)))
        cv2.line(image, p1, p2, (255, 255, 0), 3, cv2.LINE_AA)
        if decision and decision.region_type == "FLOOR_FINISH_GRID_REGION":
            cv2.line(image, p1, p2, (0, 255, 255), 3, cv2.LINE_AA)


def _candidate_line(candidate) -> LineString:
    return LineString([
        (float(candidate.start.x), float(candidate.start.y)),
        (float(candidate.end.x), float(candidate.end.y)),
    ])




# ---------------------------------------------------------------------------
# Common solver probe. EXACTLY the same logic for every LoadedLevel.
# ---------------------------------------------------------------------------

def _relation_maps(graph):
    hard = defaultdict(set)
    positive = defaultdict(list)
    all_relations = defaultdict(list)
    for relation in graph.relations:
        all_relations[relation.a_id].append(relation)
        all_relations[relation.b_id].append(relation)
        if relation.hard_conflict:
            hard[relation.a_id].add(relation.b_id)
            hard[relation.b_id].add(relation.a_id)
        elif relation.relation_type in {"JUNCTION", "CONTINUATION"}:
            positive[relation.a_id].append(relation)
            positive[relation.b_id].append(relation)
    return hard, positive, all_relations


def _relation_bonus_to_selected(*, solver, candidate_id: str, selected_ids: set[str], positive_relations) -> float:
    return float(solver._incremental_relation_bonus(
        candidate_id=candidate_id,
        selected=frozenset(selected_ids),
        relations=positive_relations,
    ))


def _topology_degree_for_solver(solver, positive_count: int) -> int:
    if hasattr(solver, "_pattern_penalty"):
        return min(2, int(positive_count))
    return int(positive_count)


def _evidence_reasons(candidate) -> list[str]:
    e = candidate.evidence
    reasons: list[str] = []
    if e.dashed_penalty >= 0.5:
        reasons.append(f"dashed_penalty={e.dashed_penalty:.3f}")
    if e.axis_support >= 0.50:
        reasons.append(f"axis_support={e.axis_support:.3f}")
    if e.pair_overlap >= 0.60:
        reasons.append(f"pair_overlap={e.pair_overlap:.3f}")
    if e.region_support >= 0.50:
        reasons.append(f"region_support={e.region_support:.3f}")
    if e.source_consensus >= 0.50:
        reasons.append(f"source_consensus={e.source_consensus:.3f}")
    if e.vector_support >= 0.50:
        reasons.append(f"vector_support={e.vector_support:.3f}")
    if e.raster_line_support >= 0.50:
        reasons.append(f"raster_support={e.raster_line_support:.3f}")
    if e.semantic_support > 0.0:
        reasons.append(f"semantic_support={e.semantic_support:.3f}")
    return reasons


def _classify_candidate(
    *,
    candidate,
    selected_ids,
    selected_candidates,
    perimeter,
    hard_conflicts,
    positive_relations,
    solver,
    baseline_topology,
):
    cid = candidate.id
    topology_degree = _topology_degree_for_solver(solver, len(positive_relations.get(cid, [])))
    unary = float(solver._unary(candidate, topology_degree))
    relation_bonus = _relation_bonus_to_selected(
        solver=solver,
        candidate_id=cid,
        selected_ids=selected_ids,
        positive_relations=positive_relations,
    )

    row = {
        "candidate_id": cid,
        "generator": candidate.generator,
        "selected": cid in selected_ids,
        "start": [float(candidate.start.x), float(candidate.start.y)],
        "end": [float(candidate.end.x), float(candidate.end.y)],
        "angle_deg": float(candidate.angle_deg),
        "length_px": float(candidate.length_px),
        "thickness_px": float(candidate.thickness_px),
        "prior_score": float(candidate.prior_score),
        "unary_score": unary,
        "relation_bonus_to_selected": relation_bonus,
        "topology_degree": topology_degree,
        "positive_relation_count": len(positive_relations.get(cid, [])),
        "evidence": candidate.evidence.model_dump(mode="json"),
        "sources": list(candidate.source_names),
        "face_ids": list(candidate.face_ids),
        "metadata": dict(candidate.metadata),
        "context_state": candidate.metadata.get("context_state", "ACTIVE"),
        "context_region_id": candidate.metadata.get("context_region_id"),
        "context_region_type": candidate.metadata.get("context_region_type"),
        "context_risk": float(candidate.metadata.get("context_risk", 0.0) or 0.0),
        "wall_identity_probability": float(solver._wall_identity_probability(candidate)),
        "observation_confidence": float(solver._observation_confidence(candidate)),
        "context_penalty": float(solver._context_penalty(candidate)),
        "structural_lineage_score": float((candidate.metadata.get("structural_lineage") or {}).get("score", 0.0) or 0.0),
        "structural_lineage_strong": bool((candidate.metadata.get("structural_lineage") or {}).get("strong", False)),
        "reasons": _evidence_reasons(candidate),
    }

    if cid in selected_ids:
        row.update(category="SELECTED", marginal_topology=None, marginal_global=None)
        return row

    inside = line_is_inside_perimeter(
        line=_candidate_line(candidate),
        polygon=perimeter_polygon(perimeter),
        tolerance_px=2.0,
    )
    if not inside:
        row.update(category="OUTSIDE_F02", marginal_topology=None, marginal_global=None)
        row["reasons"].append("line_is_inside_perimeter=False")
        return row

    conflicts = sorted(hard_conflicts.get(cid, set()) & selected_ids)
    row["hard_conflicts_with_selected"] = conflicts
    if conflicts:
        row.update(category="CONFLICT_WITH_SELECTED", marginal_topology=None, marginal_global=None)
        row["reasons"].append("hard_conflict_with_selected=" + ",".join(conflicts))
        return row

    topology_if_added = solver.topology.analyze(
        perimeter=perimeter,
        candidates=[*selected_candidates, candidate],
    )
    marginal_topology = float(topology_if_added.topology_score - baseline_topology.topology_score)
    marginal_global = float(unary + relation_bonus + marginal_topology)
    row["marginal_topology"] = marginal_topology
    row["marginal_global"] = marginal_global
    row["topology_if_added"] = topology_if_added.model_dump(mode="json")

    e = candidate.evidence
    reference_like = (
        e.dashed_penalty >= 0.5
        or (e.axis_support >= 0.60 and e.region_support < 0.45 and e.pair_overlap < 0.60)
    )
    if reference_like:
        row["category"] = "DASHED_REFERENCE"
        row["reasons"].append("reference_like_geometry")
    elif unary <= 0.0:
        row["category"] = "WEAK_EVIDENCE"
        row["reasons"].append("unary_score<=0")
    elif marginal_topology < -0.02 and marginal_global <= 0.0:
        row["category"] = "TOPOLOGY_REJECTED"
        row["reasons"].append("candidate_worsens_global_topology")
    elif marginal_global > 0.0:
        row["category"] = "SEARCH_COMPETITION"
        row["reasons"].append("positive_marginal_vs_final_solution_but_not_selected")
    else:
        row["category"] = "OTHER_REJECTED"
        row["reasons"].append("non_positive_marginal_without_primary_rule")
    return row


PROBE_COLORS = {
    "SELECTED": (255, 0, 0),
    "CONFLICT_WITH_SELECTED": (0, 180, 0),
    "OUTSIDE_F02": (0, 255, 255),
    "DASHED_REFERENCE": (0, 140, 255),
    "WEAK_EVIDENCE": (180, 0, 180),
    "TOPOLOGY_REJECTED": (0, 0, 255),
    "SEARCH_COMPETITION": (255, 255, 0),
    "OTHER_REJECTED": (150, 150, 150),
}


def _summarize_numeric(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    if not values:
        return {"count": 0, "mean": None, "min": None, "max": None}
    return {
        "count": len(values),
        "mean": sum(values) / len(values),
        "min": min(values),
        "max": max(values),
    }


def _draw_probe(image: np.ndarray, rows: list[dict[str, Any]], perimeter) -> None:
    _draw_perimeter(image, perimeter)
    ordered = sorted(rows, key=lambda row: row["category"] == "SELECTED")
    for row in ordered:
        color = PROBE_COLORS[row["category"]]
        # En la única imagen de salida, un SELECTED que todavía proviene de
        # REVIEW se distingue en naranja. Así podemos medir contaminación
        # contextual sin generar imágenes diagnósticas adicionales.
        if row["category"] == "SELECTED" and row.get("context_state") == "REVIEW":
            color = (0, 165, 255)
        p1 = (int(round(row["start"][0])), int(round(row["start"][1])))
        p2 = (int(round(row["end"][0])), int(round(row["end"][1])))
        thickness = 4 if row["category"] == "SELECTED" else 2
        cv2.line(image, p1, p2, color, thickness, cv2.LINE_AA)


def _run_level_probe(*, pipeline: ProposalCReconstructionPipeline, loaded: LoadedLevel, output_dir: Path, case_id: str, scale_profile) -> dict[str, Any]:
    level_result = loaded.level_result
    assert level_result.perimeter.state in {"VALID", "REVIEW"}
    perimeter = level_result.perimeter.editable_perimeter
    assert perimeter is not None
    raw_evidence = list(level_result.evidence.evidence)

    full = pipeline.run(
        level_view=level_result.level_view,
        perimeter=perimeter,
        evidence=raw_evidence,
        scale_profile=scale_profile,
    )
    assert full.diagnostics.graph_candidate_limit_drop_count == 0

    graph = full.candidate_graph
    solution = full.solution
    selected_ids = set(solution.selected_candidate_ids)
    by_id = {candidate.id: candidate for candidate in graph.candidates}
    selected_candidates = [by_id[cid] for cid in solution.selected_candidate_ids]
    hard, positive, all_relations = _relation_maps(graph)

    rows: list[dict[str, Any]] = []
    for candidate in graph.candidates:
        row = _classify_candidate(
            candidate=candidate,
            selected_ids=selected_ids,
            selected_candidates=selected_candidates,
            perimeter=perimeter,
            hard_conflicts=hard,
            positive_relations=positive,
            solver=pipeline.solver,
            baseline_topology=solution.topology,
        )
        row["relations"] = [
            {
                "other_id": relation.b_id if relation.a_id == candidate.id else relation.a_id,
                "type": relation.relation_type,
                "strength": float(relation.strength),
                "hard_conflict": bool(relation.hard_conflict),
            }
            for relation in all_relations.get(candidate.id, [])
        ]
        rows.append(row)

    counts = Counter(row["category"] for row in rows)
    rejected = [row for row in rows if row["category"] != "SELECTED"]
    weak = [row for row in rows if row["category"] == "WEAK_EVIDENCE"]
    semantic_nonzero = sum(1 for c in graph.candidates if float(c.evidence.semantic_support) > 0.0)
    vector_nonzero = sum(1 for c in graph.candidates if float(c.evidence.vector_support) > 0.0)
    selected_rows = [row for row in rows if row["category"] == "SELECTED"]
    selected_active = sum(1 for row in selected_rows if row["context_state"] == "ACTIVE")
    selected_protected = sum(1 for row in selected_rows if row["context_state"] == "PROTECTED")
    selected_review = sum(1 for row in selected_rows if row["context_state"] == "REVIEW")
    review_rows = [row for row in rows if row["context_state"] == "REVIEW"]
    protected_rows = [row for row in rows if row["context_state"] == "PROTECTED"]
    region_type_counts = Counter(region.region_type for region in full.context_gate.regions)
    decision_state_counts = Counter(decision.state for decision in full.context_gate.decisions)
    decision_region_type_counts = Counter(
        decision.region_type
        for decision in full.context_gate.decisions
        if decision.region_type is not None
    )
    selected_context_type_counts = Counter(
        row.get("context_region_type") or "NONE"
        for row in selected_rows
    )
    surface_types = {"FLOOR_FINISH_GRID_REGION", "HATCH_FILL_REGION", "FURNITURE_MODULE_REGION"}
    surface_rows = [row for row in rows if row.get("context_region_type") in surface_types]
    surface_selected = [row for row in selected_rows if row.get("context_region_type") in surface_types]
    strong_lineage_rows = [row for row in rows if row.get("structural_lineage_strong")]
    strong_lineage_selected = [row for row in selected_rows if row.get("structural_lineage_strong")]

    # ONE visual output per LevelView. No before/regions/decisions/graph-input set.
    image = _decode(level_result.level_view.raster_bytes)
    _draw_context_regions(image, full)
    _draw_probe(image, rows, perimeter)
    safe = _safe_name(loaded.level_name)
    overlay_path = output_dir / f"{case_id}__{safe}__context_classification_v6_2.png"
    _save(overlay_path, image)

    assert len(rows) == len(graph.candidates)
    assert sum(counts.values()) == len(graph.candidates)
    assert counts.get("SELECTED", 0) == len(selected_ids)

    return {
        "level": loaded.level_name,
        "replay": loaded.replay_source,
        "visual_file": str(overlay_path),
        "discovered_count": full.diagnostics.discovered_wall_candidate_count,
        "quarantined_count": full.diagnostics.quarantined_wall_candidate_count,
        "review_count": full.diagnostics.review_wall_candidate_count,
        "solver_candidate_count": len(graph.candidates),
        "selected_count": len(selected_ids),
        "rejected_count": len(rejected),
        "selected_ratio": (len(selected_ids) / len(graph.candidates)) if graph.candidates else 0.0,
        "category_counts": dict(sorted(counts.items())),
        "weak_evidence_ratio_of_rejected": (len(weak) / len(rejected)) if rejected else 0.0,
        "semantic_support_nonzero_count": semantic_nonzero,
        "vector_support_nonzero_count": vector_nonzero,
        "selected_active_count": selected_active,
        "selected_protected_count": selected_protected,
        "selected_review_count": selected_review,
        "protected_solver_count": len(protected_rows),
        "review_solver_count": len(review_rows),
        "review_selected_ratio": (selected_review / len(review_rows)) if review_rows else 0.0,
        "region_type_counts": dict(sorted(region_type_counts.items())),
        "decision_state_counts": dict(sorted(decision_state_counts.items())),
        "decision_region_type_counts": dict(sorted(decision_region_type_counts.items())),
        "selected_context_type_counts": dict(sorted(selected_context_type_counts.items())),
        "surface_context_solver_count": len(surface_rows),
        "surface_context_selected_count": len(surface_selected),
        "strong_structural_lineage_solver_count": len(strong_lineage_rows),
        "strong_structural_lineage_selected_count": len(strong_lineage_selected),
        "structural_lineage_all": _summarize_numeric(rows, "structural_lineage_score"),
        "structural_lineage_selected": _summarize_numeric(selected_rows, "structural_lineage_score"),
        "context_risk_all": _summarize_numeric(rows, "context_risk"),
        "context_risk_selected": _summarize_numeric(selected_rows, "context_risk"),
        "context_penalty_selected": _summarize_numeric(selected_rows, "context_penalty"),
        "wall_identity_selected": _summarize_numeric(selected_rows, "wall_identity_probability"),
        "wall_identity_rejected": _summarize_numeric(rejected, "wall_identity_probability"),
        "unary_all": _summarize_numeric(rows, "unary_score"),
        "unary_rejected": _summarize_numeric(rejected, "unary_score"),
        "prior_rejected": _summarize_numeric(rejected, "prior_score"),
        "solver_diagnostics": solution.diagnostics.model_dump(mode="json"),
        "baseline_topology": solution.topology.model_dump(mode="json"),
        "scale_profile": scale_profile.model_dump(mode="json"),
        "candidates": rows,
    }


def _run_standardized_case(case: LoadedCase) -> None:
    output_dir = OUTPUT_ROOT / case.case_id
    _reset_case_output(output_dir)
    pipeline = ProposalCReconstructionPipeline()

    scale_inputs = []
    for loaded in case.levels:
        level_result = loaded.level_result
        perimeter = level_result.perimeter.editable_perimeter
        assert perimeter is not None
        scale_inputs.append((level_result.level_view, perimeter))
    project_scale = pipeline.build_project_scale_context(levels=scale_inputs)

    report: dict[str, Any] = {
        "probe_version": RULE_VERSION,
        "base": BASELINE,
        "case_id": case.case_id,
        "case": case.case_name,
        "gemini_calls": 0,
        "visual_output_policy": "ONE_PNG_PER_LEVEL_VIEW",
        "project_scale": project_scale.model_dump(mode="json"),
        "levels": [],
    }

    for loaded in case.levels:
        scale_profile = project_scale.for_level(loaded.level_result.level_view.id)
        level_report = _run_level_probe(
            pipeline=pipeline,
            loaded=loaded,
            output_dir=output_dir,
            case_id=case.case_id,
            scale_profile=scale_profile,
        )
        report["levels"].append(level_report)

        print("\n" + "=" * 96)
        print(f"CONTEXT CLASSIFICATION V6.2 STANDARDIZED — {case.case_name} — {loaded.level_name}")
        print({
            "solver_candidates": level_report["solver_candidate_count"],
            "selected": level_report["selected_count"],
            "categories": level_report["category_counts"],
            "weak_ratio_of_rejected": round(level_report["weak_evidence_ratio_of_rejected"], 4),
            "region_types": level_report["region_type_counts"],
            "decision_states": level_report["decision_state_counts"],
            "scale_state": level_report["scale_profile"]["state"],
            "local_m_per_px": level_report["scale_profile"]["local_m_per_px"],
            "scale_factor": round(level_report["scale_profile"]["scale_factor_to_canonical"], 6),
            "visual": level_report["visual_file"],
        })

    # Unique JSON per case. No collision between Casa Viri / Miguel H / Miguel V.
    json_path = output_dir / f"{case.case_id}__context_classification_v6_2.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    pngs = sorted(output_dir.glob("*.png"))
    assert len(pngs) == len(case.levels), (
        f"La prueba debe producir exactamente 1 PNG por LevelView; "
        f"levels={len(case.levels)}, pngs={len(pngs)}"
    )

    print(f"\nJSON: {json_path}")
    print(f"Visuales: {len(pngs)} = 1 por LevelView")


@pytest.mark.parametrize("case_id", ["casa_viri", "miguel_h", "miguel_v"])
def test_context_classification_v6_2_standardized(case_id: str) -> None:
    # A partir de LoadedCase, los tres casos ejecutan exactamente el mismo probe.
    case = CASE_LOADERS[case_id]()
    _run_standardized_case(case)


if __name__ == "__main__":
    for _case_id in ("casa_viri", "miguel_h", "miguel_v"):
        _run_standardized_case(CASE_LOADERS[_case_id]())
