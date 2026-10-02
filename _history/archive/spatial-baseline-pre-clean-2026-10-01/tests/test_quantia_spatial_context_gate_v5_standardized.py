from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import pytest
from shapely.geometry import LineString

from app.quantia_spatialV1.engine import QuantiaSpatialEngine
from app.quantia_spatialV1.phase_01_level.pymupdf_level_source import PyMuPDFLevelSource
from app.quantia_spatialV1.phase_015_evidence.gemini_evidence_adapter import GeminiEvidenceAdapter
from app.quantia_spatialV1.phase_015_evidence.gemini_semantic_history import GeminiSemanticHistory
from app.quantia_spatialV1.reconstruction_core import ProposalCReconstructionPipeline
from app.quantia_spatialV1.reconstruction_core.perimeter_adapter import (
    line_is_inside_perimeter,
    perimeter_polygon,
)
from app.prompts.quantia_extraction_prompt import QUANTIA_EXTRACTION_PROMPT
from app.schemas.gemini_extraction_transport import get_gemini_extraction_transport_schema


TESTS_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = TESTS_DIR / "output"

RULE_VERSION = "DRAWING_CONTEXT_GATE_V5_STANDARDIZED_SOLVER_PROBE"
BASELINE = "Proposal C A1.2 + Context Gate V5 + solver exclusions probe"

MIGUEL_H_PDF_PATH = Path(
    r"D:\03 INGENIEIRA SISTEMAS\03 RESIDENCIAS PROFESIONALES"
    r"\REFERENCIAS Y ANEXOS Quantia General"
    r"\Plano Migue H sin muebles.pdf"
)
MIGUEL_H_RENDER_SCALE = 0.7935


@dataclass(frozen=True)
class LoadedLevel:
    level_name: str
    level_result: Any
    replay_source: dict[str, Any] | None = None


@dataclass(frozen=True)
class LoadedCase:
    case_id: str
    case_name: str
    levels: tuple[LoadedLevel, ...]


class _ReplayOnlyProvider:
    models = ("replay-only",)

    def analyze(self, **kwargs):
        raise AssertionError("La prueba estandarizada V5 es replay-only; no permite llamadas Gemini reales.")


# ---------------------------------------------------------------------------
# Common low-level helpers
# ---------------------------------------------------------------------------

def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_json(value: dict[str, Any]) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


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


def _candidate_line(candidate) -> LineString:
    return LineString([
        (float(candidate.start.x), float(candidate.start.y)),
        (float(candidate.end.x), float(candidate.end.y)),
    ])


# ---------------------------------------------------------------------------
# Exact replay loaders — only input preparation differs between cases.
# ---------------------------------------------------------------------------

def _find_casa_localization_replay(*, history_path: Path, raster_bytes: bytes) -> dict[str, Any] | None:
    records = _read_jsonl(history_path)
    raster_hash = _sha256_bytes(raster_bytes)
    started_by_key = {
        str(row.get("call_key")): row
        for row in records
        if row.get("event") == "STARTED" and row.get("call_key")
    }
    matches: list[dict[str, Any]] = []
    for row in records:
        if row.get("event") != "SUCCEEDED":
            continue
        data = row.get("data")
        if not isinstance(data, dict) or not isinstance(data.get("niveles"), list):
            continue
        started = started_by_key.get(str(row.get("call_key")))
        if started and str(started.get("raster_sha256")) == raster_hash:
            matches.append(row)
    return matches[-1]["data"] if matches else None


def _find_casa_semantic_replay(
    *,
    history_path: Path,
    source_document_id: str,
    source_page_number: int,
    raster_bytes: bytes,
) -> dict[str, Any] | None:
    records = _read_jsonl(history_path)
    raster_hash = _sha256_bytes(raster_bytes)
    started_by_call_id = {
        str(row.get("call_id")): row
        for row in records
        if row.get("event") == "STARTED" and row.get("call_id")
    }
    matches: list[dict[str, Any]] = []
    for row in records:
        if row.get("event") != "SUCCEEDED":
            continue
        if str(row.get("source_document_id")) != source_document_id:
            continue
        if int(row.get("source_page_number") or -1) != source_page_number:
            continue
        payload = row.get("response_payload")
        if not isinstance(payload, dict):
            continue
        started = started_by_call_id.get(str(row.get("call_id")))
        if started and str(started.get("raster_sha256")) == raster_hash:
            matches.append(row)
    return matches[-1]["response_payload"] if matches else None


def _find_exact_extraction_replay(
    *,
    history_path: Path,
    source_document_id: str,
    source_page_number: int,
    raster_bytes: bytes,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    records = _read_jsonl(history_path)
    prompt_hash = _sha256_text(QUANTIA_EXTRACTION_PROMPT)
    schema_hash = _sha256_json(get_gemini_extraction_transport_schema())
    raster_hash = _sha256_bytes(raster_bytes)

    started: dict[str, dict[str, Any]] = {}
    for record in records:
        call_id = str(record.get("call_id") or "").strip()
        if call_id and record.get("event") == "STARTED":
            started[call_id] = record

    for record in reversed(records):
        if record.get("event") != "SUCCEEDED":
            continue
        call_id = str(record.get("call_id") or "").strip()
        begin = started.get(call_id)
        if begin is None:
            continue
        if str(begin.get("source_document_id") or "") != source_document_id:
            continue
        if int(begin.get("source_page_number") or 0) != source_page_number:
            continue
        if begin.get("prompt_sha256") != prompt_hash:
            continue
        if begin.get("schema_sha256") != schema_hash:
            continue
        if begin.get("raster_sha256") != raster_hash:
            continue
        payload = record.get("response_payload")
        if not isinstance(payload, dict):
            continue

        replay = {
            "result": {
                "data": payload,
                "model": record.get("model"),
                "fallback_used": bool(record.get("fallback_used", False)),
            }
        }
        replay_source = {
            "call_id": call_id,
            "provider": record.get("provider"),
            "model": record.get("model"),
            "timestamp_utc": record.get("timestamp_utc"),
        }
        return replay, replay_source

    return None, None


# ---------------------------------------------------------------------------
# Case adapters. They ONLY convert source documents into LoadedLevel[];
# everything after that is exactly the same for the three examples.
# ---------------------------------------------------------------------------

def _load_casa_viri() -> LoadedCase:
    case_dir = TESTS_DIR / "data" / "casa_viri"
    config = _read_json(case_dir / "input" / "case.json")
    phase_01_history = case_dir / "output" / "phase_01_vision_history.jsonl"
    semantic_history = case_dir / "output" / "gemini_semantic_history.jsonl"

    pdf_path = Path(str(config["pdf_path"]))
    if not pdf_path.exists():
        pytest.skip(f"PDF Casa Viri no disponible: {pdf_path}")

    document_bytes = pdf_path.read_bytes()
    render_scale = float(config["render_scale"])
    source_document_id = str(config["source_document_id"])
    page_number = 1

    source = PyMuPDFLevelSource().read(document_bytes=document_bytes, render_scale=render_scale)
    page = source.pages[page_number - 1]

    localization = _find_casa_localization_replay(
        history_path=phase_01_history,
        raster_bytes=page.raster_bytes,
    )
    assert localization is not None, "Falta replay exacto F01 Casa Viri."

    extraction = _find_casa_semantic_replay(
        history_path=semantic_history,
        source_document_id=source_document_id,
        source_page_number=page_number,
        raster_bytes=page.raster_bytes,
    )
    assert extraction is not None, "Falta replay exacto F01.5 Casa Viri."

    level_names = [
        str(item.get("nombre"))
        for item in localization.get("niveles", [])
        if isinstance(item, dict) and item.get("nombre")
    ]
    assert level_names

    provider = _ReplayOnlyProvider()
    engine = QuantiaSpatialEngine(
        vision_provider=provider,
        gemini_evidence_adapter=GeminiEvidenceAdapter(
            provider=provider,
            semantic_history=GeminiSemanticHistory(history_path=None),
            prompt="LEGACY_REPLAY_ONLY",
            response_json_schema={},
        ),
    )
    engine_result = engine.run(
        document_bytes=document_bytes,
        media_mime_type=str(config["media_mime_type"]),
        source_document_id=source_document_id,
        render_scale=render_scale,
        known_level_names_by_page={page_number: level_names},
        localization_payloads_by_page={page_number: localization},
        gemini_extraction_payloads_by_page={page_number: extraction},
        enable_gemini_discovery=False,
        enable_gemini_extraction=True,
    )
    by_name = {item.level_view.level_name: item for item in engine_result.levels}
    assert set(by_name) == set(level_names)

    return LoadedCase(
        case_id="casa_viri",
        case_name="Casa Viri",
        levels=tuple(LoadedLevel(name, by_name[name]) for name in level_names),
    )


def _load_miguel_h() -> LoadedCase:
    if not MIGUEL_H_PDF_PATH.exists():
        pytest.skip(f"PDF Miguel H limpio no disponible: {MIGUEL_H_PDF_PATH}")

    fixture_dir = TESTS_DIR / "data" / "miguel_h"
    localization = _read_json(fixture_dir / "miguel_h_gemini_localization_replay.json")["result"]["data"]
    extraction = _read_json(fixture_dir / "miguel_h_gemini_extraction_replay.json")
    level_names = ("Planta Baja", "Planta Alta")

    provider = _ReplayOnlyProvider()
    engine = QuantiaSpatialEngine(
        vision_provider=provider,
        gemini_evidence_adapter=GeminiEvidenceAdapter(
            provider=provider,
            semantic_history=GeminiSemanticHistory(history_path=None),
            prompt="LEGACY_REPLAY_ONLY",
            response_json_schema={},
        ),
    )
    engine_result = engine.run(
        document_bytes=MIGUEL_H_PDF_PATH.read_bytes(),
        media_mime_type="application/pdf",
        source_document_id="MIGUEL_H_CLEAN_CONTEXT_GATE_V5_STANDARDIZED",
        render_scale=MIGUEL_H_RENDER_SCALE,
        known_level_names_by_page={1: list(level_names)},
        localization_payloads_by_page={1: localization},
        gemini_extraction_payloads_by_page={1: extraction},
        enable_gemini_discovery=False,
        enable_gemini_extraction=True,
    )
    by_name = {item.level_view.level_name: item for item in engine_result.levels}
    assert set(level_names).issubset(by_name)

    return LoadedCase(
        case_id="miguel_h",
        case_name="Miguel H limpio",
        levels=tuple(LoadedLevel(name, by_name[name]) for name in level_names),
    )


def _load_miguel_v() -> LoadedCase:
    case_dir = TESTS_DIR / "data" / "miguel_v"
    config = _read_json(case_dir / "input" / "case.json")
    documents = config.get("documents")
    if not isinstance(documents, list):
        raise ValueError("case.json de Miguel V no contiene documents[].")

    semantic_history = case_dir / "replay" / "gemini_semantic_history.jsonl"
    pdf_source = PyMuPDFLevelSource()
    loaded: list[LoadedLevel] = []

    for item in documents:
        pdf_path = Path(str(item["pdf_path"]))
        if not pdf_path.exists():
            pytest.skip(f"PDF Miguel V no disponible: {pdf_path}")

        document_bytes = pdf_path.read_bytes()
        render_scale = float(config["render_scale"])
        page = pdf_source.read(document_bytes=document_bytes, render_scale=render_scale).pages[0]

        replay_payload, replay_source = _find_exact_extraction_replay(
            history_path=semantic_history,
            source_document_id=str(item["document_id"]),
            source_page_number=int(item["source_page_number"]),
            raster_bytes=page.raster_bytes,
        )
        if replay_payload is None:
            pytest.skip(f"No hay replay exacto para {item['document_id']}; no se hará llamada Gemini.")

        engine = QuantiaSpatialEngine()
        engine_result = engine.run(
            document_bytes=document_bytes,
            media_mime_type=str(config["media_mime_type"]),
            source_document_id=str(item["document_id"]),
            render_scale=render_scale,
            known_level_names_by_page={1: [str(item["level_name"])]},
            isolated_pages={1},
            enable_gemini_discovery=False,
            enable_gemini_extraction=True,
            gemini_extraction_payloads_by_page={1: replay_payload},
        )
        assert len(engine_result.levels) == 1
        loaded.append(
            LoadedLevel(
                level_name=str(item["level_name"]),
                level_result=engine_result.levels[0],
                replay_source=replay_source,
            )
        )

    return LoadedCase(
        case_id="miguel_v",
        case_name="Miguel V",
        levels=tuple(loaded),
    )


CASE_LOADERS: dict[str, Callable[[], LoadedCase]] = {
    "casa_viri": _load_casa_viri,
    "miguel_h": _load_miguel_h,
    "miguel_v": _load_miguel_v,
}


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
        p1 = (int(round(row["start"][0])), int(round(row["start"][1])))
        p2 = (int(round(row["end"][0])), int(round(row["end"][1])))
        thickness = 4 if row["category"] == "SELECTED" else 2
        cv2.line(image, p1, p2, color, thickness, cv2.LINE_AA)


def _run_level_probe(*, pipeline: ProposalCReconstructionPipeline, loaded: LoadedLevel, output_dir: Path, case_id: str) -> dict[str, Any]:
    level_result = loaded.level_result
    assert level_result.perimeter.state in {"VALID", "REVIEW"}
    perimeter = level_result.perimeter.editable_perimeter
    assert perimeter is not None
    raw_evidence = list(level_result.evidence.evidence)

    full = pipeline.run(
        level_view=level_result.level_view,
        perimeter=perimeter,
        evidence=raw_evidence,
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

    # ONE visual output per LevelView. No before/regions/decisions/graph-input set.
    image = _decode(level_result.level_view.raster_bytes)
    _draw_probe(image, rows, perimeter)
    safe = _safe_name(loaded.level_name)
    overlay_path = output_dir / f"{case_id}__{safe}__solver_exclusions_probe_v5.png"
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
        "unary_all": _summarize_numeric(rows, "unary_score"),
        "unary_rejected": _summarize_numeric(rejected, "unary_score"),
        "prior_rejected": _summarize_numeric(rejected, "prior_score"),
        "solver_diagnostics": solution.diagnostics.model_dump(mode="json"),
        "baseline_topology": solution.topology.model_dump(mode="json"),
        "candidates": rows,
    }


def _run_standardized_case(case: LoadedCase) -> None:
    output_dir = OUTPUT_ROOT / case.case_id
    _reset_case_output(output_dir)
    pipeline = ProposalCReconstructionPipeline()

    report: dict[str, Any] = {
        "probe_version": RULE_VERSION,
        "base": BASELINE,
        "case_id": case.case_id,
        "case": case.case_name,
        "gemini_calls": 0,
        "visual_output_policy": "ONE_PNG_PER_LEVEL_VIEW",
        "levels": [],
    }

    for loaded in case.levels:
        level_report = _run_level_probe(
            pipeline=pipeline,
            loaded=loaded,
            output_dir=output_dir,
            case_id=case.case_id,
        )
        report["levels"].append(level_report)

        print("\n" + "=" * 96)
        print(f"CONTEXT GATE V5 STANDARDIZED — {case.case_name} — {loaded.level_name}")
        print({
            "solver_candidates": level_report["solver_candidate_count"],
            "selected": level_report["selected_count"],
            "categories": level_report["category_counts"],
            "weak_ratio_of_rejected": round(level_report["weak_evidence_ratio_of_rejected"], 4),
            "visual": level_report["visual_file"],
        })

    # Unique JSON per case. No collision between Casa Viri / Miguel H / Miguel V.
    json_path = output_dir / f"{case.case_id}__solver_exclusions_probe_v5.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    pngs = sorted(output_dir.glob("*.png"))
    assert len(pngs) == len(case.levels), (
        f"La prueba debe producir exactamente 1 PNG por LevelView; "
        f"levels={len(case.levels)}, pngs={len(pngs)}"
    )

    print(f"\nJSON: {json_path}")
    print(f"Visuales: {len(pngs)} = 1 por LevelView")


@pytest.mark.parametrize("case_id", ["casa_viri", "miguel_h", "miguel_v"])
def test_context_gate_v5_standardized(case_id: str) -> None:
    # A partir de LoadedCase, los tres casos ejecutan exactamente el mismo probe.
    case = CASE_LOADERS[case_id]()
    _run_standardized_case(case)


if __name__ == "__main__":
    for _case_id in ("casa_viri", "miguel_h", "miguel_v"):
        _run_standardized_case(CASE_LOADERS[_case_id]())
