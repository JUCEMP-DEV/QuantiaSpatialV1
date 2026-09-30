from __future__ import annotations

import json
import math
import shutil
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any

import cv2
import numpy as np
import pytest

from app.quantia_spatialV1.engine import QuantiaSpatialEngine
from app.quantia_spatialV1.phase_01_level.pymupdf_level_source import PyMuPDFLevelSource
from app.quantia_spatialV1.phase_015_evidence.gemini_evidence_adapter import GeminiEvidenceAdapter
from app.quantia_spatialV1.phase_015_evidence.gemini_semantic_history import GeminiSemanticHistory
from app.quantia_spatialV1.reconstruction_core import ProposalCReconstructionPipeline
from app.quantia_spatialV1.reconstruction_core.level_scale_normalizer import LevelScaleProfile
from app.quantia_spatialV1.reconstruction_core.raster_density_policy import RasterDensityPolicy
from app.quantia_spatialV1.reconstruction_core.scale_evidence_resolver import ScaleEvidenceResolver
from app.quantia_spatialV1.tests.quantia_case_loader import (
    CASE_LOADERS,
    DATA_DIR,
    MIGUEL_H_PDF_PATH,
    MIGUEL_H_RENDER_SCALE,
    LoadedCase,
    LoadedLevel,
    _find_casa_localization_replay,
    _find_casa_semantic_replay,
    _find_exact_extraction_replay,
    _first_existing,
    _read_json,
)


TESTS_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = TESTS_DIR / "output"
CASE_IDS = ("casa_viri", "miguel_h", "miguel_v")

PROBE_VERSION = "METRIC_RERASTER_V1_3_STANDARDIZED"
REQUIRED_SCALE_RESOLVER_VERSION = "SCALE_EVIDENCE_RESOLVER_V1_6"
TARGET_PX_PER_M = 90.0
TARGET_M_PER_PX = 1.0 / TARGET_PX_PER_M
MAX_TARGET_REL_ERROR = 0.05


class _ReplayOnlyProvider:
    models = ("replay-only",)

    def analyze(self, **kwargs):
        raise AssertionError(
            "METRIC_RERASTER_V1_3 es replay-only; no permite llamadas Gemini reales."
        )


def _engine_replay_only() -> QuantiaSpatialEngine:
    provider = _ReplayOnlyProvider()
    return QuantiaSpatialEngine(
        vision_provider=provider,
        gemini_evidence_adapter=GeminiEvidenceAdapter(
            provider=provider,
            semantic_history=GeminiSemanticHistory(history_path=None),
            prompt="LEGACY_REPLAY_ONLY_METRIC_RERASTER_V1_3",
            response_json_schema={},
        ),
    )


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


def _cleanup_own_outputs(case_id: str) -> Path:
    out_dir = OUTPUT_ROOT / case_id
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in out_dir.glob(f"{case_id}__*__metric_reraster_v1_3.png"):
        path.unlink(missing_ok=True)
    (out_dir / f"{case_id}__metric_reraster_v1_3.json").unlink(missing_ok=True)
    return out_dir


def _resolve_scale(
    level: LoadedLevel,
    resolver: ScaleEvidenceResolver,
    *,
    perimeter_override=None,
):
    result = level.level_result
    perimeter = perimeter_override
    if perimeter is None:
        perimeter = result.perimeter.editable_perimeter
    return resolver.resolve(
        level_view=result.level_view,
        evidence=result.evidence.evidence,
        perimeter=perimeter,
    )


def _scaled_perimeter_to_view(*, perimeter, source_view, target_view):
    """Conserva F02 como referencia inmutable y solo cambia coordenadas raster.

    La topología/ids/orden de muros no cambian. Únicamente se aplica la
    transformación afín del LevelView fuente al LevelView rerasterizado.
    """
    sx = float(target_view.raster_width_px) / float(source_view.raster_width_px)
    sy = float(target_view.raster_height_px) / float(source_view.raster_height_px)
    result = perimeter.model_copy(deep=True)
    # El F02 congelado conserva su identidad/geom. original, pero LevelView.id
    # es una identidad de ejecución y cambia entre replays/rerasterizados.
    # La copia adaptada debe pertenecer al LevelView destino para que el contrato
    # F02->Scale/Reconstruction siga siendo válido sin alterar la geometría fuente.
    result.level_view_id = target_view.id

    for vertex in result.vertices:
        point = vertex.point_px
        vertex.point_px = point.model_copy(
            update={
                "x": float(point.x) * sx,
                "y": float(point.y) * sy,
            }
        )

    by_id = {vertex.id: vertex.point_px for vertex in result.vertices}
    for wall in result.walls:
        a = by_id[wall.start_vertex_id]
        b = by_id[wall.end_vertex_id]
        wall.length_px = math.hypot(float(b.x) - float(a.x), float(b.y) - float(a.y))

    metric = result.metric_summary
    if metric.metric_scale_m_per_px is not None:
        uniform = (sx + sy) / 2.0
        metric.metric_scale_m_per_px = float(metric.metric_scale_m_per_px) / uniform

    result.geometry_revision += 1
    return result


def _miguel_h_canonical_f02_reference() -> dict[str, tuple[Any, Any]]:
    """Fallback canónico solo si el PDF actual no publica F02 en un nivel.

    Usa el raster canónico ya versionado de Miguel H y los mismos replays.
    No hace Gemini real y no modifica F02.
    """
    case_dir = DATA_DIR / "miguel_h"
    raster_path = case_dir / "miguel_h_canonical_raster.png"
    if not raster_path.exists():
        raise FileNotFoundError(f"Falta raster canónico Miguel H: {raster_path}")

    localization = _read_json(case_dir / "miguel_h_gemini_localization_replay.json")["result"]["data"]
    extraction = _read_json(case_dir / "miguel_h_gemini_extraction_replay.json")
    level_names = ("Planta Baja", "Planta Alta")

    result = _engine_replay_only().run(
        document_bytes=raster_path.read_bytes(),
        media_mime_type="image/png",
        source_document_id="MIGUEL_H_CANONICAL_F02_REFERENCE_METRIC_RERASTER_V1_3",
        known_level_names_by_page={1: list(level_names)},
        localization_payloads_by_page={1: localization},
        gemini_extraction_payloads_by_page={1: extraction},
        enable_gemini_discovery=False,
        enable_gemini_extraction=True,
    )
    by_name = {item.level_view.level_name: item for item in result.levels}
    refs: dict[str, tuple[Any, Any]] = {}
    for name in level_names:
        item = by_name[name]
        perimeter = item.perimeter.editable_perimeter
        assert perimeter is not None, (
            f"Miguel H/{name}: ni el PDF actual ni el raster canónico publicaron F02. "
            f"state={item.perimeter.state}"
        )
        refs[name] = (perimeter, item.level_view)
    return refs


def _freeze_f02_references(*, case_id: str, bootstrap: LoadedCase):
    canonical_h = None
    refs = {}
    for loaded in bootstrap.levels:
        runtime = loaded.level_result.perimeter.editable_perimeter
        if runtime is not None:
            refs[loaded.level_name] = {
                "perimeter": runtime.model_copy(deep=True),
                "source_view": loaded.level_result.level_view,
                "source": "BOOTSTRAP_F02",
            }
            continue

        if case_id != "miguel_h":
            raise AssertionError(
                f"{case_id}/{loaded.level_name}: F02 bootstrap no publicó EditablePerimeterModel. "
                f"state={loaded.level_result.perimeter.state}"
            )
        if canonical_h is None:
            canonical_h = _miguel_h_canonical_f02_reference()
        perimeter, source_view = canonical_h[loaded.level_name]
        refs[loaded.level_name] = {
            "perimeter": perimeter.model_copy(deep=True),
            "source_view": source_view,
            "source": "MIGUEL_H_CANONICAL_RASTER_F02_FALLBACK",
        }
    return refs


def _frozen_perimeter_for_loaded(*, frozen_ref, loaded: LoadedLevel):
    return _scaled_perimeter_to_view(
        perimeter=frozen_ref["perimeter"],
        source_view=frozen_ref["source_view"],
        target_view=loaded.level_result.level_view,
    )


def _scale_diagnostics(scale) -> dict[str, Any]:
    numeric = [item for item in scale.candidates if item.m_per_px is not None]
    return {
        "state": scale.state,
        "selected_m_per_px": scale.selected_m_per_px,
        "selected_method": scale.selected_method,
        "warnings": list(scale.warnings),
        "candidates": [
            {
                "method": item.method,
                "m_per_px": item.m_per_px,
                "confidence": item.confidence,
                "orientation": item.semantic_orientation,
                "side": item.graphic_side,
                "notes": list(item.notes),
            }
            for item in numeric
        ],
    }


def _build_profiles(
    *,
    levels: tuple[LoadedLevel, ...],
    scales_by_level_name: dict[str, Any],
) -> dict[str, LevelScaleProfile]:
    normalized_dims: list[float] = []
    normalized_areas: list[float] = []

    prepared: list[tuple[LoadedLevel, Any, float, float, float]] = []
    for loaded in levels:
        scale = scales_by_level_name[loaded.level_name]
        assert scale.state == "RESOLVED"
        assert scale.selected_m_per_px is not None
        local_m_per_px = float(scale.selected_m_per_px)
        factor = local_m_per_px / TARGET_M_PER_PX
        view = loaded.level_result.level_view
        local_min_dim = float(min(view.raster_width_px, view.raster_height_px))
        normalized_min_dim = local_min_dim * factor
        normalized_area = float(view.raster_width_px * view.raster_height_px) * factor * factor
        normalized_dims.append(normalized_min_dim)
        normalized_areas.append(normalized_area)
        prepared.append((loaded, scale, factor, local_min_dim, normalized_min_dim))

    reference_dim = float(median(normalized_dims))
    reference_area = float(median(normalized_areas))

    profiles: dict[str, LevelScaleProfile] = {}
    for loaded, scale, factor, local_min_dim, normalized_min_dim in prepared:
        view = loaded.level_result.level_view
        profiles[loaded.level_name] = LevelScaleProfile(
            level_view_id=view.id,
            state="F02_METRIC",
            local_m_per_px=float(scale.selected_m_per_px),
            canonical_m_per_px=TARGET_M_PER_PX,
            scale_factor_to_canonical=factor,
            local_min_dim_px=local_min_dim,
            normalized_min_dim_px=normalized_min_dim,
            project_reference_min_dim_normalized_px=reference_dim,
            project_reference_area_normalized_px2=reference_area,
            source="SCALE_FOUNDATION_V1_6_METRIC_RERASTER_V1_3",
            confidence=float(scale.selected_confidence),
        )
    return profiles


def _classification_summary(full) -> dict[str, Any]:
    graph = full.candidate_graph
    selected_ids = set(full.solution.selected_candidate_ids)
    selected = [candidate for candidate in graph.candidates if candidate.id in selected_ids]

    region_types = Counter(region.region_type for region in full.context_gate.regions)
    decision_states = Counter(decision.state for decision in full.context_gate.decisions)
    decision_region_types = Counter(
        decision.region_type
        for decision in full.context_gate.decisions
        if decision.region_type is not None
    )
    selected_context_types = Counter(
        str(candidate.metadata.get("context_region_type") or "NONE")
        for candidate in selected
    )

    surface_types = {
        "FLOOR_FINISH_GRID_REGION",
        "HATCH_FILL_REGION",
        "FURNITURE_MODULE_REGION",
    }
    surface_solver = [
        candidate for candidate in graph.candidates
        if candidate.metadata.get("context_region_type") in surface_types
    ]
    surface_selected = [
        candidate for candidate in selected
        if candidate.metadata.get("context_region_type") in surface_types
    ]
    strong_lineage_selected = [
        candidate for candidate in selected
        if bool((candidate.metadata.get("structural_lineage") or {}).get("strong", False))
    ]

    return {
        "drawing_line_count": full.diagnostics.drawing_line_count,
        "discovered_count": full.diagnostics.discovered_wall_candidate_count,
        "quarantined_count": full.diagnostics.quarantined_wall_candidate_count,
        "review_count": full.diagnostics.review_wall_candidate_count,
        "solver_candidate_count": len(graph.candidates),
        "selected_count": len(selected_ids),
        "selected_ratio": (len(selected_ids) / len(graph.candidates)) if graph.candidates else 0.0,
        "region_type_counts": dict(sorted(region_types.items())),
        "decision_state_counts": dict(sorted(decision_states.items())),
        "decision_region_type_counts": dict(sorted(decision_region_types.items())),
        "selected_context_type_counts": dict(sorted(selected_context_types.items())),
        "surface_context_solver_count": len(surface_solver),
        "surface_context_selected_count": len(surface_selected),
        "strong_structural_lineage_selected_count": len(strong_lineage_selected),
        "perimeter_wall_count": full.diagnostics.perimeter_wall_count,
        "divider_wall_count": full.diagnostics.parametric_divider_wall_count,
        "topology": full.solution.topology.model_dump(mode="json"),
        "solver_diagnostics": full.solution.diagnostics.model_dump(mode="json"),
    }


def _run_classification(*, loaded: LoadedLevel, profile: LevelScaleProfile, perimeter):
    result = loaded.level_result
    pipeline = ProposalCReconstructionPipeline()
    full = pipeline.run(
        level_view=result.level_view,
        perimeter=perimeter,
        evidence=result.evidence.evidence,
        scale_profile=profile,
    )
    assert full.diagnostics.graph_candidate_limit_drop_count == 0
    return full, _classification_summary(full)


def _draw_final_overlay(*, loaded: LoadedLevel, full, perimeter, output_path: Path) -> None:
    image = _decode(loaded.level_result.level_view.raster_bytes)

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
        color = (255, 255, 0)
        if decision and decision.region_type == "FLOOR_FINISH_GRID_REGION":
            color = (0, 255, 255)
        cv2.line(image, p1, p2, color, 3, cv2.LINE_AA)

    selected_ids = set(full.solution.selected_candidate_ids)
    for candidate in full.candidate_graph.candidates:
        if candidate.id not in selected_ids:
            continue
        p1 = (int(round(candidate.start.x)), int(round(candidate.start.y)))
        p2 = (int(round(candidate.end.x)), int(round(candidate.end.y)))
        state = str(candidate.metadata.get("context_state") or "ACTIVE")
        color = (0, 165, 255) if state == "REVIEW" else (255, 0, 0)
        cv2.line(image, p1, p2, color, 4, cv2.LINE_AA)

    _save(output_path, image)


def _rerun_casa_viri(*, render_scale: float) -> LoadedCase:
    case_dir = DATA_DIR / "casa_viri"
    config_path = _first_existing(case_dir / "case.json", case_dir / "input" / "case.json")
    config = _read_json(config_path)
    phase_01_history = _first_existing(
        case_dir / "replay" / "phase_01_vision_history.jsonl",
        case_dir / "output" / "phase_01_vision_history.jsonl",
    )
    semantic_history = _first_existing(
        case_dir / "replay" / "gemini_semantic_history.jsonl",
        case_dir / "output" / "gemini_semantic_history.jsonl",
    )

    pdf_path = Path(str(config["pdf_path"]))
    if not pdf_path.exists():
        pytest.skip(f"PDF Casa Viri no disponible: {pdf_path}")
    document_bytes = pdf_path.read_bytes()
    original_scale = float(config["render_scale"])
    source_document_id = str(config["source_document_id"])

    original_page = PyMuPDFLevelSource().read(
        document_bytes=document_bytes,
        render_scale=original_scale,
    ).pages[0]
    localization = _find_casa_localization_replay(
        history_path=phase_01_history,
        raster_bytes=original_page.raster_bytes,
    )
    assert localization is not None, "Falta replay F01 Casa Viri del raster bootstrap."
    extraction = _find_casa_semantic_replay(
        history_path=semantic_history,
        source_document_id=source_document_id,
        source_page_number=1,
        raster_bytes=original_page.raster_bytes,
    )
    assert extraction is not None, "Falta replay F01.5 Casa Viri del raster bootstrap."

    level_names = [
        str(item.get("nombre"))
        for item in localization.get("niveles", [])
        if isinstance(item, dict) and item.get("nombre")
    ]
    assert level_names

    result = _engine_replay_only().run(
        document_bytes=document_bytes,
        media_mime_type=str(config["media_mime_type"]),
        source_document_id=source_document_id,
        render_scale=float(render_scale),
        known_level_names_by_page={1: level_names},
        localization_payloads_by_page={1: localization},
        gemini_extraction_payloads_by_page={1: extraction},
        enable_gemini_discovery=False,
        enable_gemini_extraction=True,
    )
    by_name = {item.level_view.level_name: item for item in result.levels}
    return LoadedCase(
        case_id="casa_viri",
        case_name="Casa Viri",
        levels=tuple(LoadedLevel(name, by_name[name]) for name in level_names),
    )


def _rerun_miguel_h(*, render_scale: float) -> LoadedCase:
    if not MIGUEL_H_PDF_PATH.exists():
        pytest.skip(f"PDF Miguel H limpio no disponible: {MIGUEL_H_PDF_PATH}")

    case_dir = DATA_DIR / "miguel_h"
    localization = _read_json(case_dir / "miguel_h_gemini_localization_replay.json")["result"]["data"]
    extraction = _read_json(case_dir / "miguel_h_gemini_extraction_replay.json")
    level_names = ("Planta Baja", "Planta Alta")

    result = _engine_replay_only().run(
        document_bytes=MIGUEL_H_PDF_PATH.read_bytes(),
        media_mime_type="application/pdf",
        source_document_id="MIGUEL_H_CLEAN_STANDARDIZED_REPLAY",
        render_scale=float(render_scale),
        known_level_names_by_page={1: list(level_names)},
        localization_payloads_by_page={1: localization},
        gemini_extraction_payloads_by_page={1: extraction},
        enable_gemini_discovery=False,
        enable_gemini_extraction=True,
    )
    by_name = {item.level_view.level_name: item for item in result.levels}
    return LoadedCase(
        case_id="miguel_h",
        case_name="Miguel H limpio",
        levels=tuple(LoadedLevel(name, by_name[name]) for name in level_names),
    )


def _rerun_miguel_v(*, render_scales_by_level: dict[str, float]) -> LoadedCase:
    case_dir = DATA_DIR / "miguel_v"
    config_path = _first_existing(case_dir / "case.json", case_dir / "input" / "case.json")
    config = _read_json(config_path)
    documents = config.get("documents")
    if not isinstance(documents, list):
        raise ValueError("case.json de Miguel V no contiene documents[].")
    semantic_history = _first_existing(
        case_dir / "replay" / "gemini_semantic_history.jsonl",
        case_dir / "output" / "gemini_semantic_history.jsonl",
    )
    original_scale = float(config["render_scale"])
    source = PyMuPDFLevelSource()

    loaded: list[LoadedLevel] = []
    for item in documents:
        level_name = str(item["level_name"])
        pdf_path = Path(str(item["pdf_path"]))
        if not pdf_path.exists():
            pytest.skip(f"PDF Miguel V no disponible: {pdf_path}")
        document_bytes = pdf_path.read_bytes()
        original_page = source.read(
            document_bytes=document_bytes,
            render_scale=original_scale,
        ).pages[0]
        replay_payload, replay_source = _find_exact_extraction_replay(
            history_path=semantic_history,
            source_document_id=str(item["document_id"]),
            source_page_number=int(item["source_page_number"]),
            raster_bytes=original_page.raster_bytes,
        )
        assert replay_payload is not None, (
            f"Falta replay exacto bootstrap para {item['document_id']}; "
            "la prueba no hará llamada Gemini."
        )

        result = _engine_replay_only().run(
            document_bytes=document_bytes,
            media_mime_type=str(config["media_mime_type"]),
            source_document_id=str(item["document_id"]),
            render_scale=float(render_scales_by_level[level_name]),
            known_level_names_by_page={1: [level_name]},
            isolated_pages={1},
            enable_gemini_discovery=False,
            enable_gemini_extraction=True,
            gemini_extraction_payloads_by_page={1: replay_payload},
        )
        assert len(result.levels) == 1
        loaded.append(LoadedLevel(level_name, result.levels[0], replay_source))

    return LoadedCase(
        case_id="miguel_v",
        case_name="Miguel V",
        levels=tuple(loaded),
    )


def _rerun_at_recommended_density(
    *,
    case_id: str,
    recommendations_by_level: dict[str, float],
) -> tuple[LoadedCase, dict[str, float]]:
    if case_id == "casa_viri":
        page_scale = float(median(recommendations_by_level.values()))
        rerun = _rerun_casa_viri(render_scale=page_scale)
        return rerun, {level.level_name: page_scale for level in rerun.levels}

    if case_id == "miguel_h":
        page_scale = float(median(recommendations_by_level.values()))
        rerun = _rerun_miguel_h(render_scale=page_scale)
        return rerun, {level.level_name: page_scale for level in rerun.levels}

    if case_id == "miguel_v":
        rerun = _rerun_miguel_v(render_scales_by_level=recommendations_by_level)
        return rerun, dict(recommendations_by_level)

    raise KeyError(case_id)


def _delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    numeric_keys = (
        "drawing_line_count",
        "discovered_count",
        "quarantined_count",
        "review_count",
        "solver_candidate_count",
        "selected_count",
        "selected_ratio",
        "surface_context_solver_count",
        "surface_context_selected_count",
        "strong_structural_lineage_selected_count",
        "perimeter_wall_count",
        "divider_wall_count",
    )
    return {
        key: float(after[key]) - float(before[key])
        for key in numeric_keys
    }


def _run_case(case_id: str) -> None:
    out_dir = _cleanup_own_outputs(case_id)
    resolver = ScaleEvidenceResolver()
    assert resolver.VERSION == REQUIRED_SCALE_RESOLVER_VERSION, (
        f"Esta prueba requiere {REQUIRED_SCALE_RESOLVER_VERSION}, "
        f"pero el repo cargó {resolver.VERSION}. Reemplaza los archivos incluidos en este paquete."
    )
    policy = RasterDensityPolicy(target_geometry_px_per_m=TARGET_PX_PER_M)

    # PASS 1 — bootstrap raster actual + replays existentes. No Gemini nuevo.
    bootstrap = CASE_LOADERS[case_id]()
    frozen_refs = _freeze_f02_references(case_id=case_id, bootstrap=bootstrap)
    bootstrap_perimeters = {
        loaded.level_name: _frozen_perimeter_for_loaded(
            frozen_ref=frozen_refs[loaded.level_name],
            loaded=loaded,
        )
        for loaded in bootstrap.levels
    }
    bootstrap_scales = {
        loaded.level_name: _resolve_scale(
            loaded,
            resolver,
            perimeter_override=bootstrap_perimeters[loaded.level_name],
        )
        for loaded in bootstrap.levels
    }
    bootstrap_failures: list[dict[str, Any]] = []
    for level_name, scale in bootstrap_scales.items():
        if scale.state != "RESOLVED":
            bootstrap_failures.append({
                "level": level_name,
                "diagnostics": _scale_diagnostics(scale),
            })
            continue
        assert scale.selected_m_per_px is not None
        assert scale.current_render_scale is not None

    assert not bootstrap_failures, (
        f"{case_id}: Scale Foundation V1.6 no cerró uno o más LevelView. "
        f"diagnostics={json.dumps(bootstrap_failures, ensure_ascii=False)}"
    )

    recommendations_by_level: dict[str, float] = {}
    recommendation_payloads: dict[str, dict[str, Any]] = {}
    for loaded in bootstrap.levels:
        scale = bootstrap_scales[loaded.level_name]
        rec = policy.recommend_from_grounded_scale(
            current_render_scale=float(scale.current_render_scale),
            current_m_per_px=float(scale.selected_m_per_px),
        )
        recommendations_by_level[loaded.level_name] = float(rec.recommended_render_scale)
        recommendation_payloads[loaded.level_name] = rec.model_dump(mode="json")

    # Métrica V6.2 sobre el raster actual usando Scale Foundation V1.5.
    before_profiles = _build_profiles(
        levels=bootstrap.levels,
        scales_by_level_name=bootstrap_scales,
    )
    before_summaries: dict[str, dict[str, Any]] = {}
    for loaded in bootstrap.levels:
        _, summary = _run_classification(
            loaded=loaded,
            profile=before_profiles[loaded.level_name],
            perimeter=bootstrap_perimeters[loaded.level_name],
        )
        before_summaries[loaded.level_name] = summary

    # PASS 2 — nuevo raster geométrico. Los payloads Gemini se reutilizan;
    # PyMuPDF/OpenCV/OCR se recalculan. F02 se conserva INMUTABLE como referencia
    # geométrica y solo se transforma a las nuevas coordenadas del LevelView.
    normalized, applied_render_scales = _rerun_at_recommended_density(
        case_id=case_id,
        recommendations_by_level=recommendations_by_level,
    )

    normalized_perimeters = {
        loaded.level_name: _frozen_perimeter_for_loaded(
            frozen_ref=frozen_refs[loaded.level_name],
            loaded=loaded,
        )
        for loaded in normalized.levels
    }
    normalized_scales = {
        loaded.level_name: _resolve_scale(
            loaded,
            resolver,
            perimeter_override=normalized_perimeters[loaded.level_name],
        )
        for loaded in normalized.levels
    }
    for loaded in normalized.levels:
        scale = normalized_scales[loaded.level_name]
        assert scale.state == "RESOLVED", (
            f"{case_id}/{loaded.level_name}: Scale Foundation V1.5 dejó de resolver tras reraster. "
            f"diagnostics={json.dumps(_scale_diagnostics(scale), ensure_ascii=False)}"
        )
        assert scale.selected_m_per_px is not None
        actual_px_per_m = 1.0 / float(scale.selected_m_per_px)
        rel_error = abs(actual_px_per_m - TARGET_PX_PER_M) / TARGET_PX_PER_M
        assert rel_error <= MAX_TARGET_REL_ERROR, (
            f"{case_id}/{loaded.level_name}: densidad {actual_px_per_m:.3f} px/m "
            f"fuera de ±{MAX_TARGET_REL_ERROR * 100:.1f}% del objetivo {TARGET_PX_PER_M}."
        )

    after_profiles = _build_profiles(
        levels=normalized.levels,
        scales_by_level_name=normalized_scales,
    )

    report: dict[str, Any] = {
        "probe_version": PROBE_VERSION,
        "case_id": normalized.case_id,
        "case": normalized.case_name,
        "gemini_calls_added": 0,
        "semantic_replay_policy": "REUSE_BOOTSTRAP_PAYLOAD_WITHOUT_NEW_GEMINI_CALL",
        "f02_policy": "IMMUTABLE_BOOTSTRAP_GEOMETRY_TRANSFORMED_TO_RERASTER_COORDINATES",
        "target_geometry_px_per_m": TARGET_PX_PER_M,
        "page_scale_policy": (
            "MEDIAN_RECOMMENDATION_PER_SHARED_PAGE"
            if case_id in {"casa_viri", "miguel_h"}
            else "PER_DOCUMENT_LEVEL_RECOMMENDATION"
        ),
        "levels": [],
    }

    normalized_by_name = {loaded.level_name: loaded for loaded in normalized.levels}
    bootstrap_by_name = {loaded.level_name: loaded for loaded in bootstrap.levels}

    for level_name in [level.level_name for level in bootstrap.levels]:
        before_loaded = bootstrap_by_name[level_name]
        after_loaded = normalized_by_name[level_name]
        before_scale = bootstrap_scales[level_name]
        after_scale = normalized_scales[level_name]
        after_full, after_summary = _run_classification(
            loaded=after_loaded,
            profile=after_profiles[level_name],
            perimeter=normalized_perimeters[level_name],
        )
        before_summary = before_summaries[level_name]

        safe = _safe_name(level_name)
        visual_path = out_dir / f"{case_id}__{safe}__metric_reraster_v1_3.png"
        _draw_final_overlay(
            loaded=after_loaded,
            full=after_full,
            perimeter=normalized_perimeters[level_name],
            output_path=visual_path,
        )

        before_view = before_loaded.level_result.level_view
        after_view = after_loaded.level_result.level_view
        before_perimeter = bootstrap_perimeters[level_name]
        after_perimeter = normalized_perimeters[level_name]

        after_px_per_m = 1.0 / float(after_scale.selected_m_per_px)
        report["levels"].append(
            {
                "level": level_name,
                "bootstrap": {
                    "render_scale": float(before_scale.current_render_scale),
                    "m_per_px": float(before_scale.selected_m_per_px),
                    "px_per_m": 1.0 / float(before_scale.selected_m_per_px),
                    "raster_size_px": [before_view.raster_width_px, before_view.raster_height_px],
                    "level_bbox_px": before_view.source_bbox_px.model_dump(mode="json"),
                    "f02_wall_count": len(before_perimeter.walls),
                    "f02_reference_source": frozen_refs[level_name]["source"],
                    "runtime_f02_state": before_loaded.level_result.perimeter.state,
                    "runtime_editable_perimeter_available": (
                        before_loaded.level_result.perimeter.editable_perimeter is not None
                    ),
                    "scale_method": before_scale.selected_method,
                },
                "recommendation": recommendation_payloads[level_name],
                "reraster": {
                    "applied_render_scale": float(applied_render_scales[level_name]),
                    "m_per_px": float(after_scale.selected_m_per_px),
                    "px_per_m": after_px_per_m,
                    "target_relative_error": abs(after_px_per_m - TARGET_PX_PER_M) / TARGET_PX_PER_M,
                    "raster_size_px": [after_view.raster_width_px, after_view.raster_height_px],
                    "level_bbox_px": after_view.source_bbox_px.model_dump(mode="json"),
                    "f02_wall_count": len(after_perimeter.walls),
                    "f02_reference_source": frozen_refs[level_name]["source"],
                    "runtime_f02_state": after_loaded.level_result.perimeter.state,
                    "runtime_editable_perimeter_available": (
                        after_loaded.level_result.perimeter.editable_perimeter is not None
                    ),
                    "scale_method": after_scale.selected_method,
                    "scale_state": after_scale.state,
                },
                "v6_2_metric_aware_before": before_summary,
                "v6_2_metric_normalized_after": after_summary,
                "delta_after_minus_before": _delta(before_summary, after_summary),
                "visual_file": str(visual_path),
            }
        )

        print("\n" + "=" * 100)
        print(f"METRIC RERASTER V1.2 — {normalized.case_name} — {level_name}")
        print({
            "render_before": round(float(before_scale.current_render_scale), 6),
            "render_after": round(float(applied_render_scales[level_name]), 6),
            "px_m_before": round(1.0 / float(before_scale.selected_m_per_px), 3),
            "px_m_after": round(after_px_per_m, 3),
            "discovered_before": before_summary["discovered_count"],
            "discovered_after": after_summary["discovered_count"],
            "quarantine_before": before_summary["quarantined_count"],
            "quarantine_after": after_summary["quarantined_count"],
            "selected_before": before_summary["selected_count"],
            "selected_after": after_summary["selected_count"],
            "surface_selected_before": before_summary["surface_context_selected_count"],
            "surface_selected_after": after_summary["surface_context_selected_count"],
            "visual": str(visual_path),
        })

    json_path = out_dir / f"{case_id}__metric_reraster_v1_3.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    own_pngs = sorted(out_dir.glob(f"{case_id}__*__metric_reraster_v1_3.png"))
    assert len(own_pngs) == len(normalized.levels), (
        "La prueba debe producir exactamente 1 PNG por LevelView para METRIC_RERASTER_V1_3."
    )
    assert json_path.exists()

    print(f"\nJSON: {json_path}")
    print(f"Visuales: {len(own_pngs)} = 1 por LevelView")


@pytest.mark.parametrize("case_id", CASE_IDS)
def test_metric_reraster_v1_3_standardized(case_id: str) -> None:
    _run_case(case_id)


if __name__ == "__main__":
    for _case_id in CASE_IDS:
        _run_case(_case_id)
