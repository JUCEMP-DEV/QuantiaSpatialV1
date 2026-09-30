from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import pytest
import requests
from dotenv import dotenv_values
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import polygonize_full, snap, unary_union

from app.quantia_spatial.providers.vision import GeminiSpatialVisionProvider
from app.quantia_spatial.tests.quantia_case_loader import CASE_LOADERS


PROBE_VERSION = "CALL2_GEOMETRY_INTERFACE04_ALL_3_V1_2_QWEN_HF"
RESPONSE_CONTRACT_VERSION = "CALL2_WALL_AUDIT_ELEMENTS_V3"
INTERFACE04_CONTRACT_VERSION = "SPATIAL_INTERFACE04_GEOMETRY_REVIEW_V1"
BASELINE = "WALL_CANONICALIZATION_ISOLATED_ALL_3_V1"

TESTS_DIR = Path(__file__).resolve().parent
CANONICAL_ROOT = TESTS_DIR / "output" / "wall_canonicalization_isolated_v1"
OUTPUT_ROOT = TESTS_DIR / "output" / "call2_geometry_interface04_v1"

CASE_IDS = ("casa_viri", "miguel_h", "miguel_v")

# Gates experimentales de esta prueba. No modifican el motor.
MIN_WALL_CORRECTION_CONFIDENCE = float(os.getenv("QUANTIA_CALL2_WALL_MIN_CONFIDENCE", "0.82"))
MIN_PERIMETER_CORRECTION_CONFIDENCE = float(os.getenv("QUANTIA_CALL2_PERIMETER_MIN_CONFIDENCE", "0.90"))
MIN_ELEMENT_KEEP_CONFIDENCE = float(os.getenv("QUANTIA_CALL2_ELEMENT_MIN_CONFIDENCE", "0.65"))
MIN_SPACE_AREA_M2 = float(os.getenv("QUANTIA_CALL2_MIN_SPACE_AREA_M2", "0.50"))
OPENING_HOST_MAX_DISTANCE_M = float(os.getenv("QUANTIA_CALL2_OPENING_HOST_MAX_DISTANCE_M", "0.30"))
PARALLEL_LATERAL_MAX_M = float(os.getenv("QUANTIA_CALL2_PARALLEL_LATERAL_MAX_M", "0.22"))
PARALLEL_ANGLE_MAX_DEG = float(os.getenv("QUANTIA_CALL2_PARALLEL_ANGLE_MAX_DEG", "4.0"))
PARALLEL_MIN_OVERLAP_RATIO = float(os.getenv("QUANTIA_CALL2_PARALLEL_MIN_OVERLAP_RATIO", "0.50"))

# Transporte experimental Qwen por Hugging Face Inference Providers.
# No modifica providers/vision.py ni el motor canónico.
HF_ROUTER_CHAT_COMPLETIONS_URL = "https://router.huggingface.co/v1/chat/completions"
HF_TRANSIENT_HTTP_STATUS = {408, 429, 500, 502, 503, 504}
HF_MAX_ATTEMPTS_PER_ROUTE = int(os.getenv("QUANTIA_CALL2_HF_MAX_ATTEMPTS", "3"))
HF_REQUEST_TIMEOUT_SECONDS = int(os.getenv("QUANTIA_CALL2_HF_TIMEOUT_SECONDS", "180"))
HF_PRIMARY_MODEL = str(
    os.getenv(
        "QUANTIA_CALL2_HF_PRIMARY_MODEL",
        "Qwen/Qwen3-VL-30B-A3B-Instruct:deepinfra",
    )
).strip()
HF_FALLBACK_MODEL = str(
    os.getenv(
        "QUANTIA_CALL2_HF_FALLBACK_MODEL",
        "Qwen/Qwen3-VL-30B-A3B-Instruct:novita",
    )
).strip()


RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "contract_version": {"type": "string"},
        "level_view_id": {"type": "string"},
        "graph_assessment": {
            "type": "string",
            "enum": ["COMPLETE", "PARTIAL", "REVIEW"],
        },
        "wall_corrections": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "correction_id": {"type": "string"},
                    "action": {
                        "type": "string",
                        "enum": [
                            "ADD_WALL",
                            "REMOVE_WALL",
                            "REPLACE_CENTERLINE",
                            "ADJUST_ENDPOINT",
                            "MERGE_PARALLEL_WALLS",
                            "SPLIT_WALL",
                        ],
                    },
                    "target_wall_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "wall_role": {
                        "type": "string",
                        "enum": ["PERIMETER", "DIVIDER", "UNKNOWN"],
                    },
                    "start_px": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 2,
                        "maxItems": 2,
                    },
                    "end_px": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 2,
                        "maxItems": 2,
                    },
                    "endpoint": {
                        "type": "string",
                        "enum": ["START", "END"],
                    },
                    "new_point_px": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 2,
                        "maxItems": 2,
                    },
                    "segments_px": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "start_px": {
                                    "type": "array",
                                    "items": {"type": "number"},
                                    "minItems": 2,
                                    "maxItems": 2,
                                },
                                "end_px": {
                                    "type": "array",
                                    "items": {"type": "number"},
                                    "minItems": 2,
                                    "maxItems": 2,
                                },
                            },
                            "required": ["start_px", "end_px"],
                        },
                    },
                    "thickness_px": {"type": "number", "minimum": 0.0},
                    "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "reason": {"type": "string"},
                },
                "required": [
                    "correction_id",
                    "action",
                    "target_wall_ids",
                    "wall_role",
                    "confidence",
                    "reason",
                ],
            },
        },
        "gap_decisions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "gap_id": {"type": "string"},
                    "classification": {
                        "type": "string",
                        "enum": [
                            "WALL_CONTINUITY",
                            "PROBABLE_OPENING",
                            "UNCERTAIN",
                            "NOT_A_GAP",
                        ],
                    },
                    "logical_wall_continuity": {"type": "boolean"},
                    "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "reason": {"type": "string"},
                },
                "required": [
                    "gap_id",
                    "classification",
                    "logical_wall_continuity",
                    "confidence",
                    "reason",
                ],
            },
        },
        "architectural_elements": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "element_id": {"type": "string"},
                    "element_type": {
                        "type": "string",
                        "enum": [
                            "DOOR",
                            "WINDOW",
                            "GARAGE_DOOR",
                            "STAIR",
                            "UNKNOWN_ARCHITECTURAL_ELEMENT",
                        ],
                    },
                    "bbox_px": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 4,
                        "maxItems": 4,
                    },
                    "polygon_px": {
                        "type": "array",
                        "items": {
                            "type": "array",
                            "items": {"type": "number"},
                            "minItems": 2,
                            "maxItems": 2,
                        },
                    },
                    "probable_host_wall_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "span_start_px": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 2,
                        "maxItems": 2,
                    },
                    "span_end_px": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 2,
                        "maxItems": 2,
                    },
                    "stair_axis_start_px": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 2,
                        "maxItems": 2,
                    },
                    "stair_axis_end_px": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 2,
                        "maxItems": 2,
                    },
                    "stair_direction": {
                        "type": "string",
                        "enum": ["UP", "DOWN", "UNKNOWN"],
                    },
                    "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "reason": {"type": "string"},
                },
                "required": [
                    "element_id",
                    "element_type",
                    "bbox_px",
                    "probable_host_wall_ids",
                    "confidence",
                    "reason",
                ],
            },
        },
        "unresolved_regions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "bbox_px": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 4,
                        "maxItems": 4,
                    },
                    "reason": {"type": "string"},
                    "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                },
                "required": ["bbox_px", "reason", "confidence"],
            },
        },
        "summary": {"type": "string"},
    },
    "required": [
        "contract_version",
        "level_view_id",
        "graph_assessment",
        "wall_corrections",
        "gap_decisions",
        "architectural_elements",
        "unresolved_regions",
        "summary",
    ],
}


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_json(data: Any) -> str:
    raw = json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return _sha256_bytes(raw)


def _slug(value: str) -> str:
    safe = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(value))
    while "__" in safe:
        safe = safe.replace("__", "_")
    return safe.strip("_") or "level"


def _point(value: Iterable[float]) -> tuple[float, float]:
    seq = list(value)
    if len(seq) != 2:
        raise ValueError(f"Punto inválido: {value!r}")
    return float(seq[0]), float(seq[1])


def _line_length(start: tuple[float, float], end: tuple[float, float]) -> float:
    return math.hypot(end[0] - start[0], end[1] - start[1])


def _in_bounds(point: tuple[float, float], *, width: int, height: int, margin: float = 1.0) -> bool:
    return -margin <= point[0] <= width - 1 + margin and -margin <= point[1] <= height - 1 + margin


def _bbox_valid(bbox: list[float], *, width: int, height: int) -> bool:
    if not isinstance(bbox, list) or len(bbox) != 4:
        return False
    x0, y0, x1, y1 = map(float, bbox)
    return 0 <= x0 <= x1 < width and 0 <= y0 <= y1 < height


def _orientation_deg(start: tuple[float, float], end: tuple[float, float]) -> float:
    return math.degrees(math.atan2(end[1] - start[1], end[0] - start[0]))


def _angle_diff_deg(a: float, b: float) -> float:
    diff = abs((a - b) % 180.0)
    return min(diff, 180.0 - diff)


def _project_t(point: tuple[float, float], start: tuple[float, float], end: tuple[float, float]) -> tuple[float, tuple[float, float], float]:
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    denom = dx * dx + dy * dy
    if denom <= 1e-9:
        return 0.0, start, _line_length(point, start)
    raw_t = ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / denom
    t = max(0.0, min(1.0, raw_t))
    projected = (start[0] + dx * t, start[1] + dy * t)
    return t, projected, _line_length(point, projected)


def _canonical_path(case_id: str) -> Path:
    return CANONICAL_ROOT / case_id / f"{case_id}__wall_canonicalization_isolated_v1.json"


def _load_canonical_report(case_id: str) -> dict[str, Any]:
    path = _canonical_path(case_id)
    assert path.exists(), (
        f"Falta baseline {BASELINE}: {path}. Ejecuta primero "
        "test_quantia_spatial_wall_canonicalization_isolated_all_3_v1.py"
    )
    return json.loads(path.read_text(encoding="utf-8"))


def _load_case(case_id: str):
    assert case_id in CASE_LOADERS, f"CASE_LOADERS no contiene {case_id!r}."
    return CASE_LOADERS[case_id]()


def _find_loaded_level(case, *, level_name: str, level_view_id: str):
    for loaded in case.levels:
        if str(loaded.level_result.level_view.id) == str(level_view_id):
            return loaded
        if str(loaded.level_name).strip().casefold() == str(level_name).strip().casefold():
            return loaded
    raise AssertionError(f"No se encontró LevelView {level_name!r}/{level_view_id!r} en {case.case_id}.")


def _baseline_walls(level: dict[str, Any]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for item in level.get("canonical_walls", []):
        output.append(
            {
                "id": str(item["id"]),
                "role": str(item.get("role") or "UNKNOWN"),
                "start_px": [float(item["start"][0]), float(item["start"][1])],
                "end_px": [float(item["end"][0]), float(item["end"][1])],
                "length_px": float(item.get("length_px") or _line_length(_point(item["start"]), _point(item["end"]))),
                "thickness_px": float(item["thickness_px"]) if item.get("thickness_px") is not None else None,
                "confidence": float(item["confidence"]) if item.get("confidence") is not None else None,
                "source_wall_ids": [str(x) for x in item.get("source_wall_ids", [])],
                "evidence_ids": [str(x) for x in item.get("evidence_ids", [])],
                "lineage": {
                    "origin": "WALL_CANONICALIZATION_ISOLATED_V1",
                    "source_wall_ids": [str(x) for x in item.get("source_wall_ids", [])],
                    "correction_ids": [],
                },
            }
        )
    return output


def _gap_capsule(level: dict[str, Any]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for index, item in enumerate(level.get("gap_candidates", []), start=1):
        output.append(
            {
                "id": f"G{index:03d}",
                "between": [str(item["first_wall_id"]), str(item["second_wall_id"])],
                "start_px": [round(float(item["start"][0]), 2), round(float(item["start"][1]), 2)],
                "end_px": [round(float(item["end"][0]), 2), round(float(item["end"][1]), 2)],
                "gap_m": round(float(item["gap_m"]), 3) if item.get("gap_m") is not None else None,
            }
        )
    return output


def _compact_semantic_evidence(loaded_level) -> list[dict[str, Any]]:
    terms = (
        "door", "puerta", "window", "ventana", "stair", "escalera",
        "garage", "garaje", "cochera", "porton", "portón", "opening", "vano",
    )
    output: list[dict[str, Any]] = []
    evidence = getattr(loaded_level.level_result.evidence, "evidence", []) or []
    for item in evidence:
        searchable = " ".join(
            [
                str(getattr(item, "text", "") or ""),
                json.dumps(getattr(item, "metadata", {}) or {}, ensure_ascii=False),
                json.dumps(
                    [getattr(param, "model_dump", lambda: {})() for param in getattr(item, "parameters", [])],
                    ensure_ascii=False,
                ),
            ]
        ).casefold()
        if not any(term in searchable for term in terms):
            continue
        geometry = getattr(item, "geometry", None)
        geometry_dump = geometry.model_dump(mode="json") if geometry is not None else None
        output.append(
            {
                "id": str(item.id),
                "source": str(item.source),
                "kind": str(item.kind),
                "text": getattr(item, "text", None),
                "confidence": getattr(item, "confidence", None),
                "geometry": geometry_dump,
            }
        )
        if len(output) >= 80:
            break
    return output


def _state_capsule(level: dict[str, Any], loaded_level, *, width: int, height: int) -> dict[str, Any]:
    px_per_m = float(level["px_per_m"])
    walls = _baseline_walls(level)
    return {
        "contract_version": RESPONSE_CONTRACT_VERSION,
        "level_view_id": str(level["level_view_id"]),
        "level_name": str(level.get("level") or ""),
        "image_size_px": [width, height],
        "coordinate_system": "LevelView local raster; origin top-left; +x right; +y down",
        "px_per_m": round(px_per_m, 8),
        "centerline_policy": "ONE_CENTERLINE_PER_PHYSICAL_WALL",
        "baseline_wall_count": len(walls),
        "baseline_walls": [
            {
                "id": wall["id"],
                "role": wall["role"],
                "start_px": [round(x, 2) for x in wall["start_px"]],
                "end_px": [round(x, 2) for x in wall["end_px"]],
                "length_m": round(wall["length_px"] / px_per_m, 3),
                "thickness_m": round(wall["thickness_px"] / px_per_m, 3) if wall["thickness_px"] else None,
                "source_wall_ids": wall["source_wall_ids"],
                "evidence_ids": wall["evidence_ids"][:20],
            }
            for wall in walls
        ],
        "gap_candidates": _gap_capsule(level),
        "semantic_element_evidence": _compact_semantic_evidence(loaded_level),
        "rules": {
            "walls_are_centerlines_not_faces": True,
            "double_face_graphics_must_not_be_returned_as_two_walls": True,
            "opening_gap_can_preserve_logical_wall_continuity": True,
            "perimeter_baseline_is_evidence_not_immutable_final_geometry": True,
            "elements_to_find": ["DOOR", "WINDOW", "GARAGE_DOOR", "STAIR"],
        },
    }


def _draw_centerlines(image: np.ndarray, walls: list[dict[str, Any]], *, with_ids: bool) -> np.ndarray:
    canvas = image.copy()
    for wall in walls:
        start = tuple(int(round(v)) for v in wall["start_px"])
        end = tuple(int(round(v)) for v in wall["end_px"])
        role = str(wall.get("role") or "UNKNOWN")
        color = (35, 35, 220) if role == "PERIMETER" else (220, 80, 30)
        cv2.line(canvas, start, end, color, 2, cv2.LINE_AA)
        if with_ids:
            mx = int(round((start[0] + end[0]) / 2))
            my = int(round((start[1] + end[1]) / 2))
            cv2.putText(canvas, wall["id"], (mx + 2, my - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.33, color, 1, cv2.LINE_AA)
    return canvas


def _compose_input(original: np.ndarray, walls: list[dict[str, Any]], gaps: list[dict[str, Any]]) -> np.ndarray:
    overlay = _draw_centerlines(original, walls, with_ids=True)
    for gap in gaps:
        p1 = tuple(int(round(v)) for v in gap["start_px"])
        p2 = tuple(int(round(v)) for v in gap["end_px"])
        cv2.line(overlay, p1, p2, (0, 140, 255), 3, cv2.LINE_AA)
        mx = int(round((p1[0] + p2[0]) / 2))
        my = int(round((p1[1] + p2[1]) / 2))
        cv2.putText(overlay, gap["id"], (mx + 2, my - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 110, 220), 1, cv2.LINE_AA)

    clean = np.full_like(original, 255)
    clean = _draw_centerlines(clean, walls, with_ids=True)
    for gap in gaps:
        p1 = tuple(int(round(v)) for v in gap["start_px"])
        p2 = tuple(int(round(v)) for v in gap["end_px"])
        cv2.line(clean, p1, p2, (0, 140, 255), 3, cv2.LINE_AA)

    def titled(panel: np.ndarray, title: str) -> np.ndarray:
        header = np.full((42, panel.shape[1], 3), 255, dtype=np.uint8)
        cv2.putText(header, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 0, 0), 2, cv2.LINE_AA)
        return np.vstack([header, panel])

    return np.hstack(
        [
            titled(original, "A - ORIGINAL LEVELVIEW"),
            titled(overlay, "B - ORIGINAL + CURRENT CENTERLINES"),
            titled(clean, "C - CURRENT CENTERLINES ONLY"),
        ]
    )


def _build_prompt(capsule: dict[str, Any]) -> str:
    capsule_json = json.dumps(capsule, ensure_ascii=False, separators=(",", ":"))
    return f"""
Eres CALL 2 de Quantia: auditoría diferencial y consolidación arquitectónica.
Tu tarea NO es volver a dibujar el plano desde cero. Debes comparar la verdad visual del LevelView original con el WallGraph actual y devolver SOLO deltas y candidatos arquitectónicos.

IMAGEN ÚNICA
A = LevelView original limpio.
B = mismo LevelView + centerlines actuales e IDs.
C = únicamente centerlines actuales e IDs.
A, B y C usan exactamente el mismo recorte, escala, orientación y coordenadas locales. Las coordenadas que devuelvas SIEMPRE pertenecen al sistema del LevelView, nunca al ancho total de la composición A/B/C.

REGLA CENTRAL DE MUROS
- Un muro físico se representa con UNA sola CENTERLINE.
- Las dos caras paralelas dibujadas de un muro NO son dos muros.
- Detecta centerlines faltantes, centerlines falsas, centerlines desplazadas, extremos incorrectos, duplicados/paralelismos residuales y junctions que no cierran.
- Busca activamente muros que existen en A y faltan en B/C, aunque no aparezcan como candidatos del estado actual.
- El perímetro previo es evidencia histórica; si A demuestra que es incorrecto puedes proponer corregirlo. No lo sobrescribes: devuelves un delta trazable.
- No inventes muros por simetría o expectativa arquitectónica.

GAPS
Debes decidir TODOS los gap_candidates:
- WALL_CONTINUITY: el hueco debe cerrarse geométricamente como muro.
- PROBABLE_OPENING: el muro mantiene continuidad lógica/topológica, pero el vano físico debe permanecer abierto.
- UNCERTAIN: evidencia insuficiente.
- NOT_A_GAP: el candidato no representa un problema real.
Una puerta/ventana/portón NO debe cerrarse físicamente con un muro sólido.

ELEMENTOS ARQUITECTÓNICOS
Identifica en TODO el LevelView, incluso si el WallGraph actual no los detectó:
- DOOR
- WINDOW
- GARAGE_DOOR (portón/puerta de cochera; puede ser un vano grande sin arco de abatimiento)
- STAIR

Para DOOR/WINDOW/GARAGE_DOOR:
- bbox_px siempre.
- probable_host_wall_ids cuando puedas asociarlo al muro lógico.
- span_start_px y span_end_px sobre el eje longitudinal del vano cuando sean observables.
- si el host aún falta, deja probable_host_wall_ids vacío; no inventes un wall_id.

Para STAIR:
- bbox_px siempre.
- polygon_px si el contorno es razonablemente visible.
- stair_axis_start_px/stair_axis_end_px si puedes inferir eje de recorrido.
- stair_direction UP/DOWN/UNKNOWN.

IMPORTANTE
Los elementos son candidatos geométricos para revisión en 04; no están confirmados por el usuario.
Call 3 podrá refinar el grounding dimensional. Aquí necesitamos suficiente geometría para visualizar, comparar y mantener host/contexto sin contaminar el WallGraph.

OPERACIONES DE MURO
ADD_WALL: agrega una centerline faltante.
REMOVE_WALL: elimina una centerline falsa.
REPLACE_CENTERLINE: sustituye completamente la centerline de un muro.
ADJUST_ENDPOINT: mueve START o END para extender/recortar/cerrar junction.
MERGE_PARALLEL_WALLS: varias centerlines representan un solo muro físico; devuelve una única centerline resultante.
SPLIT_WALL: divide una centerline cuando una sola entidad actual representa tramos que deben mantenerse separados.

Devuelve contract_version exactamente {RESPONSE_CONTRACT_VERSION!r} y JSON conforme al schema.
No repitas como correcciones los muros que ya están bien.

STATE CAPSULE
{capsule_json}
""".strip()


def _extract_gemini_text(raw: dict[str, Any]) -> str:
    candidates = raw.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        return ""
    content = candidates[0].get("content") if isinstance(candidates[0], dict) else None
    parts = content.get("parts") if isinstance(content, dict) else None
    if not isinstance(parts, list):
        return ""
    return "".join(
        part.get("text", "")
        for part in parts
        if isinstance(part, dict) and isinstance(part.get("text"), str)
    ).strip()


def _call_gemini(*, prompt: str, image_bytes: bytes) -> tuple[dict[str, Any], str, str, list[dict[str, Any]]]:
    """
    Ejecuta Call 2 usando el proveedor Gemini canónico de Quantia.

    No replica la política HTTP dentro del probe: GeminiSpatialVisionProvider
    ya implementa reintentos para errores transitorios y fallback de modelo.
    Así evitamos que el test tenga una política de resiliencia distinta al motor.
    """
    provider = GeminiSpatialVisionProvider()
    assert provider.api_key, "GEMINI_API_KEY no está configurada."

    started = datetime.now(timezone.utc).isoformat()
    try:
        result = provider.analyze(
            prompt=prompt,
            media_bytes=image_bytes,
            media_mime_type="image/png",
            response_json_schema=RESPONSE_SCHEMA,
        )
    except Exception as exc:
        policy = {
            "provider": "gemini",
            "timestamp_utc": started,
            "ok": False,
            "models": list(provider.models),
            "max_attempts_per_model": int(provider.MAX_ATTEMPTS_PER_MODEL),
            "transient_http_status": sorted(int(x) for x in provider.TRANSIENT_HTTP_STATUS),
            "error": str(exc),
        }
        raise AssertionError(
            "Call 2 Gemini falló después de aplicar la política canónica "
            f"de reintentos/fallback. audit={policy}"
        ) from exc

    raw = result.raw if isinstance(result.raw, dict) else {"raw": result.raw}
    text = str(result.text or "").strip()
    assert text, "Gemini devolvió respuesta vacía en Call 2."

    attempts = [
        {
            "provider": "gemini",
            "timestamp_utc": started,
            "ok": True,
            "models": list(provider.models),
            "max_attempts_per_model": int(provider.MAX_ATTEMPTS_PER_MODEL),
            "selected_model": str(result.model),
            "fallback_used": bool(result.fallback_used),
            "policy": "GeminiSpatialVisionProvider.analyze",
        }
    ]
    return raw, text, str(result.model), attempts




def _resolve_hf_token() -> tuple[str, str]:
    """Prioriza HF_TOKEN real y evita reutilizar QWEN_API_KEY ambiguo."""
    process_token = str(os.getenv("HF_TOKEN") or "").strip()
    if process_token:
        return process_token, "process_env:HF_TOKEN"

    backend_dir = TESTS_DIR.parents[2]
    for env_name in (".env.local", ".env"):
        env_path = backend_dir / env_name
        if not env_path.exists():
            continue
        values = dotenv_values(env_path)
        token = str(values.get("HF_TOKEN") or "").strip()
        if token:
            return token, f"{env_name}:HF_TOKEN"
    return "", "missing"

def _extract_hf_message_text(raw: dict[str, Any]) -> str:
    choices = raw.get("choices") or []
    assert choices, f"Hugging Face devolvió respuesta sin choices: {raw}"
    message = (choices[0] or {}).get("message") or {}
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text_value = item.get("text") or item.get("content")
                if isinstance(text_value, str):
                    parts.append(text_value)
        return "\n".join(part for part in parts if part).strip()
    return ""


def _hf_retry_delay_seconds(response: requests.Response | None, attempt_index: int) -> float:
    if response is not None:
        retry_after = str(response.headers.get("Retry-After") or "").strip()
        if retry_after:
            try:
                return max(0.0, min(float(retry_after), 30.0))
            except ValueError:
                pass
    return float(min(2 ** attempt_index, 8))


def _call_qwen_hf(*, prompt: str, image_bytes: bytes) -> tuple[dict[str, Any], str, str, list[dict[str, Any]]]:
    """
    Ejecuta exactamente la misma Call 2 mediante Hugging Face Inference Providers.

    Entrada compartida con Gemini:
      - mismo prompt
      - misma composición PNG A/B/C
      - mismo RESPONSE_SCHEMA

    Diferencia de transporte:
      - imagen: data:image/png;base64,... dentro de image_url
      - salida estructurada: response_format=json_schema
      - respuesta útil: choices[0].message.content

    Esta ruta es experimental y permanece dentro del probe; no modifica el
    provider canónico de Quantia.
    """
    hf_token, credential_source = _resolve_hf_token()
    assert hf_token, (
        "HF_TOKEN no está configurado ni en el entorno ni en Backend/.env.local/.env. "
        "Para esta prueba no se reutiliza QWEN_API_KEY para evitar credenciales ambiguas."
    )

    models: list[str] = []
    for candidate in (HF_PRIMARY_MODEL, HF_FALLBACK_MODEL):
        candidate = str(candidate or "").strip()
        if candidate and candidate not in models:
            models.append(candidate)
    assert models, "No hay rutas/modelos Qwen configurados para Hugging Face."

    data_url = "data:image/png;base64," + base64.b64encode(image_bytes).decode("ascii")
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "QuantiaCall2WallAuditElementsV3",
            "schema": RESPONSE_SCHEMA,
            "strict": True,
        },
    }

    attempts: list[dict[str, Any]] = []
    last_error: str | None = None
    for model in models:
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
            "response_format": response_format,
        }

        for attempt_index in range(HF_MAX_ATTEMPTS_PER_ROUTE):
            started = datetime.now(timezone.utc).isoformat()
            response: requests.Response | None = None
            try:
                response = requests.post(
                    HF_ROUTER_CHAT_COMPLETIONS_URL,
                    headers={
                        "Authorization": f"Bearer {hf_token}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=HF_REQUEST_TIMEOUT_SECONDS,
                )
            except requests.RequestException as exc:
                last_error = str(exc)
                attempts.append(
                    {
                        "provider": "huggingface",
                        "credential_source": credential_source,
                        "provider_route": model,
                        "attempt": attempt_index + 1,
                        "timestamp_utc": started,
                        "ok": False,
                        "transient": True,
                        "error": last_error,
                    }
                )
                if attempt_index + 1 < HF_MAX_ATTEMPTS_PER_ROUTE:
                    time.sleep(_hf_retry_delay_seconds(None, attempt_index))
                    continue
                break

            status = int(response.status_code)
            if status == 200:
                raw = response.json()
                text = _extract_hf_message_text(raw)
                assert text, f"Hugging Face/Qwen devolvió contenido vacío. RAW={raw}"
                usage = raw.get("usage") if isinstance(raw, dict) else None
                attempts.append(
                    {
                        "provider": "huggingface",
                        "credential_source": credential_source,
                        "provider_route": model,
                        "attempt": attempt_index + 1,
                        "timestamp_utc": started,
                        "ok": True,
                        "http_status": status,
                        "response_model": str(raw.get("model") or model),
                        "usage": usage,
                        "response_format": "json_schema",
                    }
                )
                return raw, text, str(raw.get("model") or model), attempts

            transient = status in HF_TRANSIENT_HTTP_STATUS
            try:
                error_payload: Any = response.json()
            except ValueError:
                error_payload = response.text[:2000]
            last_error = json.dumps(error_payload, ensure_ascii=False) if isinstance(error_payload, (dict, list)) else str(error_payload)
            attempts.append(
                {
                    "provider": "huggingface",
                        "credential_source": credential_source,
                    "provider_route": model,
                    "attempt": attempt_index + 1,
                    "timestamp_utc": started,
                    "ok": False,
                    "http_status": status,
                    "transient": transient,
                    "error": last_error[:2000],
                }
            )
            if transient and attempt_index + 1 < HF_MAX_ATTEMPTS_PER_ROUTE:
                time.sleep(_hf_retry_delay_seconds(response, attempt_index))
                continue
            break

    raise AssertionError(
        "Call 2 Qwen/Hugging Face falló después de agotar rutas/reintentos. "
        f"models={models} attempts={attempts} error={last_error}"
    )

def _call_groq(*, prompt: str, image_bytes: bytes) -> tuple[dict[str, Any], str, str, list[dict[str, Any]]]:
    api_key = str(os.getenv("GROQ_API_KEY") or "").strip()
    assert api_key, "GROQ_API_KEY no está configurada."
    model = str(os.getenv("QUANTIA_CALL2_GROQ_MODEL") or "qwen/qwen3.8-27b").strip()
    schema_text = json.dumps(RESPONSE_SCHEMA, ensure_ascii=False, separators=(",", ":"))
    data_url = "data:image/png;base64," + base64.b64encode(image_bytes).decode("ascii")
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt + "\n\nJSON_SCHEMA\n" + schema_text},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ],
        "response_format": {"type": "json_object"},
    }
    attempts: list[dict[str, Any]] = []
    url = "https://api.groq.com/openai/v1/chat/completions"
    last_error: str | None = None
    for attempt in range(2):
        started = datetime.now(timezone.utc).isoformat()
        try:
            response = requests.post(
                url,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json=payload,
                timeout=180,
            )
        except requests.RequestException as exc:
            attempts.append({"model": model, "timestamp_utc": started, "ok": False, "error": str(exc), "transient": True})
            last_error = str(exc)
            if attempt == 0:
                time.sleep(2.0)
                continue
            break
        ok = response.status_code == 200
        attempts.append({
            "model": model,
            "timestamp_utc": started,
            "ok": ok,
            "http_status": response.status_code,
            "transient": response.status_code in {408, 429, 500, 502, 503, 504},
        })
        if ok:
            raw = response.json()
            choices = raw.get("choices") or []
            text = ""
            if choices and isinstance(choices[0], dict):
                text = str(((choices[0].get("message") or {}).get("content")) or "").strip()
            return raw, text, model, attempts
        last_error = response.text[:1500]
        if response.status_code not in {408, 429, 500, 502, 503, 504} or attempt == 1:
            break
        retry_after = response.headers.get("Retry-After")
        delay = float(retry_after) if retry_after and retry_after.replace(".", "", 1).isdigit() else 2.0
        time.sleep(min(delay, 10.0))
    raise AssertionError(f"Call 2 Groq falló. attempts={attempts} error={last_error}")


def _call_model(*, prompt: str, image_bytes: bytes) -> tuple[dict[str, Any], str, str, str, list[dict[str, Any]]]:
    provider = str(os.getenv("QUANTIA_CALL2_PROVIDER") or "gemini").strip().casefold()
    if provider in {"qwen_hf", "huggingface", "hf"}:
        raw, text, model, attempts = _call_qwen_hf(prompt=prompt, image_bytes=image_bytes)
        return raw, text, "qwen_hf", model, attempts
    if provider == "groq":
        raw, text, model, attempts = _call_groq(prompt=prompt, image_bytes=image_bytes)
        return raw, text, "groq", model, attempts
    if provider == "gemini":
        raw, text, model, attempts = _call_gemini(prompt=prompt, image_bytes=image_bytes)
        return raw, text, "gemini", model, attempts
    raise AssertionError("QUANTIA_CALL2_PROVIDER debe ser 'gemini', 'qwen_hf' o 'groq'.")


def _validate_result(*, capsule: dict[str, Any], result: dict[str, Any], width: int, height: int) -> list[str]:
    failures: list[str] = []
    wall_ids = {wall["id"] for wall in capsule["baseline_walls"]}
    gap_ids = [gap["id"] for gap in capsule["gap_candidates"]]

    if result.get("contract_version") != RESPONSE_CONTRACT_VERSION:
        failures.append(f"CONTRACT_VERSION:{result.get('contract_version')}")
    if result.get("level_view_id") != capsule["level_view_id"]:
        failures.append("LEVEL_VIEW_ID_MISMATCH")

    returned_gap_ids = [str(item.get("gap_id") or "") for item in result.get("gap_decisions", [])]
    if sorted(returned_gap_ids) != sorted(gap_ids):
        failures.append(f"GAP_COVERAGE expected={gap_ids} returned={returned_gap_ids}")
    if len(returned_gap_ids) != len(set(returned_gap_ids)):
        failures.append("DUPLICATE_GAP_DECISION")

    correction_ids: set[str] = set()
    for correction in result.get("wall_corrections", []):
        cid = str(correction.get("correction_id") or "")
        if not cid or cid in correction_ids:
            failures.append(f"INVALID_OR_DUPLICATE_CORRECTION_ID:{cid}")
        correction_ids.add(cid)
        action = str(correction.get("action") or "")
        targets = [str(x) for x in correction.get("target_wall_ids", [])]
        if action != "ADD_WALL" and any(target not in wall_ids for target in targets):
            failures.append(f"UNKNOWN_CORRECTION_TARGET:{cid}:{targets}")
        for key in ("start_px", "end_px", "new_point_px"):
            if key in correction:
                try:
                    point = _point(correction[key])
                except Exception:
                    failures.append(f"INVALID_POINT:{cid}:{key}")
                    continue
                if not _in_bounds(point, width=width, height=height):
                    failures.append(f"OUT_OF_BOUNDS:{cid}:{key}:{correction[key]}")
        for segment in correction.get("segments_px", []) or []:
            for key in ("start_px", "end_px"):
                try:
                    point = _point(segment[key])
                except Exception:
                    failures.append(f"INVALID_SPLIT_POINT:{cid}:{key}")
                    continue
                if not _in_bounds(point, width=width, height=height):
                    failures.append(f"OUT_OF_BOUNDS_SPLIT:{cid}:{key}:{segment[key]}")

    element_ids: set[str] = set()
    for element in result.get("architectural_elements", []):
        eid = str(element.get("element_id") or "")
        if not eid or eid in element_ids:
            failures.append(f"INVALID_OR_DUPLICATE_ELEMENT_ID:{eid}")
        element_ids.add(eid)
        if not _bbox_valid(element.get("bbox_px", []), width=width, height=height):
            failures.append(f"INVALID_ELEMENT_BBOX:{eid}:{element.get('bbox_px')}")
        for host_id in element.get("probable_host_wall_ids", []) or []:
            if str(host_id) not in wall_ids:
                # Puede apuntar a un muro que la misma respuesta propone agregar/mergear.
                correction_result_ids = {
                    str(c.get("correction_id") or "") for c in result.get("wall_corrections", [])
                }
                if str(host_id) not in correction_result_ids:
                    failures.append(f"UNKNOWN_ELEMENT_HOST:{eid}:{host_id}")
        for key in ("span_start_px", "span_end_px", "stair_axis_start_px", "stair_axis_end_px"):
            if key in element:
                try:
                    point = _point(element[key])
                except Exception:
                    failures.append(f"INVALID_ELEMENT_POINT:{eid}:{key}")
                    continue
                if not _in_bounds(point, width=width, height=height):
                    failures.append(f"OUT_OF_BOUNDS_ELEMENT_POINT:{eid}:{key}")
        for vertex in element.get("polygon_px", []) or []:
            try:
                point = _point(vertex)
            except Exception:
                failures.append(f"INVALID_ELEMENT_POLYGON:{eid}")
                continue
            if not _in_bounds(point, width=width, height=height):
                failures.append(f"OUT_OF_BOUNDS_ELEMENT_POLYGON:{eid}:{vertex}")

    for item in result.get("unresolved_regions", []):
        if not _bbox_valid(item.get("bbox_px", []), width=width, height=height):
            failures.append(f"INVALID_UNRESOLVED_BBOX:{item.get('bbox_px')}")
    return failures


def _wall_threshold(wall_role: str) -> float:
    return MIN_PERIMETER_CORRECTION_CONFIDENCE if wall_role == "PERIMETER" else MIN_WALL_CORRECTION_CONFIDENCE


def _parallel_geometry(first: dict[str, Any], second: dict[str, Any], *, px_per_m: float) -> dict[str, float]:
    a0, a1 = _point(first["start_px"]), _point(first["end_px"])
    b0, b1 = _point(second["start_px"]), _point(second["end_px"])
    a_angle = _orientation_deg(a0, a1)
    b_angle = _orientation_deg(b0, b1)
    angle = _angle_diff_deg(a_angle, b_angle)
    length_a = _line_length(a0, a1)
    length_b = _line_length(b0, b1)
    if min(length_a, length_b) <= 1e-9:
        return {"angle_deg": 180.0, "lateral_m": float("inf"), "overlap_ratio": 0.0}
    ux = (a1[0] - a0[0]) / length_a
    uy = (a1[1] - a0[1]) / length_a
    nx, ny = -uy, ux
    offsets_a = [a0[0] * nx + a0[1] * ny, a1[0] * nx + a1[1] * ny]
    offsets_b = [b0[0] * nx + b0[1] * ny, b1[0] * nx + b1[1] * ny]
    lateral_px = abs(sum(offsets_a) / 2 - sum(offsets_b) / 2)
    proj_a = sorted([a0[0] * ux + a0[1] * uy, a1[0] * ux + a1[1] * uy])
    proj_b = sorted([b0[0] * ux + b0[1] * uy, b1[0] * ux + b1[1] * uy])
    overlap = max(0.0, min(proj_a[1], proj_b[1]) - max(proj_a[0], proj_b[0]))
    return {
        "angle_deg": angle,
        "lateral_m": lateral_px / px_per_m,
        "overlap_ratio": overlap / min(length_a, length_b),
    }


def _can_merge_parallel(targets: list[dict[str, Any]], *, px_per_m: float) -> bool:
    if len(targets) < 2:
        return False
    roles = {wall["role"] for wall in targets}
    if len(roles) != 1:
        return False
    reference = targets[0]
    for other in targets[1:]:
        geometry = _parallel_geometry(reference, other, px_per_m=px_per_m)
        if geometry["angle_deg"] > PARALLEL_ANGLE_MAX_DEG:
            return False
        if geometry["lateral_m"] > PARALLEL_LATERAL_MAX_M:
            return False
        if geometry["overlap_ratio"] < PARALLEL_MIN_OVERLAP_RATIO:
            return False
    return True


def _make_wall(
    *,
    wall_id: str,
    role: str,
    start: tuple[float, float],
    end: tuple[float, float],
    thickness_px: float | None,
    confidence: float | None,
    lineage: dict[str, Any],
) -> dict[str, Any]:
    return {
        "id": wall_id,
        "role": role,
        "start_px": [float(start[0]), float(start[1])],
        "end_px": [float(end[0]), float(end[1])],
        "length_px": _line_length(start, end),
        "thickness_px": thickness_px,
        "confidence": confidence,
        "source_wall_ids": list(lineage.get("source_wall_ids", [])),
        "evidence_ids": list(lineage.get("evidence_ids", [])),
        "lineage": lineage,
    }


def _apply_corrections(
    *,
    baseline: list[dict[str, Any]],
    corrections: list[dict[str, Any]],
    px_per_m: float,
    width: int,
    height: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    walls = {wall["id"]: json.loads(json.dumps(wall)) for wall in baseline}
    applied: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    def reject(item: dict[str, Any], reason: str) -> None:
        rejected.append({"correction_id": item.get("correction_id"), "action": item.get("action"), "reason": reason})

    for item in corrections:
        cid = str(item["correction_id"])
        action = str(item["action"])
        targets = [str(x) for x in item.get("target_wall_ids", [])]
        confidence = float(item.get("confidence") or 0.0)
        role_hint = str(item.get("wall_role") or "UNKNOWN")
        target_walls = [walls[x] for x in targets if x in walls]
        # Para correcciones sobre muros existentes prevalece el rol REAL del baseline/final graph.
        # Evita que una respuesta etiquete un PERIMETER como DIVIDER para eludir el gate estricto.
        effective_role = (
            target_walls[0]["role"]
            if target_walls
            else (role_hint if role_hint in {"PERIMETER", "DIVIDER"} else "DIVIDER")
        )
        threshold = _wall_threshold(effective_role)
        if confidence < threshold:
            reject(item, f"LOW_CONFIDENCE {confidence:.3f} < {threshold:.3f}")
            continue

        if action == "ADD_WALL":
            if "start_px" not in item or "end_px" not in item:
                reject(item, "ADD_WALL_REQUIRES_START_END")
                continue
            start, end = _point(item["start_px"]), _point(item["end_px"])
            if not (_in_bounds(start, width=width, height=height) and _in_bounds(end, width=width, height=height)):
                reject(item, "OUT_OF_BOUNDS")
                continue
            if _line_length(start, end) < 0.10 * px_per_m:
                reject(item, "WALL_TOO_SHORT")
                continue
            new_id = f"C2_ADD_{_slug(cid).upper()}"
            walls[new_id] = _make_wall(
                wall_id=new_id,
                role=effective_role,
                start=start,
                end=end,
                thickness_px=float(item["thickness_px"]) if item.get("thickness_px") else None,
                confidence=confidence,
                lineage={"origin": "CALL2_ADD", "source_wall_ids": [], "correction_ids": [cid], "evidence_ids": []},
            )
            applied.append({"correction_id": cid, "action": action, "result_wall_ids": [new_id]})
            continue

        if not targets or any(target not in walls for target in targets):
            reject(item, "UNKNOWN_OR_EMPTY_TARGETS")
            continue

        if action == "REMOVE_WALL":
            removed = []
            for target in targets:
                removed.append(target)
                walls.pop(target, None)
            applied.append({"correction_id": cid, "action": action, "removed_wall_ids": removed})
            continue

        if action == "REPLACE_CENTERLINE":
            if len(targets) != 1 or "start_px" not in item or "end_px" not in item:
                reject(item, "REPLACE_REQUIRES_ONE_TARGET_START_END")
                continue
            start, end = _point(item["start_px"]), _point(item["end_px"])
            if _line_length(start, end) < 0.10 * px_per_m:
                reject(item, "REPLACEMENT_TOO_SHORT")
                continue
            wall = walls[targets[0]]
            wall["start_px"] = list(start)
            wall["end_px"] = list(end)
            wall["length_px"] = _line_length(start, end)
            if item.get("thickness_px"):
                wall["thickness_px"] = float(item["thickness_px"])
            wall["confidence"] = confidence
            wall["lineage"].setdefault("correction_ids", []).append(cid)
            applied.append({"correction_id": cid, "action": action, "result_wall_ids": [wall["id"]]})
            continue

        if action == "ADJUST_ENDPOINT":
            if len(targets) != 1 or "endpoint" not in item or "new_point_px" not in item:
                reject(item, "ADJUST_ENDPOINT_REQUIRES_ONE_TARGET_ENDPOINT_POINT")
                continue
            wall = walls[targets[0]]
            new_point = _point(item["new_point_px"])
            if not _in_bounds(new_point, width=width, height=height):
                reject(item, "OUT_OF_BOUNDS")
                continue
            next_start = new_point if item["endpoint"] == "START" else _point(wall["start_px"])
            next_end = new_point if item["endpoint"] == "END" else _point(wall["end_px"])
            next_length = _line_length(next_start, next_end)
            if next_length < 0.10 * px_per_m:
                reject(item, "ADJUSTMENT_COLLAPSES_WALL")
                continue
            wall["start_px"] = list(next_start)
            wall["end_px"] = list(next_end)
            wall["length_px"] = next_length
            wall["confidence"] = confidence
            wall["lineage"].setdefault("correction_ids", []).append(cid)
            applied.append({"correction_id": cid, "action": action, "result_wall_ids": [wall["id"]]})
            continue

        if action == "MERGE_PARALLEL_WALLS":
            if len(targets) < 2 or "start_px" not in item or "end_px" not in item:
                reject(item, "MERGE_REQUIRES_2_TARGETS_AND_RESULT_CENTERLINE")
                continue
            target_walls = [walls[x] for x in targets]
            if not _can_merge_parallel(target_walls, px_per_m=px_per_m):
                reject(item, "TARGETS_NOT_PARALLEL_COMPATIBLE")
                continue
            start, end = _point(item["start_px"]), _point(item["end_px"])
            if _line_length(start, end) < 0.10 * px_per_m:
                reject(item, "MERGE_RESULT_TOO_SHORT")
                continue
            source_ids = sorted({sid for wall in target_walls for sid in ([wall["id"]] + wall.get("source_wall_ids", []))})
            thickness_values = [wall["thickness_px"] for wall in target_walls if wall.get("thickness_px")]
            thickness = float(np.median(thickness_values)) if thickness_values else None
            for target in targets:
                walls.pop(target, None)
            new_id = f"C2_MERGE_{_slug(cid).upper()}"
            walls[new_id] = _make_wall(
                wall_id=new_id,
                role=effective_role,
                start=start,
                end=end,
                thickness_px=float(item.get("thickness_px") or thickness) if (item.get("thickness_px") or thickness) else None,
                confidence=confidence,
                lineage={"origin": "CALL2_MERGE", "source_wall_ids": source_ids, "correction_ids": [cid], "evidence_ids": []},
            )
            applied.append({"correction_id": cid, "action": action, "removed_wall_ids": targets, "result_wall_ids": [new_id]})
            continue

        if action == "SPLIT_WALL":
            segments = item.get("segments_px") or []
            if len(targets) != 1 or len(segments) < 2:
                reject(item, "SPLIT_REQUIRES_ONE_TARGET_AND_2_SEGMENTS")
                continue
            source = walls[targets[0]]
            pieces: list[dict[str, Any]] = []
            valid = True
            for idx, segment in enumerate(segments, start=1):
                start, end = _point(segment["start_px"]), _point(segment["end_px"])
                if _line_length(start, end) < 0.08 * px_per_m:
                    valid = False
                    break
                piece_id = f"{source['id']}__C2S{idx:02d}"
                pieces.append(
                    _make_wall(
                        wall_id=piece_id,
                        role=source["role"],
                        start=start,
                        end=end,
                        thickness_px=source.get("thickness_px"),
                        confidence=confidence,
                        lineage={
                            "origin": "CALL2_SPLIT",
                            "source_wall_ids": [source["id"], *source.get("source_wall_ids", [])],
                            "correction_ids": [cid],
                            "evidence_ids": source.get("evidence_ids", []),
                        },
                    )
                )
            if not valid:
                reject(item, "INVALID_SPLIT_SEGMENT")
                continue
            walls.pop(source["id"], None)
            for piece in pieces:
                walls[piece["id"]] = piece
            applied.append({"correction_id": cid, "action": action, "removed_wall_ids": [source["id"]], "result_wall_ids": [p["id"] for p in pieces]})
            continue

        reject(item, f"UNSUPPORTED_ACTION:{action}")

    return list(walls.values()), applied, rejected


def _dedupe_exact_walls(walls: list[dict[str, Any]], *, precision: int = 3) -> tuple[list[dict[str, Any]], list[tuple[str, str]]]:
    seen: dict[tuple[Any, ...], str] = {}
    output: list[dict[str, Any]] = []
    duplicates: list[tuple[str, str]] = []
    for wall in walls:
        a = tuple(round(float(v), precision) for v in wall["start_px"])
        b = tuple(round(float(v), precision) for v in wall["end_px"])
        key = tuple(sorted((a, b))) + (wall["role"],)
        if key in seen:
            duplicates.append((seen[key], wall["id"]))
            continue
        seen[key] = wall["id"]
        output.append(wall)
    return output, duplicates


def _parallel_anomalies(walls: list[dict[str, Any]], *, px_per_m: float) -> list[dict[str, Any]]:
    anomalies: list[dict[str, Any]] = []
    for i in range(len(walls)):
        for j in range(i + 1, len(walls)):
            a, b = walls[i], walls[j]
            if a["role"] != b["role"]:
                continue
            geometry = _parallel_geometry(a, b, px_per_m=px_per_m)
            if (
                geometry["angle_deg"] <= PARALLEL_ANGLE_MAX_DEG
                and geometry["lateral_m"] <= PARALLEL_LATERAL_MAX_M
                and geometry["overlap_ratio"] >= PARALLEL_MIN_OVERLAP_RATIO
            ):
                anomalies.append({"wall_ids": [a["id"], b["id"]], **geometry})
    return anomalies


def _split_at_junctions(walls: list[dict[str, Any]], *, px_per_m: float) -> list[dict[str, Any]]:
    tolerance = max(1.5, 0.025 * px_per_m)
    lines = {wall["id"]: LineString([_point(wall["start_px"]), _point(wall["end_px"])]) for wall in walls}
    output: list[dict[str, Any]] = []
    for wall in walls:
        line = lines[wall["id"]]
        points: list[tuple[float, float, float]] = [(0.0, *line.coords[0]), (1.0, *line.coords[-1])]
        for other_id, other in lines.items():
            if other_id == wall["id"]:
                continue
            intersection = line.intersection(other)
            candidates: list[Point] = []
            if intersection.geom_type == "Point":
                candidates = [intersection]
            elif intersection.geom_type == "MultiPoint":
                candidates = list(intersection.geoms)
            for point in candidates:
                t = line.project(point) / max(line.length, 1e-9)
                if tolerance / max(line.length, 1e-9) < t < 1.0 - tolerance / max(line.length, 1e-9):
                    points.append((t, float(point.x), float(point.y)))
        points.sort(key=lambda item: item[0])
        unique: list[tuple[float, float, float]] = []
        for item in points:
            if not unique or abs(item[0] - unique[-1][0]) > 1e-5:
                unique.append(item)
        if len(unique) <= 2:
            copy = json.loads(json.dumps(wall))
            copy["parent_wall_id"] = wall["id"]
            output.append(copy)
            continue
        for idx in range(len(unique) - 1):
            start = (unique[idx][1], unique[idx][2])
            end = (unique[idx + 1][1], unique[idx + 1][2])
            if _line_length(start, end) < tolerance:
                continue
            piece = json.loads(json.dumps(wall))
            piece["id"] = f"{wall['id']}__J{idx + 1:02d}"
            piece["parent_wall_id"] = wall["id"]
            piece["start_px"] = list(start)
            piece["end_px"] = list(end)
            piece["length_px"] = _line_length(start, end)
            piece["lineage"].setdefault("junction_split_parent", wall["id"])
            output.append(piece)
    return output


def _gap_index(capsule: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {item["id"]: item for item in capsule["gap_candidates"]}


def _derive_spaces(
    *,
    walls: list[dict[str, Any]],
    gap_decisions: list[dict[str, Any]],
    gaps: dict[str, dict[str, Any]],
    px_per_m: float,
    level_id: str,
) -> tuple[list[dict[str, Any]], dict[str, list[str]], dict[str, Any]]:
    lines: list[LineString] = [LineString([_point(w["start_px"]), _point(w["end_px"])]) for w in walls]
    logical_gap_ids: list[str] = []
    for decision in gap_decisions:
        if not decision.get("logical_wall_continuity"):
            continue
        if decision.get("classification") not in {"WALL_CONTINUITY", "PROBABLE_OPENING"}:
            continue
        gap = gaps.get(str(decision.get("gap_id")))
        if not gap:
            continue
        lines.append(LineString([_point(gap["start_px"]), _point(gap["end_px"])]))
        logical_gap_ids.append(str(decision["gap_id"]))

    if not lines:
        return [], {}, {"logical_gap_ids": logical_gap_ids, "dangle_length_px": 0.0, "cut_length_px": 0.0}

    network = unary_union(lines)
    snapped = snap(network, network, max(1.5, 0.03 * px_per_m))
    polygons, cuts, dangles, invalid = polygonize_full(snapped)
    min_area_px2 = MIN_SPACE_AREA_M2 * px_per_m * px_per_m
    room_polygons = [poly for poly in polygons.geoms if isinstance(poly, Polygon) and poly.area >= min_area_px2]
    room_polygons.sort(key=lambda poly: (round(poly.centroid.y, 3), round(poly.centroid.x, 3)))

    spaces: list[dict[str, Any]] = []
    space_polygons: dict[str, Polygon] = {}
    for idx, poly in enumerate(room_polygons, start=1):
        sid = f"{level_id}__SPACE_{idx:03d}"
        coords_px = list(poly.exterior.coords)[:-1]
        vertices_m = [{"x": float(x) / px_per_m, "y": float(y) / px_per_m} for x, y in coords_px]
        minx, miny, maxx, maxy = poly.bounds
        spaces.append(
            {
                "id": sid,
                "name": "",
                "usageCode": "",
                "usageLabel": "",
                "category": "",
                "levelId": level_id,
                "geometry": {
                    "type": "polygon",
                    "vertices": vertices_m,
                    "areaM2": float(poly.area) / (px_per_m * px_per_m),
                    "perimeterM": float(poly.length) / px_per_m,
                    "bbox": {
                        "minX": minx / px_per_m,
                        "minY": miny / px_per_m,
                        "maxX": maxx / px_per_m,
                        "maxY": maxy / px_per_m,
                        "width": (maxx - minx) / px_per_m,
                        "height": (maxy - miny) / px_per_m,
                    },
                },
                "areaM2": float(poly.area) / (px_per_m * px_per_m),
                "perimeterM": float(poly.length) / px_per_m,
                "doubleHeight": False,
                "heightM": None,
                "confirmed": False,
                "source": {
                    "type": "ai",
                    "documentId": None,
                    "documentName": None,
                    "page": None,
                    "confidence": None,
                    "state": "INFERIDO",
                    "note": "Derivado topológicamente de centerlines Call 2 + gaps con continuidad lógica.",
                },
                "wallIds": [],
                "doorIds": [],
                "windowIds": [],
                "validation": {"isValid": True, "conflicts": [], "warnings": []},
            }
        )
        space_polygons[sid] = poly

    wall_owners: dict[str, list[str]] = {}
    boundary_tolerance = max(1.5, 0.04 * px_per_m)
    for wall in walls:
        line = LineString([_point(wall["start_px"]), _point(wall["end_px"])])
        hits: list[tuple[float, str]] = []
        for sid, poly in space_polygons.items():
            overlap = poly.boundary.buffer(boundary_tolerance, cap_style=2, join_style=2).intersection(line).length
            if overlap >= max(0.15 * line.length, 0.08 * px_per_m):
                hits.append((overlap, sid))
        hits.sort(reverse=True)
        wall_owners[wall["id"]] = [sid for _, sid in hits[:2]]

    spaces_by_id = {space["id"]: space for space in spaces}
    for wall_id, owners in wall_owners.items():
        for sid in owners:
            spaces_by_id[sid]["wallIds"].append(wall_id)

    topology = {
        "logical_gap_ids": logical_gap_ids,
        "polygon_count": len(list(polygons.geoms)),
        "meaningful_space_count": len(spaces),
        "dangle_length_px": float(sum(geom.length for geom in dangles.geoms)),
        "cut_length_px": float(sum(geom.length for geom in cuts.geoms)),
        "invalid_ring_length_px": float(sum(geom.length for geom in invalid.geoms)),
    }
    return spaces, wall_owners, topology


def _level_key(name: str) -> str:
    normalized = str(name).strip().casefold()
    if "baja" in normalized or "ground" in normalized:
        return "planta_baja"
    if "alta" in normalized or "segunda" in normalized or "upper" in normalized:
        return "segunda_planta"
    if "terc" in normalized:
        return "tercera_planta"
    if "azotea" in normalized:
        return "planta_azotea"
    return "planta_baja"


def _wall_to_editor(wall: dict[str, Any], *, level_id: str, px_per_m: float, owners: list[str]) -> dict[str, Any]:
    role = wall["role"]
    wall_type = "interior" if len(owners) >= 2 else ("exterior" if role == "PERIMETER" else "uncertain")
    start = _point(wall["start_px"])
    end = _point(wall["end_px"])
    return {
        "id": wall["id"],
        "levelId": level_id,
        "start": {"x": start[0] / px_per_m, "y": start[1] / px_per_m},
        "end": {"x": end[0] / px_per_m, "y": end[1] / px_per_m},
        "lengthM": _line_length(start, end) / px_per_m,
        "thicknessM": (float(wall["thickness_px"]) / px_per_m) if wall.get("thickness_px") else None,
        "heightM": None,
        "type": wall_type,
        "spaceAId": owners[0] if owners else None,
        "spaceBId": owners[1] if len(owners) > 1 else None,
        "doorIds": [],
        "windowIds": [],
        "confirmed": False,
        "source": {
            "type": "ai",
            "documentId": None,
            "documentName": None,
            "page": None,
            "confidence": wall.get("confidence"),
            "state": "INFERIDO",
            "note": f"Call 2 centerline; parent={wall.get('parent_wall_id') or wall['id']}; lineage={wall.get('lineage', {})}",
        },
        "validation": {"isValid": True, "conflicts": [], "warnings": []},
        "_parentWallId": wall.get("parent_wall_id") or wall["id"],
        "_role": wall["role"],
        "_lineage": wall.get("lineage", {}),
    }


def _host_candidates_for_element(element: dict[str, Any], segmented_walls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    requested = {str(x) for x in element.get("probable_host_wall_ids", [])}
    if not requested:
        return []
    output: list[dict[str, Any]] = []
    for wall in segmented_walls:
        identifiers = {
            wall["id"],
            str(wall.get("parent_wall_id") or ""),
            *[str(x) for x in wall.get("source_wall_ids", [])],
            *[str(x) for x in (wall.get("lineage") or {}).get("source_wall_ids", [])],
            *[str(x) for x in (wall.get("lineage") or {}).get("correction_ids", [])],
        }
        if identifiers & requested:
            output.append(wall)
    return output


def _ground_opening_for_editor(
    *,
    element: dict[str, Any],
    segmented_walls: list[dict[str, Any]],
    level_id: str,
    px_per_m: float,
) -> tuple[dict[str, Any] | None, str | None]:
    if "span_start_px" not in element or "span_end_px" not in element:
        return None, "MISSING_SPAN"
    span_start = _point(element["span_start_px"])
    span_end = _point(element["span_end_px"])
    center = ((span_start[0] + span_end[0]) / 2, (span_start[1] + span_end[1]) / 2)
    candidates = _host_candidates_for_element(element, segmented_walls)
    if not candidates:
        return None, "NO_HOST_CANDIDATE"

    ranked: list[tuple[float, dict[str, Any], float, float, tuple[float, float], tuple[float, float]]] = []
    for wall in candidates:
        start, end = _point(wall["start_px"]), _point(wall["end_px"])
        tc, _, center_dist = _project_t(center, start, end)
        t0, p0, d0 = _project_t(span_start, start, end)
        t1, p1, d1 = _project_t(span_end, start, end)
        distance = max(center_dist, d0, d1) / px_per_m
        ranked.append((distance, wall, tc, abs(t1 - t0), p0, p1))
    ranked.sort(key=lambda item: item[0])
    distance_m, wall, position, span_t, p0, p1 = ranked[0]
    if distance_m > OPENING_HOST_MAX_DISTANCE_M:
        return None, f"HOST_TOO_FAR:{distance_m:.3f}m"
    width_m = _line_length(p0, p1) / px_per_m
    if width_m <= 0.10:
        return None, f"OPENING_WIDTH_TOO_SMALL:{width_m:.3f}m"

    common = {
        "id": str(element["element_id"]),
        "levelId": level_id,
        "wallId": wall["id"],
        "position": float(position),
        "widthM": float(width_m),
        "heightM": None,
        "confirmed": False,
        "source": {
            "type": "ai",
            "documentId": None,
            "documentName": None,
            "page": None,
            "confidence": float(element.get("confidence") or 0.0),
            "state": "INFERIDO",
            "note": f"Call 2 candidate {element['element_type']}; host_distance_m={distance_m:.3f}; Call 3 debe confirmar/refinar.",
        },
        "validation": {
            "isValid": True,
            "conflicts": [],
            "warnings": ["Elemento inferido por Call 2; pendiente de confirmación/grounding final."],
        },
    }
    if element["element_type"] == "WINDOW":
        return {**common, "sillHeightM": None}, None
    return {
        **common,
        "swingDirection": "garage" if element["element_type"] == "GARAGE_DOOR" else "unknown",
        "flipSide": False,
    }, None


def _stair_to_editor(
    *,
    element: dict[str, Any],
    level_id: str,
    level_order: list[str],
    px_per_m: float,
) -> dict[str, Any]:
    polygon_px = element.get("polygon_px") or []
    if len(polygon_px) < 3:
        x0, y0, x1, y1 = map(float, element["bbox_px"])
        polygon_px = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
    vertices_m = [{"x": float(x) / px_per_m, "y": float(y) / px_per_m} for x, y in polygon_px]
    xs = [p["x"] for p in vertices_m]
    ys = [p["y"] for p in vertices_m]
    position = {"x": sum(xs) / len(xs), "y": sum(ys) / len(ys)}
    rotation = 0.0
    if "stair_axis_start_px" in element and "stair_axis_end_px" in element:
        rotation = _orientation_deg(_point(element["stair_axis_start_px"]), _point(element["stair_axis_end_px"]))
    target = ""
    if level_id in level_order:
        idx = level_order.index(level_id)
        direction = str(element.get("stair_direction") or "UNKNOWN")
        if direction == "UP" and idx + 1 < len(level_order):
            target = level_order[idx + 1]
        elif direction == "DOWN" and idx - 1 >= 0:
            target = level_order[idx - 1]
    width_m = min(max(xs) - min(xs), max(ys) - min(ys)) if xs and ys else None
    return {
        "id": str(element["element_id"]),
        "levelFromId": level_id,
        "levelToId": target,
        "position": position,
        "rotationDeg": rotation,
        "widthM": width_m,
        "geometry": {"type": "polygon", "vertices": vertices_m},
        "type": "unknown",
        "confirmed": False,
        "source": {
            "type": "ai",
            "documentId": None,
            "documentName": None,
            "page": None,
            "confidence": float(element.get("confidence") or 0.0),
            "state": "INFERIDO",
            "note": f"Call 2 STAIR candidate direction={element.get('stair_direction', 'UNKNOWN')}; pendiente de confirmación.",
        },
    }


def _build_interface04(
    *,
    case_id: str,
    case_name: str,
    canonical_report: dict[str, Any],
    level: dict[str, Any],
    loaded_level,
    result: dict[str, Any],
    final_walls: list[dict[str, Any]],
    applied: list[dict[str, Any]],
    rejected: list[dict[str, Any]],
    exact_duplicates: list[tuple[str, str]],
    parallel_anomalies: list[dict[str, Any]],
    comparison_image: Path,
    original_image: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    px_per_m = float(level["px_per_m"])
    level_id = str(level["level_view_id"])
    level_names = [str(item.get("level") or "") for item in canonical_report.get("levels", [])]
    level_ids = [str(item.get("level_view_id") or "") for item in canonical_report.get("levels", [])]
    order_map = {level_ids[i]: i for i in range(len(level_ids))}

    segmented = _split_at_junctions(final_walls, px_per_m=px_per_m)
    spaces, wall_owners, topology = _derive_spaces(
        walls=segmented,
        gap_decisions=result.get("gap_decisions", []),
        gaps=_gap_index({"gap_candidates": _gap_capsule(level)}),
        px_per_m=px_per_m,
        level_id=level_id,
    )

    editor_walls = [_wall_to_editor(wall, level_id=level_id, px_per_m=px_per_m, owners=wall_owners.get(wall["id"], [])) for wall in segmented]
    editor_wall_map = {wall["id"]: wall for wall in editor_walls}

    doors: list[dict[str, Any]] = []
    windows: list[dict[str, Any]] = []
    stairs: list[dict[str, Any]] = []
    ungrounded_elements: list[dict[str, Any]] = []

    kept_elements = [
        element
        for element in result.get("architectural_elements", [])
        if float(element.get("confidence") or 0.0) >= MIN_ELEMENT_KEEP_CONFIDENCE
    ]
    for element in kept_elements:
        element_type = str(element.get("element_type") or "")
        if element_type in {"DOOR", "WINDOW", "GARAGE_DOOR"}:
            grounded, reason = _ground_opening_for_editor(
                element=element,
                segmented_walls=segmented,
                level_id=level_id,
                px_per_m=px_per_m,
            )
            if grounded is None:
                ungrounded_elements.append({**element, "grounding_status": reason})
                continue
            if element_type == "WINDOW":
                windows.append(grounded)
                editor_wall_map[grounded["wallId"]]["windowIds"].append(grounded["id"])
            else:
                doors.append(grounded)
                editor_wall_map[grounded["wallId"]]["doorIds"].append(grounded["id"])
            continue
        if element_type == "STAIR":
            stairs.append(
                _stair_to_editor(
                    element=element,
                    level_id=level_id,
                    level_order=level_ids,
                    px_per_m=px_per_m,
                )
            )
            continue
        ungrounded_elements.append({**element, "grounding_status": "REVIEW_ONLY"})

    for space in spaces:
        boundary = Polygon([(v["x"] * px_per_m, v["y"] * px_per_m) for v in space["geometry"]["vertices"]]).boundary
        space["doorIds"] = []
        space["windowIds"] = []
        for opening, key in [*( (d, "doorIds") for d in doors), *( (w, "windowIds") for w in windows)]:
            wall = editor_wall_map.get(opening["wallId"])
            if not wall:
                continue
            point_m = (
                wall["start"]["x"] + (wall["end"]["x"] - wall["start"]["x"]) * opening["position"],
                wall["start"]["y"] + (wall["end"]["y"] - wall["start"]["y"]) * opening["position"],
            )
            point_px = Point(point_m[0] * px_per_m, point_m[1] * px_per_m)
            if boundary.distance(point_px) <= max(2.0, 0.06 * px_per_m):
                space[key].append(opening["id"])

    level_obj = {
        "id": level_id,
        "key": _level_key(str(level.get("level") or "")),
        "name": str(level.get("level") or ""),
        "order": order_map.get(level_id, 0),
        "elevationM": None,
        "heightM": None,
        "slabThicknessM": None,
        "visible": True,
        "locked": False,
        "confirmed": False,
        "source": {
            "type": "ai",
            "documentId": loaded_level.level_result.level_view.source_document_id,
            "documentName": None,
            "page": loaded_level.level_result.level_view.source_page_number,
            "confidence": None,
            "state": "INFERIDO",
            "note": "LevelView reconstruido para revisión en 04.",
        },
    }

    structure = {
        "schemaVersion": "quantia-editor-1.0",
        "projectId": None,
        "activeLevelId": level_id,
        "sourceMode": "plan",
        "metadata": {
            "sourceMode": "plan",
            "sourceDocumentId": loaded_level.level_result.level_view.source_document_id,
            "sourcePageNumber": loaded_level.level_result.level_view.source_page_number,
            "sourceLevelViewId": level_id,
            "sourceLevelName": str(level.get("level") or ""),
            "pixelsPerMeter": px_per_m,
            "call2ContractVersion": RESPONSE_CONTRACT_VERSION,
            "interface04ContractVersion": INTERFACE04_CONTRACT_VERSION,
            "localOriginalRaster": str(original_image),
            "localComparisonRaster": str(comparison_image),
            "rasterUrlReady": False,
        },
        "planoBase": {
            "referencia": None,
            "levelViewId": level_id,
            "widthPx": loaded_level.level_result.level_view.raster_width_px,
            "heightPx": loaded_level.level_result.level_view.raster_height_px,
            "localPath": str(original_image),
        },
        "niveles": [level_obj],
        "espacios": spaces,
        "muros": list(editor_wall_map.values()),
        "puertas": doors,
        "ventanas": windows,
        "escaleras": stairs,
        "anotaciones": [],
        "validacion": {
            "isValid": len(exact_duplicates) == 0,
            "conflicts": [],
            "warnings": [
                "Contrato experimental de revisión. Ningún elemento está confirmado por usuario.",
                "GARAGE_DOOR se serializa como puerta con swingDirection='garage' y conserva su tipo real en review.architecturalCandidates.",
            ],
        },
    }

    interface = {
        "contractVersion": INTERFACE04_CONTRACT_VERSION,
        "caseId": case_id,
        "caseName": case_name,
        "levelViewId": level_id,
        "levelName": str(level.get("level") or ""),
        "coordinateSystems": {
            "source": "LevelView raster px",
            "editor": "local metric m from px_per_m",
            "pxPerM": px_per_m,
        },
        "readiness": {
            "geometryReview": bool(editor_walls),
            "workflowContinuation": False,
            "quantification": False,
            "reasons": [
                "Call 2 geometry is reviewable but not user-confirmed.",
                "Openings/elements remain candidates until Call 3 and/or user confirmation.",
                "Heights are not resolved by this probe.",
            ],
        },
        "estructuraEspacial": structure,
        "review": {
            "graphAssessment": result.get("graph_assessment"),
            "wallCorrectionsProposed": result.get("wall_corrections", []),
            "wallCorrectionsApplied": applied,
            "wallCorrectionsRejected": rejected,
            "gapDecisions": result.get("gap_decisions", []),
            "architecturalCandidates": result.get("architectural_elements", []),
            "ungroundedArchitecturalCandidates": ungrounded_elements,
            "unresolvedRegions": result.get("unresolved_regions", []),
            "exactDuplicateCenterlines": exact_duplicates,
            "parallelCenterlineAnomalies": parallel_anomalies,
            "topology": topology,
        },
    }

    metrics = {
        "baseline_wall_count": len(level.get("canonical_walls", [])),
        "final_logical_wall_count": len(final_walls),
        "editor_wall_segment_count": len(editor_walls),
        "applied_correction_count": len(applied),
        "rejected_correction_count": len(rejected),
        "space_count": len(spaces),
        "door_count": len(doors),
        "window_count": len(windows),
        "garage_door_count": sum(1 for door in doors if door.get("swingDirection") == "garage"),
        "stair_count": len(stairs),
        "ungrounded_element_count": len(ungrounded_elements),
        "exact_duplicate_count": len(exact_duplicates),
        "parallel_anomaly_count": len(parallel_anomalies),
        "topology": topology,
    }
    return interface, metrics



def _validate_interface04_structure(structure: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    levels = structure.get("niveles", []) or []
    walls = structure.get("muros", []) or []
    spaces = structure.get("espacios", []) or []
    doors = structure.get("puertas", []) or []
    windows = structure.get("ventanas", []) or []
    stairs = structure.get("escaleras", []) or []

    level_ids = {str(item.get("id")) for item in levels if item.get("id")}
    wall_ids = {str(item.get("id")) for item in walls if item.get("id")}
    space_ids = {str(item.get("id")) for item in spaces if item.get("id")}

    if structure.get("schemaVersion") != "quantia-editor-1.0":
        failures.append(f"EDITOR_SCHEMA_VERSION:{structure.get('schemaVersion')}")
    if not level_ids:
        failures.append("NO_LEVELS")

    for wall in walls:
        wid = str(wall.get("id") or "")
        if str(wall.get("levelId") or "") not in level_ids:
            failures.append(f"WALL_LEVEL:{wid}:{wall.get('levelId')}")
        for key in ("start", "end"):
            point = wall.get(key) or {}
            if not (math.isfinite(float(point.get("x", float("nan")))) and math.isfinite(float(point.get("y", float("nan"))))):
                failures.append(f"WALL_POINT:{wid}:{key}")
        for owner_key in ("spaceAId", "spaceBId"):
            owner = wall.get(owner_key)
            if owner and str(owner) not in space_ids:
                failures.append(f"WALL_SPACE_REF:{wid}:{owner_key}:{owner}")

    for space in spaces:
        sid = str(space.get("id") or "")
        if str(space.get("levelId") or "") not in level_ids:
            failures.append(f"SPACE_LEVEL:{sid}:{space.get('levelId')}")
        for wid in space.get("wallIds", []) or []:
            if str(wid) not in wall_ids:
                failures.append(f"SPACE_WALL_REF:{sid}:{wid}")
        geometry = space.get("geometry") or {}
        vertices = geometry.get("vertices") or []
        if len(vertices) < 3:
            failures.append(f"SPACE_GEOMETRY:{sid}")

    for label, openings in (("DOOR", doors), ("WINDOW", windows)):
        for opening in openings:
            oid = str(opening.get("id") or "")
            if str(opening.get("levelId") or "") not in level_ids:
                failures.append(f"{label}_LEVEL:{oid}:{opening.get('levelId')}")
            if str(opening.get("wallId") or "") not in wall_ids:
                failures.append(f"{label}_WALL:{oid}:{opening.get('wallId')}")
            position = opening.get("position")
            if position is None or not (0.0 <= float(position) <= 1.0):
                failures.append(f"{label}_POSITION:{oid}:{position}")
            width = opening.get("widthM")
            if width is None or float(width) <= 0:
                failures.append(f"{label}_WIDTH:{oid}:{width}")

    for stair in stairs:
        sid = str(stair.get("id") or "")
        if str(stair.get("levelFromId") or "") not in level_ids:
            failures.append(f"STAIR_LEVEL_FROM:{sid}:{stair.get('levelFromId')}")
        target = str(stair.get("levelToId") or "")
        # levelTo puede referir a otro nivel del mismo proyecto no incluido en el payload unitario.
        if target and target not in level_ids:
            pass
        geometry = stair.get("geometry") or {}
        if len(geometry.get("vertices") or []) < 3:
            failures.append(f"STAIR_GEOMETRY:{sid}")
    return failures


def _render_result_comparison(
    *,
    original: np.ndarray,
    baseline: list[dict[str, Any]],
    final_walls: list[dict[str, Any]],
    elements: list[dict[str, Any]],
) -> np.ndarray:
    baseline_img = _draw_centerlines(original, baseline, with_ids=False)
    final_img = _draw_centerlines(original, final_walls, with_ids=False)
    for element in elements:
        if not _bbox_valid(element.get("bbox_px", []), width=original.shape[1], height=original.shape[0]):
            continue
        x0, y0, x1, y1 = [int(round(float(v))) for v in element["bbox_px"]]
        etype = str(element.get("element_type") or "UNKNOWN")
        color = {
            "DOOR": (0, 120, 255),
            "WINDOW": (255, 100, 0),
            "GARAGE_DOOR": (140, 0, 220),
            "STAIR": (0, 160, 0),
        }.get(etype, (90, 90, 90))
        cv2.rectangle(final_img, (x0, y0), (x1, y1), color, 2)
        cv2.putText(final_img, etype, (x0, max(12, y0 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    def titled(panel: np.ndarray, title: str) -> np.ndarray:
        header = np.full((42, panel.shape[1], 3), 255, dtype=np.uint8)
        cv2.putText(header, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (0, 0, 0), 2, cv2.LINE_AA)
        return np.vstack([header, panel])

    return np.hstack(
        [
            titled(original, "ORIGINAL"),
            titled(baseline_img, "BASELINE CENTERLINES"),
            titled(final_img, "CALL 2 FINAL + ELEMENTS"),
        ]
    )


def _run_level(*, case_id: str, case, canonical_report: dict[str, Any], level: dict[str, Any]) -> dict[str, Any]:
    level_name = str(level.get("level") or "")
    level_view_id = str(level["level_view_id"])
    loaded = _find_loaded_level(case, level_name=level_name, level_view_id=level_view_id)
    level_view = loaded.level_result.level_view
    assert str(level_view.id) == level_view_id

    image_bytes = bytes(level_view.raster_bytes)
    image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert image is not None, f"No se pudo decodificar raster {level_view_id}."
    height, width = image.shape[:2]

    baseline = _baseline_walls(level)
    gaps = _gap_capsule(level)
    capsule = _state_capsule(level, loaded, width=width, height=height)
    composed = _compose_input(image, baseline, gaps)
    ok, composed_bytes = cv2.imencode(".png", composed)
    assert ok
    composed_bytes_raw = bytes(composed_bytes)
    prompt = _build_prompt(capsule)

    provider_output_key = str(os.getenv("QUANTIA_CALL2_PROVIDER") or "gemini").strip().casefold()
    level_dir = OUTPUT_ROOT / provider_output_key / case_id / _slug(level_name)
    level_dir.mkdir(parents=True, exist_ok=True)
    original_path = level_dir / "original_levelview.png"
    input_path = level_dir / "call2_input_abc.png"
    prompt_path = level_dir / "prompt.txt"
    schema_path = level_dir / "schema.json"
    capsule_path = level_dir / "state_capsule.json"
    cv2.imwrite(str(original_path), image)
    cv2.imwrite(str(input_path), composed)
    prompt_path.write_text(prompt, encoding="utf-8")
    schema_path.write_text(json.dumps(RESPONSE_SCHEMA, ensure_ascii=False, indent=2), encoding="utf-8")
    capsule_path.write_text(json.dumps(capsule, ensure_ascii=False, indent=2), encoding="utf-8")

    raw, text, provider_name, model, attempts = _call_model(prompt=prompt, image_bytes=composed_bytes_raw)
    raw_path = level_dir / "call2_raw.json"
    text_path = level_dir / "call2_text.txt"
    attempts_path = level_dir / "transport_attempts.json"
    raw_path.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
    text_path.write_text(text, encoding="utf-8")
    attempts_path.write_text(json.dumps(attempts, ensure_ascii=False, indent=2), encoding="utf-8")

    try:
        result = json.loads(text)
    except json.JSONDecodeError as exc:
        raise AssertionError(f"Call 2 devolvió JSON inválido: {exc}. RAW={raw_path}") from exc

    contract_failures = _validate_result(capsule=capsule, result=result, width=width, height=height)
    result_path = level_dir / "call2_result.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    if contract_failures:
        (level_dir / "contract_failures.json").write_text(
            json.dumps(contract_failures, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        raise AssertionError("Call 2 contract invalid: " + " | ".join(contract_failures))

    final_walls, applied, rejected = _apply_corrections(
        baseline=baseline,
        corrections=result.get("wall_corrections", []),
        px_per_m=float(level["px_per_m"]),
        width=width,
        height=height,
    )
    final_walls, exact_duplicates = _dedupe_exact_walls(final_walls)
    parallel = _parallel_anomalies(final_walls, px_per_m=float(level["px_per_m"]))

    comparison = _render_result_comparison(
        original=image,
        baseline=baseline,
        final_walls=final_walls,
        elements=result.get("architectural_elements", []),
    )
    comparison_path = level_dir / "comparison_original_baseline_call2.png"
    cv2.imwrite(str(comparison_path), comparison)

    interface04, metrics = _build_interface04(
        case_id=case_id,
        case_name=case.case_name,
        canonical_report=canonical_report,
        level=level,
        loaded_level=loaded,
        result=result,
        final_walls=final_walls,
        applied=applied,
        rejected=rejected,
        exact_duplicates=exact_duplicates,
        parallel_anomalies=parallel,
        comparison_image=comparison_path,
        original_image=original_path,
    )
    interface_failures = _validate_interface04_structure(interface04["estructuraEspacial"])
    interface_path = level_dir / "interface04_geometry.json"
    metrics_path = level_dir / "comparison_metrics.json"
    final_graph_path = level_dir / "final_centerline_graph.json"
    interface_path.write_text(json.dumps(interface04, ensure_ascii=False, indent=2), encoding="utf-8")
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    final_graph_path.write_text(json.dumps(final_walls, ensure_ascii=False, indent=2), encoding="utf-8")

    audit = {
        "probe_version": PROBE_VERSION,
        "baseline": BASELINE,
        "response_contract_version": RESPONSE_CONTRACT_VERSION,
        "interface04_contract_version": INTERFACE04_CONTRACT_VERSION,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "case_id": case_id,
        "case_name": case.case_name,
        "level_name": level_name,
        "level_view_id": level_view_id,
        "provider": provider_name,
        "model": model,
        "transport_attempts": attempts,
        "prompt_sha256": _sha256_bytes(prompt.encode("utf-8")),
        "schema_sha256": _sha256_json(RESPONSE_SCHEMA),
        "state_capsule_sha256": _sha256_json(capsule),
        "original_image_sha256": _sha256_bytes(image_bytes),
        "composed_image_sha256": _sha256_bytes(composed_bytes_raw),
        "normalized_response_sha256": _sha256_json(result),
        "contract_failures": contract_failures,
        "interface04_failures": interface_failures,
        "metrics": metrics,
        "files": {
            "original": str(original_path),
            "input_abc": str(input_path),
            "prompt": str(prompt_path),
            "schema": str(schema_path),
            "state_capsule": str(capsule_path),
            "raw": str(raw_path),
            "normalized": str(result_path),
            "final_graph": str(final_graph_path),
            "interface04": str(interface_path),
            "comparison": str(comparison_path),
            "metrics": str(metrics_path),
        },
    }
    audit_path = level_dir / "audit.json"
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")

    history_path = OUTPUT_ROOT / provider_name / "call2_geometry_history.jsonl"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(audit, ensure_ascii=False) + "\n")

    print("\n" + "=" * 110)
    print(f"CALL2 GEOMETRY -> INTERFACE04 V1 | {case.case_name} | {level_name}")
    print(
        {
            "provider": provider_name,
            "model": model,
            "baseline_walls": metrics["baseline_wall_count"],
            "final_walls": metrics["final_logical_wall_count"],
            "editor_segments": metrics["editor_wall_segment_count"],
            "spaces": metrics["space_count"],
            "doors": metrics["door_count"],
            "windows": metrics["window_count"],
            "garage_doors": metrics["garage_door_count"],
            "stairs": metrics["stair_count"],
            "ungrounded_elements": metrics["ungrounded_element_count"],
            "parallel_anomalies": metrics["parallel_anomaly_count"],
            "contract_failures": contract_failures,
            "interface04_failures": interface_failures,
            "comparison": str(comparison_path),
            "interface04": str(interface_path),
        }
    )

    assert not contract_failures, " | ".join(contract_failures)
    assert not interface_failures, " | ".join(interface_failures)
    assert metrics["exact_duplicate_count"] == 0, f"Persisten duplicados exactos: {exact_duplicates}"
    if str(os.getenv("QUANTIA_CALL2_STRICT_QUALITY") or "0").strip() == "1":
        assert metrics["parallel_anomaly_count"] == 0, f"Persisten paralelismos candidatos: {parallel}"
        assert metrics["space_count"] > 0, "No se generó ningún espacio cerrado para 04."

    return audit


@pytest.mark.parametrize("case_id", CASE_IDS)
def test_quantia_spatial_call2_geometry_interface04_all_3_v1(case_id: str) -> None:
    requested_cases = {
        item.strip()
        for item in str(os.getenv("QUANTIA_CALL2_CASES") or "all").split(",")
        if item.strip()
    }
    if "all" not in requested_cases and case_id not in requested_cases:
        pytest.skip(f"Caso {case_id} fuera de QUANTIA_CALL2_CASES={requested_cases}")

    canonical_report = _load_canonical_report(case_id)
    case = _load_case(case_id)
    level_filter = str(os.getenv("QUANTIA_CALL2_LEVEL") or "").strip().casefold()
    audits: list[dict[str, Any]] = []

    for level in canonical_report.get("levels", []):
        level_name = str(level.get("level") or "")
        if level_filter and level_filter not in level_name.casefold():
            continue
        audits.append(
            _run_level(
                case_id=case_id,
                case=case,
                canonical_report=canonical_report,
                level=level,
            )
        )

    assert audits, f"No se seleccionó ningún LevelView de {case_id}."
    case_summary = {
        "probe_version": PROBE_VERSION,
        "case_id": case_id,
        "case_name": case.case_name,
        "level_count": len(audits),
        "total_baseline_walls": sum(a["metrics"]["baseline_wall_count"] for a in audits),
        "total_final_walls": sum(a["metrics"]["final_logical_wall_count"] for a in audits),
        "total_spaces": sum(a["metrics"]["space_count"] for a in audits),
        "total_doors": sum(a["metrics"]["door_count"] for a in audits),
        "total_windows": sum(a["metrics"]["window_count"] for a in audits),
        "total_garage_doors": sum(a["metrics"]["garage_door_count"] for a in audits),
        "total_stairs": sum(a["metrics"]["stair_count"] for a in audits),
        "total_parallel_anomalies": sum(a["metrics"]["parallel_anomaly_count"] for a in audits),
        "levels": [
            {
                "level_name": a["level_name"],
                "level_view_id": a["level_view_id"],
                "metrics": a["metrics"],
                "comparison": a["files"]["comparison"],
                "interface04": a["files"]["interface04"],
            }
            for a in audits
        ],
    }
    provider_output_key = str(os.getenv("QUANTIA_CALL2_PROVIDER") or "gemini").strip().casefold()
    case_out = OUTPUT_ROOT / provider_output_key / case_id
    case_out.mkdir(parents=True, exist_ok=True)
    (case_out / "case_comparison_summary.json").write_text(
        json.dumps(case_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
