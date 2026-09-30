from __future__ import annotations

import base64
import copy
import hashlib
import json
import math
import os
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
import pytest
import requests

from app.quantia_spatialV1.adaptive_reconstruction.call2_schema import CALL2_SCHEMA
from app.quantia_spatialV1.adaptive_reconstruction.call2_validation import Call2DeltaValidator
from app.quantia_spatialV1.adaptive_reconstruction.contracts import MultimodalWallReview, SingleLineWallGraph
from app.quantia_spatialV1.providers.call2_providers import (
    Call2ProviderError,
    Call2ProviderResult,
    Call2StrictSchemaAdapter,
    _EnvKeyResolver,
)
from app.quantia_spatialV1.tests.call2_editor_snapshot import ROOT, load_snapshot
from app.quantia_spatialV1.tests.call2_interface04_export import export_review


PROBE_VERSION = "CALL2_QWEN_HF_INTERFACE04_GEOMETRY_PROBE_V1_1"
SCHEMA_VERSION = "CALL2_QWEN_HF_GEOMETRY_SCHEMA_V1"
OUTPUT_ROOT = ROOT / "tests/output/call2_qwen_hf_interface04_geometry_v1"
HF_URL = "https://router.huggingface.co/v1/chat/completions"
HF_PRIMARY_MODEL = str(
    os.getenv("QUANTIA_HF_QWEN_PRIMARY_MODEL", "Qwen/Qwen3-VL-30B-A3B-Instruct:deepinfra")
).strip()
HF_FALLBACK_MODEL = str(
    os.getenv("QUANTIA_HF_QWEN_FALLBACK_MODEL", "Qwen/Qwen3.8-27B:deepinfra")
).strip()
HF_MAX_ATTEMPTS = max(1, int(os.getenv("QUANTIA_HF_QWEN_MAX_ATTEMPTS", "2")))
HF_TIMEOUT_SECONDS = max(30, int(os.getenv("QUANTIA_HF_QWEN_TIMEOUT_SECONDS", "180")))
MAX_OUTPUT_TOKENS = max(1024, int(os.getenv("QUANTIA_CALL2_MAX_OUTPUT_TOKENS", "16384")))
RUN_LIVE = str(os.getenv("QUANTIA_RUN_QWEN_HF_CALL2", "0") or "0").strip() == "1"
TRANSIENT_STATUS = {408, 429, 500, 502, 503, 504}
MIN_SPACE_AREA_M2 = float(os.getenv("QUANTIA_CALL2_MIN_SPACE_AREA_M2", "0.50"))
OPENING_HOST_MAX_DISTANCE_M = float(os.getenv("QUANTIA_CALL2_OPENING_HOST_MAX_DISTANCE_M", "0.45"))


def _extended_schema() -> dict[str, Any]:
    """Extiende sólo el contrato de la prueba; no modifica CALL2_SCHEMA productivo."""
    schema = copy.deepcopy(CALL2_SCHEMA)
    props = schema["properties"]

    gap_schema = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "gap_id": {"type": "string"},
                "classification": {
                    "type": "string",
                    "enum": ["WALL_CONTINUITY", "PROBABLE_OPENING", "UNCERTAIN", "NOT_A_GAP"],
                },
                "host_wall_continuity": {"type": "boolean"},
                "solid_wall_present": {"type": "boolean"},
                "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "reason": {"type": "string"},
            },
            "required": [
                "gap_id",
                "classification",
                "host_wall_continuity",
                "solid_wall_present",
                "confidence",
                "reason",
            ],
        },
    }
    props["gap_decisions"] = gap_schema
    if "gap_decisions" not in schema["required"]:
        schema["required"].append("gap_decisions")

    region = props["non_wall_architectural_regions"]["items"]
    region_props = region["properties"]
    region_props["family_hint"]["enum"] = [
        "DOOR",
        "WINDOW",
        "GARAGE_DOOR",
        "STAIR",
        "OTHER",
        "UNCERTAIN",
    ]
    region_props.update({
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
        "stair_direction": {"type": "string", "enum": ["UP", "DOWN", "UNKNOWN"]},
    })
    return schema


QWEN_CALL2_SCHEMA = _extended_schema()


def _load_snapshot_portable(case: str, level: str):
    """Usa el loader canónico; sólo normaliza separadores al validar ZIP fuera de Windows."""
    try:
        return load_snapshot(case, level)
    except (ValueError, KeyError):
        if os.name == "nt":
            raise

    from app.quantia_spatialV1.tests import call2_editor_snapshot as snap

    validation = json.loads(
        (ROOT / "documentation/LOCAL_VALIDATION_CANONICAL_WALLGRAPH_INTEGRITY_V2.json").read_text(encoding="utf-8")
    )
    run_rel = str(validation["output_directory"]).replace("\\", "/")
    run = ROOT / Path(run_rel)
    paths = [p for p in (run / case).glob("*__canonical_wallgraph.json") if level in p.name]
    if len(paths) != 1:
        raise ValueError(f"Select exactly one saved LevelView: case={case} level={level} matches={len(paths)}")
    graph_path = paths[0]
    stem = graph_path.name.removesuffix("__canonical_wallgraph.json")
    bundle_path = graph_path.with_name(stem + "__reconstruction_evidence_bundle.json")
    post_path = graph_path.with_name(stem + "__filter_audit.json")

    manifest = validation["artifacts_sha256"]
    for path in (graph_path, bundle_path, post_path):
        rel = str(path.relative_to(ROOT))
        keys = (rel, rel.replace("/", "\\"), rel.replace("\\", "/"))
        expected = next((manifest[k] for k in keys if k in manifest), None)
        if expected is None:
            raise KeyError(f"Artifact no está en manifest: {rel}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Snapshot hash mismatch: {path.name}")

    graph = snap.SingleLineWallGraph.model_validate_json(graph_path.read_text(encoding="utf-8"))
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    post = snap.PostReconstructionFilterResult.model_validate_json(post_path.read_text(encoding="utf-8"))
    raster = (ROOT / "tests/output/canonical_visual_review" / run.name / (stem + "__original.png")).read_bytes()
    if hashlib.sha256(raster).hexdigest() != bundle["source_raster_sha256"]:
        raise ValueError("Original LevelView raster hash mismatch")
    view = snap.SavedLevelRaster(
        graph.level_view_id, graph.level_name, bundle["source_document_id"], bundle["source_page_number"],
        snap.PixelBBox(**bundle["source_bbox_px"]), *graph.image_size_px, raster
    )
    adaptive = post.filtered_wall_graph.model_copy(update={
        "walls": [snap.SingleLineWall(**row) for row in bundle["adaptive_walls"]],
        "logical_gaps": [snap.LogicalGap(**row) for row in bundle["adaptive_logical_gaps"]],
    }, deep=True)
    artifacts = {name: bundle[name] for name in (
        "f03_seed_candidates", "discovered_candidates", "quarantined_candidates",
        "hybrid_candidates", "context_decisions", "context_regions"
    )}
    for name, saved in bundle["preserved_artifacts"].items():
        if isinstance(saved, dict) and saved.get("kind") == "ndarray":
            import zlib
            raw = zlib.decompress(base64.b64decode(saved["data"]))
            if hashlib.sha256(raw).hexdigest() != saved["sha256"]:
                raise ValueError("Corrupt saved mask")
            artifacts[name] = np.frombuffer(raw, dtype=saved["dtype"]).reshape(saved["shape"]).copy()
        else:
            artifacts[name] = saved
    ids = {wall.id for wall in adaptive.walls}
    tracks = [track for track in snap.AdaptiveReconstructionEngine._physical_walls(
        [snap.WallCandidate(**row) for row in bundle["hybrid_candidates"]]) if track.wall_id in ids]
    if {track.wall_id for track in tracks} != ids:
        raise ValueError("Incomplete source wall tracks")
    runtime = snap.AdaptiveReconstructionRuntime(
        graph=adaptive, wall_tracks=tracks, bridges=[],
        topology=snap.AdaptiveReconstructionEngine._space_topology(artifacts["logical_mask"], px_per_m=graph.px_per_m),
        context_quarantined_count=adaptive.diagnostics.quarantined_candidate_count, artifacts=artifacts
    )
    evidence = [snap.RawEvidence(**row) for row in bundle["raw_evidence"]]
    return view, graph, runtime, post, evidence, stem


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha_json(data: Any) -> str:
    raw = json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return _sha_bytes(raw)


def _point(value: Any) -> tuple[float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    try:
        return float(value[0]), float(value[1])
    except (TypeError, ValueError):
        return None


def _bbox(value: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        x0, y0, x1, y1 = map(float, value)
    except (TypeError, ValueError):
        return None
    if x1 < x0 or y1 < y0:
        return None
    return x0, y0, x1, y1


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


class HuggingFaceQwenCall2Provider:
    """Transporte experimental exclusivo del test; no modifica providers productivos."""

    provider_name = "huggingface"

    def __init__(self) -> None:
        self.api_key = _EnvKeyResolver.get("HF_TOKEN")
        if not self.api_key:
            raise Call2ProviderError("HF_TOKEN no está configurado.")
        self.routes = [route for route in (HF_PRIMARY_MODEL, HF_FALLBACK_MODEL) if route]
        if not self.routes:
            raise Call2ProviderError("No hay modelos Qwen/HF configurados.")
        self.model = self.routes[0]
        self.last_attempts: list[dict[str, Any]] = []
        self.last_usage: dict[str, Any] = {}
        self.last_raw: dict[str, Any] | None = None
        self.last_text: str = ""
        self.last_finish_reason: str | None = None

    @staticmethod
    def _data_url(media_bytes: bytes, media_mime_type: str) -> str:
        return f"data:{media_mime_type};base64,{base64.b64encode(media_bytes).decode('ascii')}"

    def build_request(
        self,
        *,
        model: str,
        prompt: str,
        media_bytes: bytes,
        media_mime_type: str,
        response_json_schema: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "model": model,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": self._data_url(media_bytes, media_mime_type)},
                    },
                ],
            }],
            "temperature": 0,
            "max_tokens": MAX_OUTPUT_TOKENS,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "quantia_call2_qwen_hf_geometry",
                    "schema": Call2StrictSchemaAdapter.adapt(response_json_schema),
                    "strict": True,
                },
            },
        }

    @staticmethod
    def _extract_text(raw: dict[str, Any]) -> str:
        try:
            content = raw["choices"][0]["message"]["content"]
        except Exception as exc:
            raise Call2ProviderError("HF sin choices[0].message.content") from exc
        if not isinstance(content, str) or not content.strip():
            raise Call2ProviderError("HF devolvió contenido no textual/vacío.")
        return content.strip()

    def analyze(
        self,
        *,
        prompt: str,
        media_bytes: bytes,
        media_mime_type: str,
        response_json_schema: dict[str, Any] | None = None,
    ) -> Call2ProviderResult:
        if not response_json_schema:
            raise Call2ProviderError("Este probe requiere JSON Schema estricto.")
        attempts: list[dict[str, Any]] = []
        errors: list[str] = []
        for route_index, route in enumerate(self.routes):
            for attempt in range(1, HF_MAX_ATTEMPTS + 1):
                payload = self.build_request(
                    model=route,
                    prompt=prompt,
                    media_bytes=media_bytes,
                    media_mime_type=media_mime_type,
                    response_json_schema=response_json_schema,
                )
                started = time.time()
                try:
                    response = requests.post(
                        HF_URL,
                        headers={
                            "Authorization": f"Bearer {self.api_key}",
                            "Content-Type": "application/json",
                        },
                        json=payload,
                        timeout=HF_TIMEOUT_SECONDS,
                    )
                except requests.RequestException as exc:
                    attempts.append({
                        "model": route,
                        "attempt": attempt,
                        "ok": False,
                        "error": type(exc).__name__,
                        "elapsed_s": round(time.time() - started, 3),
                    })
                    errors.append(f"{route} attempt={attempt}: {exc}")
                    if attempt < HF_MAX_ATTEMPTS:
                        time.sleep(min(2 ** (attempt - 1), 4))
                    continue

                status = int(response.status_code)
                attempts.append({
                    "model": route,
                    "attempt": attempt,
                    "ok": response.ok,
                    "http_status": status,
                    "elapsed_s": round(time.time() - started, 3),
                })
                if response.ok:
                    raw = response.json()
                    text = self._extract_text(raw)
                    finish_reason = None
                    try:
                        finish_reason = raw["choices"][0].get("finish_reason")
                    except Exception:
                        finish_reason = None
                    self.last_raw = raw if isinstance(raw, dict) else {"response": raw}
                    self.last_text = text
                    self.last_finish_reason = str(finish_reason) if finish_reason is not None else None
                    self.last_usage = raw.get("usage", {}) if isinstance(raw, dict) else {}
                    try:
                        data = json.loads(text)
                    except json.JSONDecodeError as exc:
                        attempts[-1]["ok"] = False
                        attempts[-1]["stage"] = "json_parse"
                        attempts[-1]["finish_reason"] = self.last_finish_reason
                        attempts[-1]["content_chars"] = len(text)
                        attempts[-1]["json_error"] = str(exc)
                        errors.append(
                            f"{route} attempt={attempt}: HTTP 200 pero JSON inválido: {exc}; "
                            f"finish_reason={self.last_finish_reason!r}; chars={len(text)}"
                        )
                        if attempt < HF_MAX_ATTEMPTS:
                            time.sleep(min(2 ** (attempt - 1), 4))
                            continue
                        break
                    if not isinstance(data, dict):
                        attempts[-1]["ok"] = False
                        attempts[-1]["stage"] = "json_type"
                        errors.append(f"{route} attempt={attempt}: JSON válido pero no es objeto.")
                        if attempt < HF_MAX_ATTEMPTS:
                            continue
                        break
                    self.model = route
                    self.last_attempts = attempts
                    return Call2ProviderResult(
                        provider=self.provider_name,
                        model=route,
                        text=text,
                        data=data,
                        fallback_used=route_index > 0,
                        raw=raw,
                    )

                try:
                    err = json.dumps(response.json(), ensure_ascii=False)
                except Exception:
                    err = response.text
                errors.append(f"{route} attempt={attempt} HTTP {status}: {err[:800]}")
                if status not in TRANSIENT_STATUS:
                    break
                if attempt < HF_MAX_ATTEMPTS:
                    time.sleep(min(2 ** (attempt - 1), 4))
            # fallback route is allowed only after current route failed.
        self.last_attempts = attempts
        raise Call2ProviderError("Todos los routes Qwen/HF fallaron. " + " || ".join(errors[-6:]))


def _decode_png(data: bytes) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise AssertionError("No se pudo decodificar raster LevelView.")
    return image


def _header(width: int, text: str) -> np.ndarray:
    band = np.full((42, width, 3), 255, dtype=np.uint8)
    cv2.putText(band, text, (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 0, 0), 2, cv2.LINE_AA)
    return band


def _build_abc_image(*, view: Any, graph: SingleLineWallGraph) -> bytes:
    """A original; B original+centerlines/IDs; C centerlines-only. Sin resize."""
    original = _decode_png(view.raster_bytes)
    height, width = original.shape[:2]
    if (width, height) != tuple(graph.image_size_px):
        raise AssertionError("Raster y WallGraph no comparten image_size_px.")

    overlay = original.copy()
    clean = np.full_like(original, 255)
    for wall in graph.walls:
        p1 = (int(round(wall.start_px[0])), int(round(wall.start_px[1])))
        p2 = (int(round(wall.end_px[0])), int(round(wall.end_px[1])))
        cv2.line(overlay, p1, p2, (255, 0, 255), 2, cv2.LINE_AA)
        cv2.line(clean, p1, p2, (0, 0, 0), 3, cv2.LINE_AA)
        mx, my = int(round((p1[0] + p2[0]) / 2)), int(round((p1[1] + p2[1]) / 2))
        cv2.putText(overlay, wall.id, (mx + 3, my - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (40, 40, 180), 1, cv2.LINE_AA)

    for gap in graph.logical_gaps:
        p1 = (int(round(gap.start_px[0])), int(round(gap.start_px[1])))
        p2 = (int(round(gap.end_px[0])), int(round(gap.end_px[1])))
        cv2.line(overlay, p1, p2, (0, 165, 255), 2, cv2.LINE_AA)

    divider = np.full((12, width, 3), 255, dtype=np.uint8)
    composite = np.vstack([
        _header(width, "A - ORIGINAL LEVELVIEW (PRIMARY TRUTH)"),
        original,
        divider,
        _header(width, "B - ORIGINAL + CURRENT CENTERLINES / IDS / GAPS"),
        overlay,
        divider,
        _header(width, "C - CURRENT CENTERLINES ONLY"),
        clean,
    ])
    ok, encoded = cv2.imencode(".png", composite)
    if not ok:
        raise AssertionError("No se pudo codificar input ABC.")
    return encoded.tobytes()


def _state_capsule(*, view: Any, graph: SingleLineWallGraph) -> dict[str, Any]:
    role_code = {"PERIMETER": "P", "DIVIDER": "D", "REVIEW": "R"}
    return {
        "probe_version": PROBE_VERSION,
        "level_view_id": graph.level_view_id,
        "level_name": graph.level_name,
        "image_size_px": list(graph.image_size_px),
        "px_per_m": round(graph.px_per_m, 5),
        "source": {
            "document_id": view.source_document_id,
            "page": view.source_page_number,
            "bbox_page_px": [
                view.source_bbox_px.x_min,
                view.source_bbox_px.y_min,
                view.source_bbox_px.x_max,
                view.source_bbox_px.y_max,
            ],
        },
        "walls": [
            [
                w.id,
                round(w.start_px[0], 1),
                round(w.start_px[1], 1),
                round(w.end_px[0], 1),
                round(w.end_px[1], 1),
                role_code.get(w.role, "R"),
                round(w.thickness_px, 1),
            ]
            for w in graph.walls
        ],
        "logical_gaps": [
            [
                g.id,
                round(g.start_px[0], 1),
                round(g.start_px[1], 1),
                round(g.end_px[0], 1),
                round(g.end_px[1], 1),
                [str(x) for x in g.wall_ids],
            ]
            for g in graph.logical_gaps
        ],
        "baseline_interior_space_count": graph.interior_space_count,
    }


def _build_prompt(*, view: Any, graph: SingleLineWallGraph) -> tuple[str, dict[str, Any]]:
    capsule = _state_capsule(view=view, graph=graph)
    state = json.dumps(capsule, ensure_ascii=False, separators=(",", ":"))
    prompt = f"""
{PROBE_VERSION}

MISSION
Audit the CURRENT Quantia SpatialV1 WallGraph against the exact original LevelView and return only corrections plus architectural candidates. The goal is a single-centerline representation per physical wall suitable for the existing SpatialV1 validator/finalizer and Interfaz 04 review contract.

VISUAL PANELS
A = exact original LevelView. PRIMARY TRUTH.
B = same original with current Quantia centerlines, IDs and logical gaps.
C = current Quantia centerlines only.
All panels preserve the same local LevelView coordinate system. Output coordinates MUST use original LevelView pixels, never composite-image offsets.

WALL RULES
1. One physical wall = one centerline. Never model the two wall faces as two walls.
2. KEEP is implicit; do not repeat correct walls.
3. REMOVE_WALL when a current centerline is not a physical wall.
4. ADD_WALL only for a clearly visible omitted physical wall.
5. EXTEND_WALL / TRIM_WALL / REPOSITION_WALL must follow visible endpoints and junctions.
6. MERGE_WALLS when multiple current centerlines represent one physical wall. SPLIT_WALL only when one current wall incorrectly spans separate physical runs.
7. Perimeter is evidence, not immutable final geometry. Correct it only when A clearly contradicts B.
8. Do not fill a visible opening with solid wall. Preserve logical wall continuity separately through gap_decisions.
9. Do not convert axes, dimensions, stair treads, railings, furniture, hatches, glazing symbols or annotation into walls.
10. If uncertain, use unresolved_regions rather than inventing geometry.

GAPS
Return one gap_decision for every logical_gap listed in STATE.
- WALL_CONTINUITY: physical solid wall should continue through this gap; host_wall_continuity=true; solid_wall_present=true.
- PROBABLE_OPENING: logical wall continuity exists but the gap is physically open; host_wall_continuity=true; solid_wall_present=false.
- NOT_A_GAP: upstream bridge hypothesis is false; host_wall_continuity=false; solid_wall_present=false.
- UNCERTAIN: evidence insufficient; host_wall_continuity=false; solid_wall_present=false.
Never invent gap IDs.

ARCHITECTURAL ELEMENTS TO IDENTIFY
Identify all visually clear DOOR, WINDOW, GARAGE_DOOR and STAIR regions that are relevant to the reconstructed geometry.
For DOOR/WINDOW/GARAGE_DOOR:
- bbox_px required.
- host_wall_id when a current/final wall is identifiable.
- host_wall_continuity=true and solid_wall_present=false when it is an opening.
- span_start_px/span_end_px should mark the opening interval projected along the host wall when visually defensible.
For STAIR:
- bbox_px required.
- stair_axis_start_px/stair_axis_end_px and UP/DOWN/UNKNOWN when visually defensible.
Do not promote weak symbols: use OTHER/UNCERTAIN or unresolved_regions.

OUTPUT
Return only JSON matching the schema. Reasons <= 16 words. summary <= 2 short sentences.
Wall tuple in STATE = [id,x1,y1,x2,y2,role,thickness_px], role P/D/R.
Gap tuple = [id,x1,y1,x2,y2,wall_ids].

STATE
{state}
""".strip()
    return prompt, capsule


def _segment_projection(point: tuple[float, float], start: tuple[float, float], end: tuple[float, float]) -> tuple[float, tuple[float, float], float]:
    dx, dy = end[0] - start[0], end[1] - start[1]
    denom = dx * dx + dy * dy
    if denom <= 1e-9:
        return 0.0, start, math.dist(point, start)
    t = ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / denom
    t_clamped = max(0.0, min(1.0, t))
    proj = (start[0] + t_clamped * dx, start[1] + t_clamped * dy)
    return t_clamped, proj, math.dist(point, proj)


def _resolve_host(region: dict[str, Any], graph: SingleLineWallGraph) -> tuple[Any | None, str | None, float | None]:
    walls = {wall.id: wall for wall in graph.walls}
    explicit = str(region.get("host_wall_id") or "").strip()
    if explicit in walls:
        return walls[explicit], "model", 0.0
    bbox = _bbox(region.get("bbox_px"))
    if bbox is None:
        return None, None, None
    x0, y0, x1, y1 = bbox
    points = [
        ((x0 + x1) / 2.0, (y0 + y1) / 2.0),
        (x0, y0), (x1, y0), (x1, y1), (x0, y1),
    ]
    best = None
    for wall in graph.walls:
        distance = min(_segment_projection(point, wall.start_px, wall.end_px)[2] for point in points)
        if best is None or distance < best[0]:
            best = (distance, wall)
    if best is None or best[0] / graph.px_per_m > OPENING_HOST_MAX_DISTANCE_M:
        return None, None, None
    return best[1], "nearest", best[0] / graph.px_per_m


def _opening_geometry(region: dict[str, Any], graph: SingleLineWallGraph) -> dict[str, Any] | None:
    wall, host_source, host_distance_m = _resolve_host(region, graph)
    if wall is None:
        return None
    length_px = math.dist(wall.start_px, wall.end_px)
    if length_px <= 1e-9:
        return None

    explicit_a = _point(region.get("span_start_px"))
    explicit_b = _point(region.get("span_end_px"))
    if explicit_a and explicit_b:
        t0, p0, _ = _segment_projection(explicit_a, wall.start_px, wall.end_px)
        t1, p1, _ = _segment_projection(explicit_b, wall.start_px, wall.end_px)
        span_source = "model_span"
    else:
        bbox = _bbox(region.get("bbox_px"))
        if bbox is None:
            return None
        x0, y0, x1, y1 = bbox
        corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        projected = [_segment_projection(point, wall.start_px, wall.end_px) for point in corners]
        projected.sort(key=lambda row: row[0])
        t0, p0 = projected[0][0], projected[0][1]
        t1, p1 = projected[-1][0], projected[-1][1]
        span_source = "bbox_projection"
    if t1 < t0:
        t0, t1, p0, p1 = t1, t0, p1, p0
    span_px = max(0.0, (t1 - t0) * length_px)
    if span_px <= 1.0:
        return None
    scale = graph.px_per_m
    return {
        "hostWallId": wall.id,
        "hostGroundingSource": host_source,
        "hostDistanceM": host_distance_m,
        "spanSource": span_source,
        "position": (t0 + t1) / 2.0,
        "offsetStartM": t0 * length_px / scale,
        "offsetEndM": t1 * length_px / scale,
        "widthM": span_px / scale,
        "rasterSegment": [{"x": p0[0], "y": p0[1]}, {"x": p1[0], "y": p1[1]}],
        "metricSegment": [{"x": p0[0] / scale, "y": p0[1] / scale}, {"x": p1[0] / scale, "y": p1[1] / scale}],
    }


def _derive_spaces(*, graph: SingleLineWallGraph, accepted_review: MultimodalWallReview) -> list[dict[str, Any]]:
    """Topología experimental para revisión 04; no promueve espacios a verdad productiva."""
    width, height = graph.image_size_px
    barrier = np.zeros((height, width), dtype=np.uint8)
    for wall in graph.walls:
        p1 = (int(round(wall.start_px[0])), int(round(wall.start_px[1])))
        p2 = (int(round(wall.end_px[0])), int(round(wall.end_px[1])))
        cv2.line(barrier, p1, p2, 255, 3, cv2.LINE_8)

    gaps = {gap.id: gap for gap in graph.logical_gaps}
    for decision in accepted_review.gap_decisions:
        if decision.classification not in {"WALL_CONTINUITY", "PROBABLE_OPENING"}:
            continue
        if not decision.host_wall_continuity:
            continue
        gap = gaps.get(decision.gap_id)
        if gap is None:
            continue
        p1 = (int(round(gap.start_px[0])), int(round(gap.start_px[1])))
        p2 = (int(round(gap.end_px[0])), int(round(gap.end_px[1])))
        cv2.line(barrier, p1, p2, 255, 2, cv2.LINE_8)

    free = (barrier == 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(free, connectivity=4)
    border_labels = set(labels[0, :]) | set(labels[-1, :]) | set(labels[:, 0]) | set(labels[:, -1])
    min_px2 = MIN_SPACE_AREA_M2 * graph.px_per_m * graph.px_per_m
    spaces: list[dict[str, Any]] = []
    for label in range(1, count):
        if label in border_labels:
            continue
        area_px2 = float(stats[label, cv2.CC_STAT_AREA])
        if area_px2 < min_px2:
            continue
        mask = np.where(labels == label, 255, 0).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        epsilon = max(1.0, 0.002 * cv2.arcLength(contour, True))
        approx = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
        if len(approx) < 3:
            continue
        raster = [{"x": float(x), "y": float(y)} for x, y in approx]
        metric = [{"x": float(x) / graph.px_per_m, "y": float(y) / graph.px_per_m} for x, y in approx]
        idx = len(spaces) + 1
        spaces.append({
            "id": f"CALL2_SPACE_{idx:03d}",
            "levelId": graph.level_view_id,
            "name": None,
            "semanticMode": "TOPOLOGY_CALL2_REVIEW",
            "areaM2": area_px2 / (graph.px_per_m * graph.px_per_m),
            "areaSource": "centerline+accepted_gap_continuity",
            "polygon": metric,
            "geometry": {"raster": {"vertices": raster}, "metric": {"vertices": metric}},
            "state": "REVIEW",
            "confirmed": False,
        })
    spaces.sort(key=lambda item: item["areaM2"], reverse=True)
    return spaces


def _build_geometry_probe(
    *,
    base_payload: dict[str, Any],
    graph: SingleLineWallGraph,
    validation: Any,
) -> dict[str, Any]:
    payload = copy.deepcopy(base_payload)
    payload["schemaVersion"] = "SPATIAL_INTERFACE04_QWEN_GEOMETRY_PROBE_V1"
    payload["sourceSchemaVersion"] = base_payload.get("schemaVersion")
    payload["puertas"] = []
    payload["ventanas"] = []
    payload["escaleras"] = []
    payload["elementosRevision"] = []

    for index, region in enumerate(validation.accepted_architectural_regions, start=1):
        family = str(region.get("family_hint") or "UNCERTAIN").upper()
        confidence = float(region.get("confidence", 0.0) or 0.0)
        region_id = f"CALL2_{family}_{index:03d}"
        if family in {"DOOR", "WINDOW", "GARAGE_DOOR"}:
            grounded = _opening_geometry(region, graph)
            item = {
                "id": region_id,
                "levelId": graph.level_view_id,
                "openingType": family,
                "bboxPx": region.get("bbox_px"),
                "confidence": confidence,
                "reason": region.get("reason"),
                "state": "REVIEW",
                "confirmed": False,
                "source": "CALL2_QWEN_HF",
            }
            if grounded:
                item.update(grounded)
            else:
                item["grounding"] = "UNRESOLVED"
            if family == "WINDOW":
                payload["ventanas"].append(item)
            else:
                if family == "GARAGE_DOOR":
                    item["doorSubtype"] = "GARAGE_DOOR"
                payload["puertas"].append(item)
        elif family == "STAIR":
            bbox = _bbox(region.get("bbox_px"))
            if bbox is None:
                continue
            x0, y0, x1, y1 = bbox
            raster = [
                {"x": x0, "y": y0}, {"x": x1, "y": y0},
                {"x": x1, "y": y1}, {"x": x0, "y": y1},
            ]
            metric = [{"x": p["x"] / graph.px_per_m, "y": p["y"] / graph.px_per_m} for p in raster]
            payload["escaleras"].append({
                "id": region_id,
                "levelId": graph.level_view_id,
                "bboxPx": list(bbox),
                "geometry": {"raster": {"vertices": raster}, "metric": {"vertices": metric}},
                "axisStartPx": region.get("stair_axis_start_px"),
                "axisEndPx": region.get("stair_axis_end_px"),
                "direction": region.get("stair_direction") or "UNKNOWN",
                "confidence": confidence,
                "reason": region.get("reason"),
                "state": "REVIEW",
                "confirmed": False,
                "source": "CALL2_QWEN_HF",
            })
        else:
            payload["elementosRevision"].append({"id": region_id, **region})

    payload["espacios"] = _derive_spaces(graph=graph, accepted_review=validation.accepted_review)
    payload["readiness"] = {
        **payload.get("readiness", {}),
        "geometryProbe": True,
        "workflowContinuation": False,
        "quantification": False,
        "missing": [
            "user validation of Call 2 geometry",
            "semantic room names",
            "final opening grounding/confirmation",
            "wall/opening heights",
        ],
    }
    return payload


def _render_comparison(*, view: Any, before: SingleLineWallGraph, after: SingleLineWallGraph, validation: Any, output: Path) -> None:
    original = _decode_png(view.raster_bytes)
    height, width = original.shape[:2]

    def canvas(graph: SingleLineWallGraph) -> np.ndarray:
        result = np.full_like(original, 255)
        for wall in graph.walls:
            p1 = (int(round(wall.start_px[0])), int(round(wall.start_px[1])))
            p2 = (int(round(wall.end_px[0])), int(round(wall.end_px[1])))
            cv2.line(result, p1, p2, (0, 0, 0), 3, cv2.LINE_AA)
        return result

    final = canvas(after)
    for region in validation.accepted_architectural_regions:
        bbox = _bbox(region.get("bbox_px"))
        if bbox is None:
            continue
        x0, y0, x1, y1 = [int(round(v)) for v in bbox]
        cv2.rectangle(final, (x0, y0), (x1, y1), (120, 120, 120), 2)
        cv2.putText(final, str(region.get("family_hint") or "ELEMENT"), (x0, max(14, y0 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (80, 80, 80), 1, cv2.LINE_AA)

    divider = np.full((12, width, 3), 255, dtype=np.uint8)
    composite = np.vstack([
        _header(width, "A - ORIGINAL"), original,
        divider,
        _header(width, "B - BASELINE SPATIALV1"), canvas(before),
        divider,
        _header(width, "C - QWEN/HF VALIDATED RESULT"), final,
    ])
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), composite):
        raise AssertionError(f"No se pudo escribir {output}")


def _metrics(*, before: SingleLineWallGraph, after: SingleLineWallGraph, validation: Any, geometry_payload: dict[str, Any]) -> dict[str, Any]:
    families = Counter(str(item.get("family_hint") or "UNCERTAIN") for item in validation.accepted_architectural_regions)
    gaps = Counter(item.gap_decision.classification for item in validation.gap_items if item.accepted)
    return {
        "walls_before": len(before.walls),
        "walls_after": len(after.walls),
        "wall_delta": len(after.walls) - len(before.walls),
        "raw_deltas": len(validation.delta_items),
        "accepted_deltas": validation.accepted_delta_count,
        "rejected_deltas": validation.rejected_delta_count,
        "accepted_gap_decisions": dict(gaps),
        "accepted_architectural_elements": dict(families),
        "doors_for_04": len(geometry_payload.get("puertas", [])),
        "windows_for_04": len(geometry_payload.get("ventanas", [])),
        "stairs_for_04": len(geometry_payload.get("escaleras", [])),
        "spaces_for_04_review": len(geometry_payload.get("espacios", [])),
        "interior_space_count_final_graph": after.interior_space_count,
    }


def _safe_name(value: str) -> str:
    import re
    return re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")


def _selected_case_level() -> tuple[str, str]:
    case = str(os.getenv("QUANTIA_CALL2_CASE", "miguel_v") or "miguel_v").strip().lower()
    level = _safe_name(str(os.getenv("QUANTIA_CALL2_LEVEL", "planta_baja") or "planta_baja"))
    if case not in {"casa_viri", "miguel_h", "miguel_v"}:
        raise AssertionError(f"QUANTIA_CALL2_CASE no soportado: {case}")
    return case, level


def test_qwen_hf_transport_builds_multimodal_strict_request(monkeypatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "test-token")
    captured: dict[str, Any] = {}

    class Response:
        status_code = 200
        ok = True
        text = ""

        @staticmethod
        def json() -> dict[str, Any]:
            content = {
                "level_view_id": "LV",
                "graph_state": "REVIEW",
                "deltas": [],
                "gap_decisions": [],
                "non_wall_architectural_regions": [],
                "unresolved_regions": [],
                "summary": "ok",
            }
            return {
                "choices": [{"message": {"content": json.dumps(content)}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            }

    def fake_post(url: str, **kwargs: Any) -> Response:
        captured["url"] = url
        captured.update(kwargs)
        return Response()

    monkeypatch.setattr(requests, "post", fake_post)
    provider = HuggingFaceQwenCall2Provider()
    result = provider.analyze(
        prompt="test",
        media_bytes=b"png",
        media_mime_type="image/png",
        response_json_schema=QWEN_CALL2_SCHEMA,
    )
    payload = captured["json"]
    assert captured["url"] == HF_URL
    assert payload["model"] == HF_PRIMARY_MODEL
    assert payload["messages"][0]["content"][1]["type"] == "image_url"
    assert payload["messages"][0]["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert payload["response_format"]["type"] == "json_schema"
    assert payload["response_format"]["json_schema"]["strict"] is True
    assert "GARAGE_DOOR" in payload["response_format"]["json_schema"]["schema"]["properties"]["non_wall_architectural_regions"]["items"]["properties"]["family_hint"]["enum"]
    assert result.data["level_view_id"] == "LV"
    assert provider.last_usage["total_tokens"] == 15


def test_qwen_hf_malformed_json_retries_and_uses_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_test")
    calls: list[str] = []

    class Response:
        status_code = 200
        ok = True
        text = ""

        def __init__(self, payload: dict[str, Any]) -> None:
            self._payload = payload

        def json(self) -> dict[str, Any]:
            return self._payload

    valid_content = {
        "level_view_id": "LV",
        "graph_state": "REVIEW",
        "deltas": [],
        "gap_decisions": [],
        "non_wall_architectural_regions": [],
        "unresolved_regions": [],
        "summary": "ok",
    }

    def fake_post(url: str, **kwargs: Any) -> Response:
        model = kwargs["json"]["model"]
        calls.append(model)
        if model == HF_PRIMARY_MODEL:
            return Response({
                "choices": [{"message": {"content": '{"level_view_id":"LV","graph_state":"REVIEW"'}, "finish_reason": "stop"}],
                "usage": {"total_tokens": 100},
            })
        return Response({
            "choices": [{"message": {"content": json.dumps(valid_content)}, "finish_reason": "stop"}],
            "usage": {"total_tokens": 50},
        })

    monkeypatch.setattr(requests, "post", fake_post)
    monkeypatch.setattr(time, "sleep", lambda *_: None)
    provider = HuggingFaceQwenCall2Provider()
    result = provider.analyze(
        prompt="test",
        media_bytes=b"png",
        media_mime_type="image/png",
        response_json_schema=QWEN_CALL2_SCHEMA,
    )
    assert calls.count(HF_PRIMARY_MODEL) == HF_MAX_ATTEMPTS
    assert HF_FALLBACK_MODEL in calls
    assert result.model == HF_FALLBACK_MODEL
    assert result.fallback_used is True
    assert result.data["level_view_id"] == "LV"
    assert any(item.get("stage") == "json_parse" for item in provider.last_attempts)


def test_qwen_hf_probe_prepares_current_spatialv1_snapshot_offline(tmp_path: Path) -> None:
    view, graph, runtime, post, evidence, stem = _load_snapshot_portable("miguel_v", "planta_baja")
    image = _build_abc_image(view=view, graph=graph)
    prompt, capsule = _build_prompt(view=view, graph=graph)
    decoded = _decode_png(image)
    assert decoded.shape[1] == graph.image_size_px[0]
    assert decoded.shape[0] > graph.image_size_px[1] * 3
    assert capsule["level_view_id"] == graph.level_view_id
    assert len(capsule["walls"]) == len(graph.walls)
    assert len(capsule["logical_gaps"]) == len(graph.logical_gaps)
    assert "GARAGE_DOOR" in prompt and "one centerline" in prompt
    assert runtime.graph.walls and post.filtered_wall_graph.walls
    assert evidence is not None and stem


@pytest.mark.skipif(not RUN_LIVE, reason="Requiere QUANTIA_RUN_QWEN_HF_CALL2=1")
def test_qwen_hf_call2_spatialv1_to_interface04_geometry_live() -> None:
    case, level = _selected_case_level()
    view, graph, runtime, post, evidence, stem = _load_snapshot_portable(case, level)
    provider = HuggingFaceQwenCall2Provider()
    image = _build_abc_image(view=view, graph=graph)
    prompt, capsule = _build_prompt(view=view, graph=graph)
    model_key = hashlib.sha256("|".join(provider.routes).encode()).hexdigest()[:12]
    out = OUTPUT_ROOT / model_key / stem
    out.mkdir(parents=True, exist_ok=True)

    (out / "original.png").write_bytes(view.raster_bytes)
    (out / "call2_input_abc.png").write_bytes(image)
    (out / "prompt.txt").write_text(prompt, encoding="utf-8")
    _write_json(out / "schema.json", QWEN_CALL2_SCHEMA)
    _write_json(out / "state_capsule.json", capsule)
    _write_json(out / "input_graph.json", graph.model_dump(mode="json"))

    try:
        result = provider.analyze(
            prompt=prompt,
            media_bytes=image,
            media_mime_type="image/png",
            response_json_schema=QWEN_CALL2_SCHEMA,
        )
        _write_json(out / "provider_raw.json", result.raw)
        (out / "provider_text.txt").write_text(result.text, encoding="utf-8")
        payload = dict(result.data or {})
        payload.setdefault("gap_decisions", [])
        review = MultimodalWallReview.model_validate(payload)
        if review.level_view_id != graph.level_view_id:
            raise AssertionError("Qwen/HF devolvió level_view_id distinto al solicitado.")
        _write_json(out / "review.json", review.model_dump(mode="json"))

        validation = Call2DeltaValidator().validate(graph=graph, review=review)
        _write_json(out / "pre_export_validation.json", validation.model_dump(mode="json"))

        base_interface = export_review(
            view=view,
            graph=graph,
            runtime=runtime,
            post=post,
            evidence=evidence,
            review=review,
            output=out,
        )
        final_graph = SingleLineWallGraph.model_validate_json((out / "final_graph.json").read_text(encoding="utf-8"))
        final_validation = json.loads((out / "validation.json").read_text(encoding="utf-8"))
        # Reuse the real validation object from the same input for probe enrichment.
        geometry_payload = _build_geometry_probe(
            base_payload=base_interface,
            graph=final_graph,
            validation=validation,
        )
        _write_json(out / "interface04_geometry_probe.json", geometry_payload)
        metrics = _metrics(before=graph, after=final_graph, validation=validation, geometry_payload=geometry_payload)
        _write_json(out / "comparison_metrics.json", metrics)
        _render_comparison(
            view=view,
            before=graph,
            after=final_graph,
            validation=validation,
            output=out / "comparison_original_baseline_qwen.png",
        )

        audit = {
            "probe_version": PROBE_VERSION,
            "schema_version": SCHEMA_VERSION,
            "provider": result.provider,
            "model": result.model,
            "fallback_used": result.fallback_used,
            "attempts": provider.last_attempts,
            "usage": provider.last_usage,
            "level_view_id": graph.level_view_id,
            "case": case,
            "level": level,
            "hashes": {
                "original_sha256": _sha_bytes(view.raster_bytes),
                "input_abc_sha256": _sha_bytes(image),
                "prompt_sha256": _sha_bytes(prompt.encode("utf-8")),
                "schema_sha256": _sha_json(QWEN_CALL2_SCHEMA),
                "input_graph_sha256": _sha_json(graph.model_dump(mode="json")),
                "final_graph_sha256": _sha_json(final_graph.model_dump(mode="json")),
            },
            "metrics": metrics,
            "interface04_base_schema": base_interface.get("schemaVersion"),
            "interface04_geometry_probe_schema": geometry_payload.get("schemaVersion"),
            "final_validation_level_view_id": final_validation.get("level_view_id"),
        }
        _write_json(out / "audit.json", audit)
        _write_json(out / "run_status.json", {"status": "EXPORTED", **audit})

        assert final_graph.walls
        assert base_interface["muros"]
        assert base_interface["readiness"]["quantification"] is False
        assert geometry_payload["readiness"]["quantification"] is False
        assert final_validation.get("level_view_id") == graph.level_view_id
    except Exception as exc:
        # Preserva la evidencia del proveedor incluso cuando HTTP=200 pero el contenido
        # no puede convertirse a JSON. No intenta reparar silenciosamente la geometría.
        if provider.last_raw is not None:
            _write_json(out / "provider_raw_failed.json", provider.last_raw)
        if provider.last_text:
            (out / "provider_text_failed.txt").write_text(provider.last_text, encoding="utf-8")
        message = str(exc)
        if provider.api_key:
            message = message.replace(provider.api_key, "[REDACTED]")
        _write_json(out / "run_status.json", {
            "status": "FAILED",
            "probe_version": PROBE_VERSION,
            "case": case,
            "level": level,
            "attempts": provider.last_attempts,
            "last_finish_reason": provider.last_finish_reason,
            "last_usage": provider.last_usage,
            "error_type": type(exc).__name__,
            "error": message,
        })
        raise


def test_qwen_hf_spatialv1_to_interface04_geometry_offline_contract(tmp_path: Path) -> None:
    """Valida toda la cadena posterior al modelo sin red ni proveedor real."""
    view, graph, runtime, post, evidence, stem = _load_snapshot_portable("miguel_v", "planta_baja")
    assert graph.walls
    wall = graph.walls[0]
    mx = (wall.start_px[0] + wall.end_px[0]) / 2.0
    my = (wall.start_px[1] + wall.end_px[1]) / 2.0
    dx = wall.end_px[0] - wall.start_px[0]
    dy = wall.end_px[1] - wall.start_px[1]
    length = max(math.hypot(dx, dy), 1.0)
    ux, uy = dx / length, dy / length
    span_a = [mx - ux * min(20.0, length * 0.1), my - uy * min(20.0, length * 0.1)]
    span_b = [mx + ux * min(20.0, length * 0.1), my + uy * min(20.0, length * 0.1)]
    x0, y0 = max(0.0, mx - 12.0), max(0.0, my - 12.0)
    x1, y1 = min(graph.image_size_px[0] - 1.0, mx + 12.0), min(graph.image_size_px[1] - 1.0, my + 12.0)

    gap_decisions = []
    if graph.logical_gaps:
        gap = graph.logical_gaps[0]
        gap_decisions.append({
            "gap_id": gap.id,
            "classification": "PROBABLE_OPENING",
            "host_wall_continuity": True,
            "solid_wall_present": False,
            "confidence": 0.95,
            "reason": "synthetic opening continuity",
        })

    review = MultimodalWallReview.model_validate({
        "level_view_id": graph.level_view_id,
        "graph_state": "REVIEW",
        "deltas": [],
        "gap_decisions": gap_decisions,
        "non_wall_architectural_regions": [
            {
                "bbox_px": [x0, y0, x1, y1],
                "family_hint": "DOOR",
                "host_wall_id": wall.id,
                "host_wall_continuity": True,
                "solid_wall_present": False,
                "span_start_px": span_a,
                "span_end_px": span_b,
                "confidence": 0.95,
                "reason": "synthetic door",
            },
            {
                "bbox_px": [20.0, 20.0, 80.0, 100.0],
                "family_hint": "STAIR",
                "host_wall_id": None,
                "host_wall_continuity": False,
                "solid_wall_present": False,
                "stair_axis_start_px": [30.0, 30.0],
                "stair_axis_end_px": [70.0, 90.0],
                "stair_direction": "UP",
                "confidence": 0.90,
                "reason": "synthetic stair",
            },
        ],
        "unresolved_regions": [],
        "summary": "Synthetic full-chain validation",
    })

    validation = Call2DeltaValidator().validate(graph=graph, review=review)
    assert validation.accepted_architectural_regions
    base_interface = export_review(
        view=view,
        graph=graph,
        runtime=runtime,
        post=post,
        evidence=evidence,
        review=review,
        output=tmp_path,
    )
    final_graph = SingleLineWallGraph.model_validate_json((tmp_path / "final_graph.json").read_text(encoding="utf-8"))
    geometry = _build_geometry_probe(base_payload=base_interface, graph=final_graph, validation=validation)
    assert geometry["muros"]
    assert len(geometry["puertas"]) == 1
    assert geometry["puertas"][0]["hostWallId"] == wall.id
    assert geometry["puertas"][0]["widthM"] > 0
    assert len(geometry["escaleras"]) == 1
    assert geometry["readiness"]["quantification"] is False
    assert (tmp_path / "integrity.json").is_file()
    assert json.loads((tmp_path / "integrity.json").read_text(encoding="utf-8"))["valid"] is True
