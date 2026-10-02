from __future__ import annotations

import hashlib
import json
import math
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

import cv2
import numpy as np

from app.quantia_spatialV1.models.level_view import LevelView
from app.quantia_spatialV1.providers.vision import GeminiSpatialVisionProvider

from .contracts import MultimodalWallReview, SingleLineWallGraph


class MultimodalReviewProvider(Protocol):
    def analyze(
        self,
        *,
        prompt: str,
        media_bytes: bytes,
        media_mime_type: str,
        response_json_schema: dict[str, Any] | None = None,
    ) -> Any: ...


CALL2_PROMPT_VERSION = "CALL2_WALL_REVIEW_V2"
CALL2_SCHEMA_VERSION = "CALL2_WALL_REVIEW_SCHEMA_V2"

# V2 intentionally does not enumerate every logical gap generated upstream.
# Wall continuity is corrected by deltas; visible openings/elements are published
# as architectural candidates for the next layer. This keeps the call focused
# and prevents hundreds of internal bridge hypotheses from consuming context.
from .call2_schema import CALL2_SCHEMA



class WallGraphMultimodalReviewer:
    """Call 2 productiva: corrección del WallGraph contra la fuente visual real.

    Fuente A: `LevelView.raster_bytes`, que LevelViewBuilder obtiene mediante un
    recorte directo del raster fuente de página sin resize ni limpieza.
    Fuente B: WallGraph walls-only filtrado/canónico en el mismo sistema local.

    El objetivo primario es corregir muros por DELTAS. La identificación de
    elementos no-muro es secundaria y sirve como candidato para la siguiente capa.
    """

    HISTORY_ENV = "QUANTIA_ADAPTIVE_CALL2_HISTORY_PATH"

    def __init__(
        self,
        *,
        provider: MultimodalReviewProvider | None = None,
        history_path: str | Path | None = None,
        persist_input_artifact: bool = True,
        enable_replay: bool = True,
    ) -> None:
        self.provider = provider or GeminiSpatialVisionProvider()
        configured = str(os.getenv(self.HISTORY_ENV, "") or "").strip()
        self.history_path = Path(history_path) if history_path else (Path(configured) if configured else None)
        self.persist_input_artifact = bool(persist_input_artifact)
        self.enable_replay = bool(enable_replay)

    @staticmethod
    def _decode(data: bytes) -> np.ndarray:
        image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError("No se pudo decodificar raster fuente del LevelView.")
        return image

    @staticmethod
    def _encode_png(image: np.ndarray) -> bytes:
        ok, encoded = cv2.imencode(".png", image)
        if not ok:
            raise RuntimeError("No se pudo codificar artefacto visual Call 2.")
        return encoded.tobytes()

    @staticmethod
    def _header(width: int, text: str) -> np.ndarray:
        band = np.full((42, width, 3), 255, dtype=np.uint8)
        cv2.putText(
            band,
            text,
            (14, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
        return band

    def build_review_image(self, *, level_view: LevelView, graph: SingleLineWallGraph) -> bytes:
        original = self._decode(level_view.raster_bytes)
        height, width = original.shape[:2]
        if (width, height) != tuple(graph.image_size_px):
            raise RuntimeError(
                "Call 2 requiere que plano fuente y WallGraph compartan exactamente image_size_px."
            )

        clean = np.full_like(original, 255)
        for wall in graph.walls:
            cv2.line(
                clean,
                (int(round(wall.start_px[0])), int(round(wall.start_px[1]))),
                (int(round(wall.end_px[0])), int(round(wall.end_px[1]))),
                (0, 0, 0),
                3,
                cv2.LINE_AA,
            )

        # Un único archivo visual; los headers quedan fuera del sistema local.
        # A y B conservan exactamente el mismo ancho/alto y coordenadas internas.
        divider = np.full((12, width, 3), 255, dtype=np.uint8)
        composite = np.vstack([
            self._header(width, "A - SOURCE LEVELVIEW (ORIGINAL)"),
            original,
            divider,
            self._header(width, "B - QUANTIA WALLGRAPH (WALLS ONLY)"),
            clean,
        ])
        return self._encode_png(composite)

    @staticmethod
    def _filter_summary(filter_context: dict[str, Any] | None) -> dict[str, Any]:
        if not filter_context:
            return {}
        families = Counter()
        for item in filter_context.get("architectural_candidates", []) or []:
            family = str(item.get("family_hint") or "UNCLASSIFIED")
            families[family] += 1
        return {
            "architectural_candidate_count": len(filter_context.get("architectural_candidates", []) or []),
            "excluded_graphic_count": len(filter_context.get("excluded_graphic_candidates", []) or []),
            "unresolved_candidate_count": len(filter_context.get("unresolved_candidates", []) or []),
            "architectural_family_counts": dict(sorted(families.items())),
        }

    @staticmethod
    def state_capsule(
        graph: SingleLineWallGraph,
        *,
        level_view: LevelView | None = None,
    ) -> dict[str, Any]:
        role_code = {"PERIMETER": "P", "DIVIDER": "D", "REVIEW": "R"}
        capsule: dict[str, Any] = {
            "level_view_id": graph.level_view_id,
            "level_name": graph.level_name,
            "image_size_px": list(graph.image_size_px),
            "px_per_m": round(graph.px_per_m, 4),
            # Compact tuple: [id,x1,y1,x2,y2,role].
            "walls": [
                [
                    wall.id,
                    round(wall.start_px[0], 1),
                    round(wall.start_px[1], 1),
                    round(wall.end_px[0], 1),
                    round(wall.end_px[1], 1),
                    role_code.get(wall.role, "R"),
                ]
                for wall in graph.walls
            ],
        }
        if level_view is not None:
            bbox = level_view.source_bbox_px
            capsule["source"] = {
                "document_id": level_view.source_document_id,
                "page": level_view.source_page_number,
                "bbox_page_px": [bbox.x_min, bbox.y_min, bbox.x_max, bbox.y_max],
                "origin": "top-left",
            }
        return capsule

    def build_prompt(
        self,
        *,
        graph: SingleLineWallGraph,
        level_view: LevelView | None = None,
        filter_context: dict[str, Any] | None = None,
    ) -> str:
        capsule = json.dumps(
            self.state_capsule(graph, level_view=level_view),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        filter_summary = self._filter_summary(filter_context)
        filter_note = (
            "\nPREVIOUS FILTER SUMMARY (counts only; NOT truth):\n"
            + json.dumps(filter_summary, ensure_ascii=False, separators=(",", ":"))
            if filter_summary
            else ""
        )

        return f"""
{CALL2_PROMPT_VERSION}

MISSION
Produce the corrected WALLGRAPH for this LevelView by returning ONLY DELTAS against B.
This is not initial extraction and not a redesign of the plan.

VISUAL INPUT — same local coordinates in both panels
A = exact original LevelView crop from the source page; primary truth.
B = current Quantia walls-only WallGraph; proposal to audit, not truth.
Priority: A > geometric evidence > B > previous filter metadata.

WALL REVIEW RULES
1. KEEP is implicit: never repeat correct walls.
2. REMOVE_WALL when B is not a physical wall in A: axis/dimension, stair/railing, roof/terrace projection, slab edge, glazing/cancel, furniture, symbol, hatch/grid or other graphic.
3. ADD_WALL only when A visibly contains a physical wall omitted by B. Require visible wall thickness/band, clear wall convention, or a convincing junction/boundary consistent with nearby walls. Never add a wall only because an axis aligns or symmetry suggests it.
4. EXTEND/TRIM/REPOSITION only to match visible wall endpoints/junctions in A.
5. MERGE/SPLIT only when B represents the same physical wall incorrectly.
6. Perimeter is NOT protected. It may be corrected or removed.
7. Do NOT draw solid wall through a visible opening. Door, window, floor-to-ceiling aluminum glazing, shutter/closure or similar opening stays physically open in the WallGraph.
8. Parapet with real wall thickness = WALL. Railing/guardrail/handrail = non-wall architectural element.
9. Roof/terrace lines are not walls unless A clearly shows an actual wall/parapet.
10. If evidence is insufficient, do not guess: report unresolved_regions.

NEXT-LAYER CANDIDATES
After wall corrections, report only visually clear non-wall architectural regions relevant to the next layer or to a wall/opening decision: DOOR, WINDOW, FLOOR_TO_CEILING_GLAZING, RAILING_GUARDRAIL, STAIR, STAIR_HANDRAIL, ROOF_COVER, TERRACE, OPENING_CLOSURE, SLAB_EDGE_LEVEL_CHANGE, COLUMN, OTHER, UNCERTAIN.
This is candidate identification, not final F04 classification/dimensioning.
If an opening candidate has a clear host wall, include host_wall_id. For door/window/glazing: host_wall_continuity=true and solid_wall_present=false.

OUTPUT DISCIPLINE
- Coordinates are LevelView pixels: origin top-left, x right, y down.
- WALL tuple format in STATE: [id,x1,y1,x2,y2,role], role P/D/R.
- Reasons must be short and evidence-based (prefer <=12 words).
- summary: maximum two short sentences.
- Do not enumerate unchanged walls.
- Do not output internal logical-gap hypotheses; continuity corrections belong in deltas.

STATE
{capsule}{filter_note}
""".strip()

    def review(
        self,
        *,
        level_view: LevelView,
        graph: SingleLineWallGraph,
        filter_context: dict[str, Any] | None = None,
    ) -> MultimodalWallReview:
        image_bytes = self.build_review_image(level_view=level_view, graph=graph)
        prompt = self.build_prompt(graph=graph, level_view=level_view, filter_context=filter_context)
        prompt_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        schema_sha256 = self._sha_json(CALL2_SCHEMA)
        image_sha256 = hashlib.sha256(image_bytes).hexdigest()
        graph_sha256 = self._sha_json(graph.model_dump(mode="json"))

        if self.enable_replay:
            cached = self._load_cached_review(
                level_view_id=graph.level_view_id,
                prompt_sha256=prompt_sha256,
                schema_sha256=schema_sha256,
                image_sha256=image_sha256,
            )
            if cached is not None:
                self._append_history({
                    "event": "REPLAY_HIT",
                    "level_view_id": graph.level_view_id,
                    "prompt_version": CALL2_PROMPT_VERSION,
                    "schema_version": CALL2_SCHEMA_VERSION,
                    "prompt_sha256": prompt_sha256,
                    "schema_sha256": schema_sha256,
                    "image_sha256": image_sha256,
                    "graph_sha256": graph_sha256,
                })
                return cached

        call_id = f"{graph.level_view_id}__CALL2V2__{uuid4().hex[:12]}"
        input_path = self._persist_input_image(call_id=call_id, image_bytes=image_bytes)

        prompt_chars = len(prompt)
        self._append_history({
            "event": "STARTED",
            "call_id": call_id,
            "prompt_version": CALL2_PROMPT_VERSION,
            "schema_version": CALL2_SCHEMA_VERSION,
            "level_view_id": graph.level_view_id,
            "level_name": graph.level_name,
            "source_document_id": level_view.source_document_id,
            "source_page_number": level_view.source_page_number,
            "source_bbox_px": level_view.source_bbox_px.model_dump(mode="json"),
            "source_raster_sha256": hashlib.sha256(level_view.raster_bytes).hexdigest(),
            "source_semantics": "direct LevelView crop of source page raster; no resize/cleanup",
            "prompt": prompt,
            "prompt_chars": prompt_chars,
            "approx_prompt_tokens_chars_div4": math.ceil(prompt_chars / 4),
            "prompt_sha256": prompt_sha256,
            "schema": CALL2_SCHEMA,
            "schema_sha256": schema_sha256,
            "image_sha256": image_sha256,
            "input_artifact_path": str(input_path) if input_path else None,
            "graph_sha256": graph_sha256,
            "wall_count": len(graph.walls),
            "filter_summary": self._filter_summary(filter_context),
        })
        try:
            result = self.provider.analyze(
                prompt=prompt,
                media_bytes=image_bytes,
                media_mime_type="image/png",
                response_json_schema=CALL2_SCHEMA,
            )
            payload = result.data
            if not isinstance(payload, dict):
                raise RuntimeError("Call 2 no devolvió objeto JSON.")
            # Schema V2 no solicita gap_decisions; contrato interno los mantiene
            # opcionales para compatibilidad con validación/aplicador existentes.
            payload.setdefault("gap_decisions", [])
            review = MultimodalWallReview.model_validate(payload)
            if review.level_view_id != graph.level_view_id:
                raise RuntimeError("Call 2 devolvió level_view_id distinto al solicitado.")
            self._append_history({
                "event": "SUCCEEDED",
                "call_id": call_id,
                "level_view_id": graph.level_view_id,
                "provider": result.provider,
                "model": result.model,
                "fallback_used": result.fallback_used,
                "normalized_response": review.model_dump(mode="json"),
                "provider_raw_response": self._sanitize(result.raw),
            })
            return review
        except Exception as exc:
            self._append_history({
                "event": "FAILED",
                "call_id": call_id,
                "level_view_id": graph.level_view_id,
                "error_type": type(exc).__name__,
                "error": str(exc),
            })
            raise

    def _load_cached_review(
        self,
        *,
        level_view_id: str,
        prompt_sha256: str,
        schema_sha256: str,
        image_sha256: str,
    ) -> MultimodalWallReview | None:
        if self.history_path is None or not self.history_path.exists():
            return None

        started: dict[str, dict[str, Any]] = {}
        succeeded: dict[str, dict[str, Any]] = {}
        try:
            rows = self.history_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return None

        for raw in rows:
            raw = raw.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            call_id = str(row.get("call_id") or "")
            if not call_id:
                continue
            event = row.get("event")
            if event == "STARTED":
                started[call_id] = row
            elif event == "SUCCEEDED":
                succeeded[call_id] = row

        for call_id, start in reversed(list(started.items())):
            if start.get("level_view_id") != level_view_id:
                continue
            if start.get("prompt_version") != CALL2_PROMPT_VERSION:
                continue
            if start.get("schema_version") != CALL2_SCHEMA_VERSION:
                continue
            if start.get("prompt_sha256") != prompt_sha256:
                continue
            if start.get("schema_sha256") != schema_sha256:
                continue
            if start.get("image_sha256") != image_sha256:
                continue
            success = succeeded.get(call_id)
            if not success:
                continue
            payload = success.get("normalized_response")
            if not isinstance(payload, dict):
                continue
            try:
                review = MultimodalWallReview.model_validate(payload)
            except Exception:
                continue
            if review.level_view_id == level_view_id:
                return review
        return None

    def _persist_input_image(self, *, call_id: str, image_bytes: bytes) -> Path | None:
        if not self.persist_input_artifact or self.history_path is None:
            return None
        target_dir = self.history_path.parent / "_call2_inputs"
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / f"{call_id}.png"
        target.write_bytes(image_bytes)
        return target

    def _append_history(self, payload: dict[str, Any]) -> None:
        if self.history_path is None:
            return
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        row = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), **self._sanitize(payload)}
        with self.history_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())

    @staticmethod
    def _sha_json(value: Any) -> str:
        raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    @classmethod
    def _sanitize(cls, value: Any) -> Any:
        secrets = {
            "api_key", "apikey", "x-goog-api-key", "authorization",
            "access_token", "refresh_token", "thoughtsignature",
        }
        if isinstance(value, dict):
            return {
                str(k): ("[REDACTED]" if str(k).lower() in secrets else cls._sanitize(v))
                for k, v in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [cls._sanitize(v) for v in value]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)
