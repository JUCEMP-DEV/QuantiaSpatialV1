from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

from app.quantia_spatialV1.providers.vision import GeminiSpatialVisionProvider


VISUAL_RESOLVER_VERSION = "VISUAL_CANDIDATE_RESOLVER_V1"

VISUAL_RESOLVER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "resolver_version": {"type": "string"},
        "levels": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "level_name": {"type": "string"},
                    "wall_groups": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "wall_id": {"type": "string"},
                                "candidate_labels": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                                "confidence": {"type": "number"},
                                "orientation": {
                                    "type": "string",
                                    "enum": ["HORIZONTAL", "VERTICAL", "DIAGONAL", "MIXED"],
                                },
                                "reason": {"type": "string"},
                            },
                            "required": [
                                "wall_id",
                                "candidate_labels",
                                "confidence",
                                "orientation",
                                "reason",
                            ],
                        },
                    },
                    "axis_candidates": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "dimension_candidates": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "uncertain_candidates": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "level_notes": {"type": "string"},
                },
                "required": [
                    "level_name",
                    "wall_groups",
                    "axis_candidates",
                    "dimension_candidates",
                    "uncertain_candidates",
                    "level_notes",
                ],
            },
        },
    },
    "required": ["resolver_version", "levels"],
}


@dataclass(frozen=True)
class VisualResolverLevelInput:
    level_name: str
    raster_bytes: bytes
    perimeter: Any
    candidate_graph: Any


@dataclass(frozen=True)
class VisualResolverRequest:
    prompt: str
    image_bytes: bytes
    label_to_candidate_id: dict[str, str]
    candidate_id_to_label: dict[str, str]
    request_hash: str
    candidate_count_by_level: dict[str, int]


@dataclass(frozen=True)
class VisualResolverResult:
    data: dict[str, Any]
    model: str
    provider: str
    fallback_used: bool
    replay_used: bool
    request_hash: str
    label_to_candidate_id: dict[str, str]
    raw_record_path: str


class VisualCandidateResolverV1:
    """Experimento aislado: una resolución visual global sobre candidatos existentes.

    No modifica F01/F01.5/F02 ni el Candidate Discovery. El modelo visual NO
    produce coordenadas: únicamente devuelve etiquetas que ya existen en el overlay.
    """

    def __init__(
        self,
        *,
        provider: GeminiSpatialVisionProvider | None = None,
        history_path: Path | None = None,
    ) -> None:
        self.provider = provider or GeminiSpatialVisionProvider()
        self.history_path = history_path

    def build_request(self, *, levels: Iterable[VisualResolverLevelInput]) -> VisualResolverRequest:
        level_list = list(levels)
        if not level_list:
            raise ValueError("VisualCandidateResolverV1 requiere al menos un nivel.")

        panels: list[np.ndarray] = []
        prompt_sections: list[str] = []
        label_to_candidate_id: dict[str, str] = {}
        candidate_id_to_label: dict[str, str] = {}
        candidate_count_by_level: dict[str, int] = {}

        for level_index, level in enumerate(level_list, 1):
            prefix = f"L{level_index}"
            original = self._decode(level.raster_bytes)
            overlay = original.copy()
            self._draw_perimeter(overlay, level.perimeter)

            rows: list[str] = []
            candidates = list(level.candidate_graph.candidates)
            candidate_count_by_level[level.level_name] = len(candidates)

            # Orden determinista para que un mismo input conserve las mismas etiquetas.
            candidates.sort(key=lambda item: (
                round(float(item.start.y), 3),
                round(float(item.start.x), 3),
                round(float(item.end.y), 3),
                round(float(item.end.x), 3),
                str(item.id),
            ))

            for candidate_index, candidate in enumerate(candidates, 1):
                label = f"{prefix}C{candidate_index:03d}"
                label_to_candidate_id[label] = str(candidate.id)
                candidate_id_to_label[str(candidate.id)] = label
                self._draw_candidate(overlay, candidate=candidate, label=label)

                e = candidate.evidence
                orientation = self._orientation(candidate)
                sources = "+".join(sorted(set(str(x) for x in candidate.source_names))) or "NONE"
                rows.append(
                    "|".join([
                        label,
                        orientation,
                        f"len={float(candidate.length_px):.1f}",
                        f"th={float(candidate.thickness_px):.1f}",
                        f"gen={candidate.generator}",
                        f"src={sources}",
                        f"pair={float(e.pair_overlap):.2f}",
                        f"region={float(e.region_support):.2f}",
                        f"dash={float(e.dashed_penalty):.2f}",
                    ])
                )

            panels.append(
                self._side_by_side_panel(
                    original=original,
                    overlay=overlay,
                    title=f"{level.level_name} — IZQ: ORIGINAL / DER: TODOS LOS CANDIDATOS",
                )
            )
            prompt_sections.append(
                f"\nNIVEL: {level.level_name}\n"
                f"Etiquetas disponibles ({len(rows)}):\n"
                + "\n".join(rows)
            )

        composite = self._stack_panels(panels)
        ok, encoded = cv2.imencode(".png", composite)
        if not ok:
            raise RuntimeError("No se pudo codificar la imagen compuesta del resolver visual.")
        image_bytes = encoded.tobytes()

        prompt = self._build_prompt(prompt_sections)
        schema_bytes = json.dumps(VISUAL_RESOLVER_SCHEMA, sort_keys=True, separators=(",", ":")).encode()
        request_hash = hashlib.sha256(
            prompt.encode("utf-8") + b"\0" + schema_bytes + b"\0" + image_bytes
        ).hexdigest()

        return VisualResolverRequest(
            prompt=prompt,
            image_bytes=image_bytes,
            label_to_candidate_id=label_to_candidate_id,
            candidate_id_to_label=candidate_id_to_label,
            request_hash=request_hash,
            candidate_count_by_level=candidate_count_by_level,
        )

    def resolve(self, *, request: VisualResolverRequest) -> VisualResolverResult:
        force = os.getenv("QUANTIA_VISUAL_RESOLVER_FORCE_CALL", "0").strip() == "1"
        if not force:
            replay = self._find_replay(request.request_hash)
            if replay is not None:
                self._validate_response(replay["normalized"], request=request)
                return VisualResolverResult(
                    data=replay["normalized"],
                    model=str(replay.get("model", "replay")),
                    provider=str(replay.get("provider", "gemini")),
                    fallback_used=bool(replay.get("fallback_used", False)),
                    replay_used=True,
                    request_hash=request.request_hash,
                    label_to_candidate_id=request.label_to_candidate_id,
                    raw_record_path=str(self.history_path or ""),
                )

        result = self.provider.analyze(
            prompt=request.prompt,
            media_bytes=request.image_bytes,
            media_mime_type="image/png",
            response_json_schema=VISUAL_RESOLVER_SCHEMA,
        )
        if not isinstance(result.data, dict):
            raise RuntimeError("Gemini no devolvió un objeto JSON para Visual Resolver V1.")

        self._validate_response(result.data, request=request)
        self._append_history(
            request=request,
            normalized=result.data,
            raw=result.raw,
            model=result.model,
            provider=result.provider,
            fallback_used=result.fallback_used,
        )
        return VisualResolverResult(
            data=result.data,
            model=result.model,
            provider=result.provider,
            fallback_used=result.fallback_used,
            replay_used=False,
            request_hash=request.request_hash,
            label_to_candidate_id=request.label_to_candidate_id,
            raw_record_path=str(self.history_path or ""),
        )

    @staticmethod
    def selected_candidate_labels(data: dict[str, Any]) -> set[str]:
        selected: set[str] = set()
        for level in data.get("levels", []):
            for group in level.get("wall_groups", []):
                selected.update(str(x) for x in group.get("candidate_labels", []))
        return selected

    @staticmethod
    def _build_prompt(prompt_sections: list[str]) -> str:
        return (
            "QUANTIA V2L — VISUAL CANDIDATE RESOLVER V1\n\n"
            "Objetivo: decidir cuáles de los candidatos geométricos YA DETECTADOS pertenecen "
            "a muros interiores reales de la vivienda. No debes inventar coordenadas ni nuevas "
            "líneas. Solo puedes usar las etiquetas visibles en el panel derecho y listadas en "
            "el texto.\n\n"
            "La imagen compuesta muestra por cada nivel: IZQUIERDA = plano original limpio; "
            "DERECHA = exactamente el mismo plano con todos los candidatos etiquetados. "
            "El contorno rojo/negro corresponde al perímetro F02 y NO debe devolverse como muro interior.\n\n"
            "REGLAS:\n"
            "1. Selecciona únicamente muros interiores arquitectónicos reales.\n"
            "2. NO selecciones ejes, retículas, líneas punteadas, cotas, textos, escaleras, puertas, "
            "ventanas, mobiliario, banqueta ni líneas del perímetro F02.\n"
            "3. Un mismo muro puede estar fragmentado: agrupa todas sus candidate_labels en un wall_group.\n"
            "4. Si dos candidatos son alternativas geométricas del mismo muro, elige solo la alternativa "
            "que mejor coincida visualmente con el muro real.\n"
            "5. La ausencia de soporte vectorial NO invalida un muro si visualmente existe en el original.\n"
            "6. No uses el prior del motor: esta prueba mide tu comprensión visual independiente.\n"
            "7. candidate_labels que parezcan ejes van a axis_candidates; cotas a dimension_candidates.\n"
            "8. Si no tienes certeza suficiente, coloca la etiqueta en uncertain_candidates.\n"
            "9. No inventes etiquetas.\n"
            "10. confidence debe estar entre 0 y 1.\n\n"
            "Devuelve exclusivamente el JSON solicitado por el schema."
            + "".join(prompt_sections)
        )

    @staticmethod
    def _decode(data: bytes) -> np.ndarray:
        image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError("No se pudo decodificar raster de LevelView.")
        return image

    @staticmethod
    def _orientation(candidate: Any) -> str:
        dx = abs(float(candidate.end.x) - float(candidate.start.x))
        dy = abs(float(candidate.end.y) - float(candidate.start.y))
        if dx >= 4.0 * max(dy, 1e-6):
            return "H"
        if dy >= 4.0 * max(dx, 1e-6):
            return "V"
        return "D"

    @staticmethod
    def _draw_perimeter(image: np.ndarray, perimeter: Any) -> None:
        vertices = {vertex.id: vertex.point_px for vertex in perimeter.vertices}
        for wall in perimeter.walls:
            a = vertices[wall.start_vertex_id]
            b = vertices[wall.end_vertex_id]
            cv2.line(
                image,
                (int(round(float(a.x))), int(round(float(a.y)))),
                (int(round(float(b.x))), int(round(float(b.y)))),
                (0, 0, 255),
                3,
                cv2.LINE_AA,
            )

    @staticmethod
    def _draw_candidate(image: np.ndarray, *, candidate: Any, label: str) -> None:
        p1 = (int(round(float(candidate.start.x))), int(round(float(candidate.start.y))))
        p2 = (int(round(float(candidate.end.x))), int(round(float(candidate.end.y))))
        dx = abs(p2[0] - p1[0])
        dy = abs(p2[1] - p1[1])
        color = (255, 180, 0) if dx >= dy else (200, 0, 200)
        cv2.line(image, p1, p2, color, 2, cv2.LINE_AA)
        mx = int(round((p1[0] + p2[0]) / 2.0))
        my = int(round((p1[1] + p2[1]) / 2.0))
        cv2.putText(
            image,
            label,
            (mx + 2, my - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.30,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            label,
            (mx + 2, my - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.30,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    @staticmethod
    def _side_by_side_panel(*, original: np.ndarray, overlay: np.ndarray, title: str) -> np.ndarray:
        target_h = max(original.shape[0], overlay.shape[0])
        def pad(image: np.ndarray) -> np.ndarray:
            if image.shape[0] == target_h:
                return image
            bottom = target_h - image.shape[0]
            return cv2.copyMakeBorder(image, 0, bottom, 0, 0, cv2.BORDER_CONSTANT, value=(255,255,255))

        body = np.hstack([pad(original), pad(overlay)])
        header = np.full((42, body.shape[1], 3), 255, dtype=np.uint8)
        cv2.putText(header, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,0,0), 2, cv2.LINE_AA)
        return np.vstack([header, body])

    @staticmethod
    def _stack_panels(panels: list[np.ndarray]) -> np.ndarray:
        width = max(panel.shape[1] for panel in panels)
        normalized: list[np.ndarray] = []
        for panel in panels:
            if panel.shape[1] < width:
                panel = cv2.copyMakeBorder(
                    panel, 0, 0, 0, width - panel.shape[1], cv2.BORDER_CONSTANT, value=(255,255,255)
                )
            normalized.append(panel)
            normalized.append(np.full((20, width, 3), 255, dtype=np.uint8))
        result = np.vstack(normalized[:-1])
        # Limita tamaño sin perder relación de aspecto.
        max_width = 4096
        if result.shape[1] > max_width:
            scale = max_width / float(result.shape[1])
            result = cv2.resize(
                result,
                (max_width, int(round(result.shape[0] * scale))),
                interpolation=cv2.INTER_AREA,
            )
        return result

    def _validate_response(self, data: dict[str, Any], *, request: VisualResolverRequest) -> None:
        allowed = set(request.label_to_candidate_id)
        used: set[str] = set()
        if not isinstance(data.get("levels"), list):
            raise RuntimeError("Visual Resolver: 'levels' no es una lista.")

        for level in data["levels"]:
            for group in level.get("wall_groups", []):
                confidence = float(group.get("confidence", 0.0))
                if not 0.0 <= confidence <= 1.0:
                    raise RuntimeError("Visual Resolver: confidence fuera de [0,1].")
                labels = [str(x) for x in group.get("candidate_labels", [])]
                if not labels:
                    raise RuntimeError("Visual Resolver: wall_group vacío.")
                for label in labels:
                    if label not in allowed:
                        raise RuntimeError(f"Visual Resolver inventó etiqueta desconocida: {label}")
                    if label in used:
                        raise RuntimeError(f"Visual Resolver duplicó etiqueta de muro: {label}")
                    used.add(label)

            for key in ("axis_candidates", "dimension_candidates", "uncertain_candidates"):
                for raw_label in level.get(key, []):
                    label = str(raw_label)
                    if label not in allowed:
                        raise RuntimeError(f"Visual Resolver inventó etiqueta desconocida: {label}")

    def _find_replay(self, request_hash: str) -> dict[str, Any] | None:
        if self.history_path is None or not self.history_path.exists():
            return None
        match: dict[str, Any] | None = None
        for line in self.history_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("request_hash") == request_hash and record.get("status") == "VALID":
                match = record
        return match

    def _append_history(
        self,
        *,
        request: VisualResolverRequest,
        normalized: dict[str, Any],
        raw: dict[str, Any],
        model: str,
        provider: str,
        fallback_used: bool,
    ) -> None:
        if self.history_path is None:
            return
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        prompt_hash = hashlib.sha256(request.prompt.encode("utf-8")).hexdigest()
        schema_hash = hashlib.sha256(
            json.dumps(VISUAL_RESOLVER_SCHEMA, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        image_hash = hashlib.sha256(request.image_bytes).hexdigest()
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "resolver_version": VISUAL_RESOLVER_VERSION,
            "status": "VALID",
            "request_hash": request.request_hash,
            "prompt_hash": prompt_hash,
            "schema_hash": schema_hash,
            "image_hash": image_hash,
            "candidate_count_by_level": request.candidate_count_by_level,
            "provider": provider,
            "model": model,
            "fallback_used": fallback_used,
            "normalized": normalized,
            "raw": raw,
        }
        with self.history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
