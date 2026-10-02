from __future__ import annotations

import hashlib
import math
from collections import Counter
from typing import Iterable, Sequence

import cv2
import numpy as np

from app.quantia_spatialV1.core.models.evidence import RawEvidence
from app.quantia_spatialV1.core.models.level_view import LevelView

from .drawing_model import (
    DrawingBBox,
    DrawingCurve,
    DrawingLine,
    DrawingModel,
    DrawingModelDiagnostics,
    DrawingPoint,
    DrawingRasterLayer,
    DrawingRegion,
    DrawingText,
    SemanticObservation,
)


class DrawingModelBuilder:
    """Normaliza F01.5 + raster local en un DrawingModel único.

    No clasifica muros. Las detecciones LSD/region son fuentes nuevas de evidencia
    y se conservan separadas de OPENCV/PyMuPDF.
    """

    def build(
        self,
        *,
        level_view: LevelView,
        evidence: Sequence[RawEvidence],
    ) -> DrawingModel:
        raw = [item for item in evidence if item.level_view_id == level_view.id]
        lines: list[DrawingLine] = []
        regions: list[DrawingRegion] = []
        curves: list[DrawingCurve] = []
        texts: list[DrawingText] = []
        semantics: list[SemanticObservation] = []
        source_counts: Counter[str] = Counter()

        for item in raw:
            source_counts[item.source] += 1
            self._from_raw_evidence(
                item=item,
                level_view=level_view,
                lines=lines,
                regions=regions,
                curves=curves,
                texts=texts,
                semantics=semantics,
            )

        gray = self._decode_gray(level_view)
        ink_mask, wall_region_mask = self._raster_masks(gray)
        lsd_lines = self._lsd_lines(level_view=level_view, gray=gray)
        region_lines = self._region_centerlines(
            level_view=level_view,
            mask=wall_region_mask,
        )
        lines.extend(lsd_lines)
        lines.extend(region_lines)

        lines = self._dedupe_exact(lines)

        raster_layers = [
            DrawingRasterLayer(
                id=f"{level_view.id}__INK_MASK",
                kind="INK_MASK",
                width_px=level_view.raster_width_px,
                height_px=level_view.raster_height_px,
                png_bytes=self._encode_png(ink_mask),
            ),
            DrawingRasterLayer(
                id=f"{level_view.id}__WALL_REGION_MASK",
                kind="WALL_REGION_MASK",
                width_px=level_view.raster_width_px,
                height_px=level_view.raster_height_px,
                png_bytes=self._encode_png(wall_region_mask),
            ),
        ]

        return DrawingModel(
            level_view_id=level_view.id,
            width_px=level_view.raster_width_px,
            height_px=level_view.raster_height_px,
            lines=lines,
            regions=regions,
            curves=curves,
            texts=texts,
            semantic_observations=semantics,
            raster_layers=raster_layers,
            diagnostics=DrawingModelDiagnostics(
                raw_evidence_count=len(raw),
                normalized_line_count=len(lines),
                normalized_region_count=len(regions),
                normalized_curve_count=len(curves),
                text_count=len(texts),
                semantic_observation_count=len(semantics),
                lsd_line_count=len(lsd_lines),
                region_centerline_count=len(region_lines),
                source_counts=dict(sorted(source_counts.items())),
            ),
        )

    def _from_raw_evidence(
        self,
        *,
        item: RawEvidence,
        level_view: LevelView,
        lines: list[DrawingLine],
        regions: list[DrawingRegion],
        curves: list[DrawingCurve],
        texts: list[DrawingText],
        semantics: list[SemanticObservation],
    ) -> None:
        geom = item.geometry

        if (
            str(item.metadata.get("primitive_type") or "").upper() == "CUBIC_BEZIER"
            and geom.bbox_px is not None
        ):
            local_points = self._bezier_local_points(item=item, level_view=level_view)
            if local_points is not None:
                raw_bbox = geom.bbox_px
                curves.append(
                    DrawingCurve(
                        id=item.id,
                        control_points=local_points,
                        bbox=DrawingBBox(
                            x_min=float(raw_bbox.x_min),
                            y_min=float(raw_bbox.y_min),
                            x_max=float(raw_bbox.x_max),
                            y_max=float(raw_bbox.y_max),
                        ),
                        evidence_ids=[item.id],
                        sources=[item.source],
                        confidence=item.confidence,
                        metadata=dict(item.metadata),
                    )
                )

        if geom.geometry_type in {"SEGMENT", "POLYLINE"} and len(geom.points) >= 2:
            pairs = (
                [(geom.points[0], geom.points[1])]
                if geom.geometry_type == "SEGMENT"
                else list(zip(geom.points[:-1], geom.points[1:]))
            )
            for index, (a, b) in enumerate(pairs):
                dx = float(b.x - a.x)
                dy = float(b.y - a.y)
                length = math.hypot(dx, dy)
                if length <= 1e-6:
                    continue
                metadata = dict(item.metadata)
                stroke = self._extract_numeric(
                    metadata,
                    ("stroke_width_px", "stroke_width", "line_width", "width"),
                )
                dashed = self._extract_dashed(metadata)
                lines.append(
                    DrawingLine(
                        id=(item.id if len(pairs) == 1 else f"{item.id}__S{index:03d}"),
                        start=DrawingPoint(x=float(a.x), y=float(a.y)),
                        end=DrawingPoint(x=float(b.x), y=float(b.y)),
                        length_px=length,
                        angle_deg=(math.degrees(math.atan2(dy, dx)) % 180.0),
                        stroke_width_px=stroke,
                        dashed=dashed,
                        evidence_ids=[item.id],
                        sources=[item.source],
                        confidence=item.confidence,
                        metadata=metadata,
                    )
                )

        if item.kind == "RASTER_CONTOUR" and geom.bbox_px is not None:
            bbox = geom.bbox_px
            area = self._extract_numeric(item.metadata, ("area_px2", "area")) or 0.0
            regions.append(
                DrawingRegion(
                    id=item.id,
                    bbox=DrawingBBox(
                        x_min=float(bbox.x_min),
                        y_min=float(bbox.y_min),
                        x_max=float(bbox.x_max),
                        y_max=float(bbox.y_max),
                    ),
                    area_px2=max(0.0, float(area)),
                    evidence_ids=[item.id],
                    sources=[item.source],
                    metadata=dict(item.metadata),
                )
            )

        if item.text and item.text.strip():
            bbox = None
            if geom.bbox_px is not None:
                raw_bbox = geom.bbox_px
                bbox = DrawingBBox(
                    x_min=float(raw_bbox.x_min),
                    y_min=float(raw_bbox.y_min),
                    x_max=float(raw_bbox.x_max),
                    y_max=float(raw_bbox.y_max),
                )
            if item.source in {"PYMUPDF", "OCR"}:
                texts.append(
                    DrawingText(
                        id=item.id,
                        text=item.text.strip(),
                        bbox=bbox,
                        evidence_ids=[item.id],
                        sources=[item.source],
                        confidence=item.confidence,
                    )
                )

        if item.source == "GEMINI":
            # F03 Element Context Isolation V4: GeminiEvidenceAdapter publica la
            # categoría normalizada en `semantic_category`. Las versiones previas
            # la ignoraban y terminaban etiquetando toda observación como
            # GEMINI_OBSERVATION, dejando inactiva la semántica DOOR/WINDOW/etc.
            family = str(
                item.metadata.get("semantic_category")
                or item.metadata.get("semantic_family")
                or item.metadata.get("family")
                or item.metadata.get("type")
                or item.kind
            )
            bbox = None
            if geom.bbox_px is not None:
                raw_bbox = geom.bbox_px
                bbox = DrawingBBox(
                    x_min=float(raw_bbox.x_min),
                    y_min=float(raw_bbox.y_min),
                    x_max=float(raw_bbox.x_max),
                    y_max=float(raw_bbox.y_max),
                )
            semantics.append(
                SemanticObservation(
                    id=item.id,
                    family=family,
                    text=item.text,
                    bbox=bbox,
                    confidence=item.confidence,
                    payload=dict(item.metadata),
                )
            )

    @staticmethod
    def _bezier_local_points(*, item: RawEvidence, level_view: LevelView) -> list[DrawingPoint] | None:
        raw_points = item.metadata.get("control_points_page_px")
        if not isinstance(raw_points, list) or len(raw_points) != 4:
            return None
        result: list[DrawingPoint] = []
        for raw in raw_points:
            if not isinstance(raw, dict):
                return None
            try:
                page_x = float(raw["x"])
                page_y = float(raw["y"])
            except (KeyError, TypeError, ValueError):
                return None
            local_x = page_x - float(level_view.source_bbox_px.x_min)
            local_y = page_y - float(level_view.source_bbox_px.y_min)
            # Se conserva la curva solo si al menos su control point cae en un
            # margen razonable del LevelView. El bbox ya fue recortado por F01.5.
            margin = max(level_view.raster_width_px, level_view.raster_height_px) * 0.05
            if not (-margin <= local_x <= level_view.raster_width_px + margin):
                return None
            if not (-margin <= local_y <= level_view.raster_height_px + margin):
                return None
            result.append(DrawingPoint(x=local_x, y=local_y))
        return result

    @staticmethod
    def _decode_gray(level_view: LevelView) -> np.ndarray:
        data = np.frombuffer(level_view.raster_bytes, dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError("No se pudo decodificar LevelView.raster_bytes.")
        return image

    @staticmethod
    def _raster_masks(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        # Se conserva el ink crudo y una versión de región muy ligera. No intenta
        # decidir paredes; únicamente cierra antialiasing/gaps pequeños.
        wall_region = cv2.morphologyEx(
            ink,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
            iterations=1,
        )
        return ink, wall_region

    def _lsd_lines(self, *, level_view: LevelView, gray: np.ndarray) -> list[DrawingLine]:
        detector = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
        result = detector.detect(gray)[0]
        if result is None:
            return []

        # OpenCV no garantiza la misma forma del ndarray entre builds/plataformas.
        # Se han observado tanto (N, 1, 4) como (N, 4). Normalizar a (-1, 4)
        # evita depender de una forma concreta sin alterar los valores LSD.
        raw_segments = np.asarray(result, dtype=np.float32)
        if raw_segments.size == 0:
            return []
        if raw_segments.size % 4 != 0:
            raise ValueError(
                f"OpenCV LSD devolvió una forma inesperada: {raw_segments.shape}."
            )
        raw_segments = raw_segments.reshape(-1, 4)

        minimum = max(10.0, min(gray.shape[:2]) * 0.012)
        lines: list[DrawingLine] = []
        for index, raw in enumerate(raw_segments):
            x1, y1, x2, y2 = map(float, raw)
            length = math.hypot(x2 - x1, y2 - y1)
            if length < minimum:
                continue
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180.0
            ident = hashlib.sha1(
                f"{level_view.id}|LSD|{index}|{x1:.2f}|{y1:.2f}|{x2:.2f}|{y2:.2f}".encode()
            ).hexdigest()[:16]
            lines.append(
                DrawingLine(
                    id=f"{level_view.id}__LSD__{ident}",
                    start=DrawingPoint(x=x1, y=y1),
                    end=DrawingPoint(x=x2, y=y2),
                    length_px=length,
                    angle_deg=angle,
                    evidence_ids=[],
                    sources=["LSD_LOCAL"],
                    confidence=None,
                    metadata={"technical_source": "opencv_lsd"},
                )
            )
        return lines

    def _region_centerlines(
        self,
        *,
        level_view: LevelView,
        mask: np.ndarray,
    ) -> list[DrawingLine]:
        height, width = mask.shape[:2]
        minimum = min(width, height)
        run = max(9, int(round(minimum * 0.016)))
        result: list[DrawingLine] = []

        for orientation, kernel in (
            ("H", cv2.getStructuringElement(cv2.MORPH_RECT, (run, 1))),
            ("V", cv2.getStructuringElement(cv2.MORPH_RECT, (1, run))),
        ):
            directional = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            count, _, stats, _ = cv2.connectedComponentsWithStats(directional, 8)
            for label in range(1, count):
                x, y, w, h, area = map(int, stats[label])
                longitudinal = w if orientation == "H" else h
                transverse = h if orientation == "H" else w
                if longitudinal < max(12, int(minimum * 0.018)):
                    continue
                if transverse <= 0:
                    continue
                # No se filtra por "parece muro"; esta línea solo es evidencia
                # regional y el solver decidirá si aporta coherencia global.
                if orientation == "H":
                    start = DrawingPoint(x=float(x), y=float(y + h / 2.0))
                    end = DrawingPoint(x=float(x + w), y=float(y + h / 2.0))
                    angle = 0.0
                else:
                    start = DrawingPoint(x=float(x + w / 2.0), y=float(y))
                    end = DrawingPoint(x=float(x + w / 2.0), y=float(y + h))
                    angle = 90.0
                ident = hashlib.sha1(
                    f"{level_view.id}|REGION|{orientation}|{x}|{y}|{w}|{h}|{area}".encode()
                ).hexdigest()[:16]
                result.append(
                    DrawingLine(
                        id=f"{level_view.id}__REGION__{ident}",
                        kind="REGION_CENTERLINE",
                        start=start,
                        end=end,
                        length_px=float(longitudinal),
                        angle_deg=angle,
                        evidence_ids=[],
                        sources=["REGION_LOCAL"],
                        confidence=None,
                        metadata={
                            "component_area_px2": area,
                            "estimated_thickness_px": float(transverse),
                            "orientation": orientation,
                        },
                    )
                )
        return result

    @staticmethod
    def _dedupe_exact(lines: Iterable[DrawingLine]) -> list[DrawingLine]:
        seen: set[tuple] = set()
        output: list[DrawingLine] = []
        for item in lines:
            key = (
                item.kind,
                tuple(item.sources),
                round(min(item.start.x, item.end.x), 2),
                round(min(item.start.y, item.end.y), 2),
                round(max(item.start.x, item.end.x), 2),
                round(max(item.start.y, item.end.y), 2),
            )
            if key in seen:
                continue
            seen.add(key)
            output.append(item)
        return output

    @staticmethod
    def _extract_numeric(metadata: dict, keys: Iterable[str]) -> float | None:
        stack = [metadata]
        while stack:
            current = stack.pop()
            if not isinstance(current, dict):
                continue
            for key in keys:
                value = current.get(key)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    return float(value)
            stack.extend(value for value in current.values() if isinstance(value, dict))
        return None

    @staticmethod
    def _extract_dashed(metadata: dict) -> bool:
        text = str(metadata).lower()
        return any(token in text for token in ("dash", "dashed", "dotted", "[3", "[2"))

    @staticmethod
    def _encode_png(mask: np.ndarray) -> bytes:
        ok, encoded = cv2.imencode(".png", mask)
        if not ok:
            raise RuntimeError("No se pudo codificar máscara temporal.")
        return encoded.tobytes()
