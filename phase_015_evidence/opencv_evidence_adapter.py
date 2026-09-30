from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import cv2
import numpy as np

from app.quantia_spatialV1.models.evidence import (
    EvidenceGeometry,
    RawEvidence,
)
from app.quantia_spatialV1.models.level_view import (
    LevelView,
    PixelBBox,
    PixelPoint,
)

# ============================================================
# CONFIGURACIÓN
# ============================================================


@dataclass(slots=True)
class OpenCVEvidenceConfig:
    """
    Parámetros técnicos heredados del extractor legacy.

    Todos los parámetros geométricos dependen del tamaño
    real del LevelView.

    No contienen medidas arquitectónicas ni conversiones
    px -> m.
    """

    canny_sigma: float = 0.33

    directional_kernel_ratio: float = 0.025

    hough_threshold_ratio: float = 0.018

    min_line_length_ratio: float = 0.025

    max_line_gap_ratio: float = 0.006

    axis_tolerance_ratio: float = 0.0025

    merge_gap_ratio: float = 0.0075

    minimum_contour_area_ratio: float = 0.0001


# ============================================================
# MODELOS INTERNOS
# ============================================================


@dataclass(slots=True)
class _RasterSegment:
    id: str

    x1: float
    y1: float
    x2: float
    y2: float

    length_px: float

    orientation: str

    angle_deg: float

    source: str

    merged_from: list[str]


@dataclass(slots=True)
class _RasterIntersection:
    id: str

    x: float
    y: float

    horizontal_segment_id: str

    vertical_segment_id: str


@dataclass(slots=True)
class _RasterContour:
    id: str

    area_px2: float

    perimeter_px: float

    points: list[tuple[int, int]]

    x: int
    y: int

    width: int
    height: int

    hierarchy_next: int | None
    hierarchy_previous: int | None
    hierarchy_child: int | None
    hierarchy_parent: int | None


# ============================================================
# ADAPTER
# ============================================================


class OpenCVEvidenceAdapter:
    """
    Fase 01.5 — extracción geométrica raster cruda.

    Entrada:
        LevelView

    Fuente:
        level_view.raster_bytes

    Extrae:
        - segmentos Hough crudos;
        - continuidad geométrica derivada;
        - intersecciones;
        - contornos.

    Toda la geometría permanece en coordenadas LOCALES
    del LevelView.

    No:
        - identifica muros;
        - identifica perímetros;
        - identifica ejes;
        - identifica espacios;
        - identifica puertas/ventanas;
        - convierte px -> m;
        - elimina evidencia por semántica;
        - confirma elementos.

    El análisis de deskew se conserva únicamente como
    diagnóstico. Fase 01.5 NO rota el LevelView porque eso
    rompería su transformación exacta con la página original.
    """

    NUMERIC_TOLERANCE = 1e-6

    def __init__(
        self,
        *,
        config: OpenCVEvidenceConfig | None = None,
    ) -> None:

        self.config = config or OpenCVEvidenceConfig()

    # ========================================================
    # API
    # ========================================================

    def extract(
        self,
        *,
        level_view: LevelView,
    ) -> list[RawEvidence]:

        image = self._decode_image(level_view.raster_bytes)

        height, width = image.shape[:2]

        if width != level_view.raster_width_px or height != level_view.raster_height_px:
            raise ValueError(
                "Las dimensiones reales del raster no coinciden con LevelView."
            )

        # ----------------------------------------------------
        # ESCALA DE GRISES
        # ----------------------------------------------------

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY,
        )

        # ----------------------------------------------------
        # BINARIZACIÓN OTSU
        # ----------------------------------------------------

        otsu_threshold, binary = cv2.threshold(
            gray,
            0,
            255,
            (cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU),
        )

        # ----------------------------------------------------
        # DIAGNÓSTICO DE INCLINACIÓN
        # ----------------------------------------------------

        deskew_angle = self._estimate_deskew_angle(binary)

        minimum_dimension = min(
            width,
            height,
        )

        # ----------------------------------------------------
        # PARÁMETROS EFECTIVOS
        # ----------------------------------------------------

        horizontal_kernel_length = max(
            3,
            int(round(width * self.config.directional_kernel_ratio)),
        )

        vertical_kernel_length = max(
            3,
            int(round(height * self.config.directional_kernel_ratio)),
        )

        min_line_length = max(
            8,
            int(round(minimum_dimension * self.config.min_line_length_ratio)),
        )

        max_line_gap = max(
            2,
            int(round(minimum_dimension * self.config.max_line_gap_ratio)),
        )

        hough_threshold = max(
            10,
            int(round(minimum_dimension * self.config.hough_threshold_ratio)),
        )

        axis_tolerance = max(
            1.0,
            (minimum_dimension * self.config.axis_tolerance_ratio),
        )

        merge_gap = max(
            2.0,
            (minimum_dimension * self.config.merge_gap_ratio),
        )

        # ----------------------------------------------------
        # MÁSCARAS DIRECCIONALES
        # ----------------------------------------------------

        horizontal_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (
                horizontal_kernel_length,
                1,
            ),
        )

        vertical_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (
                1,
                vertical_kernel_length,
            ),
        )

        horizontal_mask = cv2.morphologyEx(
            binary,
            cv2.MORPH_OPEN,
            horizontal_kernel,
        )

        vertical_mask = cv2.morphologyEx(
            binary,
            cv2.MORPH_OPEN,
            vertical_kernel,
        )

        line_mask = cv2.bitwise_or(
            horizontal_mask,
            vertical_mask,
        )

        line_mask = cv2.morphologyEx(
            line_mask,
            cv2.MORPH_CLOSE,
            np.ones(
                (3, 3),
                dtype=np.uint8,
            ),
        )

        # ----------------------------------------------------
        # CANNY
        # ----------------------------------------------------

        (
            canny_low,
            canny_high,
        ) = self._automatic_canny_thresholds(gray)

        edges = cv2.Canny(
            line_mask,
            canny_low,
            canny_high,
            apertureSize=3,
            L2gradient=True,
        )

        # ----------------------------------------------------
        # HOUGH
        # ----------------------------------------------------

        hough = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=(np.pi / 180.0),
            threshold=hough_threshold,
            minLineLength=min_line_length,
            maxLineGap=max_line_gap,
        )

        raw_segments = self._build_hough_segments(hough)

        raw_horizontal = [
            segment for segment in raw_segments if (segment.orientation == "horizontal")
        ]

        raw_vertical = [
            segment for segment in raw_segments if (segment.orientation == "vertical")
        ]

        # ----------------------------------------------------
        # CONTINUIDAD DERIVADA
        # ----------------------------------------------------

        merged_horizontal = self._merge_horizontal_segments(
            raw_horizontal,
            axis_tolerance=axis_tolerance,
            merge_gap=merge_gap,
        )

        merged_vertical = self._merge_vertical_segments(
            raw_vertical,
            axis_tolerance=axis_tolerance,
            merge_gap=merge_gap,
        )

        # ----------------------------------------------------
        # INTERSECCIONES
        # ----------------------------------------------------

        intersections = self._find_intersections(
            horizontal=merged_horizontal,
            vertical=merged_vertical,
            tolerance=axis_tolerance,
        )

        # ----------------------------------------------------
        # CONTORNOS
        # ----------------------------------------------------

        contours = self._extract_contours(
            line_mask,
            image_width=width,
            image_height=height,
        )

        # ----------------------------------------------------
        # DIAGNÓSTICOS COMUNES
        # ----------------------------------------------------

        dark_pixel_ratio = float(np.count_nonzero(binary)) / float(binary.size)

        common_metadata = {
            "source_document_id": (level_view.source_document_id),
            "source_page_number": (level_view.source_page_number),
            "level_view_width_px": (width),
            "level_view_height_px": (height),
            "opencv": {
                "otsu_threshold": (float(otsu_threshold)),
                "canny_low": (canny_low),
                "canny_high": (canny_high),
                "directional_kernel_horizontal": (horizontal_kernel_length),
                "directional_kernel_vertical": (vertical_kernel_length),
                "hough_threshold": (hough_threshold),
                "min_line_length_px": (min_line_length),
                "max_line_gap_px": (max_line_gap),
                "axis_tolerance_px": (axis_tolerance),
                "merge_gap_px": (merge_gap),
                "deskew_angle_deg": (deskew_angle),
                "deskew_applied": (False),
                "dark_pixel_ratio": (dark_pixel_ratio),
            },
        }

        result: list[RawEvidence] = []

        # ----------------------------------------------------
        # PUBLICAR SEGMENTOS CRUDOS
        # ----------------------------------------------------

        for segment in raw_segments:
            result.append(
                self._segment_to_evidence(
                    level_view=level_view,
                    segment=segment,
                    stage="raw_hough",
                    common_metadata=common_metadata,
                )
            )

        # ----------------------------------------------------
        # PUBLICAR CONTINUIDAD DERIVADA
        # ----------------------------------------------------

        for segment in [
            *merged_horizontal,
            *merged_vertical,
        ]:
            result.append(
                self._segment_to_evidence(
                    level_view=level_view,
                    segment=segment,
                    stage="continuity_merged",
                    common_metadata=common_metadata,
                )
            )

        # ----------------------------------------------------
        # PUBLICAR INTERSECCIONES
        # ----------------------------------------------------

        for intersection in intersections:
            result.append(
                self._intersection_to_evidence(
                    level_view=level_view,
                    intersection=intersection,
                    common_metadata=common_metadata,
                )
            )

        # ----------------------------------------------------
        # PUBLICAR CONTORNOS
        # ----------------------------------------------------

        for contour in contours:
            result.append(
                self._contour_to_evidence(
                    level_view=level_view,
                    contour=contour,
                    common_metadata=common_metadata,
                )
            )

        return result

    # ========================================================
    # DECODIFICACIÓN
    # ========================================================

    @staticmethod
    def _decode_image(
        image_bytes: bytes,
    ) -> np.ndarray:

        if not image_bytes:
            raise ValueError("El raster del LevelView está vacío.")

        buffer = np.frombuffer(
            image_bytes,
            dtype=np.uint8,
        )

        image = cv2.imdecode(
            buffer,
            cv2.IMREAD_COLOR,
        )

        if image is None:
            raise ValueError("OpenCV no pudo decodificar el raster del LevelView.")

        return image

    # ========================================================
    # CANNY AUTOMÁTICO
    # ========================================================

    def _automatic_canny_thresholds(
        self,
        gray: np.ndarray,
    ) -> tuple[int, int]:

        median = float(np.median(gray))

        sigma = float(self.config.canny_sigma)

        lower = int(
            max(
                0,
                (1.0 - sigma) * median,
            )
        )

        upper = int(
            min(
                255,
                (1.0 + sigma) * median,
            )
        )

        if upper <= lower:
            lower = 50
            upper = 150

        return (
            lower,
            upper,
        )

    # ========================================================
    # DESKEW — SOLO DIAGNÓSTICO
    # ========================================================

    @staticmethod
    def _estimate_deskew_angle(
        binary: np.ndarray,
    ) -> float:

        edges = cv2.Canny(
            binary,
            50,
            150,
            apertureSize=3,
        )

        minimum_dimension = min(binary.shape[:2])

        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=(np.pi / 180.0),
            threshold=max(
                30,
                int(minimum_dimension * 0.03),
            ),
            minLineLength=max(
                20,
                int(minimum_dimension * 0.08),
            ),
            maxLineGap=max(
                3,
                int(minimum_dimension * 0.005),
            ),
        )

        if lines is None:
            return 0.0

        normalized_lines = np.asarray(lines).reshape(
            -1,
            4,
        )

        deviations: list[float] = []

        for (
            x1,
            y1,
            x2,
            y2,
        ) in normalized_lines:
            dx = float(x2 - x1)

            dy = float(y2 - y1)

            if abs(dx) <= 1e-9 and abs(dy) <= 1e-9:
                continue

            angle = math.degrees(
                math.atan2(
                    dy,
                    dx,
                )
            )

            normalized = ((angle + 45.0) % 90.0) - 45.0

            if abs(normalized) <= 10.0:
                deviations.append(normalized)

        if not deviations:
            return 0.0

        return float(
            np.median(
                np.asarray(
                    deviations,
                    dtype=np.float64,
                )
            )
        )

    # ========================================================
    # HOUGH → SEGMENTOS CRUDOS
    # ========================================================

    def _build_hough_segments(
        self,
        lines: object,
    ) -> list[_RasterSegment]:

        if lines is None:
            return []

        normalized_lines = np.asarray(lines).reshape(
            -1,
            4,
        )

        result: list[_RasterSegment] = []

        for index, (
            raw_x1,
            raw_y1,
            raw_x2,
            raw_y2,
        ) in enumerate(
            normalized_lines,
            start=1,
        ):
            x1 = float(raw_x1)

            y1 = float(raw_y1)

            x2 = float(raw_x2)

            y2 = float(raw_y2)

            dx = x2 - x1

            dy = y2 - y1

            length = math.hypot(
                dx,
                dy,
            )

            if length <= self.NUMERIC_TOLERANCE:
                continue

            angle = math.degrees(
                math.atan2(
                    dy,
                    dx,
                )
            )

            orientation = self._orientation_from_angle(angle)

            if orientation == "horizontal" and x2 < x1:
                x1, x2 = (
                    x2,
                    x1,
                )

                y1, y2 = (
                    y2,
                    y1,
                )

            elif orientation == "vertical" and y2 < y1:
                x1, x2 = (
                    x2,
                    x1,
                )

                y1, y2 = (
                    y2,
                    y1,
                )

            result.append(
                _RasterSegment(
                    id=(f"RAW_{index}"),
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    length_px=length,
                    orientation=orientation,
                    angle_deg=angle,
                    source="opencv_hough",
                    merged_from=[],
                )
            )

        return result

    # ========================================================
    # ORIENTACIÓN
    # ========================================================

    @staticmethod
    def _orientation_from_angle(
        angle_deg: float,
    ) -> str:

        normalized = angle_deg % 180.0

        horizontal_distance = min(
            abs(normalized),
            abs(normalized - 180.0),
        )

        vertical_distance = abs(normalized - 90.0)

        if horizontal_distance <= 5.0:
            return "horizontal"

        if vertical_distance <= 5.0:
            return "vertical"

        return "other"

    # ========================================================
    # MERGE HORIZONTAL
    # ========================================================

    def _merge_horizontal_segments(
        self,
        segments: list[_RasterSegment],
        *,
        axis_tolerance: float,
        merge_gap: float,
    ) -> list[_RasterSegment]:

        if not segments:
            return []

        ordered = sorted(
            segments,
            key=lambda item: (
                (item.y1 + item.y2) / 2.0,
                item.x1,
            ),
        )

        groups: list[list[_RasterSegment]] = []

        for segment in ordered:
            segment_mid_y = (segment.y1 + segment.y2) / 2.0

            assigned = False

            for group in groups:
                reference_y = float(
                    np.mean([(item.y1 + item.y2) / 2.0 for item in group])
                )

                if abs(segment_mid_y - reference_y) > axis_tolerance:
                    continue

                group_min_x = min(
                    min(
                        item.x1,
                        item.x2,
                    )
                    for item in group
                )

                group_max_x = max(
                    max(
                        item.x1,
                        item.x2,
                    )
                    for item in group
                )

                segment_min_x = min(
                    segment.x1,
                    segment.x2,
                )

                segment_max_x = max(
                    segment.x1,
                    segment.x2,
                )

                gap = max(
                    0.0,
                    max(
                        (segment_min_x - group_max_x),
                        (group_min_x - segment_max_x),
                    ),
                )

                if gap <= merge_gap:
                    group.append(segment)

                    assigned = True

                    break

            if not assigned:
                groups.append([segment])

        result: list[_RasterSegment] = []

        for index, group in enumerate(
            groups,
            start=1,
        ):
            x1 = min(
                min(
                    item.x1,
                    item.x2,
                )
                for item in group
            )

            x2 = max(
                max(
                    item.x1,
                    item.x2,
                )
                for item in group
            )

            y = float(np.mean([(item.y1 + item.y2) / 2.0 for item in group]))

            result.append(
                _RasterSegment(
                    id=(f"MERGED_H_{index}"),
                    x1=x1,
                    y1=y,
                    x2=x2,
                    y2=y,
                    length_px=abs(x2 - x1),
                    orientation="horizontal",
                    angle_deg=0.0,
                    source="opencv_continuity",
                    merged_from=[item.id for item in group],
                )
            )

        return result

    # ========================================================
    # MERGE VERTICAL
    # ========================================================

    def _merge_vertical_segments(
        self,
        segments: list[_RasterSegment],
        *,
        axis_tolerance: float,
        merge_gap: float,
    ) -> list[_RasterSegment]:

        if not segments:
            return []

        ordered = sorted(
            segments,
            key=lambda item: (
                (item.x1 + item.x2) / 2.0,
                item.y1,
            ),
        )

        groups: list[list[_RasterSegment]] = []

        for segment in ordered:
            segment_mid_x = (segment.x1 + segment.x2) / 2.0

            assigned = False

            for group in groups:
                reference_x = float(
                    np.mean([(item.x1 + item.x2) / 2.0 for item in group])
                )

                if abs(segment_mid_x - reference_x) > axis_tolerance:
                    continue

                group_min_y = min(
                    min(
                        item.y1,
                        item.y2,
                    )
                    for item in group
                )

                group_max_y = max(
                    max(
                        item.y1,
                        item.y2,
                    )
                    for item in group
                )

                segment_min_y = min(
                    segment.y1,
                    segment.y2,
                )

                segment_max_y = max(
                    segment.y1,
                    segment.y2,
                )

                gap = max(
                    0.0,
                    max(
                        (segment_min_y - group_max_y),
                        (group_min_y - segment_max_y),
                    ),
                )

                if gap <= merge_gap:
                    group.append(segment)

                    assigned = True

                    break

            if not assigned:
                groups.append([segment])

        result: list[_RasterSegment] = []

        for index, group in enumerate(
            groups,
            start=1,
        ):
            y1 = min(
                min(
                    item.y1,
                    item.y2,
                )
                for item in group
            )

            y2 = max(
                max(
                    item.y1,
                    item.y2,
                )
                for item in group
            )

            x = float(np.mean([(item.x1 + item.x2) / 2.0 for item in group]))

            result.append(
                _RasterSegment(
                    id=(f"MERGED_V_{index}"),
                    x1=x,
                    y1=y1,
                    x2=x,
                    y2=y2,
                    length_px=abs(y2 - y1),
                    orientation="vertical",
                    angle_deg=90.0,
                    source="opencv_continuity",
                    merged_from=[item.id for item in group],
                )
            )

        return result

    # ========================================================
    # INTERSECCIONES
    # ========================================================

    @staticmethod
    def _find_intersections(
        *,
        horizontal: list[_RasterSegment],
        vertical: list[_RasterSegment],
        tolerance: float,
    ) -> list[_RasterIntersection]:

        result: list[_RasterIntersection] = []

        seen: set[
            tuple[
                int,
                int,
            ]
        ] = set()

        counter = 0

        for h_segment in horizontal:
            hx_min = min(
                h_segment.x1,
                h_segment.x2,
            )

            hx_max = max(
                h_segment.x1,
                h_segment.x2,
            )

            hy = (h_segment.y1 + h_segment.y2) / 2.0

            for v_segment in vertical:
                vx = (v_segment.x1 + v_segment.x2) / 2.0

                vy_min = min(
                    v_segment.y1,
                    v_segment.y2,
                )

                vy_max = max(
                    v_segment.y1,
                    v_segment.y2,
                )

                if not ((hx_min - tolerance) <= vx <= (hx_max + tolerance)):
                    continue

                if not ((vy_min - tolerance) <= hy <= (vy_max + tolerance)):
                    continue

                rounded_key = (
                    int(round(vx)),
                    int(round(hy)),
                )

                if rounded_key in seen:
                    continue

                seen.add(rounded_key)

                counter += 1

                result.append(
                    _RasterIntersection(
                        id=(f"INTERSECTION_{counter}"),
                        x=vx,
                        y=hy,
                        horizontal_segment_id=(h_segment.id),
                        vertical_segment_id=(v_segment.id),
                    )
                )

        return result

    # ========================================================
    # CONTORNOS
    # ========================================================

    def _extract_contours(
        self,
        line_mask: np.ndarray,
        *,
        image_width: int,
        image_height: int,
    ) -> list[_RasterContour]:

        contours, hierarchy = cv2.findContours(
            line_mask,
            cv2.RETR_TREE,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        image_area = float(image_width * image_height)
        minimum_area = image_area * self.config.minimum_contour_area_ratio

        hierarchy_rows = (
            hierarchy[0].tolist()
            if hierarchy is not None and len(hierarchy) > 0
            else []
        )

        result: list[_RasterContour] = []

        for contour_index, contour in enumerate(contours):
            area = float(cv2.contourArea(contour))
            if area < minimum_area:
                continue

            perimeter = float(cv2.arcLength(contour, True))
            x, y, width, height = cv2.boundingRect(contour)

            if width <= 1 or height <= 1:
                continue

            raw_points = [
                (int(point[0][0]), int(point[0][1]))
                for point in contour
            ]

            if len(raw_points) < 2:
                continue

            if raw_points[0] != raw_points[-1]:
                raw_points.append(raw_points[0])

            hierarchy_row = (
                hierarchy_rows[contour_index]
                if contour_index < len(hierarchy_rows)
                else [-1, -1, -1, -1]
            )

            def index_or_none(value: int) -> int | None:
                return None if int(value) < 0 else int(value)

            result.append(
                _RasterContour(
                    id=f"CONTOUR_{contour_index + 1}",
                    area_px2=area,
                    perimeter_px=perimeter,
                    points=raw_points,
                    x=int(x),
                    y=int(y),
                    width=int(width),
                    height=int(height),
                    hierarchy_next=index_or_none(hierarchy_row[0]),
                    hierarchy_previous=index_or_none(hierarchy_row[1]),
                    hierarchy_child=index_or_none(hierarchy_row[2]),
                    hierarchy_parent=index_or_none(hierarchy_row[3]),
                )
            )

        result.sort(key=lambda item: (-item.area_px2, item.id))
        return result

    # ========================================================
    # SEGMENTO → RAW EVIDENCE
    # ========================================================

    def _segment_to_evidence(
        self,
        *,
        level_view: LevelView,
        segment: _RasterSegment,
        stage: str,
        common_metadata: dict,
    ) -> RawEvidence:

        x1 = self._clamp_x(
            segment.x1,
            level_view,
        )

        y1 = self._clamp_y(
            segment.y1,
            level_view,
        )

        x2 = self._clamp_x(
            segment.x2,
            level_view,
        )

        y2 = self._clamp_y(
            segment.y2,
            level_view,
        )

        evidence_id = self._stable_id(
            prefix="OPENCV_LINE",
            level_view_id=level_view.id,
            payload=(
                stage,
                segment.id,
                segment.x1,
                segment.y1,
                segment.x2,
                segment.y2,
            ),
        )

        metadata = dict(common_metadata)

        metadata.update(
            {
                "stage": (stage),
                "source_reference": (segment.id),
                "detector": (segment.source),
                "orientation": (segment.orientation),
                "angle_deg": (segment.angle_deg),
                "length_px": (segment.length_px),
                "merged_from": list(segment.merged_from),
                "raw_geometry_px": {
                    "x1": (segment.x1),
                    "y1": (segment.y1),
                    "x2": (segment.x2),
                    "y2": (segment.y2),
                },
            }
        )

        return RawEvidence(
            id=evidence_id,
            level_view_id=level_view.id,
            source="OPENCV",
            kind="RASTER_LINE",
            geometry=EvidenceGeometry(
                geometry_type="SEGMENT",
                points=[
                    PixelPoint(
                        x=x1,
                        y=y1,
                    ),
                    PixelPoint(
                        x=x2,
                        y=y2,
                    ),
                ],
            ),
            text=None,
            confidence=None,
            metadata=metadata,
            confirmed=False,
        )

    # ========================================================
    # INTERSECCIÓN → RAW EVIDENCE
    # ========================================================

    def _intersection_to_evidence(
        self,
        *,
        level_view: LevelView,
        intersection: _RasterIntersection,
        common_metadata: dict,
    ) -> RawEvidence:

        x = self._clamp_x(
            intersection.x,
            level_view,
        )

        y = self._clamp_y(
            intersection.y,
            level_view,
        )

        evidence_id = self._stable_id(
            prefix="OPENCV_INTERSECTION",
            level_view_id=level_view.id,
            payload=(
                intersection.id,
                intersection.x,
                intersection.y,
                intersection.horizontal_segment_id,
                intersection.vertical_segment_id,
            ),
        )

        metadata = dict(common_metadata)

        metadata.update(
            {
                "source_reference": (intersection.id),
                "horizontal_segment_id": (intersection.horizontal_segment_id),
                "vertical_segment_id": (intersection.vertical_segment_id),
                "raw_geometry_px": {
                    "x": (intersection.x),
                    "y": (intersection.y),
                },
            }
        )

        return RawEvidence(
            id=evidence_id,
            level_view_id=level_view.id,
            source="OPENCV",
            kind="RASTER_INTERSECTION",
            geometry=EvidenceGeometry(
                geometry_type="POINT",
                points=[
                    PixelPoint(
                        x=x,
                        y=y,
                    )
                ],
            ),
            text=None,
            confidence=None,
            metadata=metadata,
            confirmed=False,
        )

    # ========================================================
    # CONTORNO → RAW EVIDENCE
    # ========================================================

    def _contour_to_evidence(
        self,
        *,
        level_view: LevelView,
        contour: _RasterContour,
        common_metadata: dict,
    ) -> RawEvidence:

        pixel_points = [
            PixelPoint(
                x=self._clamp_x(x, level_view),
                y=self._clamp_y(y, level_view),
            )
            for x, y in contour.points
        ]

        evidence_id = self._stable_id(
            prefix="OPENCV_CONTOUR",
            level_view_id=level_view.id,
            payload=(
                contour.id,
                tuple(contour.points),
                contour.area_px2,
                contour.perimeter_px,
            ),
        )

        metadata = dict(common_metadata)
        metadata.update(
            {
                "source_reference": contour.id,
                "area_px2": contour.area_px2,
                "perimeter_px": contour.perimeter_px,
                "closed": True,
                "contour_retrieval_mode": "RETR_TREE",
                "contour_chain_mode": "CHAIN_APPROX_SIMPLE",
                "hierarchy": {
                    "next": contour.hierarchy_next,
                    "previous": contour.hierarchy_previous,
                    "child": contour.hierarchy_child,
                    "parent": contour.hierarchy_parent,
                },
                "raw_bbox_px": {
                    "x": contour.x,
                    "y": contour.y,
                    "width": contour.width,
                    "height": contour.height,
                },
                "raw_contour_points_px": [
                    {"x": x, "y": y}
                    for x, y in contour.points
                ],
            }
        )

        return RawEvidence(
            id=evidence_id,
            level_view_id=level_view.id,
            source="OPENCV",
            kind="RASTER_CONTOUR",
            geometry=EvidenceGeometry(
                geometry_type="POLYLINE",
                points=pixel_points,
            ),
            text=None,
            confidence=None,
            metadata=metadata,
            confirmed=False,
        )

    # ========================================================
    # COORDENADAS
    # ========================================================

    @staticmethod
    def _clamp_x(
        value: float,
        level_view: LevelView,
    ) -> int:

        return max(
            0,
            min(
                level_view.raster_width_px,
                int(round(value)),
            ),
        )

    @staticmethod
    def _clamp_y(
        value: float,
        level_view: LevelView,
    ) -> int:

        return max(
            0,
            min(
                level_view.raster_height_px,
                int(round(value)),
            ),
        )

    # ========================================================
    # ID ESTABLE
    # ========================================================

    @staticmethod
    def _stable_id(
        *,
        prefix: str,
        level_view_id: str,
        payload: tuple,
    ) -> str:

        raw = (f"{level_view_id}|{repr(payload)}").encode("utf-8")

        digest = hashlib.sha1(raw).hexdigest()[:16]

        return f"{prefix}_{digest}"
