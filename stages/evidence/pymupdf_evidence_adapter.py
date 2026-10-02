from __future__ import annotations

import hashlib
import math

import pymupdf

from app.quantia_spatialV1.core.models.evidence import (
    EvidenceGeometry,
    RawEvidence,
)
from app.quantia_spatialV1.core.models.level_view import (
    LevelView,
    PixelBBox,
    PixelPoint,
)


class PyMuPDFEvidenceAdapter:
    """
    Fase 01.5 — extracción vectorial cruda por LevelView.

    Entrada:
        document_bytes
        LevelView

    Extrae:
        - líneas vectoriales;
        - texto vectorial;
        - coordenadas originales;
        - metadatos documentales.

    Todas las geometrías publicadas quedan expresadas
    en coordenadas LOCALES del LevelView.

    No:
        - identifica muros;
        - identifica boundaries;
        - identifica ejes;
        - interpreta cotas;
        - convierte px -> m;
        - elimina evidencia;
        - confirma elementos.
    """

    # ========================================================
    # API
    # ========================================================

    def extract(
        self,
        *,
        document_bytes: bytes,
        level_view: LevelView,
    ) -> list[RawEvidence]:

        if not document_bytes:
            raise ValueError("PyMuPDFEvidenceAdapter requiere document_bytes.")

        page_number = int(level_view.source_page_number)

        if page_number <= 0:
            raise ValueError("LevelView.source_page_number inválido.")

        try:
            document = pymupdf.open(
                stream=document_bytes,
                filetype="pdf",
            )

        except Exception as exc:
            raise ValueError("No fue posible abrir el PDF con PyMuPDF.") from exc

        try:
            if page_number > document.page_count:
                raise ValueError(
                    "LevelView referencia una página inexistente en el PDF."
                )

            page = document.load_page(page_number - 1)

            page_width = float(page.rect.width)

            page_height = float(page.rect.height)

            if page_width <= 0 or page_height <= 0:
                raise ValueError("La página PDF tiene dimensiones inválidas.")

            scale_x = level_view.source_page_width_px / page_width

            scale_y = level_view.source_page_height_px / page_height

            result: list[RawEvidence] = []

            result.extend(
                self._extract_vector_lines(
                    page=page,
                    level_view=level_view,
                    scale_x=scale_x,
                    scale_y=scale_y,
                )
            )

            result.extend(
                self._extract_vector_text(
                    page=page,
                    level_view=level_view,
                    scale_x=scale_x,
                    scale_y=scale_y,
                )
            )

            return result

        finally:
            document.close()

    # ========================================================
    # VECTOR LINES
    # ========================================================

    def _extract_vector_lines(
        self,
        *,
        page: pymupdf.Page,
        level_view: LevelView,
        scale_x: float,
        scale_y: float,
    ) -> list[RawEvidence]:

        result: list[RawEvidence] = []

        try:
            drawings = page.get_drawings()
        except Exception:
            return []

        rotation_matrix = page.rotation_matrix
        level_bbox = level_view.source_bbox_px

        for drawing_index, drawing in enumerate(drawings):
            if not isinstance(drawing, dict):
                continue

            items = drawing.get("items", [])
            if not isinstance(items, list):
                continue

            drawing_metadata = {
                "drawing_type": drawing.get("type"),
                "stroke_color": drawing.get("color"),
                "fill_color": drawing.get("fill"),
                "stroke_width": drawing.get("width"),
                "dashes": drawing.get("dashes"),
                "line_cap": drawing.get("lineCap"),
                "line_join": drawing.get("lineJoin"),
                "close_path": drawing.get("closePath"),
            }

            for item_index, item in enumerate(items):
                if not isinstance(item, tuple) or not item:
                    continue

                command = str(item[0])

                # Líneas, rectángulos y quads pueden conservarse exactamente
                # como segmentos de borde sin inventar una discretización.
                segments = self._drawing_item_segments(item)

                for segment_index, (p1, p2) in enumerate(segments):
                    visual_p1 = pymupdf.Point(float(p1.x), float(p1.y)) * rotation_matrix
                    visual_p2 = pymupdf.Point(float(p2.x), float(p2.y)) * rotation_matrix

                    page_x1 = float(visual_p1.x) * scale_x
                    page_y1 = float(visual_p1.y) * scale_y
                    page_x2 = float(visual_p2.x) * scale_x
                    page_y2 = float(visual_p2.y) * scale_y

                    clipped = self._clip_segment_to_bbox(
                        x1=page_x1,
                        y1=page_y1,
                        x2=page_x2,
                        y2=page_y2,
                        bbox=level_bbox,
                    )
                    if clipped is None:
                        continue

                    clipped_x1, clipped_y1, clipped_x2, clipped_y2 = clipped
                    local_start = self._page_point_to_local(
                        x=clipped_x1,
                        y=clipped_y1,
                        level_view=level_view,
                    )
                    local_end = self._page_point_to_local(
                        x=clipped_x2,
                        y=clipped_y2,
                        level_view=level_view,
                    )
                    if local_start == local_end:
                        continue

                    evidence_id = self._stable_id(
                        prefix="PYMUPDF_LINE",
                        level_view_id=level_view.id,
                        payload=(
                            page.number,
                            drawing_index,
                            item_index,
                            segment_index,
                            command,
                            page_x1,
                            page_y1,
                            page_x2,
                            page_y2,
                        ),
                    )

                    result.append(
                        RawEvidence(
                            id=evidence_id,
                            level_view_id=level_view.id,
                            source="PYMUPDF",
                            kind="VECTOR_LINE",
                            geometry=EvidenceGeometry(
                                geometry_type="SEGMENT",
                                points=[local_start, local_end],
                            ),
                            text=None,
                            confidence=None,
                            metadata={
                                "source_document_id": level_view.source_document_id,
                                "source_page_number": level_view.source_page_number,
                                "drawing_index": drawing_index,
                                "item_index": item_index,
                                "segment_index": segment_index,
                                "vector_command": command,
                                "drawing": drawing_metadata,
                                "pdf_visual": {
                                    "x1": float(visual_p1.x),
                                    "y1": float(visual_p1.y),
                                    "x2": float(visual_p2.x),
                                    "y2": float(visual_p2.y),
                                },
                                "page_px": {
                                    "x1": page_x1,
                                    "y1": page_y1,
                                    "x2": page_x2,
                                    "y2": page_y2,
                                },
                                "clipped_page_px": {
                                    "x1": clipped_x1,
                                    "y1": clipped_y1,
                                    "x2": clipped_x2,
                                    "y2": clipped_y2,
                                },
                                "clipped_to_level": not self._same_segment(
                                    (page_x1, page_y1, page_x2, page_y2),
                                    clipped,
                                ),
                                "page_to_raster_scale_x": scale_x,
                                "page_to_raster_scale_y": scale_y,
                            },
                            confirmed=False,
                        )
                    )

                # Curvas Bézier no se convierten a líneas. Se conserva su
                # geometría de control original y una bbox local para que la
                # información no desaparezca de F01.5.
                if command == "c":
                    primitive = self._curve_primitive_evidence(
                        page=page,
                        item=item,
                        drawing_index=drawing_index,
                        item_index=item_index,
                        drawing_metadata=drawing_metadata,
                        rotation_matrix=rotation_matrix,
                        level_view=level_view,
                        scale_x=scale_x,
                        scale_y=scale_y,
                    )
                    if primitive is not None:
                        result.append(primitive)

        return result

    @staticmethod
    def _drawing_item_segments(
        item: tuple,
    ) -> list[tuple[pymupdf.Point, pymupdf.Point]]:
        command = str(item[0])

        if command == "l" and len(item) >= 3:
            p1, p2 = item[1], item[2]
            if all(hasattr(p, "x") and hasattr(p, "y") for p in (p1, p2)):
                return [(p1, p2)]
            return []

        if command == "re" and len(item) >= 2:
            rect = item[1]
            if not isinstance(rect, pymupdf.Rect):
                return []
            points = [rect.tl, rect.tr, rect.br, rect.bl]
            return [
                (points[index], points[(index + 1) % 4])
                for index in range(4)
            ]

        if command == "qu" and len(item) >= 2:
            quad = item[1]
            names = ("ul", "ur", "lr", "ll")
            if not all(hasattr(quad, name) for name in names):
                return []
            points = [quad.ul, quad.ur, quad.lr, quad.ll]
            return [
                (points[index], points[(index + 1) % 4])
                for index in range(4)
            ]

        return []

    def _curve_primitive_evidence(
        self,
        *,
        page: pymupdf.Page,
        item: tuple,
        drawing_index: int,
        item_index: int,
        drawing_metadata: dict,
        rotation_matrix: pymupdf.Matrix,
        level_view: LevelView,
        scale_x: float,
        scale_y: float,
    ) -> RawEvidence | None:
        if len(item) < 5:
            return None

        control_points = item[1:5]
        if not all(
            hasattr(point, "x") and hasattr(point, "y")
            for point in control_points
        ):
            return None

        visual_points = [
            pymupdf.Point(float(point.x), float(point.y)) * rotation_matrix
            for point in control_points
        ]
        page_points = [
            (float(point.x) * scale_x, float(point.y) * scale_y)
            for point in visual_points
        ]

        page_bbox = (
            min(x for x, _ in page_points),
            min(y for _, y in page_points),
            max(x for x, _ in page_points),
            max(y for _, y in page_points),
        )
        clipped_bbox = self._intersect_bbox(
            bbox=page_bbox,
            level_bbox=level_view.source_bbox_px,
        )
        if clipped_bbox is None:
            return None

        local_bbox = self._page_bbox_to_local(
            bbox=clipped_bbox,
            level_view=level_view,
        )
        if local_bbox is None:
            return None

        evidence_id = self._stable_id(
            prefix="PYMUPDF_PRIMITIVE",
            level_view_id=level_view.id,
            payload=(
                page.number,
                drawing_index,
                item_index,
                "c",
                tuple(page_points),
            ),
        )

        return RawEvidence(
            id=evidence_id,
            level_view_id=level_view.id,
            source="PYMUPDF",
            kind="VECTOR_PRIMITIVE",
            geometry=EvidenceGeometry(
                geometry_type="BBOX",
                bbox_px=local_bbox,
            ),
            text=None,
            confidence=None,
            metadata={
                "source_document_id": level_view.source_document_id,
                "source_page_number": level_view.source_page_number,
                "drawing_index": drawing_index,
                "item_index": item_index,
                "vector_command": "c",
                "primitive_type": "CUBIC_BEZIER",
                "drawing": drawing_metadata,
                "control_points_pdf_visual": [
                    {"x": float(point.x), "y": float(point.y)}
                    for point in visual_points
                ],
                "control_points_page_px": [
                    {"x": x, "y": y}
                    for x, y in page_points
                ],
                "page_px_bbox": {
                    "x_min": page_bbox[0],
                    "y_min": page_bbox[1],
                    "x_max": page_bbox[2],
                    "y_max": page_bbox[3],
                },
                "page_to_raster_scale_x": scale_x,
                "page_to_raster_scale_y": scale_y,
            },
            confirmed=False,
        )

    # ========================================================
    # VECTOR TEXT
    # ========================================================

    def _extract_vector_text(
        self,
        *,
        page: pymupdf.Page,
        level_view: LevelView,
        scale_x: float,
        scale_y: float,
    ) -> list[RawEvidence]:

        result: list[RawEvidence] = []

        raw = page.get_text("dict")

        rotation_matrix = page.rotation_matrix

        level_bbox = level_view.source_bbox_px

        span_index = 0

        for block in raw.get(
            "blocks",
            [],
        ):
            if not isinstance(
                block,
                dict,
            ):
                continue

            lines = block.get(
                "lines",
                [],
            )

            if not isinstance(
                lines,
                list,
            ):
                continue

            for line in lines:
                if not isinstance(
                    line,
                    dict,
                ):
                    continue

                spans = line.get(
                    "spans",
                    [],
                )

                if not isinstance(
                    spans,
                    list,
                ):
                    continue

                for span in spans:
                    if not isinstance(
                        span,
                        dict,
                    ):
                        continue

                    span_index += 1

                    text = str(
                        span.get(
                            "text",
                            "",
                        )
                    ).strip()

                    if not text:
                        continue

                    bbox = span.get("bbox")

                    if (
                        not isinstance(
                            bbox,
                            (list, tuple),
                        )
                        or len(bbox) != 4
                    ):
                        continue

                    pdf_rect = pymupdf.Rect(
                        float(bbox[0]),
                        float(bbox[1]),
                        float(bbox[2]),
                        float(bbox[3]),
                    )

                    visual_rect = pdf_rect * rotation_matrix

                    page_bbox = self._visual_rect_to_page_px(
                        rect=visual_rect,
                        scale_x=scale_x,
                        scale_y=scale_y,
                    )

                    clipped_bbox = self._intersect_bbox(
                        bbox=page_bbox,
                        level_bbox=level_bbox,
                    )

                    if clipped_bbox is None:
                        continue

                    local_bbox = self._page_bbox_to_local(
                        bbox=clipped_bbox,
                        level_view=level_view,
                    )

                    if local_bbox is None:
                        continue

                    evidence_id = self._stable_id(
                        prefix="PYMUPDF_TEXT",
                        level_view_id=level_view.id,
                        payload=(
                            page.number,
                            span_index,
                            text,
                            float(visual_rect.x0),
                            float(visual_rect.y0),
                            float(visual_rect.x1),
                            float(visual_rect.y1),
                        ),
                    )

                    size = span.get("size")

                    result.append(
                        RawEvidence(
                            id=evidence_id,
                            level_view_id=(level_view.id),
                            source="PYMUPDF",
                            kind="VECTOR_TEXT",
                            geometry=EvidenceGeometry(
                                geometry_type="BBOX",
                                bbox_px=local_bbox,
                            ),
                            text=text,
                            confidence=None,
                            metadata={
                                "source_document_id": (level_view.source_document_id),
                                "source_page_number": (level_view.source_page_number),
                                "font": (
                                    str(span.get("font")) if span.get("font") else None
                                ),
                                "font_size": (
                                    float(size)
                                    if isinstance(
                                        size,
                                        (int, float),
                                    )
                                    else None
                                ),
                                "font_flags": span.get("flags"),
                                "char_flags": span.get("char_flags"),
                                "origin": span.get("origin"),
                                "ascender": span.get("ascender"),
                                "descender": span.get("descender"),
                                "line_direction": line.get("dir"),
                                "line_wmode": line.get("wmode"),
                                "pdf_visual_bbox": {
                                    "x0": float(visual_rect.x0),
                                    "y0": float(visual_rect.y0),
                                    "x1": float(visual_rect.x1),
                                    "y1": float(visual_rect.y1),
                                },
                                "page_px_bbox": {
                                    "x_min": (page_bbox[0]),
                                    "y_min": (page_bbox[1]),
                                    "x_max": (page_bbox[2]),
                                    "y_max": (page_bbox[3]),
                                },
                                "page_to_raster_scale_x": (scale_x),
                                "page_to_raster_scale_y": (scale_y),
                            },
                            confirmed=False,
                        )
                    )

        return result

    # ========================================================
    # PDF RECT -> PAGE PIXELS
    # ========================================================

    @staticmethod
    def _visual_rect_to_page_px(
        *,
        rect: pymupdf.Rect,
        scale_x: float,
        scale_y: float,
    ) -> tuple[
        float,
        float,
        float,
        float,
    ]:

        return (
            float(rect.x0) * scale_x,
            float(rect.y0) * scale_y,
            float(rect.x1) * scale_x,
            float(rect.y1) * scale_y,
        )

    # ========================================================
    # PAGE POINT -> LOCAL LEVEL VIEW
    # ========================================================

    @staticmethod
    def _page_point_to_local(
        *,
        x: float,
        y: float,
        level_view: LevelView,
    ) -> PixelPoint:

        local_x = int(round(x - level_view.source_bbox_px.x_min))

        local_y = int(round(y - level_view.source_bbox_px.y_min))

        local_x = max(
            0,
            min(
                level_view.raster_width_px,
                local_x,
            ),
        )

        local_y = max(
            0,
            min(
                level_view.raster_height_px,
                local_y,
            ),
        )

        return PixelPoint(
            x=local_x,
            y=local_y,
        )

    # ========================================================
    # PAGE BBOX -> LOCAL LEVEL VIEW
    # ========================================================

    @staticmethod
    def _page_bbox_to_local(
        *,
        bbox: tuple[
            float,
            float,
            float,
            float,
        ],
        level_view: LevelView,
    ) -> PixelBBox | None:

        offset_x = level_view.source_bbox_px.x_min

        offset_y = level_view.source_bbox_px.y_min

        x_min = int(math.floor(bbox[0] - offset_x))

        y_min = int(math.floor(bbox[1] - offset_y))

        x_max = int(math.ceil(bbox[2] - offset_x))

        y_max = int(math.ceil(bbox[3] - offset_y))

        x_min = max(
            0,
            min(
                level_view.raster_width_px,
                x_min,
            ),
        )

        y_min = max(
            0,
            min(
                level_view.raster_height_px,
                y_min,
            ),
        )

        x_max = max(
            0,
            min(
                level_view.raster_width_px,
                x_max,
            ),
        )

        y_max = max(
            0,
            min(
                level_view.raster_height_px,
                y_max,
            ),
        )

        if x_max <= x_min or y_max <= y_min:
            return None

        return PixelBBox(
            x_min=x_min,
            y_min=y_min,
            x_max=x_max,
            y_max=y_max,
        )

    # ========================================================
    # INTERSECCIÓN DE BBOX
    # ========================================================

    @staticmethod
    def _intersect_bbox(
        *,
        bbox: tuple[
            float,
            float,
            float,
            float,
        ],
        level_bbox: PixelBBox,
    ) -> (
        tuple[
            float,
            float,
            float,
            float,
        ]
        | None
    ):

        x_min = max(
            bbox[0],
            float(level_bbox.x_min),
        )

        y_min = max(
            bbox[1],
            float(level_bbox.y_min),
        )

        x_max = min(
            bbox[2],
            float(level_bbox.x_max),
        )

        y_max = min(
            bbox[3],
            float(level_bbox.y_max),
        )

        if x_max <= x_min or y_max <= y_min:
            return None

        return (
            x_min,
            y_min,
            x_max,
            y_max,
        )

    # ========================================================
    # CLIP SEGMENT TO LEVEL BBOX
    # ========================================================

    @staticmethod
    def _clip_segment_to_bbox(
        *,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        bbox: PixelBBox,
    ) -> (
        tuple[
            float,
            float,
            float,
            float,
        ]
        | None
    ):
        """
        Liang-Barsky.

        Conserva el tramo de una línea que realmente
        pertenece al LevelView.

        Esto evita:
        - duplicar una línea completa entre niveles;
        - perder una línea que cruza parcialmente el bbox.
        """

        dx = x2 - x1
        dy = y2 - y1

        p = (
            -dx,
            dx,
            -dy,
            dy,
        )

        q = (
            x1 - bbox.x_min,
            bbox.x_max - x1,
            y1 - bbox.y_min,
            bbox.y_max - y1,
        )

        u1 = 0.0
        u2 = 1.0

        for pi, qi in zip(
            p,
            q,
        ):
            if abs(pi) < 1e-12:
                if qi < 0:
                    return None

                continue

            ratio = qi / pi

            if pi < 0:
                u1 = max(
                    u1,
                    ratio,
                )

            else:
                u2 = min(
                    u2,
                    ratio,
                )

            if u1 > u2:
                return None

        return (
            x1 + u1 * dx,
            y1 + u1 * dy,
            x1 + u2 * dx,
            y1 + u2 * dy,
        )

    # ========================================================
    # SEGMENT COMPARISON
    # ========================================================

    @staticmethod
    def _same_segment(
        first: tuple[
            float,
            float,
            float,
            float,
        ],
        second: tuple[
            float,
            float,
            float,
            float,
        ],
    ) -> bool:

        return all(
            math.isclose(
                a,
                b,
                rel_tol=0.0,
                abs_tol=1e-6,
            )
            for a, b in zip(
                first,
                second,
            )
        )

    # ========================================================
    # STABLE ID
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
