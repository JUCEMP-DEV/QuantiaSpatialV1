from __future__ import annotations

import hashlib
import os
from io import BytesIO
from typing import Any

import pytesseract
from PIL import Image, ImageOps
from pytesseract import Output

from app.quantia_spatialV1.core.models.evidence import (
    EvidenceGeometry,
    RawEvidence,
)
from app.quantia_spatialV1.core.models.level_view import (
    LevelView,
    PixelBBox,
)


class OCREvidenceAdapter:
    """
    Fase 01.5 — extracción textual raster cruda.

    Entrada:
        LevelView

    Fuente:
        level_view.raster_bytes

    Extrae:
        - tokens reconocidos por Tesseract;
        - bbox local de cada token;
        - confianza OCR;
        - estructura block / paragraph / line / word;
        - líneas OCR agrupadas según estructura Tesseract.

    Toda la geometría permanece en coordenadas LOCALES
    del LevelView.

    No:
        - identifica cotas;
        - identifica ejes;
        - identifica espacios;
        - interpreta arquitectura;
        - convierte texto numérico a medida real;
        - convierte px -> m;
        - elimina evidencia por confianza;
        - confirma elementos.
    """

    def __init__(
        self,
        *,
        tesseract_cmd: str | None = None,
        preferred_language: str | None = None,
    ) -> None:

        configured_cmd = tesseract_cmd or os.getenv("TESSERACT_CMD")

        if configured_cmd:
            pytesseract.pytesseract.tesseract_cmd = configured_cmd

        self.preferred_language = preferred_language or os.getenv("TESSERACT_LANG")

    # ========================================================
    # API
    # ========================================================

    def extract(
        self,
        *,
        level_view: LevelView,
    ) -> list[RawEvidence]:

        image = self._open_image(level_view)

        width, height = image.size

        if width != level_view.raster_width_px or height != level_view.raster_height_px:
            raise ValueError(
                "Las dimensiones OCR no coinciden con el raster del LevelView."
            )

        # Solo cambia intensidad.
        # No modifica tamaño ni sistema de coordenadas.
        grayscale = ImageOps.grayscale(image)

        language = self._select_language()

        kwargs: dict[str, Any] = {
            "output_type": Output.DICT,
        }

        if language:
            kwargs["lang"] = language

        try:
            raw = pytesseract.image_to_data(
                grayscale,
                **kwargs,
            )

        except pytesseract.TesseractError as exc:
            raise ValueError(
                "Tesseract no pudo procesar el raster del LevelView."
            ) from exc

        token_evidence: list[RawEvidence] = []

        line_groups: dict[
            tuple[int, int, int],
            list[RawEvidence],
        ] = {}

        count = len(
            raw.get(
                "text",
                [],
            )
        )

        # ----------------------------------------------------
        # TOKENS
        # ----------------------------------------------------

        for index in range(count):
            text = self._safe_text(
                raw,
                "text",
                index,
            )

            if not text:
                continue

            raw_bbox = self._read_bbox(
                raw=raw,
                index=index,
            )

            confidence, confidence_raw = self._read_confidence(
                raw=raw,
                index=index,
            )

            block_num = self._safe_int(
                raw,
                "block_num",
                index,
            )

            par_num = self._safe_int(
                raw,
                "par_num",
                index,
            )

            line_num = self._safe_int(
                raw,
                "line_num",
                index,
            )

            word_num = self._safe_int(
                raw,
                "word_num",
                index,
            )

            geometry, geometry_metadata = self._build_geometry(
                raw_bbox=raw_bbox,
                level_view=level_view,
            )

            evidence_id = self._stable_id(
                prefix="OCR_TOKEN",
                level_view_id=level_view.id,
                payload=(
                    index,
                    text,
                    raw_bbox,
                    block_num,
                    par_num,
                    line_num,
                    word_num,
                ),
            )

            metadata = {
                "source_document_id": (level_view.source_document_id),
                "source_page_number": (level_view.source_page_number),
                "ocr_engine": "tesseract",
                "language": language,
                "granularity": "TOKEN",
                "tesseract_index": index,
                "block_num": block_num,
                "paragraph_num": par_num,
                "line_num": line_num,
                "word_num": word_num,
                "confidence_raw": confidence_raw,
                "raw_bbox_px": {
                    "x": raw_bbox[0],
                    "y": raw_bbox[1],
                    "width": raw_bbox[2],
                    "height": raw_bbox[3],
                },
                **geometry_metadata,
            }

            evidence = RawEvidence(
                id=evidence_id,
                level_view_id=level_view.id,
                source="OCR",
                kind="OCR_TEXT",
                geometry=geometry,
                text=text,
                confidence=confidence,
                metadata=metadata,
                confirmed=False,
            )

            token_evidence.append(evidence)

            group_key = (
                block_num,
                par_num,
                line_num,
            )

            line_groups.setdefault(
                group_key,
                [],
            ).append(evidence)

        # ----------------------------------------------------
        # LÍNEAS
        # ----------------------------------------------------

        line_evidence = self._build_line_evidence(
            level_view=level_view,
            groups=line_groups,
            language=language,
        )

        return [
            *token_evidence,
            *line_evidence,
        ]

    # ========================================================
    # ABRIR RASTER
    # ========================================================

    @staticmethod
    def _open_image(
        level_view: LevelView,
    ) -> Image.Image:

        if not level_view.raster_bytes:
            raise ValueError("El raster OCR del LevelView está vacío.")

        try:
            with Image.open(BytesIO(level_view.raster_bytes)) as source:
                return source.convert("RGB")

        except Exception as exc:
            raise ValueError(
                "No fue posible abrir el raster del LevelView para OCR."
            ) from exc

    # ========================================================
    # IDIOMA
    # ========================================================

    def _select_language(
        self,
    ) -> str | None:

        try:
            available = set(pytesseract.get_languages(config=""))

        except Exception:
            available = set()

        preferred = str(self.preferred_language or "").strip()

        if preferred and preferred in available:
            return preferred

        if "spa" in available:
            return "spa"

        if "eng" in available:
            return "eng"

        return None

    # ========================================================
    # BBOX ORIGINAL TESSERACT
    # ========================================================

    @staticmethod
    def _read_bbox(
        *,
        raw: dict[str, Any],
        index: int,
    ) -> tuple[
        int,
        int,
        int,
        int,
    ]:

        x = OCREvidenceAdapter._safe_int(
            raw,
            "left",
            index,
        )

        y = OCREvidenceAdapter._safe_int(
            raw,
            "top",
            index,
        )

        width = OCREvidenceAdapter._safe_int(
            raw,
            "width",
            index,
        )

        height = OCREvidenceAdapter._safe_int(
            raw,
            "height",
            index,
        )

        return (
            x,
            y,
            width,
            height,
        )

    # ========================================================
    # GEOMETRÍA LOCAL
    # ========================================================

    @staticmethod
    def _build_geometry(
        *,
        raw_bbox: tuple[
            int,
            int,
            int,
            int,
        ],
        level_view: LevelView,
    ) -> tuple[
        EvidenceGeometry,
        dict[str, Any],
    ]:

        (
            x,
            y,
            width,
            height,
        ) = raw_bbox

        raw_x_max = x + width
        raw_y_max = y + height

        x_min = max(
            0,
            x,
        )

        y_min = max(
            0,
            y,
        )

        x_max = min(
            level_view.raster_width_px,
            raw_x_max,
        )

        y_max = min(
            level_view.raster_height_px,
            raw_y_max,
        )

        # No eliminamos el texto si Tesseract produjo
        # una geometría anómala. Conservamos el bbox original
        # en metadata y dejamos geometry=NONE.

        if width <= 0 or height <= 0 or x_max <= x_min or y_max <= y_min:
            return (
                EvidenceGeometry(
                    geometry_type="NONE",
                ),
                {
                    "geometry_status": ("INVALID_OR_OUTSIDE_RASTER"),
                    "bbox_clipped": False,
                },
            )

        clipped = x_min != x or y_min != y or x_max != raw_x_max or y_max != raw_y_max

        return (
            EvidenceGeometry(
                geometry_type="BBOX",
                bbox_px=PixelBBox(
                    x_min=x_min,
                    y_min=y_min,
                    x_max=x_max,
                    y_max=y_max,
                ),
            ),
            {
                "geometry_status": "VALID",
                "bbox_clipped": clipped,
            },
        )

    # ========================================================
    # CONFIANZA
    # ========================================================

    @staticmethod
    def _read_confidence(
        *,
        raw: dict[str, Any],
        index: int,
    ) -> tuple[
        float | None,
        float | None,
    ]:

        try:
            raw_value = float(raw["conf"][index])

        except (
            KeyError,
            IndexError,
            TypeError,
            ValueError,
        ):
            return (
                None,
                None,
            )

        # Tesseract usa -1 para elementos sin confidence
        # válida. La evidencia textual no se elimina.

        if raw_value < 0:
            return (
                None,
                raw_value,
            )

        normalized = min(
            1.0,
            max(
                0.0,
                raw_value / 100.0,
            ),
        )

        return (
            normalized,
            raw_value,
        )

    # ========================================================
    # LÍNEAS OCR
    # ========================================================

    def _build_line_evidence(
        self,
        *,
        level_view: LevelView,
        groups: dict[
            tuple[int, int, int],
            list[RawEvidence],
        ],
        language: str | None,
    ) -> list[RawEvidence]:

        result: list[RawEvidence] = []

        for (
            block_num,
            par_num,
            line_num,
        ), items in groups.items():
            if not items:
                continue

            ordered = sorted(
                items,
                key=self._token_sort_x,
            )

            text = " ".join(item.text for item in ordered if item.text).strip()

            if not text:
                continue

            geometry = self._line_geometry(
                items=ordered,
                level_view=level_view,
            )

            confidence_values = [
                item.confidence for item in ordered if item.confidence is not None
            ]

            confidence = (
                (sum(confidence_values) / len(confidence_values))
                if confidence_values
                else None
            )

            token_ids = [item.id for item in ordered]

            evidence_id = self._stable_id(
                prefix="OCR_LINE",
                level_view_id=level_view.id,
                payload=(
                    block_num,
                    par_num,
                    line_num,
                    text,
                    tuple(token_ids),
                ),
            )

            result.append(
                RawEvidence(
                    id=evidence_id,
                    level_view_id=level_view.id,
                    source="OCR",
                    kind="OCR_TEXT",
                    geometry=geometry,
                    text=text,
                    confidence=confidence,
                    metadata={
                        "source_document_id": (level_view.source_document_id),
                        "source_page_number": (level_view.source_page_number),
                        "ocr_engine": "tesseract",
                        "language": language,
                        "granularity": "LINE",
                        "block_num": block_num,
                        "paragraph_num": par_num,
                        "line_num": line_num,
                        "token_ids": token_ids,
                        "token_count": len(token_ids),
                    },
                    confirmed=False,
                )
            )

        return result

    # ========================================================
    # GEOMETRÍA DE LÍNEA OCR
    # ========================================================

    @staticmethod
    def _line_geometry(
        *,
        items: list[RawEvidence],
        level_view: LevelView,
    ) -> EvidenceGeometry:

        bboxes = [
            item.geometry.bbox_px
            for item in items
            if (
                item.geometry.geometry_type == "BBOX"
                and item.geometry.bbox_px is not None
            )
        ]

        if not bboxes:
            return EvidenceGeometry(
                geometry_type="NONE",
            )

        x_min = min(bbox.x_min for bbox in bboxes)

        y_min = min(bbox.y_min for bbox in bboxes)

        x_max = max(bbox.x_max for bbox in bboxes)

        y_max = max(bbox.y_max for bbox in bboxes)

        x_min = max(
            0,
            min(
                x_min,
                level_view.raster_width_px,
            ),
        )

        y_min = max(
            0,
            min(
                y_min,
                level_view.raster_height_px,
            ),
        )

        x_max = max(
            0,
            min(
                x_max,
                level_view.raster_width_px,
            ),
        )

        y_max = max(
            0,
            min(
                y_max,
                level_view.raster_height_px,
            ),
        )

        if x_max <= x_min or y_max <= y_min:
            return EvidenceGeometry(
                geometry_type="NONE",
            )

        return EvidenceGeometry(
            geometry_type="BBOX",
            bbox_px=PixelBBox(
                x_min=x_min,
                y_min=y_min,
                x_max=x_max,
                y_max=y_max,
            ),
        )

    # ========================================================
    # ORDEN DE TOKEN
    # ========================================================

    @staticmethod
    def _token_sort_x(
        item: RawEvidence,
    ) -> int:

        bbox = item.geometry.bbox_px

        if bbox is None:
            return 0

        return bbox.x_min

    # ========================================================
    # SAFE TEXT
    # ========================================================

    @staticmethod
    def _safe_text(
        payload: dict[str, Any],
        key: str,
        index: int,
    ) -> str:

        try:
            value = payload[key][index]

        except (
            KeyError,
            IndexError,
            TypeError,
        ):
            return ""

        return str(value or "").strip()

    # ========================================================
    # SAFE INT
    # ========================================================

    @staticmethod
    def _safe_int(
        payload: dict[str, Any],
        key: str,
        index: int,
    ) -> int:

        try:
            return int(payload[key][index])

        except (
            KeyError,
            IndexError,
            TypeError,
            ValueError,
        ):
            return 0

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
