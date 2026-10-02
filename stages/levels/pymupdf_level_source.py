from __future__ import annotations

import math
import unicodedata
from typing import Literal

import pymupdf
from app.quantia_spatialV1.core.models.level_view import (
    PixelBBox,
)
from pydantic import (
    BaseModel,
    Field,
    model_validator,
)

# ============================================================
# BBOX PDF VISUAL
# ============================================================


class VisualPDFBBox(BaseModel):
    """
    Bounding box en el sistema VISUAL de la página PDF.

    Unidad:
        puntos PDF.

    NO representa:
        - metros;
        - píxeles;
        - geometría arquitectónica.
    """

    x0: float
    y0: float
    x1: float
    y1: float

    @model_validator(mode="after")
    def validate_bbox(
        self,
    ) -> "VisualPDFBBox":

        if self.x1 <= self.x0:
            raise ValueError("VisualPDFBBox.x1 debe ser mayor que x0.")

        if self.y1 <= self.y0:
            raise ValueError("VisualPDFBBox.y1 debe ser mayor que y0.")

        return self


# ============================================================
# MARCADOR DE NIVEL
# ============================================================


class PyMuPDFLevelMarker(BaseModel):
    """
    Evidencia textual determinística de nivel.

    Ejemplos:

        Planta Baja
        Planta Alta
        PB
        PA
        Azotea
        Nivel 1

    candidato != verdad final
    """

    page_number: int

    text: str

    normalized_text: str

    bbox_pdf_visual: VisualPDFBBox

    bbox_px: PixelBBox

    source: Literal["pymupdf_vector_text"] = "pymupdf_vector_text"

    confidence: Literal[1.0] = 1.0


# ============================================================
# PÁGINA FUENTE
# ============================================================


class PyMuPDFSourcePage(BaseModel):
    """
    Página PDF preparada para Fase 1.

    Contiene:

        raster canónico de página
        +
        marcadores vectoriales de nivel
        +
        relación PDF visual -> raster
    """

    page_number: int

    width_pdf_visual: float
    height_pdf_visual: float

    rotation: int

    raster_width_px: int
    raster_height_px: int

    raster_mime_type: Literal["image/png"] = "image/png"

    raster_bytes: bytes = Field(
        exclude=True,
        repr=False,
    )

    level_markers: list[PyMuPDFLevelMarker] = Field(default_factory=list)


# ============================================================
# RESULTADO DOCUMENTAL
# ============================================================


class PyMuPDFLevelSourceResult(BaseModel):
    page_count: int

    pages: list[PyMuPDFSourcePage] = Field(default_factory=list)

    vector_level_markers_available: bool

    warnings: list[str] = Field(default_factory=list)


# ============================================================
# PYMuPDF LEVEL SOURCE
# ============================================================


class PyMuPDFLevelSource:
    """
    Entrada PDF real de Fase 1.

    Responsabilidades:

        PDF bytes
            ↓
        PyMuPDF
            ↓
        páginas
            ↓
        raster PNG por página
            +
        marcadores textuales de nivel

    NO:

        - ejecuta Gemini;
        - ejecuta OpenCV;
        - identifica muros;
        - identifica espacios;
        - convierte px -> m;
        - importa servicios legacy;
        - genera LevelView;
        - genera contrato 03.2 -> 04.

    El objetivo es producir evidencia documental real
    para LevelDetector.
    """

    PDF_MIME_TYPE = "application/pdf"

    LEVEL_MARKERS = {
        "planta baja",
        "planta alta",
        "azotea",
        "nivel 1",
        "nivel 2",
        "nivel 3",
        "nivel 4",
        "pb",
        "pa",
    }

    # ========================================================
    # API PRINCIPAL
    # ========================================================

    def read(
        self,
        *,
        document_bytes: bytes,
        render_scale: float,
    ) -> PyMuPDFLevelSourceResult:
        """
        Abre un PDF y prepara todas sus páginas.

        render_scale es obligatorio y explícito.

        No se introduce resolución PDF silenciosa.
        """

        if not document_bytes:
            raise ValueError("document_bytes no puede estar vacío.")

        self._validate_render_scale(render_scale)

        try:
            document = pymupdf.open(
                stream=document_bytes,
                filetype="pdf",
            )

        except Exception as exc:
            raise ValueError("No fue posible abrir el PDF con PyMuPDF.") from exc

        pages: list[PyMuPDFSourcePage] = []

        warnings: list[str] = []

        total_markers = 0

        try:
            for page_index in range(document.page_count):
                page = document.load_page(page_index)

                source_page = self._read_page(
                    page=page,
                    page_number=page_index + 1,
                    render_scale=render_scale,
                )

                total_markers += len(source_page.level_markers)

                pages.append(source_page)

        finally:
            document.close()

        if not pages:
            warnings.append("El PDF no contiene páginas.")

        if total_markers == 0:
            warnings.append(
                "PyMuPDF no encontró marcadores vectoriales "
                "de nivel. La identificación deberá apoyarse "
                "en otras evidencias."
            )

        return PyMuPDFLevelSourceResult(
            page_count=len(pages),
            pages=pages,
            vector_level_markers_available=(total_markers > 0),
            warnings=warnings,
        )

    # ========================================================
    # PÁGINA
    # ========================================================

    def _read_page(
        self,
        *,
        page: pymupdf.Page,
        page_number: int,
        render_scale: float,
    ) -> PyMuPDFSourcePage:

        visual_rect = page.rect

        width_pdf_visual = float(visual_rect.width)

        height_pdf_visual = float(visual_rect.height)

        if width_pdf_visual <= 0 or height_pdf_visual <= 0:
            raise ValueError(
                f"La página {page_number} posee dimensiones PDF inválidas."
            )

        # ----------------------------------------------------
        # RASTER CANÓNICO DE PÁGINA
        # ----------------------------------------------------

        pixmap = page.get_pixmap(
            matrix=pymupdf.Matrix(
                render_scale,
                render_scale,
            ),
            alpha=False,
        )

        raster_width_px = int(pixmap.width)

        raster_height_px = int(pixmap.height)

        if raster_width_px <= 0 or raster_height_px <= 0:
            raise ValueError(f"La página {page_number} produjo un raster inválido.")

        raster_bytes = pixmap.tobytes("png")

        # ----------------------------------------------------
        # EVIDENCIA VECTORIAL
        # ----------------------------------------------------

        level_markers = self._extract_level_markers(
            page=page,
            page_number=page_number,
            width_pdf_visual=(width_pdf_visual),
            height_pdf_visual=(height_pdf_visual),
            raster_width_px=(raster_width_px),
            raster_height_px=(raster_height_px),
        )

        return PyMuPDFSourcePage(
            page_number=page_number,
            width_pdf_visual=(width_pdf_visual),
            height_pdf_visual=(height_pdf_visual),
            rotation=int(page.rotation),
            raster_width_px=(raster_width_px),
            raster_height_px=(raster_height_px),
            raster_bytes=(raster_bytes),
            level_markers=(level_markers),
        )

    # ========================================================
    # TEXTO VECTORIAL
    # ========================================================

    def _extract_level_markers(
        self,
        *,
        page: pymupdf.Page,
        page_number: int,
        width_pdf_visual: float,
        height_pdf_visual: float,
        raster_width_px: int,
        raster_height_px: int,
    ) -> list[PyMuPDFLevelMarker]:

        result: list[PyMuPDFLevelMarker] = []

        try:
            raw = page.get_text("dict")

        except Exception:
            return []

        rotation_matrix = page.rotation_matrix

        for block in raw.get(
            "blocks",
            [],
        ):
            if not isinstance(
                block,
                dict,
            ):
                continue

            lines = block.get("lines")

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

                spans = line.get("spans")

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

                    text = str(
                        span.get(
                            "text",
                            "",
                        )
                    ).strip()

                    if not text:
                        continue

                    normalized = self._normalize_text(text)

                    if normalized not in self.LEVEL_MARKERS:
                        continue

                    raw_bbox = span.get("bbox")

                    if (
                        not isinstance(
                            raw_bbox,
                            (list, tuple),
                        )
                        or len(raw_bbox) != 4
                    ):
                        continue

                    original_rect = pymupdf.Rect(
                        float(raw_bbox[0]),
                        float(raw_bbox[1]),
                        float(raw_bbox[2]),
                        float(raw_bbox[3]),
                    )

                    visual_bbox = original_rect * rotation_matrix

                    pdf_bbox = VisualPDFBBox(
                        x0=float(visual_bbox.x0),
                        y0=float(visual_bbox.y0),
                        x1=float(visual_bbox.x1),
                        y1=float(visual_bbox.y1),
                    )

                    pixel_bbox = self._pdf_visual_to_pixel_bbox(
                        bbox=pdf_bbox,
                        width_pdf_visual=(width_pdf_visual),
                        height_pdf_visual=(height_pdf_visual),
                        raster_width_px=(raster_width_px),
                        raster_height_px=(raster_height_px),
                    )

                    result.append(
                        PyMuPDFLevelMarker(
                            page_number=(page_number),
                            text=text,
                            normalized_text=(normalized),
                            bbox_pdf_visual=(pdf_bbox),
                            bbox_px=(pixel_bbox),
                        )
                    )

        return result

    # ========================================================
    # PDF VISUAL -> RASTER
    # ========================================================

    @staticmethod
    def _pdf_visual_to_pixel_bbox(
        *,
        bbox: VisualPDFBBox,
        width_pdf_visual: float,
        height_pdf_visual: float,
        raster_width_px: int,
        raster_height_px: int,
    ) -> PixelBBox:
        """
        Convierte una posición PDF VISUAL al raster generado
        de esa misma página.

        Esto NO es px -> m.

        Es únicamente una transformación entre dos
        representaciones de la misma página.
        """

        if width_pdf_visual <= 0 or height_pdf_visual <= 0:
            raise ValueError("Las dimensiones PDF deben ser positivas.")

        if raster_width_px <= 0 or raster_height_px <= 0:
            raise ValueError("Las dimensiones raster deben ser positivas.")

        scale_x = raster_width_px / width_pdf_visual

        scale_y = raster_height_px / height_pdf_visual

        x_min = math.floor(bbox.x0 * scale_x)

        y_min = math.floor(bbox.y0 * scale_y)

        x_max = math.ceil(bbox.x1 * scale_x)

        y_max = math.ceil(bbox.y1 * scale_y)

        x_min = max(
            0,
            min(
                raster_width_px - 1,
                x_min,
            ),
        )

        y_min = max(
            0,
            min(
                raster_height_px - 1,
                y_min,
            ),
        )

        x_max = max(
            x_min + 1,
            min(
                raster_width_px,
                x_max,
            ),
        )

        y_max = max(
            y_min + 1,
            min(
                raster_height_px,
                y_max,
            ),
        )

        return PixelBBox(
            x_min=x_min,
            y_min=y_min,
            x_max=x_max,
            y_max=y_max,
        )

    # ========================================================
    # NORMALIZACIÓN
    # ========================================================

    @staticmethod
    def _normalize_text(
        value: str,
    ) -> str:

        raw = str(value or "").strip().lower()

        decomposed = unicodedata.normalize(
            "NFD",
            raw,
        )

        without_accents = "".join(
            character
            for character in decomposed
            if unicodedata.category(character) != "Mn"
        )

        return " ".join(without_accents.split())

    # ========================================================
    # RENDER SCALE
    # ========================================================

    @staticmethod
    def _validate_render_scale(
        render_scale: float,
    ) -> None:

        if isinstance(
            render_scale,
            bool,
        ):
            raise ValueError("render_scale inválido.")

        if not isinstance(
            render_scale,
            (int, float),
        ):
            raise ValueError("render_scale debe ser numérico.")

        if float(render_scale) <= 0:
            raise ValueError("render_scale debe ser mayor que cero.")


# ============================================================
# FACTORY INTERNA
# ============================================================


def get_pymupdf_level_source() -> PyMuPDFLevelSource:
    return PyMuPDFLevelSource()
