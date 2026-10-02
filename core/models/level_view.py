from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

# ============================================================
# ESTADOS INTERNOS DE IDENTIFICACIÓN
# ============================================================


LevelIdentificationState = Literal[
    "DETECTADO",
    "INFERIDO",
    "NO_IDENTIFICADO",
    "CONFLICTO",
    "CANDIDATO",
    "PENDIENTE",
]


# ============================================================
# GEOMETRÍA RASTER DE FASE 1
# ============================================================


class PixelPoint(BaseModel):
    """
    Punto expresado exclusivamente en coordenadas raster.

    No representa metros ni coordenadas PDF.
    """

    x: int
    y: int


class PixelBBox(BaseModel):
    """
    Bounding box dentro del raster original de la página.

    Convención:

        x_min, y_min -> esquina superior izquierda
        x_max, y_max -> esquina inferior derecha

    Las coordenadas están expresadas en píxeles.
    """

    x_min: int
    y_min: int
    x_max: int
    y_max: int

    @model_validator(mode="after")
    def validate_bbox(self) -> "PixelBBox":
        if self.x_min < 0 or self.y_min < 0:
            raise ValueError("PixelBBox no admite coordenadas negativas.")

        if self.x_max <= self.x_min:
            raise ValueError("PixelBBox.x_max debe ser mayor que x_min.")

        if self.y_max <= self.y_min:
            raise ValueError("PixelBBox.y_max debe ser mayor que y_min.")

        return self

    @property
    def width(self) -> int:
        return self.x_max - self.x_min

    @property
    def height(self) -> int:
        return self.y_max - self.y_min


# ============================================================
# TRANSFORMACIÓN LEVEL VIEW <-> PÁGINA
# ============================================================


class LevelViewTransform(BaseModel):
    """
    Transformación entre la vista local del nivel y el raster
    original de la página.

    FASE 1 NO reescala la región aislada.

    Por lo tanto:

        page_x = local_x + offset_x_px
        page_y = local_y + offset_y_px

    Mantener únicamente traslación permite conservar una
    correspondencia exacta entre la geometría detectada dentro
    del LevelView y la página original.

    No existe conversión px -> m en este modelo.
    """

    offset_x_px: int
    offset_y_px: int

    source_page_width_px: int
    source_page_height_px: int

    local_width_px: int
    local_height_px: int

    @model_validator(mode="after")
    def validate_transform(self) -> "LevelViewTransform":
        if self.offset_x_px < 0 or self.offset_y_px < 0:
            raise ValueError("Los offsets del LevelView no pueden ser negativos.")

        if self.source_page_width_px <= 0:
            raise ValueError("source_page_width_px debe ser mayor que cero.")

        if self.source_page_height_px <= 0:
            raise ValueError("source_page_height_px debe ser mayor que cero.")

        if self.local_width_px <= 0:
            raise ValueError("local_width_px debe ser mayor que cero.")

        if self.local_height_px <= 0:
            raise ValueError("local_height_px debe ser mayor que cero.")

        if self.offset_x_px + self.local_width_px > self.source_page_width_px:
            raise ValueError("El LevelView excede el ancho de la página original.")

        if self.offset_y_px + self.local_height_px > self.source_page_height_px:
            raise ValueError("El LevelView excede el alto de la página original.")

        return self

    def local_to_page(
        self,
        point: PixelPoint,
    ) -> PixelPoint:
        """
        Convierte un punto del LevelView a coordenadas
        de la página raster original.
        """

        if (
            point.x < 0
            or point.y < 0
            or point.x > self.local_width_px
            or point.y > self.local_height_px
        ):
            raise ValueError("El punto local está fuera de los límites del LevelView.")

        return PixelPoint(
            x=point.x + self.offset_x_px,
            y=point.y + self.offset_y_px,
        )

    def page_to_local(
        self,
        point: PixelPoint,
    ) -> PixelPoint:
        """
        Convierte un punto de la página raster original
        a coordenadas locales del LevelView.
        """

        local_x = point.x - self.offset_x_px
        local_y = point.y - self.offset_y_px

        if (
            local_x < 0
            or local_y < 0
            or local_x > self.local_width_px
            or local_y > self.local_height_px
        ):
            raise ValueError("El punto de página no pertenece a este LevelView.")

        return PixelPoint(
            x=local_x,
            y=local_y,
        )


# ============================================================
# EVIDENCIA DE IDENTIFICACIÓN
# ============================================================


class LevelViewEvidence(BaseModel):
    """
    Evidencia utilizada para identificar o delimitar un nivel.

    Ejemplos de fuente:

        PyMuPDF
        Gemini
        OpenCV

    La evidencia no define por sí misma la geometría final.
    """

    source: str

    reference: str | None = None

    text: str | None = None

    page_number: int

    bbox_px: PixelBBox | None = None

    confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )


# ============================================================
# LEVEL VIEW
# ============================================================


class LevelView(BaseModel):
    """
    Unidad natural de reconstrucción del nuevo motor 03.2.

    Una página puede producir:

        1 LevelView
        N LevelView

    y cada LevelView será procesado posteriormente de manera
    independiente por:

        Fase 1.5
        Fase 2
        Fase 3
        Fase 4
        Fase 5

    Este objeto NO contiene geometría arquitectónica final.

    Únicamente conserva:

        - identidad del nivel;
        - región aislada;
        - raster local;
        - transformación hacia la página original;
        - evidencia;
        - estado de identificación.

    confirmed permanece siempre False porque el resultado
    automático todavía no ha sido confirmado por el usuario.
    """

    id: str

    level_name: str | None = None

    source_document_id: str | None = None

    source_page_number: int

    source_bbox_px: PixelBBox

    source_page_width_px: int

    source_page_height_px: int

    raster_width_px: int

    raster_height_px: int

    raster_mime_type: str

    # Escala métrica del sistema de coordenadas px publicado por F01.
    # Se llena únicamente después de la normalización métrica canónica.
    # F02 conserva su propia verdad geométrica y no es responsable de
    # transportar esta relación hacia F03.
    metric_scale_m_per_px: float | None = Field(default=None, gt=0.0)
    metric_scale_source: str | None = None

    raster_bytes: bytes = Field(
        exclude=True,
        repr=False,
    )

    transform: LevelViewTransform

    state: LevelIdentificationState

    confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    evidence: list[LevelViewEvidence] = Field(default_factory=list)

    confirmed: Literal[False] = False

    @model_validator(mode="after")
    def validate_level_view(self) -> "LevelView":
        if self.source_page_number <= 0:
            raise ValueError("source_page_number debe ser mayor que cero.")

        if self.source_page_width_px <= 0:
            raise ValueError("source_page_width_px debe ser mayor que cero.")

        if self.source_page_height_px <= 0:
            raise ValueError("source_page_height_px debe ser mayor que cero.")

        if self.raster_width_px <= 0:
            raise ValueError("raster_width_px debe ser mayor que cero.")

        if self.raster_height_px <= 0:
            raise ValueError("raster_height_px debe ser mayor que cero.")

        if not self.raster_bytes:
            raise ValueError("LevelView requiere raster_bytes.")

        if self.source_bbox_px.x_max > self.source_page_width_px:
            raise ValueError("source_bbox_px excede el ancho de la página.")

        if self.source_bbox_px.y_max > self.source_page_height_px:
            raise ValueError("source_bbox_px excede el alto de la página.")

        if self.raster_width_px != self.source_bbox_px.width:
            raise ValueError(
                "El ancho del raster aislado debe coincidir "
                "con el ancho de source_bbox_px."
            )

        if self.raster_height_px != self.source_bbox_px.height:
            raise ValueError(
                "El alto del raster aislado debe coincidir "
                "con el alto de source_bbox_px."
            )

        if self.transform.offset_x_px != self.source_bbox_px.x_min:
            raise ValueError(
                "transform.offset_x_px debe coincidir con source_bbox_px.x_min."
            )

        if self.transform.offset_y_px != self.source_bbox_px.y_min:
            raise ValueError(
                "transform.offset_y_px debe coincidir con source_bbox_px.y_min."
            )

        if self.transform.local_width_px != self.raster_width_px:
            raise ValueError(
                "transform.local_width_px debe coincidir con raster_width_px."
            )

        if self.transform.local_height_px != self.raster_height_px:
            raise ValueError(
                "transform.local_height_px debe coincidir con raster_height_px."
            )

        if self.transform.source_page_width_px != self.source_page_width_px:
            raise ValueError(
                "El ancho de página de transform no coincide con LevelView."
            )

        if self.transform.source_page_height_px != self.source_page_height_px:
            raise ValueError(
                "El alto de página de transform no coincide con LevelView."
            )

        return self
