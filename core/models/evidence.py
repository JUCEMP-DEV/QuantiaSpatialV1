from __future__ import annotations

from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)

from app.quantia_spatialV1.models.level_view import (
    PixelBBox,
    PixelPoint,
)

# ============================================================
# TIPOS DE EVIDENCIA CRUDA
# ============================================================


EvidenceKind = Literal[
    # --------------------------------------------------------
    # PYMUPDF
    # --------------------------------------------------------
    "VECTOR_LINE",
    "VECTOR_PRIMITIVE",
    "VECTOR_TEXT",
    # --------------------------------------------------------
    # OPENCV
    # --------------------------------------------------------
    "RASTER_LINE",
    "RASTER_INTERSECTION",
    "RASTER_CONTOUR",
    # --------------------------------------------------------
    # OCR
    # --------------------------------------------------------
    "OCR_TEXT",
    # --------------------------------------------------------
    # GEMINI
    # --------------------------------------------------------
    "GEMINI_OBSERVATION",
]


# ============================================================
# FUENTE
# ============================================================


EvidenceSource = Literal[
    "PYMUPDF",
    "OPENCV",
    "OCR",
    "GEMINI",
]


# ============================================================
# TIPO DE GEOMETRÍA
# ============================================================


EvidenceGeometryType = Literal[
    "POINT",
    "SEGMENT",
    "POLYLINE",
    "BBOX",
    "NONE",
]


# ============================================================
# GEOMETRÍA DE EVIDENCIA
# ============================================================


class EvidenceGeometry(BaseModel):
    """
    Geometría observada en coordenadas LOCALES
    del LevelView.

    No representa todavía un objeto arquitectónico.

    Puede representar:

        POINT
            una posición observada.

        SEGMENT
            un trazo lineal.

        POLYLINE
            una secuencia de trazos.

        BBOX
            una región rectangular.

        NONE
            evidencia sin localización geométrica fiable.

    Nunca contiene coordenadas métricas.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    geometry_type: EvidenceGeometryType

    points: list[PixelPoint] = Field(default_factory=list)

    bbox_px: PixelBBox | None = None

    # ========================================================
    # VALIDACIÓN ESTRUCTURAL
    # ========================================================

    @model_validator(mode="after")
    def validate_geometry(
        self,
    ) -> "EvidenceGeometry":

        # ----------------------------------------------------
        # NONE
        # ----------------------------------------------------

        if self.geometry_type == "NONE":
            if self.points:
                raise ValueError("geometry_type=NONE no puede contener points.")

            if self.bbox_px is not None:
                raise ValueError("geometry_type=NONE no puede contener bbox_px.")

            return self

        # ----------------------------------------------------
        # POINT
        # ----------------------------------------------------

        if self.geometry_type == "POINT":
            if len(self.points) != 1:
                raise ValueError("geometry_type=POINT requiere exactamente un punto.")

            return self

        # ----------------------------------------------------
        # SEGMENT
        # ----------------------------------------------------

        if self.geometry_type == "SEGMENT":
            if len(self.points) != 2:
                raise ValueError(
                    "geometry_type=SEGMENT requiere exactamente dos puntos."
                )

            return self

        # ----------------------------------------------------
        # POLYLINE
        # ----------------------------------------------------

        if self.geometry_type == "POLYLINE":
            if len(self.points) < 2:
                raise ValueError("geometry_type=POLYLINE requiere al menos dos puntos.")

            return self

        # ----------------------------------------------------
        # BBOX
        # ----------------------------------------------------

        if self.geometry_type == "BBOX":
            if self.bbox_px is None:
                raise ValueError("geometry_type=BBOX requiere bbox_px.")

            return self

        return self


# ============================================================
# PARÁMETRO NORMALIZADO DE EVIDENCIA
# ============================================================


class EvidenceParameter(BaseModel):
    """
    Vista paramétrica común de una RawEvidence.

    `source_path` apunta al campo crudo de geometry/text/metadata del cual
    procede el parámetro. No convierte evidencia en verdad arquitectónica.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    group: str = Field(min_length=1)
    name: str = Field(min_length=1)
    value: Any

    unit: str | None = None
    state: str | None = None
    confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )
    source_path: str | None = None


# ============================================================
# EVIDENCIA CRUDA NORMALIZADA
# ============================================================


class RawEvidence(BaseModel):
    """
    Contrato común de evidencia cruda de Quantia.

    Todas las fuentes de Fase 01.5 terminan aquí:

        PyMuPDF
        OpenCV
        OCR
        Gemini

                ↓

            RawEvidence

    Cada elemento conserva:

        - identidad propia;
        - LevelView al que pertenece;
        - fuente real;
        - tipo técnico de evidencia;
        - geometría local;
        - texto cuando existe;
        - confianza cuando existe;
        - metadata/procedencia original;
        - parameters: vista paramétrica normalizada de la misma evidencia.

    geometry/text/metadata siguen siendo la capa cruda. `parameters` no los
    sustituye; permite que F02–F06 consuman primero una representación común
    y regresen al crudo mediante `source_path` cuando necesiten más detalle.

    RawEvidence NO representa todavía un objeto arquitectónico FINAL.

    Puede conservar observaciones semánticas de Gemini sobre muros,
    perímetros, ejes, cotas, espacios, puertas, ventanas o escaleras,
    siempre identificadas como evidencia y confirmed=False.

    La reconciliación y publicación de esas verdades pertenece a Fases 02–06.

    La evidencia automática nunca puede quedar confirmada.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    # --------------------------------------------------------
    # IDENTIDAD
    # --------------------------------------------------------

    id: str = Field(min_length=1)

    level_view_id: str = Field(min_length=1)

    # --------------------------------------------------------
    # PROCEDENCIA
    # --------------------------------------------------------

    source: EvidenceSource

    kind: EvidenceKind

    # --------------------------------------------------------
    # CONTENIDO
    # --------------------------------------------------------

    geometry: EvidenceGeometry

    text: str | None = None

    confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    # --------------------------------------------------------
    # INFORMACIÓN ORIGINAL / DIAGNÓSTICA
    # --------------------------------------------------------

    metadata: dict[str, Any] = Field(default_factory=dict)

    # --------------------------------------------------------
    # VISTA PARAMÉTRICA COMÚN
    # --------------------------------------------------------

    parameters: list[EvidenceParameter] = Field(default_factory=list)

    # --------------------------------------------------------
    # CONFIRMACIÓN
    # --------------------------------------------------------

    confirmed: Literal[False] = False
