from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


DrawingSource = Literal[
    "PYMUPDF",
    "OPENCV",
    "OCR",
    "GEMINI",
    "LSD_LOCAL",
    "REGION_LOCAL",
]

DrawingPrimitiveKind = Literal[
    "LINE",
    "REGION_CENTERLINE",
]

DrawingCurveKind = Literal["CUBIC_BEZIER"]


class DrawingPoint(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    x: float
    y: float


class DrawingBBox(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    x_min: float
    y_min: float
    x_max: float
    y_max: float


class DrawingLine(BaseModel):
    """Primitiva geométrica normalizada; todavía no representa un muro."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    kind: DrawingPrimitiveKind = "LINE"
    start: DrawingPoint
    end: DrawingPoint
    length_px: float = Field(gt=0.0)
    angle_deg: float = Field(ge=0.0, lt=180.0)
    stroke_width_px: float | None = Field(default=None, ge=0.0)
    dashed: bool = False
    evidence_ids: list[str] = Field(default_factory=list)
    sources: list[DrawingSource] = Field(default_factory=list)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class DrawingCurve(BaseModel):
    """Curva vectorial normalizada conservada desde F01.5.

    No representa todavía una puerta, ventana u otro elemento. El objetivo es
    impedir que primitivas vectoriales informativas se pierdan antes de F03.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    kind: DrawingCurveKind = "CUBIC_BEZIER"
    control_points: list[DrawingPoint] = Field(min_length=4, max_length=4)
    bbox: DrawingBBox
    evidence_ids: list[str] = Field(default_factory=list)
    sources: list[DrawingSource] = Field(default_factory=list)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class DrawingRegion(BaseModel):
    """Región observada; se conserva como evidencia geométrica, no semántica."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    bbox: DrawingBBox
    area_px2: float = Field(ge=0.0)
    evidence_ids: list[str] = Field(default_factory=list)
    sources: list[DrawingSource] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class DrawingText(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    bbox: DrawingBBox | None = None
    evidence_ids: list[str] = Field(default_factory=list)
    sources: list[DrawingSource] = Field(default_factory=list)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class SemanticObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    family: str = Field(min_length=1)
    text: str | None = None
    bbox: DrawingBBox | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    payload: dict[str, Any] = Field(default_factory=dict)


class DrawingRasterLayer(BaseModel):
    """Máscara técnica temporal para el Reconstruction Core.

    `png_bytes` queda excluido al serializar resultados para evitar contaminar el
    contrato final; la máscara no forma parte del modelo paramétrico publicado.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    kind: Literal["INK_MASK", "WALL_REGION_MASK"]
    width_px: int = Field(gt=0)
    height_px: int = Field(gt=0)
    png_bytes: bytes = Field(exclude=True, repr=False)


class DrawingModelDiagnostics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    raw_evidence_count: int = Field(ge=0)
    normalized_line_count: int = Field(ge=0)
    normalized_region_count: int = Field(ge=0)
    normalized_curve_count: int = Field(default=0, ge=0)
    text_count: int = Field(ge=0)
    semantic_observation_count: int = Field(ge=0)
    lsd_line_count: int = Field(ge=0)
    region_centerline_count: int = Field(ge=0)
    source_counts: dict[str, int] = Field(default_factory=dict)


class DrawingModel(BaseModel):
    """Representación intermedia única entre evidencia y reconstrucción."""

    model_config = ConfigDict(extra="forbid")

    level_view_id: str = Field(min_length=1)
    width_px: int = Field(gt=0)
    height_px: int = Field(gt=0)
    lines: list[DrawingLine] = Field(default_factory=list)
    regions: list[DrawingRegion] = Field(default_factory=list)
    curves: list[DrawingCurve] = Field(default_factory=list)
    texts: list[DrawingText] = Field(default_factory=list)
    semantic_observations: list[SemanticObservation] = Field(default_factory=list)
    raster_layers: list[DrawingRasterLayer] = Field(default_factory=list)
    diagnostics: DrawingModelDiagnostics
    warnings: list[str] = Field(default_factory=list)
