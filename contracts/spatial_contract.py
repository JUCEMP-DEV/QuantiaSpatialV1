from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

# ============================================================
# ESTADOS
# ============================================================


ContractValidationState = Literal[
    "VALID",
    "REVIEW",
    "INVALID",
]


ContractElementState = Literal[
    "DETECTADO",
    "INFERIDO",
    "NO_IDENTIFICADO",
    "CONFLICTO",
    "CANDIDATO",
    "PENDIENTE",
]


# ============================================================
# GEOMETRÍA
# ============================================================


class ContractPoint(BaseModel):
    x: float
    y: float


class ContractBBox(BaseModel):
    x_min: float
    y_min: float
    x_max: float
    y_max: float


class ContractRasterGeometry(BaseModel):
    """
    Geometría estrictamente raster.

    Nunca representa metros.
    """

    coordinate_system: Literal["raster_px"] = "raster_px"

    geometry_type: Literal[
        "POINT",
        "SEGMENT",
        "POLYLINE",
        "POLYGON",
        "BBOX",
    ]

    vertices: list[ContractPoint] = Field(
        default_factory=list,
    )

    bbox: ContractBBox | None = None

    area_px2: float | None = None
    perimeter_px: float | None = None


class ContractMetricGeometry(BaseModel):
    """
    Geometría métrica únicamente cuando existe
    grounding arquitectónico válido.
    """

    coordinate_system: Literal["metric_m"] = "metric_m"

    geometry_type: Literal[
        "POINT",
        "SEGMENT",
        "POLYLINE",
        "POLYGON",
        "BBOX",
    ]

    vertices: list[ContractPoint] = Field(
        default_factory=list,
    )

    area_m2: float | None = None
    perimeter_m: float | None = None


class ContractGeometry(BaseModel):
    raster: ContractRasterGeometry | None = None

    metric: ContractMetricGeometry | None = None


# ============================================================
# EVIDENCIA
# ============================================================


class ContractEvidence(BaseModel):
    evidence_id: str | None = None

    source: str

    description: str | None = None

    text: str | None = None

    confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    metadata: dict[
        str,
        Any,
    ] = Field(
        default_factory=dict,
    )


# ============================================================
# BASE LAYER
# ============================================================


class ContractBaseLayer(BaseModel):
    level_view_id: str

    source_document_id: str

    page_number: int

    mime_type: str

    width_px: int = Field(gt=0)
    height_px: int = Field(gt=0)

    source_bbox_px: ContractBBox | None = None

    offset_x_px: float = 0.0
    offset_y_px: float = 0.0


# ============================================================
# COTAS
# ============================================================


class ContractDimension(BaseModel):
    id: str

    value_m: float | None = Field(
        default=None,
        gt=0,
    )

    pixel_span: float | None = Field(
        default=None,
        gt=0,
    )

    pixels_per_meter: float | None = Field(
        default=None,
        gt=0,
    )

    start_axis_id: str | None = None
    end_axis_id: str | None = None

    start_axis_label: str | None = None
    end_axis_label: str | None = None

    evidence: list[ContractEvidence] = Field(
        default_factory=list,
    )

    confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    state: ContractElementState

    confirmed: bool = False


# ============================================================
# BOUNDARY
# ============================================================


class ContractBoundary(BaseModel):
    id: str

    kind: Literal[
        "SITE",
        "LEVEL_BOUNDARY",
    ]

    geometry: ContractGeometry

    dimensions: list[ContractDimension] = Field(
        default_factory=list,
    )

    evidence: list[ContractEvidence] = Field(
        default_factory=list,
    )

    confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    state: ContractElementState

    confirmed: bool = False


# ============================================================
# SITE
# ============================================================


class ContractSite(BaseModel):
    boundary: ContractBoundary | None = None

    evidence: list[ContractEvidence] = Field(
        default_factory=list,
    )

    confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    state: ContractElementState

    confirmed: bool = False


# ============================================================
# EJES
# ============================================================


class ContractAxis(BaseModel):
    id: str

    label: str | None = None

    orientation: Literal[
        "horizontal",
        "vertical",
    ]

    coordinate_px: float

    coordinate_m: float | None = None

    evidence: list[ContractEvidence] = Field(
        default_factory=list,
    )

    confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    state: ContractElementState

    confirmed: bool = False


# ============================================================
# MUROS
# ============================================================


class ContractWall(BaseModel):
    id: str

    wall_type: Literal[
        "PERIMETRAL",
        "DIVISORIO",
        "NO_IDENTIFICADO",
    ]

    orientation: Literal[
        "horizontal",
        "vertical",
        "oblicuo",
        "NO_IDENTIFICADO",
    ]

    geometry: ContractGeometry

    length_px: float = Field(gt=0)

    thickness_m: float | None = Field(
        default=None,
        gt=0,
    )

    thickness_state: ContractElementState

    thickness_source: str | None = None

    axis_ids: list[str] = Field(
        default_factory=list,
    )

    dimension_ids: list[str] = Field(
        default_factory=list,
    )

    evidence: list[ContractEvidence] = Field(
        default_factory=list,
    )

    confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    state: ContractElementState

    confirmed: bool = False


# ============================================================
# ZONAS
# ============================================================


class ContractSemanticZone(BaseModel):
    id: str

    room_id: str

    name: str

    geometry: ContractGeometry | None = None

    evidence_ids: list[str] = Field(
        default_factory=list,
    )

    confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    state: ContractElementState

    confirmed: bool = False


# ============================================================
# OPENINGS
# ============================================================


class ContractOpening(BaseModel):
    id: str

    opening_type: Literal[
        "DOOR",
        "WINDOW",
    ]

    host_wall_id: str | None = None

    connected_room_ids: list[str] = Field(
        default_factory=list,
    )

    offset_px: float | None = None

    span_px: float | None = Field(
        default=None,
        ge=0,
    )

    width_m: float | None = Field(
        default=None,
        gt=0,
    )

    width_source: str | None = None

    evidence_ids: list[str] = Field(
        default_factory=list,
    )

    confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    state: ContractElementState

    confirmed: bool = False


# ============================================================
# ESCALERAS
# ============================================================


class ContractStair(BaseModel):
    id: str

    room_id: str | None = None

    geometry: ContractGeometry | None = None

    target_level_id: str | None = None

    direction: str | None = None

    evidence_ids: list[str] = Field(
        default_factory=list,
    )

    confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    state: ContractElementState

    confirmed: bool = False


# ============================================================
# ESPACIOS
# ============================================================


class ContractSpace(BaseModel):
    id: str

    name: str | None = None

    semantic_mode: str

    geometry: ContractGeometry

    area_m2: float | None = Field(
        default=None,
        gt=0,
    )

    area_source: str | None = None

    wall_ids: list[str] = Field(
        default_factory=list,
    )

    adjacent_space_ids: list[str] = Field(
        default_factory=list,
    )

    zone_ids: list[str] = Field(
        default_factory=list,
    )

    opening_ids: list[str] = Field(
        default_factory=list,
    )

    stair_ids: list[str] = Field(
        default_factory=list,
    )

    evidence: list[ContractEvidence] = Field(
        default_factory=list,
    )

    confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    state: ContractElementState

    confirmed: bool = False

    @model_validator(mode="after")
    def validate_metric_area(
        self,
    ) -> "ContractSpace":

        if self.area_m2 is not None and not self.area_source:
            raise ValueError("area_m2 requiere area_source.")

        return self


# ============================================================
# NIVEL
# ============================================================


class ContractLevel(BaseModel):
    id: str

    name: str

    page_number: int

    base_layer: ContractBaseLayer

    level_boundary: ContractBoundary

    walls: list[ContractWall] = Field(
        default_factory=list,
    )

    axes: list[ContractAxis] = Field(
        default_factory=list,
    )

    dimensions: list[ContractDimension] = Field(
        default_factory=list,
    )

    spaces: list[ContractSpace] = Field(
        default_factory=list,
    )

    zones: list[ContractSemanticZone] = Field(
        default_factory=list,
    )

    doors: list[ContractOpening] = Field(
        default_factory=list,
    )

    windows: list[ContractOpening] = Field(
        default_factory=list,
    )

    stairs: list[ContractStair] = Field(
        default_factory=list,
    )

    evidence: list[ContractEvidence] = Field(
        default_factory=list,
    )

    confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    status: ContractValidationState

    confirmed: bool = False


# ============================================================
# REVISION HISTORY
# ============================================================


class ContractRevision(BaseModel):
    revision_id: str

    source: str

    description: str | None = None

    entity_ids: list[str] = Field(
        default_factory=list,
    )

    metadata: dict[
        str,
        Any,
    ] = Field(
        default_factory=dict,
    )


# ============================================================
# QUANTIA SPATIAL CONTRACT
# ============================================================


class QuantiaSpatialContract(BaseModel):
    """
    Contrato interno canónico del nuevo motor.

    No depende del router ni del schema legacy.

    Fase 06 solamente publica información ya producida
    y validada por las fases anteriores.
    """

    contract_version: str

    site: ContractSite | None = None

    levels: list[ContractLevel] = Field(
        default_factory=list,
    )

    confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    status: ContractValidationState

    evidence: list[ContractEvidence] = Field(
        default_factory=list,
    )

    revision_history: list[ContractRevision] = Field(
        default_factory=list,
    )

    confirmed: bool = False
