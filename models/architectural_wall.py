from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


ArchitecturalWallState = Literal[
    "DETECTADO",
    "INFERIDO",
    "NO_IDENTIFICADO",
    "CONFLICTO",
    "CANDIDATO",
    "PENDIENTE",
]

ArchitecturalWallType = Literal[
    "PERIMETRAL",
    "DIVISORIO",
    "NO_IDENTIFICADO",
]

ArchitecturalWallOrientation = Literal[
    "horizontal",
    "vertical",
    "oblicuo",
    "NO_IDENTIFICADO",
]


class WallPointPx(BaseModel):
    x: float
    y: float


class ArchitecturalWallEvidence(BaseModel):
    """
    Evidencia utilizada para sustentar un muro arquitectónico.

    La evidencia puede provenir de:

    - geometría vectorial;
    - OpenCV;
    - ejes;
    - cotas;
    - grounding;
    - topología;
    - otras fuentes internas del motor.

    Una evidencia no confirma por sí sola el muro.
    """

    source: str

    reference_id: str | None = None

    description: str | None = None

    confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    @field_validator("source")
    @classmethod
    def validate_source(
        cls,
        value: str,
    ) -> str:
        normalized = value.strip()

        if not normalized:
            raise ValueError(
                "ArchitecturalWallEvidence.source no puede estar vacío."
            )

        return normalized


class ArchitecturalWall(BaseModel):
    """
    Objeto canónico de muro de Quantia Spatial.

    Este objeto representa el muro arquitectónico resultante
    de Fase 3.

    No representa:

    - una línea OpenCV;
    - un BoundaryCandidate;
    - un wall run;
    - un centerline;
    - un GraphEdge;
    - una face Shapely.

    Todas esas estructuras pueden existir durante el proceso,
    pero permanecen como evidencia o estructuras internas.

    Ningún muro generado automáticamente puede quedar
    confirmed=True.
    """

    id: str

    level_view_id: str

    wall_type: ArchitecturalWallType = "NO_IDENTIFICADO"

    orientation: ArchitecturalWallOrientation = "NO_IDENTIFICADO"

    start: WallPointPx

    end: WallPointPx

    length_px: float = Field(
        gt=0.0,
    )

    thickness_m: float | None = Field(
        default=None,
        gt=0.0,
    )

    thickness_state: ArchitecturalWallState = "PENDIENTE"

    thickness_source: str | None = None

    axis_ids: list[str] = Field(
        default_factory=list,
    )

    dimension_ids: list[str] = Field(
        default_factory=list,
    )

    evidence: list[ArchitecturalWallEvidence] = Field(
        default_factory=list,
    )

    confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    state: ArchitecturalWallState = "CANDIDATO"

    confirmed: Literal[False] = False

    @field_validator(
        "id",
        "level_view_id",
    )
    @classmethod
    def validate_required_text(
        cls,
        value: str,
    ) -> str:
        normalized = value.strip()

        if not normalized:
            raise ValueError(
                "Los identificadores de ArchitecturalWall "
                "no pueden estar vacíos."
            )

        return normalized

    @field_validator(
        "axis_ids",
        "dimension_ids",
    )
    @classmethod
    def normalize_reference_ids(
        cls,
        values: list[str],
    ) -> list[str]:
        result: list[str] = []

        for value in values:
            normalized = str(value).strip()

            if not normalized:
                continue

            if normalized not in result:
                result.append(normalized)

        return result

    @field_validator("thickness_source")
    @classmethod
    def normalize_thickness_source(
        cls,
        value: str | None,
    ) -> str | None:
        if value is None:
            return None

        normalized = value.strip()

        return normalized or None

    @model_validator(mode="after")
    def validate_geometry(
        self,
    ) -> "ArchitecturalWall":
        dx = self.end.x - self.start.x
        dy = self.end.y - self.start.y

        geometric_length = math.hypot(
            dx,
            dy,
        )

        if geometric_length <= 0.0:
            raise ValueError(
                "ArchitecturalWall requiere puntos start/end distintos."
            )

        if (
            self.orientation == "horizontal"
            and not math.isclose(
                self.start.y,
                self.end.y,
                rel_tol=0.0,
                abs_tol=1e-6,
            )
        ):
            raise ValueError(
                "Un ArchitecturalWall horizontal debe conservar "
                "la misma coordenada Y en start/end."
            )

        if (
            self.orientation == "vertical"
            and not math.isclose(
                self.start.x,
                self.end.x,
                rel_tol=0.0,
                abs_tol=1e-6,
            )
        ):
            raise ValueError(
                "Un ArchitecturalWall vertical debe conservar "
                "la misma coordenada X en start/end."
            )

        return self

    @model_validator(mode="after")
    def validate_thickness(
        self,
    ) -> "ArchitecturalWall":
        if self.thickness_m is None:
            if self.thickness_state == "DETECTADO":
                raise ValueError(
                    "thickness_state=DETECTADO requiere thickness_m."
                )

            return self

        if self.thickness_state in {
            "NO_IDENTIFICADO",
            "PENDIENTE",
        }:
            raise ValueError(
                "Un muro con thickness_m no puede conservar "
                "un thickness_state sin resolver."
            )

        return self