from __future__ import annotations

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)

# ============================================================
# BBOX NORMALIZADO GEMINI
# ============================================================


class GeminiNormalizedBBox(BaseModel):
    """
    Región visual propuesta por Gemini.

    Sistema:
        0.0 -> 1.0

    No representa:

        - metros;
        - píxeles;
        - puntos PDF;
        - geometría arquitectónica final.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    x_min: float = Field(
        ge=0.0,
        le=1.0,
    )

    y_min: float = Field(
        ge=0.0,
        le=1.0,
    )

    x_max: float = Field(
        ge=0.0,
        le=1.0,
    )

    y_max: float = Field(
        ge=0.0,
        le=1.0,
    )

    @model_validator(mode="after")
    def validate_bbox(
        self,
    ) -> "GeminiNormalizedBBox":

        if self.x_max <= self.x_min:
            raise ValueError("x_max debe ser mayor que x_min.")

        if self.y_max <= self.y_min:
            raise ValueError("y_max debe ser mayor que y_min.")

        return self


# ============================================================
# NIVEL LOCALIZADO
# ============================================================


class GeminiLevelLocalizationItem(BaseModel):
    """
    Resultado Gemini para un nivel esperado.

    Gemini puede:

        localizarlo
        o
        declarar que no puede localizarlo.

    Gemini NO puede:

        - inventar geometría arquitectónica;
        - convertir px -> m;
        - confirmar elementos;
        - modificar el nombre esperado silenciosamente.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    nombre: str = Field(
        min_length=1,
    )

    localizado: bool

    bbox_normalizado: GeminiNormalizedBBox | None = None

    confianza: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    # Evidencia mínima de escala capturada en la MISMA llamada de localización.
    # No es una conversión métrica ni autoriza inferir una escala ausente.
    escala_declarada: str | None = None

    confianza_escala: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    @model_validator(mode="after")
    def validate_localization(
        self,
    ) -> "GeminiLevelLocalizationItem":

        name = self.nombre.strip()

        if not name:
            raise ValueError("nombre no puede estar vacío.")

        self.nombre = name

        if self.localizado and self.bbox_normalizado is None:
            raise ValueError("Un nivel localizado requiere bbox_normalizado.")

        if not self.localizado and self.bbox_normalizado is not None:
            raise ValueError(
                "Un nivel no localizado no debe publicar bbox_normalizado."
            )

        if self.escala_declarada is not None:
            scale_text = self.escala_declarada.strip()
            self.escala_declarada = scale_text or None
        if self.escala_declarada is None and self.confianza_escala is not None:
            raise ValueError(
                "confianza_escala requiere una escala_declarada visible."
            )

        return self


# ============================================================
# RESPUESTA GEMINI DE PÁGINA
# ============================================================


class GeminiLevelLocalizationResponse(BaseModel):
    """
    Transporte estructurado exclusivo de:

        FASE 1 — IDENTIFICACIÓN DE NIVEL

    Una respuesta corresponde a una sola página.

    `niveles` es obligatorio en el transporte.
    Puede ser una lista vacía, pero Gemini debe
    publicarlo explícitamente.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    pagina: int = Field(
        ge=1,
    )

    niveles: list[GeminiLevelLocalizationItem] = Field()

    @model_validator(mode="after")
    def validate_unique_names(
        self,
    ) -> "GeminiLevelLocalizationResponse":
        """
        No acepta dos resultados con exactamente
        el mismo nombre textual normalizado.

        La equivalencia semántica PB/Planta Baja continúa
        siendo responsabilidad de LevelDetector.
        """

        seen: set[str] = set()

        for level in self.niveles:
            key = " ".join(level.nombre.strip().casefold().split())

            if key in seen:
                raise ValueError(
                    f"Gemini devolvió un nivel textualmente duplicado: {level.nombre}."
                )

            seen.add(key)

        return self

    def to_detector_payload(
        self,
    ) -> dict:
        """
        Produce exactamente el payload consumido actualmente
        por LevelDetector.

        Esto evita acoplar LevelDetector al provider Gemini.
        """

        return self.model_dump(
            mode="python",
        )


# ============================================================
# JSON SCHEMA PARA STRUCTURED OUTPUT
# ============================================================


def get_gemini_level_localization_schema() -> dict:
    """
    Schema JSON que puede entregarse al provider Gemini.

    El transporte pertenece exclusivamente al nuevo
    app.quantia_spatialV1.
    """

    return GeminiLevelLocalizationResponse.model_json_schema()
