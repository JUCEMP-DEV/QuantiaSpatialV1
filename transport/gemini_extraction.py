from __future__ import annotations

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)

# ============================================================
# NIVEL DESCUBIERTO
# ============================================================


class GeminiDiscoveredLevel(BaseModel):
    """
    Nivel arquitectónico identificado semánticamente
    dentro de una página o imagen.

    Ejemplos válidos:

        Planta Baja
        Planta Alta
        PB
        PA
        Azotea
        Nivel 1

    Este objeto NO contiene geometría.

    Su responsabilidad es únicamente responder:

        ¿qué nivel aparece?
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    nombre: str = Field(
        min_length=1,
    )

    confianza: float = Field(
        ge=0.0,
        le=1.0,
    )

    @model_validator(mode="after")
    def validate_name(
        self,
    ) -> "GeminiDiscoveredLevel":

        normalized = self.nombre.strip()

        if not normalized:
            raise ValueError("nombre no puede estar vacío.")

        self.nombre = normalized

        return self


# ============================================================
# RESPUESTA DE DESCUBRIMIENTO
# ============================================================


class GeminiLevelDiscoveryResponse(BaseModel):
    """
    Resultado semántico de Fase 1 para una sola página.

    `niveles` contiene únicamente niveles cuyo nombre
    puede sostenerse con evidencia visual/textual.

    `plantas_sin_nombre` permite conservar la evidencia
    de plantas visibles cuyo nivel no puede identificarse
    sin inventar un nombre.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    pagina: int = Field(
        ge=1,
    )

    niveles: list[GeminiDiscoveredLevel] = Field()

    plantas_sin_nombre: int = Field(
        ge=0,
    )

    # ========================================================
    # VALIDACIÓN
    # ========================================================

    @model_validator(mode="after")
    def validate_unique_levels(
        self,
    ) -> "GeminiLevelDiscoveryResponse":

        seen: set[str] = set()

        for level in self.niveles:
            key = self._name_key(level.nombre)

            if key in seen:
                raise ValueError(
                    f"Gemini devolvió un nivel textualmente duplicado: {level.nombre}."
                )

            seen.add(key)

        return self

    # ========================================================
    # PROPIEDADES
    # ========================================================

    @property
    def named_level_count(
        self,
    ) -> int:
        return len(self.niveles)

    @property
    def total_visible_level_count(
        self,
    ) -> int:
        return len(self.niveles) + self.plantas_sin_nombre

    @property
    def has_named_levels(
        self,
    ) -> bool:
        return bool(self.niveles)

    @property
    def has_unnamed_levels(
        self,
    ) -> bool:
        return self.plantas_sin_nombre > 0

    # ========================================================
    # SALIDA PARA FASE 1
    # ========================================================

    def to_known_level_names(
        self,
    ) -> list[str]:
        """
        Devuelve únicamente nombres respaldados
        por la respuesta semántica.

        Las plantas sin nombre NO reciben ningún
        identificador artificial.
        """

        return [level.nombre for level in self.niveles]

    # ========================================================
    # NORMALIZACIÓN
    # ========================================================

    @staticmethod
    def _name_key(
        value: str,
    ) -> str:

        return " ".join(str(value or "").strip().casefold().split())


# ============================================================
# JSON SCHEMA
# ============================================================


def get_gemini_level_discovery_schema() -> dict:
    """
    JSON Schema utilizado exclusivamente para
    descubrimiento semántico de niveles.
    """

    return GeminiLevelDiscoveryResponse.model_json_schema()
