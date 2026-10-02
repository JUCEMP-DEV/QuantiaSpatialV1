from __future__ import annotations

from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator


class ProjectSiteContext(BaseModel):
    """
    Contexto declarado por el usuario antes de ejecutar QuantiaSpatial.

    Es evidencia independiente del plano:
    - no sustituye observaciones visuales/documentales;
    - no confirma por si sola la geometria del predio;
    - puede reforzar coincidencias y exponer conflictos;
    - debe permanecer disponible para correlacion posterior.
    """

    model_config = ConfigDict(
        populate_by_name=True,
        extra="forbid",
    )

    version: Literal["PROJECT_SITE_CONTEXT_V1"] = "PROJECT_SITE_CONTEXT_V1"
    source: Literal["USER_DECLARED"] = "USER_DECLARED"

    project_location: str | None = Field(
        default=None,
        validation_alias=AliasChoices("project_location", "ubicacionProyecto"),
    )

    site_width_m: float | None = Field(
        default=None,
        gt=0.0,
        validation_alias=AliasChoices("site_width_m", "anchoTerrenoM"),
    )
    site_length_m: float | None = Field(
        default=None,
        gt=0.0,
        validation_alias=AliasChoices("site_length_m", "largoTerrenoM"),
    )
    site_area_m2: float | None = Field(
        default=None,
        gt=0.0,
        validation_alias=AliasChoices("site_area_m2", "areaTerrenoM2"),
    )

    proposed_construction_area_m2: float | None = Field(
        default=None,
        gt=0.0,
        validation_alias=AliasChoices(
            "proposed_construction_area_m2",
            "areaConstruccionPropuestaM2",
        ),
    )
    construction_area_m2: float | None = Field(
        default=None,
        gt=0.0,
        validation_alias=AliasChoices("construction_area_m2", "areaConstruccionM2"),
    )

    level_count: int | None = Field(
        default=None,
        ge=1,
        validation_alias=AliasChoices("level_count", "niveles"),
    )
    level_1_height_m: float | None = Field(
        default=None,
        gt=0.0,
        validation_alias=AliasChoices("level_1_height_m", "alturaNivel1M"),
    )
    level_2_height_m: float | None = Field(
        default=None,
        gt=0.0,
        validation_alias=AliasChoices("level_2_height_m", "alturaNivel2M"),
    )
    level_3_height_m: float | None = Field(
        default=None,
        gt=0.0,
        validation_alias=AliasChoices("level_3_height_m", "alturaNivel3M"),
    )
    average_level_height_m: float | None = Field(
        default=None,
        gt=0.0,
        validation_alias=AliasChoices("average_level_height_m", "alturaPromedioM"),
    )

    structural_system: str | None = Field(
        default=None,
        validation_alias=AliasChoices("structural_system", "sistemaEstructural"),
    )
    foundation_type: str | None = Field(
        default=None,
        validation_alias=AliasChoices("foundation_type", "tipoCimentacion"),
    )
    complexity_level: str | None = Field(
        default=None,
        validation_alias=AliasChoices("complexity_level", "nivelComplejidad"),
    )
    special_conditions: str | None = Field(
        default=None,
        validation_alias=AliasChoices("special_conditions", "condicionesEspeciales"),
    )
    adjustment_factor: float | None = Field(
        default=None,
        gt=0.0,
        validation_alias=AliasChoices("adjustment_factor", "factorAjuste"),
    )
    notes: str | None = Field(
        default=None,
        validation_alias=AliasChoices("notes", "notas"),
    )

    @field_validator(
        "site_width_m",
        "site_length_m",
        "site_area_m2",
        "proposed_construction_area_m2",
        "construction_area_m2",
        "level_count",
        "level_1_height_m",
        "level_2_height_m",
        "level_3_height_m",
        "average_level_height_m",
        "adjustment_factor",
        mode="before",
    )
    @classmethod
    def blank_numeric_to_none(cls, value: Any) -> Any:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator(
        "project_location",
        "structural_system",
        "foundation_type",
        "complexity_level",
        "special_conditions",
        "notes",
        mode="before",
    )
    @classmethod
    def blank_text_to_none(cls, value: Any) -> Any:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @property
    def has_site_dimensions(self) -> bool:
        return any(
            value is not None
            for value in (
                self.site_width_m,
                self.site_length_m,
                self.site_area_m2,
            )
        )

    @property
    def has_meaningful_data(self) -> bool:
        payload = self.model_dump(
            exclude={"version", "source"},
            exclude_none=True,
        )
        return bool(payload)

    def as_prompt_payload(self) -> dict[str, Any]:
        return self.model_dump(exclude_none=True)

    @classmethod
    def coerce(
        cls,
        value: "ProjectSiteContext | dict[str, Any] | None",
    ) -> "ProjectSiteContext | None":
        if value is None:
            return None
        if isinstance(value, cls):
            return value
        if isinstance(value, dict):
            return cls.model_validate(value)
        raise TypeError(
            "project_site_context debe ser ProjectSiteContext, dict o None."
        )
