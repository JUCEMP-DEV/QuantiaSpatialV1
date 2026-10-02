from __future__ import annotations

from statistics import median
from typing import Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field

from app.quantia_spatialV1.models.level_view import LevelView
from app.quantia_spatialV1.phase_02_boundaries.perimeter_delivery import EditablePerimeterModel


ScaleResolutionState = Literal["RESOLVED", "PARTIAL", "UNRESOLVED"]
LevelScaleState = Literal["RESOLVED", "UNRESOLVED"]


class LevelScaleProfile(BaseModel):
    """Escala normalizada de un LevelView dentro de su proyecto.

    Las coordenadas originales nunca se modifican. `scale_factor_to_canonical`
    permite expresar cualquier longitud nativa en el espacio normalizado común:

        normalized_px = native_px * scale_factor_to_canonical

    cuando F02 dispone de `metric_scale_m_per_px`.
    """

    model_config = ConfigDict(extra="forbid")

    level_view_id: str = Field(min_length=1)
    state: LevelScaleState
    local_m_per_px: float | None = Field(default=None, gt=0.0)
    canonical_m_per_px: float | None = Field(default=None, gt=0.0)
    scale_factor_to_canonical: float = Field(default=1.0, gt=0.0)
    local_min_dim_px: float = Field(gt=0.0)
    normalized_min_dim_px: float = Field(gt=0.0)
    project_reference_min_dim_normalized_px: float = Field(gt=0.0)
    project_reference_area_normalized_px2: float = Field(gt=0.0)
    source: str = Field(min_length=1)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    @property
    def is_resolved(self) -> bool:
        return self.local_m_per_px is not None and self.canonical_m_per_px is not None


class ProjectScaleContext(BaseModel):
    """Contexto común de escala para todas las plantas relacionadas."""

    model_config = ConfigDict(extra="forbid")

    state: ScaleResolutionState
    canonical_m_per_px: float | None = Field(default=None, gt=0.0)
    project_reference_min_dim_normalized_px: float = Field(gt=0.0)
    project_reference_area_normalized_px2: float = Field(gt=0.0)
    resolved_level_count: int = Field(ge=0)
    unresolved_level_ids: list[str] = Field(default_factory=list)
    profiles: list[LevelScaleProfile] = Field(default_factory=list)
    source: str = Field(default="PROJECT_LEVEL_SCALE_NORMALIZER_V6_3", min_length=1)

    def for_level(self, level_view_id: str) -> LevelScaleProfile:
        for profile in self.profiles:
            if profile.level_view_id == level_view_id:
                return profile
        raise KeyError(f"LevelView sin perfil de escala: {level_view_id}")


class ProjectLevelScaleNormalizer:
    """Relaciona escalas de plantas sin asumir que fueron dibujadas igual.

    Fuente canónica: primero la escala publicada por el `LevelView` después de
    la normalización raster; F02 `metric_scale_m_per_px` queda como fallback para
    flujos antiguos o sin rerasterización. Si una planta no tiene escala métrica
    fundamentada, permanece UNRESOLVED; no hereda a ciegas la escala de otra.
    """

    VERSION = "PROJECT_LEVEL_SCALE_NORMALIZER_V6_3"

    def build(
        self,
        *,
        levels: Sequence[tuple[LevelView, EditablePerimeterModel]],
    ) -> ProjectScaleContext:
        if not levels:
            raise ValueError("ProjectLevelScaleNormalizer requiere al menos un LevelView.")

        local_scales: dict[str, float] = {}
        local_dims: dict[str, float] = {}
        local_areas: dict[str, float] = {}
        for level_view, perimeter in levels:
            if perimeter.level_view_id != level_view.id:
                raise ValueError("Perimeter y LevelView no pertenecen al mismo nivel.")
            local_dims[level_view.id] = float(min(level_view.raster_width_px, level_view.raster_height_px))
            local_areas[level_view.id] = float(level_view.raster_width_px * level_view.raster_height_px)
            metric_scale = level_view.metric_scale_m_per_px
            if metric_scale is None:
                metric_scale = perimeter.metric_summary.metric_scale_m_per_px
            if metric_scale is not None:
                local_scales[level_view.id] = float(metric_scale)

        canonical = float(median(local_scales.values())) if local_scales else None

        normalized_dims: dict[str, float] = {}
        normalized_areas: dict[str, float] = {}
        if canonical is not None:
            for level_id, scale in local_scales.items():
                factor = scale / canonical
                normalized_dims[level_id] = local_dims[level_id] * factor
                normalized_areas[level_id] = local_areas[level_id] * factor * factor

        # La referencia común se calcula solo con plantas cuya escala está
        # resuelta. Si ninguna lo está, se usa la mediana de dimensiones nativas
        # únicamente como fallback técnico y el estado queda UNRESOLVED.
        if normalized_dims:
            reference_min_dim = float(median(normalized_dims.values()))
            reference_area = float(median(normalized_areas.values()))
        else:
            reference_min_dim = float(median(local_dims.values()))
            reference_area = float(median(local_areas.values()))

        profiles: list[LevelScaleProfile] = []
        unresolved: list[str] = []
        for level_view, _ in levels:
            level_id = level_view.id
            local_dim = local_dims[level_id]
            local_scale = local_scales.get(level_id)
            if canonical is not None and local_scale is not None:
                factor = local_scale / canonical
                profiles.append(
                    LevelScaleProfile(
                        level_view_id=level_id,
                        state="RESOLVED",
                        local_m_per_px=local_scale,
                        canonical_m_per_px=canonical,
                        scale_factor_to_canonical=factor,
                        local_min_dim_px=local_dim,
                        normalized_min_dim_px=local_dim * factor,
                        project_reference_min_dim_normalized_px=reference_min_dim,
                        project_reference_area_normalized_px2=reference_area,
                        source=(
                            level_view.metric_scale_source
                            or ("LEVELVIEW_METRIC_SCALE" if level_view.metric_scale_m_per_px is not None else "F02_METRIC_SCALE")
                        ),
                        confidence=1.0,
                    )
                )
            else:
                unresolved.append(level_id)
                profiles.append(
                    LevelScaleProfile(
                        level_view_id=level_id,
                        state="UNRESOLVED",
                        local_m_per_px=None,
                        canonical_m_per_px=canonical,
                        scale_factor_to_canonical=1.0,
                        local_min_dim_px=local_dim,
                        normalized_min_dim_px=local_dim,
                        project_reference_min_dim_normalized_px=reference_min_dim,
                        project_reference_area_normalized_px2=reference_area,
                        source="NO_GROUNDED_SCALE",
                        confidence=0.0,
                    )
                )

        if not local_scales:
            state: ScaleResolutionState = "UNRESOLVED"
        elif unresolved:
            state = "PARTIAL"
        else:
            state = "RESOLVED"

        return ProjectScaleContext(
            state=state,
            canonical_m_per_px=canonical,
            project_reference_min_dim_normalized_px=reference_min_dim,
            project_reference_area_normalized_px2=reference_area,
            resolved_level_count=len(local_scales),
            unresolved_level_ids=unresolved,
            profiles=profiles,
            source=self.VERSION,
        )
