from __future__ import annotations

from itertools import combinations
from typing import Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field

from .scale_evidence_resolver import LevelScaleEvidenceResult


ProjectScaleState = Literal["RESOLVED", "PARTIAL", "CONFLICT", "UNRESOLVED"]


class ProjectScaleProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    level_view_id: str = Field(min_length=1)
    state: str = Field(min_length=1)
    local_m_per_px: float | None = Field(default=None, gt=0.0)
    canonical_m_per_px: float = Field(gt=0.0)
    scale_factor_to_canonical: float = Field(default=1.0, gt=0.0)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    source_method: str | None = None


class ProjectScaleRelation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    level_a_id: str = Field(min_length=1)
    level_b_id: str = Field(min_length=1)
    m_per_px_ratio_a_to_b: float = Field(gt=0.0)
    expected_pixel_density_ratio_b_to_a: float = Field(gt=0.0)


class ReconciledProjectScale(BaseModel):
    model_config = ConfigDict(extra="forbid")

    state: ProjectScaleState
    canonical_m_per_px: float = Field(gt=0.0)
    target_px_per_m: float = Field(gt=0.0)
    resolved_level_count: int = Field(ge=0)
    unresolved_level_ids: list[str] = Field(default_factory=list)
    conflict_level_ids: list[str] = Field(default_factory=list)
    profiles: list[ProjectScaleProfile] = Field(default_factory=list)
    relations: list[ProjectScaleRelation] = Field(default_factory=list)
    source: str = Field(default="PROJECT_SCALE_RECONCILER_V1", min_length=1)

    def for_level(self, level_view_id: str) -> ProjectScaleProfile:
        for profile in self.profiles:
            if profile.level_view_id == level_view_id:
                return profile
        raise KeyError(f"No existe perfil reconciliado para {level_view_id}.")


class ProjectScaleReconciler:
    """Interrelaciona las plantas sin asumir que todas vienen a la misma escala.

    Cada LevelView conserva su escala local. La escala canónica es una densidad
    arquitectónica objetivo común (por defecto 90 px/m), no la mediana de los
    rasters de entrada. Así una planta 1:25 y otra 1:50 pueden compararse en un
    espacio normalizado común sin alterar las coordenadas nativas.
    """

    VERSION = "PROJECT_SCALE_RECONCILER_V1"

    def reconcile(
        self,
        *,
        levels: Sequence[LevelScaleEvidenceResult],
        target_px_per_m: float = 90.0,
    ) -> ReconciledProjectScale:
        results = list(levels)
        if not results:
            raise ValueError("ProjectScaleReconciler requiere al menos una planta.")
        if target_px_per_m <= 0:
            raise ValueError("target_px_per_m debe ser mayor que cero.")
        ids = [item.level_view_id for item in results]
        if len(ids) != len(set(ids)):
            raise ValueError("ProjectScaleReconciler recibió LevelView duplicados.")

        canonical_m_per_px = 1.0 / float(target_px_per_m)
        profiles: list[ProjectScaleProfile] = []
        unresolved: list[str] = []
        conflicts: list[str] = []

        for item in results:
            if item.state == "CONFLICT":
                conflicts.append(item.level_view_id)
            if item.selected_m_per_px is None or item.state not in {"RESOLVED", "CONFLICT"}:
                unresolved.append(item.level_view_id)
                profiles.append(
                    ProjectScaleProfile(
                        level_view_id=item.level_view_id,
                        state=item.state,
                        local_m_per_px=None,
                        canonical_m_per_px=canonical_m_per_px,
                        scale_factor_to_canonical=1.0,
                        confidence=item.selected_confidence,
                        source_method=item.selected_method,
                    )
                )
                continue

            local = float(item.selected_m_per_px)
            profiles.append(
                ProjectScaleProfile(
                    level_view_id=item.level_view_id,
                    state=item.state,
                    local_m_per_px=local,
                    canonical_m_per_px=canonical_m_per_px,
                    scale_factor_to_canonical=local / canonical_m_per_px,
                    confidence=item.selected_confidence,
                    source_method=item.selected_method,
                )
            )

        relations: list[ProjectScaleRelation] = []
        resolved_profiles = [item for item in profiles if item.local_m_per_px is not None]
        for a, b in combinations(resolved_profiles, 2):
            ratio = float(a.local_m_per_px) / float(b.local_m_per_px)
            relations.append(
                ProjectScaleRelation(
                    level_a_id=a.level_view_id,
                    level_b_id=b.level_view_id,
                    m_per_px_ratio_a_to_b=ratio,
                    expected_pixel_density_ratio_b_to_a=ratio,
                )
            )

        resolved_count = len(resolved_profiles)
        if conflicts:
            state: ProjectScaleState = "CONFLICT"
        elif resolved_count == len(results):
            state = "RESOLVED"
        elif resolved_count > 0:
            state = "PARTIAL"
        else:
            state = "UNRESOLVED"

        return ReconciledProjectScale(
            state=state,
            canonical_m_per_px=canonical_m_per_px,
            target_px_per_m=float(target_px_per_m),
            resolved_level_count=resolved_count,
            unresolved_level_ids=unresolved,
            conflict_level_ids=conflicts,
            profiles=profiles,
            relations=relations,
            source=self.VERSION,
        )
