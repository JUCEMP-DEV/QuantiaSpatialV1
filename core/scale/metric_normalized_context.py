from __future__ import annotations

from dataclasses import dataclass

from app.quantia_spatialV1.stages.walls.core.drawing_model import DrawingModel
from .level_scale_normalizer import LevelScaleProfile


@dataclass(frozen=True)
class MetricNormalizedContext:
    """Conversor no destructivo entre píxeles nativos y escala común de proyecto."""

    drawing: DrawingModel
    profile: LevelScaleProfile | None = None

    @property
    def resolved(self) -> bool:
        return self.profile is not None and self.profile.is_resolved

    @property
    def scale_factor(self) -> float:
        if not self.resolved:
            return 1.0
        return float(self.profile.scale_factor_to_canonical)

    @property
    def reference_min_dim_native_px(self) -> float:
        if not self.resolved:
            return float(min(self.drawing.width_px, self.drawing.height_px))
        return float(self.profile.project_reference_min_dim_normalized_px) / self.scale_factor

    @property
    def reference_area_native_px2(self) -> float:
        if not self.resolved:
            return float(self.drawing.width_px * self.drawing.height_px)
        return float(self.profile.project_reference_area_normalized_px2) / (self.scale_factor ** 2)

    @property
    def local_m_per_px(self) -> float | None:
        if not self.resolved:
            return None
        return float(self.profile.local_m_per_px)

    def to_normalized_px(self, native_px: float) -> float:
        return float(native_px) * self.scale_factor

    def to_native_px(self, normalized_px: float) -> float:
        return float(normalized_px) / self.scale_factor

    def to_meters(self, native_px: float) -> float | None:
        if self.local_m_per_px is None:
            return None
        return float(native_px) * self.local_m_per_px
