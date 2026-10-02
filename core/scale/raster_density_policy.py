from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class RasterDensityRecommendation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target_geometry_px_per_m: float = Field(gt=0.0)
    current_render_scale: float | None = Field(default=None, gt=0.0)
    current_m_per_px: float | None = Field(default=None, gt=0.0)
    current_px_per_m: float | None = Field(default=None, gt=0.0)
    declared_scale_denominator: float | None = Field(default=None, gt=0.0)
    recommended_render_scale: float = Field(gt=0.0)
    expected_m_per_px: float = Field(gt=0.0)
    expected_px_per_m: float = Field(gt=0.0)
    equivalent_pdf_dpi: float = Field(gt=0.0)
    ocr_local_dpi: int = Field(default=300, ge=72)
    state: str = Field(min_length=1)
    notes: list[str] = Field(default_factory=list)


class RasterDensityPolicy:
    """Política de raster separada de la escala del dibujo y de Gemini.

    La geometría se normaliza por densidad arquitectónica (px/m).
    OCR mantiene una política de alta resolución localizada independiente.
    Gemini no hereda automáticamente este raster para evitar costo visual inútil.
    """

    VERSION = "RASTER_DENSITY_POLICY_V1"

    def __init__(
        self,
        *,
        target_geometry_px_per_m: float = 90.0,
        ocr_local_dpi: int = 300,
        min_render_scale: float = 0.50,
        max_render_scale: float = 4.00,
    ) -> None:
        if target_geometry_px_per_m <= 0:
            raise ValueError("target_geometry_px_per_m debe ser mayor que cero.")
        if ocr_local_dpi < 72:
            raise ValueError("ocr_local_dpi debe ser al menos 72.")
        if min_render_scale <= 0 or max_render_scale < min_render_scale:
            raise ValueError("Rango de render_scale inválido.")
        self.target_geometry_px_per_m = float(target_geometry_px_per_m)
        self.ocr_local_dpi = int(ocr_local_dpi)
        self.min_render_scale = float(min_render_scale)
        self.max_render_scale = float(max_render_scale)

    def recommend_from_grounded_scale(
        self,
        *,
        current_render_scale: float,
        current_m_per_px: float,
    ) -> RasterDensityRecommendation:
        if current_render_scale <= 0 or current_m_per_px <= 0:
            raise ValueError("current_render_scale y current_m_per_px deben ser positivos.")
        current_px_per_m = 1.0 / float(current_m_per_px)
        raw = float(current_render_scale) * self.target_geometry_px_per_m / current_px_per_m
        recommended, state, notes = self._bounded(raw)
        expected_px_per_m = current_px_per_m * recommended / float(current_render_scale)
        return RasterDensityRecommendation(
            target_geometry_px_per_m=self.target_geometry_px_per_m,
            current_render_scale=float(current_render_scale),
            current_m_per_px=float(current_m_per_px),
            current_px_per_m=current_px_per_m,
            recommended_render_scale=recommended,
            expected_m_per_px=1.0 / expected_px_per_m,
            expected_px_per_m=expected_px_per_m,
            equivalent_pdf_dpi=72.0 * recommended,
            ocr_local_dpi=self.ocr_local_dpi,
            state=state,
            notes=notes,
        )

    def recommend_from_declared_scale(
        self,
        *,
        denominator: float,
    ) -> RasterDensityRecommendation:
        if denominator <= 0:
            raise ValueError("denominator debe ser positivo.")
        raw = self.target_geometry_px_per_m * float(denominator) * 0.0254 / 72.0
        recommended, state, notes = self._bounded(raw)
        expected_px_per_m = 72.0 * recommended / (float(denominator) * 0.0254)
        return RasterDensityRecommendation(
            target_geometry_px_per_m=self.target_geometry_px_per_m,
            declared_scale_denominator=float(denominator),
            recommended_render_scale=recommended,
            expected_m_per_px=1.0 / expected_px_per_m,
            expected_px_per_m=expected_px_per_m,
            equivalent_pdf_dpi=72.0 * recommended,
            ocr_local_dpi=self.ocr_local_dpi,
            state=state,
            notes=notes,
        )

    def _bounded(self, raw: float) -> tuple[float, str, list[str]]:
        if raw < self.min_render_scale:
            return self.min_render_scale, "CLAMPED_MIN", [
                f"render_scale calculado {raw:.4f} limitado al mínimo operativo."
            ]
        if raw > self.max_render_scale:
            return self.max_render_scale, "CLAMPED_MAX", [
                f"render_scale calculado {raw:.4f} limitado al máximo operativo."
            ]
        return raw, "RESOLVED", []
