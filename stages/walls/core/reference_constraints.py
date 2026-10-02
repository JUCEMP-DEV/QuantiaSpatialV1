from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from app.quantia_spatialV1.core.models.evidence import RawEvidence
from app.quantia_spatialV1.core.models.level_view import LevelView


class ReferenceAxisConstraint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    label: str | None = None
    orientation: str = Field(min_length=1)
    coordinate_px: float
    evidence_ids: list[str] = Field(default_factory=list)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class ReferenceConstraintResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    level_view_id: str = Field(min_length=1)
    axes: list[ReferenceAxisConstraint] = Field(default_factory=list)
    state: str = Field(min_length=1)
    warnings: list[str] = Field(default_factory=list)


class ReferenceConstraintBuilder:
    """Adapta la retícula vigente como RESTRICCIÓN, nunca como generador de muro.

    Proposal C no importa WallGeometryDetector/StructurePipeline de F03. La lógica
    de retícula validada fue aislada en `reference_axis_resolver.py` para que los
    ejes funcionen como referencia geométrica y no como generador de paredes.
    """

    def build(
        self,
        *,
        level_view: LevelView,
        evidence: list[RawEvidence],
    ) -> ReferenceConstraintResult:
        from .reference_axis_resolver import AxisGridResolver

        resolved = AxisGridResolver().resolve(level_view=level_view, evidence=evidence)
        axes = [
            ReferenceAxisConstraint(
                id=item.id,
                label=item.label,
                orientation=item.orientation,
                coordinate_px=float(item.coordinate_px),
                evidence_ids=sorted({
                    *item.source_dimension_evidence_ids,
                    *item.source_axis_evidence_ids,
                    *item.source_geometric_evidence_ids,
                }),
                confidence=item.confidence,
            )
            for item in resolved.axes
        ]
        return ReferenceConstraintResult(
            level_view_id=level_view.id,
            axes=axes,
            state=resolved.state,
            warnings=list(resolved.warnings),
        )
