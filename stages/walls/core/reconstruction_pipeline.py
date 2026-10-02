from __future__ import annotations

from typing import Sequence

from pydantic import BaseModel, ConfigDict, Field

from app.quantia_spatialV1.core.models.evidence import RawEvidence
from app.quantia_spatialV1.core.models.level_view import LevelView
from app.quantia_spatialV1.core.models.parametric import QuantiaParametricModel
from app.quantia_spatialV1.stages.perimeter.perimeter_delivery import EditablePerimeterModel

from .candidate_context_gate import CandidateContextGate
from .candidate_graph import WallCandidateGraphBuilder
from .candidate_models import WallCandidateGraph
from .context_models import CandidateContextGateResult
from .context_region_detector import ContextRegionDetector
from .drawing_builder import DrawingModelBuilder
from .drawing_model import DrawingModel
from .global_topology_solver import GlobalTopologySolution, GlobalTopologySolver
from app.quantia_spatialV1.core.scale.level_scale_normalizer import LevelScaleProfile, ProjectLevelScaleNormalizer, ProjectScaleContext
from .reference_constraints import ReferenceConstraintBuilder, ReferenceConstraintResult
from .wall_candidate_generator import WallCandidateGenerator
from .wall_reconstructor import ParametricWallReconstructor


class ReconstructionCoreDiagnostics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    drawing_line_count: int = Field(ge=0)
    discovered_wall_candidate_count: int = Field(default=0, ge=0)
    context_region_count: int = Field(default=0, ge=0)
    element_context_region_count: int = Field(default=0, ge=0)
    opening_context_region_count: int = Field(default=0, ge=0)
    quarantined_wall_candidate_count: int = Field(default=0, ge=0)
    review_wall_candidate_count: int = Field(default=0, ge=0)
    graph_candidate_limit_drop_count: int = Field(default=0, ge=0)
    wall_candidate_count: int = Field(ge=0)
    selected_wall_candidate_count: int = Field(ge=0)
    perimeter_wall_count: int = Field(ge=0)
    parametric_divider_wall_count: int = Field(ge=0)
    f02_model_id: str = Field(min_length=1)
    f02_geometry_revision: int = Field(ge=0)


class ReconstructionCoreResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    level_view_id: str = Field(min_length=1)
    drawing_model: DrawingModel
    context_gate: CandidateContextGateResult
    candidate_graph: WallCandidateGraph
    reference_constraints: ReferenceConstraintResult
    solution: GlobalTopologySolution
    parametric_model: QuantiaParametricModel
    diagnostics: ReconstructionCoreDiagnostics
    warnings: list[str] = Field(default_factory=list)


class QuantiaReconstructionPipeline:
    """Proposal C + Element Context Isolation V4 sobre base métrica V2.3.

    QUARANTINE queda fuera del grafo. REVIEW permanece provisional con riesgo
    explícito. El linaje estructural se calcula después de clasificar contexto y
    la escala canónica aporta plausibilidad métrica suave de espesor.
    """

    def __init__(self) -> None:
        self.drawing_builder = DrawingModelBuilder()
        self.constraint_builder = ReferenceConstraintBuilder()
        self.candidate_generator = WallCandidateGenerator()
        self.context_region_detector = ContextRegionDetector()
        self.context_gate = CandidateContextGate()
        self.graph_builder = WallCandidateGraphBuilder()
        self.solver = GlobalTopologySolver()
        self.reconstructor = ParametricWallReconstructor()
        self.scale_normalizer = ProjectLevelScaleNormalizer()

    @staticmethod
    def _annotate_metric_thickness(
        *,
        candidates,
        scale_profile: LevelScaleProfile | None,
    ):
        """Adjunta evidencia física continua sin imponer un espesor único.

        Selection Contracts V1.2 separa tres conceptos que antes estaban
        mezclados en una sola meseta 0.06–0.40 m:
        - plausibilidad del ESPESOR físico de la banda;
        - esbeltez geométrica longitud/espesor;
        - soporte combinado de banda de muro.

        Ninguno es un veto duro. Un muro atípico todavía puede sobrevivir por
        topología/semántica/evidencia independiente, pero una pareja de líneas de
        baldosa o un trazo gráfico ya no recibe soporte 1.0 por caer dentro de una
        ventana excesivamente amplia.
        """
        if (
            scale_profile is None
            or scale_profile.state != "RESOLVED"
            or scale_profile.local_m_per_px is None
            or scale_profile.local_m_per_px <= 0.0
        ):
            return list(candidates)

        def thickness_support(thickness_m: float) -> float:
            if thickness_m <= 0.025:
                return 0.05
            if thickness_m < 0.050:
                return 0.05 + 0.50 * ((thickness_m - 0.025) / 0.025)
            if thickness_m < 0.070:
                return 0.55 + 0.35 * ((thickness_m - 0.050) / 0.020)
            if thickness_m <= 0.240:
                return 1.0
            if thickness_m < 0.300:
                return 1.0 - 0.28 * ((thickness_m - 0.240) / 0.060)
            if thickness_m < 0.380:
                return 0.72 - 0.34 * ((thickness_m - 0.300) / 0.080)
            if thickness_m < 0.500:
                return 0.38 - 0.20 * ((thickness_m - 0.380) / 0.120)
            return max(0.08, 0.18 - 0.10 * min(1.0, (thickness_m - 0.500) / 0.300))

        def slenderness_support(length_px: float, thickness_px: float) -> tuple[float, float]:
            ratio = float(length_px) / max(float(thickness_px), 1e-6)
            if ratio <= 2.5:
                support = 0.05
            elif ratio < 4.0:
                support = 0.05 + 0.40 * ((ratio - 2.5) / 1.5)
            elif ratio < 6.0:
                support = 0.45 + 0.30 * ((ratio - 4.0) / 2.0)
            elif ratio < 10.0:
                support = 0.75 + 0.25 * ((ratio - 6.0) / 4.0)
            else:
                support = 1.0
            return ratio, max(0.0, min(1.0, support))

        m_per_px = float(scale_profile.local_m_per_px)
        output = []
        for candidate in candidates:
            thickness_m = float(candidate.thickness_px) * m_per_px
            length_m = float(candidate.length_px) * m_per_px
            thickness_score = max(0.0, min(1.0, thickness_support(thickness_m)))
            slenderness_ratio, slenderness_score = slenderness_support(
                candidate.length_px,
                candidate.thickness_px,
            )
            band_support = max(
                0.05,
                min(1.0, 0.72 * thickness_score + 0.28 * slenderness_score),
            )
            metadata = dict(candidate.metadata)
            metadata.update({
                "metric_thickness_m": thickness_m,
                "metric_length_m": length_m,
                "metric_thickness_support": thickness_score,
                "metric_slenderness_ratio": slenderness_ratio,
                "metric_slenderness_support": slenderness_score,
                "metric_wall_band_support": band_support,
                "metric_scale_source": scale_profile.source,
                "metric_parameter_model": "F03_SELECTION_CONTRACTS_V1_2",
            })
            output.append(candidate.model_copy(update={"metadata": metadata}))
        return output

    def build_project_scale_context(
        self,
        *,
        levels: Sequence[tuple[LevelView, EditablePerimeterModel]],
    ) -> ProjectScaleContext:
        """Construye una referencia de escala común antes de clasificar plantas."""
        return self.scale_normalizer.build(levels=levels)

    def run(
        self,
        *,
        level_view: LevelView,
        perimeter: EditablePerimeterModel,
        evidence: Sequence[RawEvidence],
        scale_profile: LevelScaleProfile | None = None,
    ) -> ReconstructionCoreResult:
        if perimeter.level_view_id != level_view.id:
            raise ValueError("EditablePerimeterModel pertenece a otro LevelView.")

        raw_evidence = list(evidence)
        drawing = self.drawing_builder.build(level_view=level_view, evidence=raw_evidence)
        constraints = self.constraint_builder.build(level_view=level_view, evidence=raw_evidence)

        # A1.2 Candidate Discovery conserva todas sus hipótesis antes de cualquier
        # regla contextual. Así el ruido repetitivo no puede ocupar prematuramente
        # el cupo operativo del CandidateGraph.
        discovered = self.candidate_generator.generate(
            drawing=drawing,
            perimeter=perimeter,
            constraints=constraints,
        )
        discovered = self._annotate_metric_thickness(
            candidates=discovered,
            scale_profile=scale_profile,
        )

        regions = self.context_region_detector.detect(
            drawing=drawing,
            perimeter=perimeter,
            scale_profile=scale_profile,
        )
        gate = self.context_gate.apply(
            candidates=discovered,
            regions=regions,
        )

        # QUARANTINE sí queda encapsulado fuera del grafo. REVIEW no es una
        # exclusión: sin verificador visual debe seguir disponible como hipótesis
        # provisional. No se aplica ningún corte por cantidad de candidatos.
        graph_candidates = sorted(
            gate.solver_candidates,
            key=lambda item: (-item.prior_score, -item.length_px, item.id),
        )
        graph_limit_drop_count = 0

        graph = self.graph_builder.build(
            level_view_id=level_view.id,
            candidates=graph_candidates,
        )
        solution = self.solver.solve(graph=graph, perimeter=perimeter)
        parametric = self.reconstructor.build(
            level_view=level_view,
            perimeter=perimeter,
            graph=graph,
            solution=solution,
        )
        level = parametric.levels[0]
        divider_count = sum(1 for wall in level.walls if wall.role == "DIVIDER")
        perimeter_count = sum(1 for wall in level.walls if wall.role == "PERIMETER")

        warnings: list[str] = []
        if gate.review_candidates:
            warnings.append(
                f"CONTEXT_REVIEW_PENDING:{len(gate.review_candidates)} candidates continúan provisionalmente en el solver."
            )

        return ReconstructionCoreResult(
            level_view_id=level_view.id,
            drawing_model=drawing,
            context_gate=gate,
            candidate_graph=graph,
            reference_constraints=constraints,
            solution=solution,
            parametric_model=parametric,
            diagnostics=ReconstructionCoreDiagnostics(
                drawing_line_count=len(drawing.lines),
                discovered_wall_candidate_count=len(discovered),
                context_region_count=len(regions),
                element_context_region_count=sum(
                    1 for region in regions
                    if str(region.metadata.get("detector") or "").startswith("F03_ELEMENT_CONTEXT_ISOLATION_V")
                ),
                opening_context_region_count=sum(
                    1 for region in regions
                    if region.region_type in {"DOOR_REGION", "WINDOW_REGION", "OPENING_REGION"}
                ),
                quarantined_wall_candidate_count=len(gate.quarantined_candidates),
                review_wall_candidate_count=len(gate.review_candidates),
                graph_candidate_limit_drop_count=graph_limit_drop_count,
                wall_candidate_count=len(graph.candidates),
                selected_wall_candidate_count=len(solution.selected_candidate_ids),
                perimeter_wall_count=perimeter_count,
                parametric_divider_wall_count=divider_count,
                f02_model_id=perimeter.id,
                f02_geometry_revision=perimeter.geometry_revision,
            ),
            warnings=warnings,
        )


ProposalCReconstructionPipeline = QuantiaReconstructionPipeline
