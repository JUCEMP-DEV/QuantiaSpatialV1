# Legacy V2 policy remains importable only for historical tests. It is not wired
# into ProposalCReconstructionPipeline V5.
from .candidate_exclusion_policy import (
    CandidateExclusionDecision,
    CandidateExclusionPolicy,
    CandidateExclusionResult,
)
from .candidate_context_gate import CandidateContextGate
from .candidate_graph import WallCandidateGraphBuilder
from .conflict_graph import ConflictGraph, ConflictGraphBuilder
from .candidate_models import (
    CandidateRelation,
    WallCandidate,
    WallCandidateGraph,
    WallEvidenceVector,
)
from .context_models import (
    CandidateContextDecision,
    CandidateContextGateResult,
    ContextRegion,
    RepetitiveAxisProfile,
)
from .context_region_detector import ContextRegionDetector
from .context_region_classifier import ContextRegionClassifier
from .repeated_cell_detector import RepeatedCellDetector
from .surface_pattern_detector import SurfacePatternDetector
from .structural_lineage import StructuralLineage, StructuralLineageAnalyzer
from .context_region_assembler import ContextRegionAssembler
from .drawing_builder import DrawingModelBuilder
from .element_context_detector import ElementContextDetector
from .physical_stroke_normalizer import PhysicalStrokeNormalizer
from .level_scale_normalizer import (
    LevelScaleProfile,
    ProjectLevelScaleNormalizer,
    ProjectScaleContext,
)
from .metric_normalized_context import MetricNormalizedContext
from .scale_evidence_resolver import (
    LevelScaleEvidenceResult,
    ScaleEvidenceCandidate,
    ScaleEvidenceResolver,
)
from .project_scale_reconciler import (
    ProjectScaleProfile,
    ProjectScaleReconciler,
    ProjectScaleRelation,
    ReconciledProjectScale,
)
from .raster_density_policy import RasterDensityPolicy, RasterDensityRecommendation
from .evidence_fusion import WallEvidenceFusion
from .drawing_model import DrawingCurve, DrawingLine, DrawingModel, DrawingPoint
from .global_topology_solver import GlobalTopologySolution, GlobalTopologySolver
from .reference_constraints import (
    ReferenceAxisConstraint,
    ReferenceConstraintBuilder,
    ReferenceConstraintResult,
)
from .reconstruction_pipeline import (
    ProposalCReconstructionPipeline,
    QuantiaReconstructionPipeline,
    ReconstructionCoreResult,
)
from .topology_analyzer import RoomTopologyAnalyzer, TopologyAnalysis
from .wall_candidate_generator import WallCandidateGenerator

__all__ = [
    "CandidateContextDecision",
    "CandidateContextGate",
    "CandidateContextGateResult",
    "CandidateExclusionDecision",
    "CandidateExclusionPolicy",
    "CandidateExclusionResult",
    "CandidateRelation",
    "ConflictGraph",
    "ConflictGraphBuilder",
    "ContextRegion",
    "ContextRegionAssembler",
    "ContextRegionDetector",
    "ContextRegionClassifier",
    "DrawingCurve",
    "DrawingLine",
    "DrawingModel",
    "DrawingModelBuilder",
    "ElementContextDetector",
    "DrawingPoint",
    "GlobalTopologySolution",
    "GlobalTopologySolver",
    "PhysicalStrokeNormalizer",
    "LevelScaleProfile",
    "MetricNormalizedContext",
    "ProjectLevelScaleNormalizer",
    "ProjectScaleContext",
    "ProposalCReconstructionPipeline",
    "RepeatedCellDetector",
    "SurfacePatternDetector",
    "StructuralLineage",
    "StructuralLineageAnalyzer",
    "ReferenceAxisConstraint",
    "ReferenceConstraintBuilder",
    "ReferenceConstraintResult",
    "RepetitiveAxisProfile",
    "QuantiaReconstructionPipeline",
    "ReconstructionCoreResult",
    "RoomTopologyAnalyzer",
    "TopologyAnalysis",
    "WallCandidate",
    "WallEvidenceFusion",
    "WallCandidateGenerator",
    "WallCandidateGraph",
    "WallCandidateGraphBuilder",
    "WallEvidenceVector",
    "LevelScaleEvidenceResult",
    "ScaleEvidenceCandidate",
    "ScaleEvidenceResolver",
    "ProjectScaleProfile",
    "ProjectScaleReconciler",
    "ProjectScaleRelation",
    "ReconciledProjectScale",
    "RasterDensityPolicy",
    "RasterDensityRecommendation",
]
