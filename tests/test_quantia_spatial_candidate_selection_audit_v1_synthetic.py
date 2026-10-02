from __future__ import annotations

from types import SimpleNamespace

from app.quantia_spatialV1.stages.walls.core.candidate_models import (
    CandidateGraphDiagnostics,
    CandidateRelation,
    WallCandidate,
    WallCandidateGraph,
    WallEvidenceVector,
)
from app.quantia_spatialV1.stages.walls.core.candidate_selection_audit import (
    CandidateSelectionAudit,
)
from app.quantia_spatialV1.stages.walls.core.drawing_model import DrawingPoint


TRACE_ID = "QSV1-CONT-20261002-A"


def _candidate(
    candidate_id: str,
    *,
    start: tuple[float, float],
    end: tuple[float, float],
    dashed: float = 0.0,
) -> WallCandidate:
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    return WallCandidate(
        id=candidate_id,
        generator="DOUBLE_FACE",
        start=DrawingPoint(x=start[0], y=start[1]),
        end=DrawingPoint(x=end[0], y=end[1]),
        angle_deg=0.0 if abs(dy) < 1e-9 else 90.0,
        length_px=(dx * dx + dy * dy) ** 0.5,
        thickness_px=10.0,
        face_ids=[f"{candidate_id}_A", f"{candidate_id}_B"],
        evidence_ids=[],
        source_names=["SYNTHETIC"],
        evidence=WallEvidenceVector(
            pair_overlap=0.90,
            thickness_support=0.90,
            raster_line_support=1.0,
            region_support=0.90,
            source_consensus=0.50,
            perimeter_containment=1.0,
            dashed_penalty=dashed,
        ),
        prior_score=0.80,
    )


def _graph(candidates, relations=None) -> WallCandidateGraph:
    relations = list(relations or [])
    return WallCandidateGraph(
        level_view_id="LV",
        candidates=list(candidates),
        relations=relations,
        diagnostics=CandidateGraphDiagnostics(
            candidate_count=len(candidates),
            relation_count=len(relations),
            hard_conflict_count=sum(item.hard_conflict for item in relations),
            junction_count=sum(item.relation_type == "JUNCTION" for item in relations),
            continuation_count=sum(
                item.relation_type == "CONTINUATION" for item in relations
            ),
            duplicate_count=sum(item.relation_type == "DUPLICATE" for item in relations),
        ),
    )


def _perimeter():
    points = [(0.0, 0.0), (300.0, 0.0), (300.0, 200.0), (0.0, 200.0)]
    return SimpleNamespace(
        vertices=[
            SimpleNamespace(
                sequence_index=index,
                point_px=SimpleNamespace(x=x, y=y),
            )
            for index, (x, y) in enumerate(points)
        ]
    )


class _Topology:
    def analyze(self, *, perimeter, candidates):
        return SimpleNamespace(
            topology_score=0.0,
            model_dump=lambda mode="json": {"topology_score": 0.0},
        )


class _Solver:
    topology = _Topology()

    @staticmethod
    def _unary(candidate, topology_degree=0):
        return -0.2 if candidate.id == "WEAK" else 0.8

    @staticmethod
    def _incremental_relation_bonus(**kwargs):
        return 0.0

    @staticmethod
    def _wall_identity_probability(candidate):
        return 0.80

    @staticmethod
    def _observation_confidence(candidate):
        return 0.90

    @staticmethod
    def _context_penalty(candidate):
        return 0.0


def _solution(selected_ids):
    return SimpleNamespace(
        selected_candidate_ids=list(selected_ids),
        topology=SimpleNamespace(topology_score=0.0),
    )


def test_audit_explains_selected_conflict_and_outside_without_mutating_graph() -> None:
    selected = _candidate("SELECTED", start=(20, 50), end=(180, 50))
    conflict = _candidate("CONFLICT", start=(20, 60), end=(180, 60))
    outside = _candidate("OUTSIDE", start=(-40, 90), end=(180, 90))
    relation = CandidateRelation(
        a_id="SELECTED",
        b_id="CONFLICT",
        relation_type="CONFLICT_ALTERNATIVE",
        hard_conflict=True,
    )
    graph = _graph([selected, conflict, outside], [relation])
    before = graph.model_dump(mode="json")

    result = CandidateSelectionAudit().build(
        graph=graph,
        solution=_solution(["SELECTED"]),
        perimeter=_perimeter(),
        solver=_Solver(),
    )

    by_id = {item.candidate_id: item for item in result.items}
    assert by_id["SELECTED"].category == "SELECTED"
    assert by_id["CONFLICT"].category == "CONFLICT_WITH_SELECTED"
    assert by_id["CONFLICT"].hard_conflicts_with_selected == ["SELECTED"]
    assert by_id["OUTSIDE"].category == "OUTSIDE_F02"
    assert result.selected_count == 1
    assert result.rejected_count == 2
    assert graph.model_dump(mode="json") == before


def test_audit_preserves_reference_and_weak_evidence_diagnostics() -> None:
    selected = _candidate("SELECTED", start=(20, 50), end=(180, 50))
    reference = _candidate(
        "REFERENCE",
        start=(20, 80),
        end=(180, 80),
        dashed=0.90,
    )
    weak = _candidate("WEAK", start=(20, 110), end=(180, 110))
    graph = _graph([selected, reference, weak])

    result = CandidateSelectionAudit().build(
        graph=graph,
        solution=_solution(["SELECTED"]),
        perimeter=_perimeter(),
        solver=_Solver(),
    )

    by_id = {item.candidate_id: item for item in result.items}
    assert by_id["REFERENCE"].category == "DASHED_REFERENCE"
    assert "reference_like_geometry" in by_id["REFERENCE"].reasons
    assert by_id["WEAK"].category == "WEAK_EVIDENCE"
    assert "unary_score<=0" in by_id["WEAK"].reasons
