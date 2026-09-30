from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from .candidate_models import WallCandidateGraph


class ConflictGraph(BaseModel):
    model_config = ConfigDict(extra="forbid")

    conflicts: dict[str, list[str]] = Field(default_factory=dict)


class ConflictGraphBuilder:
    """Materializa las incompatibilidades duras del CandidateGraph."""

    def build(self, graph: WallCandidateGraph) -> ConflictGraph:
        result: dict[str, set[str]] = {item.id: set() for item in graph.candidates}
        for relation in graph.relations:
            if not relation.hard_conflict:
                continue
            result.setdefault(relation.a_id, set()).add(relation.b_id)
            result.setdefault(relation.b_id, set()).add(relation.a_id)
        return ConflictGraph(
            conflicts={key: sorted(values) for key, values in sorted(result.items())}
        )
