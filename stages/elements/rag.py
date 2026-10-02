from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .contracts import ArchitecturalRAGExample, ArchitecturalRAGHit, ElementFeatureVector


class ArchitecturalRAGIndex:
    """RAG local, auditable y sin embeddings obligatorios.

    V2 usa similitud de atributos interpretables para no introducir otra caja
    negra. El contrato permite sustituir el índice por embeddings multimodales
    más adelante sin cambiar el mini-motor.
    """

    def __init__(self, examples: Iterable[ArchitecturalRAGExample]) -> None:
        self.examples = list(examples)
        if not self.examples:
            raise ValueError("ArchitecturalRAGIndex requiere al menos un ejemplo.")

    @classmethod
    def from_json(cls, path: str | Path) -> "ArchitecturalRAGIndex":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        examples = [ArchitecturalRAGExample.model_validate(item) for item in payload]
        return cls(examples)

    @classmethod
    def default(cls) -> "ArchitecturalRAGIndex":
        path = Path(__file__).with_name("knowledge") / "architectural_openings_v2.json"
        return cls.from_json(path)

    def retrieve(self, features: ElementFeatureVector, *, top_k: int = 4) -> list[ArchitecturalRAGHit]:
        query = features.as_retrieval_dict()
        query_tags = set(features.tags)
        ranked: list[ArchitecturalRAGHit] = []
        for example in self.examples:
            similarity = self._similarity(
                query=query,
                query_tags=query_tags,
                example=example,
            )
            ranked.append(
                ArchitecturalRAGHit(
                    example_id=example.id,
                    label=example.label,
                    similarity=similarity,
                    description=example.description,
                    source=example.source,
                )
            )
        ranked.sort(key=lambda item: (-item.similarity, item.example_id))
        return ranked[: max(1, int(top_k))]

    @staticmethod
    def _similarity(
        *,
        query: dict[str, float | bool | str],
        query_tags: set[str],
        example: ArchitecturalRAGExample,
    ) -> float:
        scores: list[float] = []
        weights: list[float] = []
        for key, target in example.features.items():
            if key not in query:
                continue
            observed = query[key]
            weight = 1.0
            if key in {"host_wall_match", "opening_verified", "expected_class"}:
                weight = 1.7
            elif key in {"wall_through_support", "wall_gap_support", "door_arc_support", "window_frame_support"}:
                weight = 1.35

            if isinstance(target, bool):
                score = 1.0 if bool(observed) is target else 0.0
            elif isinstance(target, (int, float)) and isinstance(observed, (int, float)):
                score = max(0.0, 1.0 - abs(float(observed) - float(target)))
            else:
                score = 1.0 if str(observed) == str(target) else 0.0
            scores.append(score * weight)
            weights.append(weight)

        feature_score = sum(scores) / sum(weights) if weights else 0.5
        example_tags = set(example.tags)
        if example_tags:
            tag_score = len(query_tags & example_tags) / len(example_tags)
            return max(0.0, min(1.0, 0.85 * feature_score + 0.15 * tag_score))
        return max(0.0, min(1.0, feature_score))
