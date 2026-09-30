from __future__ import annotations

from typing import Protocol

from .contracts import ElementFeatureVector, ElementSemanticResolution, ArchitecturalRAGHit


class ElementSemanticResolver(Protocol):
    """Puerto para Gemini/VLM sin acoplar el mini-motor a un proveedor."""

    def resolve(
        self,
        *,
        image_png: bytes,
        features: ElementFeatureVector,
        rag_hits: list[ArchitecturalRAGHit],
    ) -> ElementSemanticResolution | None:
        ...
