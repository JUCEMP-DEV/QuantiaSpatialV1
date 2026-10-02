from __future__ import annotations

from collections.abc import Sequence

from shapely.geometry import Polygon

from app.quantia_spatialV1.models.evidence import RawEvidence
from app.quantia_spatialV1.models.level_view import LevelView

from .perimeter_models import (
    PerimeterCandidate,
    PerimeterEvidenceReconciliationResult,
    PerimeterResolutionResult,
)
from .perimeter_wall_graph import PerimeterWallGraphResult


class PerimeterWallResolver:
    """
    Resuelve el perímetro usando únicamente el resultado propio del motor.

    Entradas motor:
        PyMuPDF + OpenCV + OCR ya reconciliados en F02.

    Gemini NO participa en la selección.

    Resolución automática conservadora:
        1. si PerimeterWallGraphResult cerró un ciclo con soporte específico
           por tramo, ese candidato se publica sin consultar Gemini;
        2. si el grafo no cierra, conserva el fallback anterior de candidatos
           multifuente/contendedores;
        3. Gemini nunca participa en la selección.

    En cualquier otra situación se conserva REVIEW con todos los candidatos.
    No se vota por área, proporción, escala Gemini ni tolerancias inventadas.
    """

    def resolve(
        self,
        *,
        level_view: LevelView,
        candidates: Sequence[PerimeterCandidate],
        evidence: Sequence[RawEvidence],
        reconciliation: PerimeterEvidenceReconciliationResult | None = None,
        wall_graph: PerimeterWallGraphResult | None = None,
    ) -> PerimeterResolutionResult:
        items = [candidate.model_copy(deep=True) for candidate in candidates]
        self._validate_input(
            level_view=level_view,
            candidates=items,
            evidence=evidence,
            reconciliation=reconciliation,
        )

        if not items:
            return PerimeterResolutionResult(
                level_view_id=level_view.id,
                selected=None,
                candidates=[],
                state="UNRESOLVED",
                notes=[
                    "El motor no reconstruyó ningún contorno cerrado candidato a perímetro."
                ],
            )

        reconciliation_by_id = {
            item.candidate_id: item
            for item in (reconciliation.candidates if reconciliation is not None else [])
        }

        if (
            wall_graph is not None
            and wall_graph.state == "RESOLVED"
            and wall_graph.candidate is not None
        ):
            graph_id = wall_graph.candidate.id
            graph_candidate = next((item for item in items if item.id == graph_id), None)
            if graph_candidate is not None:
                return self._resolved(
                    level_view=level_view,
                    selected=graph_candidate,
                    candidates=items,
                    note=(
                        "El grafo estructural reconstruyó un ciclo exterior cerrado con "
                        "soporte geométrico específico por tramo. Gemini no intervino."
                    ),
                )

        if len(items) == 1:
            reconciled = reconciliation_by_id.get(items[0].id)
            if reconciled is not None and reconciled.state == "SUPPORTED":
                return self._resolved(
                    level_view=level_view,
                    selected=items[0],
                    candidates=items,
                    note=(
                        "El motor reconstruyó un único candidato cerrado y la reconciliación "
                        "confirmó corroboración geométrica multifuente. Gemini no intervino."
                    ),
                )
            return PerimeterResolutionResult(
                level_view_id=level_view.id,
                selected=None,
                candidates=items,
                state="REVIEW",
                notes=[
                    "Existe un único candidato geométrico, pero no tiene corroboración multifuente suficiente para publicarlo automáticamente.",
                    "Se conserva completo para fine tuning; Gemini no intervino en la resolución.",
                ],
            )

        hybrid_supported = [
            candidate
            for candidate in items
            if candidate.geometry_source == "HYBRID"
            and reconciliation_by_id.get(candidate.id) is not None
            and reconciliation_by_id[candidate.id].state == "SUPPORTED"
        ]
        if len(hybrid_supported) == 1:
            return self._resolved(
                level_view=level_view,
                selected=hybrid_supported[0],
                candidates=items,
                note=(
                    "El motor encontró un único candidato reconstruido exactamente por "
                    "fuentes VECTOR y RASTER y con cadena perimetral soportada."
                ),
            )

        polygons = {
            candidate.id: Polygon(
                [(point.x, point.y) for point in candidate.geometry.points]
            )
            for candidate in items
        }

        containers: list[PerimeterCandidate] = []
        for candidate in items:
            reconciled = reconciliation_by_id.get(candidate.id)
            if reconciled is None or reconciled.state != "SUPPORTED":
                continue

            polygon = polygons[candidate.id]
            covers_all = True
            for other in items:
                if other.id == candidate.id:
                    continue
                try:
                    if not polygon.covers(polygons[other.id]):
                        covers_all = False
                        break
                except Exception:
                    covers_all = False
                    break
            if covers_all:
                containers.append(candidate)

        if len(containers) == 1:
            return self._resolved(
                level_view=level_view,
                selected=containers[0],
                candidates=items,
                note=(
                    "El motor resolvió un único contorno exterior que contiene los demás "
                    "candidatos y cuya cadena está completamente soportada por evidencia "
                    "geométrica propia del motor."
                ),
            )

        notes = [
            "El motor produjo múltiples candidatos y no existe evidencia multifuente suficiente para seleccionar uno automáticamente.",
            "Todos los candidatos y sus tramos soportados/incompletos se conservan para fine tuning.",
            "Gemini no intervino en la resolución del perímetro.",
        ]
        if reconciliation is not None:
            supported = sum(1 for item in reconciliation.candidates if item.state == "SUPPORTED")
            partial = sum(1 for item in reconciliation.candidates if item.state == "PARTIAL")
            unresolved = sum(
                1 for item in reconciliation.candidates if item.state == "UNRESOLVED"
            )
            notes.append(
                f"Reconciliación multifuente: SUPPORTED={supported}, PARTIAL={partial}, UNRESOLVED={unresolved}."
            )

        return PerimeterResolutionResult(
            level_view_id=level_view.id,
            selected=None,
            candidates=items,
            state="REVIEW",
            notes=notes,
        )

    @staticmethod
    def _resolved(
        *,
        level_view: LevelView,
        selected: PerimeterCandidate,
        candidates: Sequence[PerimeterCandidate],
        note: str,
    ) -> PerimeterResolutionResult:
        return PerimeterResolutionResult(
            level_view_id=level_view.id,
            selected=selected.model_copy(deep=True),
            candidates=[item.model_copy(deep=True) for item in candidates],
            state="RESOLVED",
            notes=[note],
        )

    @staticmethod
    def _validate_input(
        *,
        level_view: LevelView,
        candidates: Sequence[PerimeterCandidate],
        evidence: Sequence[RawEvidence],
        reconciliation: PerimeterEvidenceReconciliationResult | None,
    ) -> None:
        candidate_ids: set[str] = set()
        for candidate in candidates:
            if candidate.level_view_id != level_view.id:
                raise ValueError("PerimeterCandidate pertenece a otro LevelView.")
            if candidate.id in candidate_ids:
                raise ValueError(f"PerimeterCandidate.id duplicado: {candidate.id}.")
            candidate_ids.add(candidate.id)
            if candidate.confirmed:
                raise ValueError(f"PerimeterCandidate {candidate.id} llegó confirmed=True.")

        evidence_ids: set[str] = set()
        for item in evidence:
            if item.level_view_id != level_view.id:
                raise ValueError(f"RawEvidence {item.id} pertenece a otro LevelView.")
            if item.id in evidence_ids:
                raise ValueError(f"RawEvidence.id duplicado: {item.id}.")
            evidence_ids.add(item.id)
            if item.confirmed:
                raise ValueError(f"RawEvidence {item.id} llegó confirmed=True.")

        if reconciliation is not None:
            if reconciliation.level_view_id != level_view.id:
                raise ValueError("PerimeterEvidenceReconciliationResult pertenece a otro LevelView.")
            reconciliation_ids = {item.candidate_id for item in reconciliation.candidates}
            if not reconciliation_ids.issubset(candidate_ids):
                raise ValueError("La reconciliación referencia candidatos inexistentes.")
