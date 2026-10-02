from __future__ import annotations

import math

from pydantic import BaseModel, ConfigDict, Field
from shapely.geometry import LineString, Polygon, box
from shapely.validation import explain_validity

from app.quantia_spatialV1.core.models.level_view import LevelView

from .perimeter_wall_graph import PerimeterWallGraphResult

from .perimeter_models import (
    PerimeterDimensionGroundingResult,
    PerimeterResolutionResult,
    PerimeterValidationState,
    PerimeterWallLayer,
)


class PerimeterWallValidationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    level_view_id: str = Field(min_length=1)
    state: PerimeterValidationState
    polygon_valid: bool | None = None
    inside_level_view: bool | None = None
    closed_wall_chain: bool | None = None
    wall_runs_on_boundary: bool | None = None
    wall_runs_have_geometry_support: bool | None = None
    metric_complete: bool | None = None
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class PerimeterWallValidator:
    """
    Valida la verdad que F02 intenta cerrar.

    VALID exige cerrar la verdad geométrica de F02 en píxeles:
        - perímetro RESOLVED;
        - geometría válida y dentro del LevelView;
        - cadena cerrada de wall_runs;
        - runs pertenecientes al boundary;
        - soporte geométrico real para cada run.

    El grounding métrico se valida por separado. Puede permanecer
    UNRESOLVED/PARTIAL/CONFLICT sin invalidar la geometría px; en esos casos
    length_m/total_length_m no se fuerzan.
    """

    def validate(
        self,
        *,
        level_view: LevelView,
        resolution: PerimeterResolutionResult,
        wall_layer: PerimeterWallLayer | None,
        grounding: PerimeterDimensionGroundingResult | None,
        wall_graph: PerimeterWallGraphResult | None = None,
    ) -> PerimeterWallValidationResult:
        if resolution.level_view_id != level_view.id:
            raise ValueError("PerimeterResolutionResult pertenece a otro LevelView.")

        if resolution.state != "RESOLVED" or resolution.selected is None:
            return PerimeterWallValidationResult(
                level_view_id=level_view.id,
                state="REVIEW",
                warnings=list(resolution.notes),
                notes=["F02 no tiene todavía un perímetro arquitectónico resuelto."],
            )

        if wall_layer is None:
            return PerimeterWallValidationResult(
                level_view_id=level_view.id,
                state="INVALID",
                errors=["Existe perímetro RESOLVED pero no se construyó PerimeterWallLayer."],
            )

        errors: list[str] = []
        warnings: list[str] = []
        notes: list[str] = []

        if wall_layer.level_view_id != level_view.id:
            errors.append("PerimeterWallLayer pertenece a otro LevelView.")
        if wall_layer.candidate_id != resolution.selected.id:
            errors.append("PerimeterWallLayer no proviene del candidato seleccionado.")
        if wall_layer.confirmed:
            errors.append("PerimeterWallLayer salió confirmed=True automáticamente.")

        polygon = Polygon([(point.x, point.y) for point in wall_layer.polygon.points])
        polygon_valid = bool(
            not polygon.is_empty and polygon.area > 0.0 and polygon.is_valid
        )
        if not polygon_valid:
            errors.append(f"Polígono perimetral inválido: {explain_validity(polygon)}")

        level_bounds = box(
            0.0,
            0.0,
            float(level_view.raster_width_px),
            float(level_view.raster_height_px),
        )
        inside_level_view = bool(level_bounds.covers(polygon)) if polygon_valid else False
        if not inside_level_view:
            errors.append("El perímetro sale de los límites del LevelView.")

        closed_wall_chain = self._validate_closed_chain(wall_layer)
        if not closed_wall_chain:
            errors.append("Los wall_runs no forman una cadena cerrada continua.")

        wall_runs_on_boundary = (
            self._validate_runs_on_boundary(wall_layer=wall_layer, polygon=polygon)
            if polygon_valid
            else False
        )
        if not wall_runs_on_boundary:
            errors.append("Uno o más wall_runs no pertenecen al boundary exterior.")

        wall_runs_have_geometry_support = all(
            bool(run.evidence_ids) for run in wall_layer.wall_runs
        )
        if wall_graph is not None and wall_graph.candidate is not None:
            if wall_graph.candidate.id == wall_layer.candidate_id and wall_graph.edges:
                wall_runs_have_geometry_support = (
                    wall_runs_have_geometry_support
                    and all(bool(edge.evidence_ids) for edge in wall_graph.edges)
                )
        if not wall_runs_have_geometry_support:
            warnings.append(
                "Uno o más wall_runs no tienen evidencia geométrica específica asociada."
            )

        if any(run.confirmed for run in wall_layer.wall_runs):
            errors.append("Uno o más PerimeterWallRun salieron confirmed=True.")
        if any(run.length_px <= 0.0 for run in wall_layer.wall_runs):
            errors.append("Existe un PerimeterWallRun con longitud px no positiva.")

        computed_total_px = sum(run.length_px for run in wall_layer.wall_runs)
        if not math.isclose(
            computed_total_px,
            wall_layer.total_length_px,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            errors.append("total_length_px no coincide con la suma de wall_runs.")

        metric_complete = self._validate_metric_consistency(
            wall_layer=wall_layer,
            grounding=grounding,
            errors=errors,
            warnings=warnings,
        )

        if errors:
            state: PerimeterValidationState = "INVALID"
        elif not wall_runs_have_geometry_support:
            state = "REVIEW"
            notes.append("Falta soporte geométrico específico en uno o más wall_runs.")
        else:
            state = "VALID"
            notes.append("Muros perimetrales geométricamente fundamentados en píxeles.")
            if grounding is None or grounding.status == "UNRESOLVED":
                notes.append(
                    "Grounding métrico no resuelto; las unidades métricas permanecen null."
                )
            elif grounding.status == "PARTIAL":
                notes.append(
                    "Grounding métrico parcial; solo se publican longitudes fundamentadas."
                )
            elif grounding.status == "CONFLICT":
                warnings.append(
                    "Grounding métrico en conflicto; no se fuerza una escala ni longitudes métricas globales."
                )
            else:
                notes.append("Grounding métrico completo y fundamentado.")

        return PerimeterWallValidationResult(
            level_view_id=level_view.id,
            state=state,
            polygon_valid=polygon_valid,
            inside_level_view=inside_level_view,
            closed_wall_chain=closed_wall_chain,
            wall_runs_on_boundary=wall_runs_on_boundary,
            wall_runs_have_geometry_support=wall_runs_have_geometry_support,
            metric_complete=metric_complete,
            errors=self._unique(errors),
            warnings=self._unique(warnings),
            notes=self._unique(notes),
        )

    @staticmethod
    def _validate_closed_chain(wall_layer: PerimeterWallLayer) -> bool:
        runs = sorted(wall_layer.wall_runs, key=lambda run: run.sequence_index)
        if len(runs) < 3:
            return False
        for index, run in enumerate(runs):
            following = runs[(index + 1) % len(runs)]
            if (run.end_px.x, run.end_px.y) != (
                following.start_px.x,
                following.start_px.y,
            ):
                return False
        return True

    @staticmethod
    def _validate_runs_on_boundary(
        *,
        wall_layer: PerimeterWallLayer,
        polygon: Polygon,
    ) -> bool:
        boundary = polygon.boundary
        for run in wall_layer.wall_runs:
            line = LineString(
                [(run.start_px.x, run.start_px.y), (run.end_px.x, run.end_px.y)]
            )
            try:
                if not boundary.covers(line):
                    return False
            except Exception:
                return False
        return True

    @staticmethod
    def _validate_metric_consistency(
        *,
        wall_layer: PerimeterWallLayer,
        grounding: PerimeterDimensionGroundingResult | None,
        errors: list[str],
        warnings: list[str],
    ) -> bool | None:
        if grounding is None:
            if wall_layer.total_length_m is not None:
                errors.append("Existe total_length_m sin resultado de grounding.")
                return False
            return None

        if grounding.level_view_id != wall_layer.level_view_id:
            errors.append("Grounding dimensional pertenece a otro LevelView.")
            return False

        if grounding.status == "CONFLICT":
            warnings.append("El grounding dimensional contiene conflictos sin resolver.")
            return False

        grounded_runs = [run for run in wall_layer.wall_runs if run.length_m is not None]
        if grounding.status == "GROUNDED":
            if len(grounded_runs) != len(wall_layer.wall_runs):
                errors.append("Grounding=GROUNDED pero existen wall_runs sin length_m.")
                return False
            if wall_layer.total_length_m is None:
                errors.append("Grounding=GROUNDED pero total_length_m es null.")
                return False

            computed_total_m = sum(float(run.length_m) for run in grounded_runs)
            if not math.isclose(
                computed_total_m,
                wall_layer.total_length_m,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                errors.append("total_length_m no coincide con la suma de wall_runs.")
                return False
            return True

        if wall_layer.metric_status not in {"UNRESOLVED", "PARTIAL", "CONFLICT"}:
            errors.append("metric_status de la capa no coincide con grounding incompleto.")
            return False
        return False if grounding.status == "CONFLICT" else None

    @staticmethod
    def _unique(messages: list[str]) -> list[str]:
        return list(dict.fromkeys(message for message in messages if message))
