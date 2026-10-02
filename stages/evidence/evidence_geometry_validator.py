from __future__ import annotations

from dataclasses import dataclass, field

from app.quantia_spatialV1.core.models.evidence import (
    RawEvidence,
)
from app.quantia_spatialV1.core.models.level_view import (
    LevelView,
    PixelBBox,
)

# ============================================================
# RESULTADO
# ============================================================


@dataclass(frozen=True, slots=True)
class EvidenceGeometryValidation:
    """
    Resultado de validación técnica de una RawEvidence.

    valid=False significa únicamente que existe una
    inconsistencia técnica que debe quedar registrada.

    NO significa:
        - NOISE;
        - evidencia descartable;
        - evidencia arquitectónicamente incorrecta.
    """

    valid: bool

    warnings: list[str] = field(default_factory=list)


# ============================================================
# VALIDADOR
# ============================================================


class EvidenceGeometryValidator:
    """
    Fase 01.5 — validación técnica de evidencia cruda.

    Comprueba exclusivamente:

        - pertenencia al LevelView;
        - coherencia del tipo geométrico;
        - coordenadas locales dentro del raster;
        - bbox dentro del raster;
        - coherencia entre points y bbox cuando ambos existen;
        - presencia esperada de geometría en evidencia
          producida por extractores geométricos.

    No:

        - clasifica arquitectura;
        - decide muros;
        - decide perímetros;
        - decide ejes;
        - decide cotas;
        - corrige coordenadas;
        - recorta geometría;
        - elimina evidencia;
        - convierte px -> m;
        - modifica RawEvidence.

    La evidencia inválida técnicamente debe conservarse.
    El pipeline únicamente agregará sus warnings.
    """

    # ========================================================
    # TIPOS CRUDOS QUE DEBEN TRAER GEOMETRÍA
    # ========================================================

    GEOMETRY_EXPECTED_KINDS: frozenset[str] = frozenset(
        {
            "VECTOR_LINE",
            "VECTOR_PRIMITIVE",
            "VECTOR_TEXT",
            "RASTER_LINE",
            "RASTER_INTERSECTION",
            "RASTER_CONTOUR",
        }
    )

    # ========================================================
    # API
    # ========================================================

    def validate(
        self,
        *,
        evidence: RawEvidence,
        level_view: LevelView,
    ) -> EvidenceGeometryValidation:

        warnings: list[str] = []

        # ----------------------------------------------------
        # IDENTIDAD DEL NIVEL
        # ----------------------------------------------------

        if evidence.level_view_id != level_view.id:
            warnings.append(
                (
                    f"Evidencia {evidence.id}: "
                    f"level_view_id={evidence.level_view_id} "
                    f"no coincide con LevelView={level_view.id}."
                )
            )

        geometry = evidence.geometry

        # ----------------------------------------------------
        # GEOMETRÍA ESPERADA
        # ----------------------------------------------------

        if (
            evidence.kind in self.GEOMETRY_EXPECTED_KINDS
            and geometry.geometry_type == "NONE"
        ):
            warnings.append(
                (
                    f"Evidencia {evidence.id}: "
                    f"{evidence.kind} debería contener "
                    "geometría local."
                )
            )

        # ----------------------------------------------------
        # NONE
        # ----------------------------------------------------

        if geometry.geometry_type == "NONE":
            return EvidenceGeometryValidation(
                valid=not warnings,
                warnings=warnings,
            )

        # ----------------------------------------------------
        # COHERENCIA DEL TIPO GEOMÉTRICO
        # ----------------------------------------------------

        self._validate_geometry_structure(
            evidence=evidence,
            warnings=warnings,
        )

        # ----------------------------------------------------
        # POINTS
        # ----------------------------------------------------

        for index, point in enumerate(geometry.points):
            if not self._point_inside_level(
                x=point.x,
                y=point.y,
                level_view=level_view,
            ):
                warnings.append(
                    (
                        f"Evidencia {evidence.id}: "
                        f"point[{index}] "
                        f"({point.x}, {point.y}) "
                        "está fuera del raster local "
                        "del LevelView."
                    )
                )

        # ----------------------------------------------------
        # BBOX
        # ----------------------------------------------------

        bbox = geometry.bbox_px

        if bbox is not None:
            if not self._bbox_inside_level(
                bbox=bbox,
                level_view=level_view,
            ):
                warnings.append(
                    (
                        f"Evidencia {evidence.id}: "
                        "bbox_px está fuera del raster "
                        "local del LevelView."
                    )
                )

            # ------------------------------------------------
            # POINTS + BBOX
            # ------------------------------------------------
            #
            # Normalmente los extractores usan uno u otro,
            # pero EvidenceGeometry permite ambos.
            # Si existen ambos, deben ser coherentes.
            # ------------------------------------------------

            for index, point in enumerate(geometry.points):
                if not self._point_inside_bbox(
                    x=point.x,
                    y=point.y,
                    bbox=bbox,
                ):
                    warnings.append(
                        (
                            f"Evidencia {evidence.id}: "
                            f"point[{index}] no pertenece "
                            "a bbox_px."
                        )
                    )

        return EvidenceGeometryValidation(
            valid=not warnings,
            warnings=warnings,
        )

    # ========================================================
    # ESTRUCTURA DE GEOMETRÍA
    # ========================================================

    @staticmethod
    def _validate_geometry_structure(
        *,
        evidence: RawEvidence,
        warnings: list[str],
    ) -> None:

        geometry = evidence.geometry

        geometry_type = geometry.geometry_type

        point_count = len(geometry.points)

        bbox = geometry.bbox_px

        # ----------------------------------------------------
        # POINT
        # ----------------------------------------------------

        if geometry_type == "POINT":
            if point_count != 1:
                warnings.append(
                    (
                        f"Evidencia {evidence.id}: "
                        "geometry_type=POINT requiere "
                        "exactamente un punto."
                    )
                )

        # ----------------------------------------------------
        # SEGMENT
        # ----------------------------------------------------

        elif geometry_type == "SEGMENT":
            if point_count != 2:
                warnings.append(
                    (
                        f"Evidencia {evidence.id}: "
                        "geometry_type=SEGMENT requiere "
                        "exactamente dos puntos."
                    )
                )

            elif (
                geometry.points[0].x == geometry.points[1].x
                and geometry.points[0].y == geometry.points[1].y
            ):
                warnings.append(
                    (f"Evidencia {evidence.id}: SEGMENT contiene dos puntos idénticos.")
                )

        # ----------------------------------------------------
        # POLYLINE
        # ----------------------------------------------------

        elif geometry_type == "POLYLINE":
            if point_count < 2:
                warnings.append(
                    (
                        f"Evidencia {evidence.id}: "
                        "geometry_type=POLYLINE requiere "
                        "al menos dos puntos."
                    )
                )

        # ----------------------------------------------------
        # BBOX
        # ----------------------------------------------------

        elif geometry_type == "BBOX":
            if bbox is None:
                warnings.append(
                    (f"Evidencia {evidence.id}: geometry_type=BBOX requiere bbox_px.")
                )

        # ----------------------------------------------------
        # DESCONOCIDO
        # ----------------------------------------------------

        elif geometry_type != "NONE":
            warnings.append(
                (
                    f"Evidencia {evidence.id}: "
                    f"geometry_type desconocido: "
                    f"{geometry_type}."
                )
            )

    # ========================================================
    # PUNTO DENTRO DEL LEVELVIEW
    # ========================================================

    @staticmethod
    def _point_inside_level(
        *,
        x: int,
        y: int,
        level_view: LevelView,
    ) -> bool:

        return (
            0 <= x <= level_view.raster_width_px
            and 0 <= y <= level_view.raster_height_px
        )

    # ========================================================
    # BBOX DENTRO DEL LEVELVIEW
    # ========================================================

    @staticmethod
    def _bbox_inside_level(
        *,
        bbox: PixelBBox,
        level_view: LevelView,
    ) -> bool:

        return (
            bbox.x_min >= 0
            and bbox.y_min >= 0
            and bbox.x_max <= level_view.raster_width_px
            and bbox.y_max <= level_view.raster_height_px
            and bbox.x_max > bbox.x_min
            and bbox.y_max > bbox.y_min
        )

    # ========================================================
    # PUNTO DENTRO DE BBOX
    # ========================================================

    @staticmethod
    def _point_inside_bbox(
        *,
        x: int,
        y: int,
        bbox: PixelBBox,
    ) -> bool:

        return bbox.x_min <= x <= bbox.x_max and bbox.y_min <= y <= bbox.y_max
