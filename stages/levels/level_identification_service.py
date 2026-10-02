from __future__ import annotations

from io import BytesIO
from typing import Any

from app.quantia_spatialV1.core.models.level_view import (
    LevelView,
)
from app.quantia_spatialV1.stages.levels.level_detector import (
    LevelDetectionResult,
    LevelDetector,
    LevelRegionDetection,
)
from app.quantia_spatialV1.stages.levels.level_view_builder import (
    LevelViewBuilder,
)
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field

# ============================================================
# RESULTADO DE FASE 1
# ============================================================


class LevelIdentificationResult(BaseModel):
    """
    Resultado completo de:

        FASE 1 — IDENTIFICACIÓN DE NIVEL

    Contiene:

        detección
            +
        LevelView construidos
            +
        regiones todavía no resolubles

    No contiene geometría arquitectónica.
    """

    source_document_id: str | None = None

    source_page_number: int

    source_page_width_px: int

    source_page_height_px: int

    detections: list[LevelRegionDetection] = Field(default_factory=list)

    level_views: list[LevelView] = Field(default_factory=list)

    unresolved: list[LevelRegionDetection] = Field(default_factory=list)

    warnings: list[str] = Field(default_factory=list)

    @property
    def has_level_views(self) -> bool:
        return bool(self.level_views)

    @property
    def level_count(self) -> int:
        return len(self.level_views)

    @property
    def unresolved_count(self) -> int:
        return len(self.unresolved)


# ============================================================
# SERVICIO DE IDENTIFICACIÓN
# ============================================================


class LevelIdentificationService:
    """
    Orquestador interno de Fase 1.

    Flujo:

        raster original
                ↓
        LevelDetector
                ↓
        LevelRegionDetection[]
                ↓
        LevelViewBuilder
                ↓
        LevelView[]

    Este servicio:

        - NO ejecuta Gemini;
        - NO ejecuta PyMuPDF;
        - NO ejecuta OpenCV;
        - NO interpreta muros;
        - NO interpreta espacios;
        - NO convierte px -> m;
        - NO depende del motor legacy;
        - NO se comunica con router/endpoints;
        - NO genera QuantiaSpatialContract.

    Únicamente consume evidencias ya disponibles y produce
    las unidades naturales de reconstrucción para las fases
    siguientes.
    """

    def __init__(
        self,
        *,
        detector: LevelDetector | None = None,
        builder: LevelViewBuilder | None = None,
    ) -> None:

        self.detector = detector or LevelDetector()

        self.builder = builder or LevelViewBuilder()

    # ========================================================
    # API PRINCIPAL
    # ========================================================

    def identify_page(
        self,
        *,
        source_raster_bytes: bytes,
        source_page_number: int,
        source_document_id: str | None = None,
        known_level_names: list[str] | None = None,
        pdf_level_markers: list[str] | None = None,
        gemini_payload: dict[str, Any] | None = None,
        single_level_isolated: bool = False,
    ) -> LevelIdentificationResult:
        """
        Ejecuta Fase 1 sobre una página raster.

        La función NO decide cómo fueron obtenidas las
        evidencias.

        Las entradas pueden provenir posteriormente de:

            PyMuPDF
            Gemini
            OpenCV
            metadata documental

        pero este servicio permanece independiente de esas
        implementaciones.
        """

        if not source_raster_bytes:
            raise ValueError("source_raster_bytes no puede estar vacío.")

        if source_page_number <= 0:
            raise ValueError("source_page_number debe ser mayor que cero.")

        (
            source_page_width_px,
            source_page_height_px,
        ) = self._read_raster_size(source_raster_bytes)

        # ----------------------------------------------------
        # 1. DETECTAR REGIONES DE NIVEL
        # ----------------------------------------------------

        detection_result = self.detector.detect(
            source_page_number=(source_page_number),
            source_page_width_px=(source_page_width_px),
            source_page_height_px=(source_page_height_px),
            known_level_names=(known_level_names),
            pdf_level_markers=(pdf_level_markers),
            gemini_payload=(gemini_payload),
            single_level_isolated=(single_level_isolated),
        )

        # ----------------------------------------------------
        # 2. CONSTRUIR LEVEL VIEWS
        # ----------------------------------------------------

        level_views: list[LevelView] = []

        unresolved: list[LevelRegionDetection] = []

        for detection in detection_result.levels:
            if not detection.is_buildable:
                unresolved.append(detection)
                continue

            if detection.bbox_px is None:
                unresolved.append(detection)
                continue

            level_view = self.builder.build(
                source_raster_bytes=(source_raster_bytes),
                source_page_number=(source_page_number),
                source_bbox_px=(detection.bbox_px),
                level_name=(detection.level_name),
                state=(detection.state),
                confidence=(detection.confidence),
                source_document_id=(source_document_id),
                evidence=(detection.evidence),
            )

            level_views.append(level_view)

        # ----------------------------------------------------
        # 3. VALIDAR IDENTIDADES
        # ----------------------------------------------------

        self._validate_unique_ids(level_views)

        # ----------------------------------------------------
        # 4. RESULTADO DE FASE
        # ----------------------------------------------------

        return LevelIdentificationResult(
            source_document_id=(source_document_id),
            source_page_number=(source_page_number),
            source_page_width_px=(source_page_width_px),
            source_page_height_px=(source_page_height_px),
            detections=list(detection_result.levels),
            level_views=level_views,
            unresolved=unresolved,
            warnings=list(detection_result.warnings),
        )

    # ========================================================
    # TAMAÑO RASTER
    # ========================================================

    @staticmethod
    def _read_raster_size(
        raster_bytes: bytes,
    ) -> tuple[int, int]:
        """
        Obtiene únicamente las dimensiones reales del raster.

        No aplica:

            resize
            rotate
            crop
            conversión métrica
        """

        try:
            buffer = BytesIO(raster_bytes)

            image = Image.open(buffer)

            image.load()

            width_px, height_px = image.size

            image.close()

        except UnidentifiedImageError as exc:
            raise ValueError(
                "Los bytes recibidos no corresponden a una imagen raster válida."
            ) from exc

        except Exception as exc:
            raise ValueError("No fue posible leer el raster fuente.") from exc

        if width_px <= 0 or height_px <= 0:
            raise ValueError("El raster fuente tiene dimensiones inválidas.")

        return (
            width_px,
            height_px,
        )

    # ========================================================
    # VALIDACIÓN DE IDENTIDADES
    # ========================================================

    @staticmethod
    def _validate_unique_ids(
        level_views: list[LevelView],
    ) -> None:
        """
        Un mismo resultado de Fase 1 no puede publicar dos
        LevelView con el mismo ID estable.
        """

        ids = [level_view.id for level_view in level_views]

        if len(ids) != len(set(ids)):
            raise ValueError("Fase 1 produjo LevelView con identificadores duplicados.")


# ============================================================
# FACTORY INTERNA
# ============================================================


def get_level_identification_service() -> LevelIdentificationService:
    """
    Factory interna del nuevo motor spatial.

    No registra dependencias exteriores.
    """

    return LevelIdentificationService()
