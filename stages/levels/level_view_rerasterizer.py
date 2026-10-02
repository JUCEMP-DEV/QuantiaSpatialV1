from __future__ import annotations

from dataclasses import dataclass

from app.quantia_spatialV1.core.models.level_view import LevelView, LevelViewEvidence, PixelBBox
from app.quantia_spatialV1.stages.levels.level_detector import LevelRegionDetection
from app.quantia_spatialV1.stages.levels.level_identification_service import LevelIdentificationResult
from app.quantia_spatialV1.stages.levels.level_view_builder import LevelViewBuilder
from app.quantia_spatialV1.stages.levels.pymupdf_level_source import PyMuPDFLevelSource


@dataclass(frozen=True, slots=True)
class RerasterizedPage:
    page_number: int
    render_scale: float
    page_raster_bytes: bytes
    page_raster_mime_type: str
    page_width_px: int
    page_height_px: int
    level_views: tuple[LevelView, ...]
    page_result: LevelIdentificationResult


class LevelViewRerasterizer:
    """Reproyecta una página F01 a un nuevo render_scale sin redetectar niveles.

    La localización ya resuelta por F01 se conserva como proporción de la página.
    Se vuelve a renderizar el PDF, se reproyectan bboxes/evidencias y se reconstruyen
    los LevelView con el mismo ID estable. No ejecuta Gemini ni reinterpreta niveles.
    """

    VERSION = "LEVEL_VIEW_RERASTERIZER_V1"

    def __init__(
        self,
        *,
        pdf_source: PyMuPDFLevelSource | None = None,
        builder: LevelViewBuilder | None = None,
    ) -> None:
        self.pdf_source = pdf_source or PyMuPDFLevelSource()
        self.builder = builder or LevelViewBuilder()

    def rerasterize_page(
        self,
        *,
        document_bytes: bytes,
        page_result: LevelIdentificationResult,
        target_render_scale: float,
    ) -> RerasterizedPage:
        if target_render_scale <= 0:
            raise ValueError("target_render_scale debe ser mayor que cero.")

        source = self.pdf_source.read(
            document_bytes=document_bytes,
            render_scale=float(target_render_scale),
        )
        page_number = int(page_result.source_page_number)
        if page_number <= 0 or page_number > source.page_count:
            raise ValueError("page_result referencia una página inexistente.")

        target_page = source.pages[page_number - 1]
        old_w = int(page_result.source_page_width_px)
        old_h = int(page_result.source_page_height_px)
        new_w = int(target_page.raster_width_px)
        new_h = int(target_page.raster_height_px)
        if min(old_w, old_h, new_w, new_h) <= 0:
            raise ValueError("Dimensiones de página inválidas para rerasterización.")

        new_views = tuple(
            self._rerasterize_level_view(
                source_view=view,
                source_raster_bytes=target_page.raster_bytes,
                old_page_width_px=old_w,
                old_page_height_px=old_h,
                new_page_width_px=new_w,
                new_page_height_px=new_h,
            )
            for view in page_result.level_views
        )
        by_id = {view.id: view for view in new_views}

        detections = [
            self._scale_detection(
                detection=item,
                old_page_width_px=old_w,
                old_page_height_px=old_h,
                new_page_width_px=new_w,
                new_page_height_px=new_h,
            )
            for item in page_result.detections
        ]
        unresolved = [
            self._scale_detection(
                detection=item,
                old_page_width_px=old_w,
                old_page_height_px=old_h,
                new_page_width_px=new_w,
                new_page_height_px=new_h,
            )
            for item in page_result.unresolved
        ]

        rebuilt_page_result = LevelIdentificationResult(
            source_document_id=page_result.source_document_id,
            source_page_number=page_number,
            source_page_width_px=new_w,
            source_page_height_px=new_h,
            detections=detections,
            level_views=[by_id[view.id] for view in page_result.level_views],
            unresolved=unresolved,
            warnings=list(page_result.warnings),
        )

        return RerasterizedPage(
            page_number=page_number,
            render_scale=float(target_render_scale),
            page_raster_bytes=target_page.raster_bytes,
            page_raster_mime_type=target_page.raster_mime_type,
            page_width_px=new_w,
            page_height_px=new_h,
            level_views=new_views,
            page_result=rebuilt_page_result,
        )

    def _rerasterize_level_view(
        self,
        *,
        source_view: LevelView,
        source_raster_bytes: bytes,
        old_page_width_px: int,
        old_page_height_px: int,
        new_page_width_px: int,
        new_page_height_px: int,
    ) -> LevelView:
        bbox = self._scale_bbox(
            bbox=source_view.source_bbox_px,
            old_page_width_px=old_page_width_px,
            old_page_height_px=old_page_height_px,
            new_page_width_px=new_page_width_px,
            new_page_height_px=new_page_height_px,
        )
        evidence = [
            self._scale_level_evidence(
                item=item,
                old_page_width_px=old_page_width_px,
                old_page_height_px=old_page_height_px,
                new_page_width_px=new_page_width_px,
                new_page_height_px=new_page_height_px,
            )
            for item in source_view.evidence
        ]
        return self.builder.build(
            source_raster_bytes=source_raster_bytes,
            source_page_number=source_view.source_page_number,
            source_bbox_px=bbox,
            level_name=source_view.level_name,
            state=source_view.state,
            confidence=source_view.confidence,
            source_document_id=source_view.source_document_id,
            evidence=evidence,
            level_view_id=source_view.id,
        )

    @classmethod
    def _scale_detection(
        cls,
        *,
        detection: LevelRegionDetection,
        old_page_width_px: int,
        old_page_height_px: int,
        new_page_width_px: int,
        new_page_height_px: int,
    ) -> LevelRegionDetection:
        bbox = None
        if detection.bbox_px is not None:
            bbox = cls._scale_bbox(
                bbox=detection.bbox_px,
                old_page_width_px=old_page_width_px,
                old_page_height_px=old_page_height_px,
                new_page_width_px=new_page_width_px,
                new_page_height_px=new_page_height_px,
            )
        evidence = [
            cls._scale_level_evidence(
                item=item,
                old_page_width_px=old_page_width_px,
                old_page_height_px=old_page_height_px,
                new_page_width_px=new_page_width_px,
                new_page_height_px=new_page_height_px,
            )
            for item in detection.evidence
        ]
        return LevelRegionDetection(
            level_name=detection.level_name,
            source_page_number=detection.source_page_number,
            bbox_px=bbox,
            state=detection.state,
            confidence=detection.confidence,
            evidence=evidence,
        )

    @classmethod
    def _scale_level_evidence(
        cls,
        *,
        item: LevelViewEvidence,
        old_page_width_px: int,
        old_page_height_px: int,
        new_page_width_px: int,
        new_page_height_px: int,
    ) -> LevelViewEvidence:
        bbox = None
        if item.bbox_px is not None:
            bbox = cls._scale_bbox(
                bbox=item.bbox_px,
                old_page_width_px=old_page_width_px,
                old_page_height_px=old_page_height_px,
                new_page_width_px=new_page_width_px,
                new_page_height_px=new_page_height_px,
            )
        return LevelViewEvidence(
            source=item.source,
            reference=item.reference,
            text=item.text,
            page_number=item.page_number,
            bbox_px=bbox,
            confidence=item.confidence,
        )

    @staticmethod
    def _scale_bbox(
        *,
        bbox: PixelBBox,
        old_page_width_px: int,
        old_page_height_px: int,
        new_page_width_px: int,
        new_page_height_px: int,
    ) -> PixelBBox:
        sx = float(new_page_width_px) / float(old_page_width_px)
        sy = float(new_page_height_px) / float(old_page_height_px)

        x_min = int(round(float(bbox.x_min) * sx))
        y_min = int(round(float(bbox.y_min) * sy))
        x_max = int(round(float(bbox.x_max) * sx))
        y_max = int(round(float(bbox.y_max) * sy))

        x_min = max(0, min(new_page_width_px - 1, x_min))
        y_min = max(0, min(new_page_height_px - 1, y_min))
        x_max = max(x_min + 1, min(new_page_width_px, x_max))
        y_max = max(y_min + 1, min(new_page_height_px, y_max))

        return PixelBBox(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max)
