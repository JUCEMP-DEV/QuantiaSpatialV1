from __future__ import annotations

from hashlib import sha1
from io import BytesIO

from app.quantia_spatialV1.models.level_view import (
    LevelIdentificationState,
    LevelView,
    LevelViewEvidence,
    LevelViewTransform,
    PixelBBox,
)
from PIL import Image, UnidentifiedImageError


class LevelViewBuilder:
    """
    Construye la vista raster aislada de un nivel.

    Responsabilidad exclusiva:

        raster de página
            +
        bbox del nivel
            ↓
        LevelView

    Reglas:

    - NO identifica niveles.
    - NO interpreta arquitectura.
    - NO convierte px -> m.
    - NO reescala el recorte.
    - NO modifica la geometría detectada.
    - Conserva transformación exacta hacia la página original.
    - Todo LevelView automático permanece confirmed=False.
    """

    # ========================================================
    # API PRINCIPAL
    # ========================================================

    def build(
        self,
        *,
        source_raster_bytes: bytes,
        source_page_number: int,
        source_bbox_px: PixelBBox,
        level_name: str | None,
        state: LevelIdentificationState,
        confidence: float | None = None,
        source_document_id: str | None = None,
        evidence: list[LevelViewEvidence] | None = None,
        level_view_id: str | None = None,
    ) -> LevelView:
        """
        Recorta una región de la página y genera un LevelView.

        El tamaño real del raster fuente se obtiene directamente
        de la imagen.

        El recorte conserva exactamente la escala raster original.
        """

        if not source_raster_bytes:
            raise ValueError("source_raster_bytes no puede estar vacío.")

        if source_page_number <= 0:
            raise ValueError("source_page_number debe ser mayor que cero.")

        image = self._open_image(source_raster_bytes)

        try:
            source_width_px, source_height_px = image.size

            self._validate_bbox_inside_page(
                bbox=source_bbox_px,
                page_width_px=source_width_px,
                page_height_px=source_height_px,
            )

            cropped_image = image.crop(
                (
                    source_bbox_px.x_min,
                    source_bbox_px.y_min,
                    source_bbox_px.x_max,
                    source_bbox_px.y_max,
                )
            )

            raster_bytes = self._encode_png(cropped_image)

        finally:
            image.close()

        raster_width_px = source_bbox_px.width
        raster_height_px = source_bbox_px.height

        transform = LevelViewTransform(
            offset_x_px=source_bbox_px.x_min,
            offset_y_px=source_bbox_px.y_min,
            source_page_width_px=source_width_px,
            source_page_height_px=source_height_px,
            local_width_px=raster_width_px,
            local_height_px=raster_height_px,
        )

        resolved_id = level_view_id or self._build_stable_id(
            source_document_id=source_document_id,
            source_page_number=source_page_number,
            level_name=level_name,
            bbox=source_bbox_px,
        )

        return LevelView(
            id=resolved_id,
            level_name=level_name,
            source_document_id=source_document_id,
            source_page_number=source_page_number,
            source_bbox_px=source_bbox_px,
            source_page_width_px=source_width_px,
            source_page_height_px=source_height_px,
            raster_width_px=raster_width_px,
            raster_height_px=raster_height_px,
            raster_mime_type="image/png",
            raster_bytes=raster_bytes,
            transform=transform,
            state=state,
            confidence=confidence,
            evidence=list(evidence or []),
            confirmed=False,
        )

    # ========================================================
    # PÁGINA COMPLETA
    # ========================================================

    def build_full_page(
        self,
        *,
        source_raster_bytes: bytes,
        source_page_number: int,
        level_name: str | None,
        state: LevelIdentificationState,
        confidence: float | None = None,
        source_document_id: str | None = None,
        evidence: list[LevelViewEvidence] | None = None,
        level_view_id: str | None = None,
    ) -> LevelView:
        """
        Construye un LevelView utilizando toda la página.

        Este caso corresponde, por ejemplo, a una planta que ya
        llega aislada y no necesita división espacial adicional.
        """

        if not source_raster_bytes:
            raise ValueError("source_raster_bytes no puede estar vacío.")

        image = self._open_image(source_raster_bytes)

        try:
            width_px, height_px = image.size
        finally:
            image.close()

        full_bbox = PixelBBox(
            x_min=0,
            y_min=0,
            x_max=width_px,
            y_max=height_px,
        )

        return self.build(
            source_raster_bytes=source_raster_bytes,
            source_page_number=source_page_number,
            source_bbox_px=full_bbox,
            level_name=level_name,
            state=state,
            confidence=confidence,
            source_document_id=source_document_id,
            evidence=evidence,
            level_view_id=level_view_id,
        )

    # ========================================================
    # IMAGEN
    # ========================================================

    @staticmethod
    def _open_image(
        raster_bytes: bytes,
    ) -> Image.Image:
        """
        Abre el raster y fuerza la carga completa antes de que el
        BytesIO desaparezca.

        No aplica resize, rotate ni transformación geométrica.
        """

        try:
            buffer = BytesIO(raster_bytes)
            image = Image.open(buffer)
            image.load()

        except UnidentifiedImageError as exc:
            raise ValueError(
                "Los bytes recibidos no corresponden a una imagen raster válida."
            ) from exc

        except Exception as exc:
            raise ValueError("No fue posible abrir el raster fuente.") from exc

        return image

    @staticmethod
    def _encode_png(
        image: Image.Image,
    ) -> bytes:
        """
        Serializa el recorte como PNG.

        PNG evita introducir pérdida adicional durante el
        aislamiento del nivel.
        """

        output = BytesIO()

        image.save(
            output,
            format="PNG",
        )

        return output.getvalue()

    # ========================================================
    # VALIDACIONES
    # ========================================================

    @staticmethod
    def _validate_bbox_inside_page(
        *,
        bbox: PixelBBox,
        page_width_px: int,
        page_height_px: int,
    ) -> None:
        """
        Garantiza que el bbox pertenece completamente al raster
        original.
        """

        if page_width_px <= 0 or page_height_px <= 0:
            raise ValueError("El raster fuente tiene dimensiones inválidas.")

        if bbox.x_max > page_width_px:
            raise ValueError("El bbox del nivel excede el ancho del raster fuente.")

        if bbox.y_max > page_height_px:
            raise ValueError("El bbox del nivel excede el alto del raster fuente.")

    # ========================================================
    # IDENTIDAD
    # ========================================================

    @staticmethod
    def _build_stable_id(
        *,
        source_document_id: str | None,
        source_page_number: int,
        level_name: str | None,
        bbox: PixelBBox,
    ) -> str:
        """
        Genera un ID determinista para la misma región del mismo
        documento/página.

        No utiliza estado mutable ni valores aleatorios.
        """

        identity = "|".join(
            [
                source_document_id or "NO_DOCUMENT_ID",
                str(source_page_number),
                level_name or "NO_LEVEL_NAME",
                str(bbox.x_min),
                str(bbox.y_min),
                str(bbox.x_max),
                str(bbox.y_max),
            ]
        )

        digest = sha1(identity.encode("utf-8")).hexdigest()[:16]

        return f"LEVEL_VIEW_{digest}"
