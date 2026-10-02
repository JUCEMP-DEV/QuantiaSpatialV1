from __future__ import annotations

import io

from PIL import Image

from app.quantia_spatialV1.core.models.level_view import LevelView
from app.quantia_spatialV1.stages.walls.core.drawing_model import DrawingBBox


class ElementVisualContextBuilder:
    """Construye el único recorte visual que un resolver externo necesitaría.

    No llama a ningún modelo. Mantiene la política Quantia de una sola imagen
    informativa por LevelView/consulta y permite persistir exactamente el input.
    """

    def crop_png(
        self,
        *,
        level_view: LevelView,
        bbox: DrawingBBox,
        padding_ratio: float = 0.55,
        min_padding_px: int = 24,
    ) -> bytes:
        image = Image.open(io.BytesIO(level_view.raster_bytes)).convert("RGB")
        width = max(1.0, bbox.x_max - bbox.x_min)
        height = max(1.0, bbox.y_max - bbox.y_min)
        pad = max(float(min_padding_px), max(width, height) * float(padding_ratio))
        left = max(0, int(round(bbox.x_min - pad)))
        top = max(0, int(round(bbox.y_min - pad)))
        right = min(image.width, int(round(bbox.x_max + pad)))
        bottom = min(image.height, int(round(bbox.y_max + pad)))
        if right <= left or bottom <= top:
            raise ValueError("BBox de elemento no produce un recorte visual válido.")
        crop = image.crop((left, top, right, bottom))
        buffer = io.BytesIO()
        crop.save(buffer, format="PNG")
        return buffer.getvalue()
