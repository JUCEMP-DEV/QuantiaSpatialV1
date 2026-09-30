from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from app.quantia_spatialV1.adaptive_reconstruction.contracts import SingleLineWallGraph
from app.quantia_spatialV1.models.level_view import LevelView


class Call2WallGraphArtifactRenderer:
    """Render exclusivo del WallGraph final corregido por Call 2."""

    @staticmethod
    def render_walls_only(
        *,
        level_view: LevelView,
        graph: SingleLineWallGraph,
        output_path: str | Path,
    ) -> str:
        image = cv2.imdecode(np.frombuffer(level_view.raster_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError("No se pudo decodificar raster para render Call 2.")
        canvas = np.full_like(image, 255)
        for wall in graph.walls:
            cv2.line(
                canvas,
                (int(round(wall.start_px[0])), int(round(wall.start_px[1]))),
                (int(round(wall.end_px[0])), int(round(wall.end_px[1]))),
                (0, 0, 0),
                3,
                cv2.LINE_AA,
            )
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(path), canvas):
            raise RuntimeError(f"No se pudo escribir {path}")
        return str(path)
