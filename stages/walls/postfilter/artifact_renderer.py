from __future__ import annotations

import cv2
import numpy as np

from app.quantia_spatialV1.stages.walls.adaptive.contracts import AdaptiveReconstructionRuntime, SingleLineWall
from app.quantia_spatialV1.core.models.level_view import LevelView

from .contracts import PostReconstructionFilterResult


class PostFilterArtifactRenderer:
    """Evidencia visual; nunca participa en decisiones.

    `render_wall_only` publica exclusivamente la reconstruccion final de muros,
    para revision visual limpia. `render` conserva la auditoria completa cuando
    sea necesaria.
    """

    @staticmethod
    def _decode(level_view: LevelView) -> np.ndarray:
        image = cv2.imdecode(np.frombuffer(level_view.raster_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError("No se pudo decodificar raster para auditoria post-filter.")
        return image

    @staticmethod
    def _line(image: np.ndarray, wall: SingleLineWall, color: tuple[int, int, int], thickness: int = 2) -> None:
        cv2.line(
            image,
            (int(round(wall.start_px[0])), int(round(wall.start_px[1]))),
            (int(round(wall.end_px[0])), int(round(wall.end_px[1]))),
            color,
            thickness,
            cv2.LINE_AA,
        )

    def render_wall_only(
        self,
        *,
        level_view: LevelView,
        result: PostReconstructionFilterResult,
    ) -> dict[str, np.ndarray]:
        original = self._decode(level_view)
        clean = np.full_like(original, 255)
        for wall in result.filtered_wall_graph.walls:
            self._line(clean, wall, (0, 0, 0), 3)
        return {"01_walls_only": clean}

    def render(
        self,
        *,
        level_view: LevelView,
        runtime: AdaptiveReconstructionRuntime,
        result: PostReconstructionFilterResult,
    ) -> dict[str, np.ndarray]:
        original = self._decode(level_view)
        by_id = {wall.id: wall for wall in runtime.graph.walls}

        input_view = np.full_like(original, 255)
        for wall in runtime.graph.walls:
            self._line(input_view, wall, (0, 0, 0), 2)

        classification = original.copy()
        palette = {
            "WALL": (0, 180, 0),
            "ARCHITECTURAL_ELEMENT": (255, 0, 255),
            "EXCLUDED_GRAPHIC": (0, 140, 255),
            "UNRESOLVED": (0, 255, 255),
        }
        for decision in result.decisions:
            wall = by_id.get(decision.source_wall_id)
            if wall is not None:
                self._line(classification, wall, palette[decision.output_class], 2)

        auxiliary = np.full_like(original, 255)
        for item in result.architectural_elements:
            self._line(auxiliary, item.source_wall, (255, 0, 255), 2)
        for item in result.excluded_graphics:
            self._line(auxiliary, item.source_wall, (0, 140, 255), 2)
        for item in result.unresolved:
            self._line(auxiliary, item.source_wall, (0, 255, 255), 2)

        clean = np.full_like(original, 255)
        for wall in result.filtered_wall_graph.walls:
            self._line(clean, wall, (0, 0, 0), 3)

        patterns = original.copy()
        pattern_palette = {
            "REPETITIVE_PARALLEL": (255, 0, 255),
            "ORTHOGONAL_GRID": (0, 140, 255),
            "ANGULAR_HUB": (255, 255, 0),
        }
        for pattern in result.pattern_evidence:
            color = pattern_palette[pattern.kind]
            for wall_id in pattern.wall_ids:
                wall = by_id.get(wall_id)
                if wall is not None:
                    self._line(patterns, wall, color, 3)

        return {
            "01_input_adaptive": input_view,
            "02_filter_classification": classification,
            "03_encapsulated_and_excluded": auxiliary,
            "04_filtered_single_line_wallgraph": clean,
            "05_pattern_evidence": patterns,
        }
