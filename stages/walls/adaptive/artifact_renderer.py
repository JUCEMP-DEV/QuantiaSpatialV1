from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from app.quantia_spatialV1.core.models.level_view import LevelView

from .contracts import AdaptiveReconstructionRuntime
from .multimodal_review import WallGraphMultimodalReviewer


class AdaptiveReconstructionArtifactRenderer:
    """Render de diagnóstico; no participa en decisiones del motor."""

    @staticmethod
    def _decode(data: bytes) -> np.ndarray:
        image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError("No se pudo decodificar raster del LevelView.")
        return image

    @staticmethod
    def _point(value) -> tuple[float, float]:
        if hasattr(value, "x"):
            return float(value.x), float(value.y)
        return float(value[0]), float(value[1])

    @classmethod
    def _line(cls, image: np.ndarray, item, color, thickness: int) -> None:
        start, end = cls._point(item.start), cls._point(item.end)
        cv2.line(
            image,
            (int(round(start[0])), int(round(start[1]))),
            (int(round(end[0])), int(round(end[1]))),
            color,
            thickness,
            cv2.LINE_AA,
        )

    @classmethod
    def render(
        cls,
        *,
        level_view: LevelView,
        runtime: AdaptiveReconstructionRuntime,
        output_dir: Path,
        stem: str,
    ) -> dict[str, str]:
        output_dir.mkdir(parents=True, exist_ok=True)
        raster = cls._decode(level_view.raster_bytes)
        white = np.full_like(raster, 255)

        seed = white.copy()
        for item in runtime.artifacts.get("f03_seed_candidates", []):
            cls._line(seed, item, (0, 0, 0), 3)

        hybrid = white.copy()
        for wall in runtime.wall_tracks:
            cls._line(hybrid, wall, (0, 0, 0), 2)

        clean = white.copy()
        for wall in runtime.graph.walls:
            cv2.line(
                clean,
                (int(round(wall.start_px[0])), int(round(wall.start_px[1]))),
                (int(round(wall.end_px[0])), int(round(wall.end_px[1]))),
                (0, 0, 0),
                3,
                cv2.LINE_AA,
            )

        rooms = white.copy()
        for label in sorted(runtime.topology.interior_labels):
            component = np.where(runtime.topology.labels == label, 255, 0).astype(np.uint8)
            contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(rooms, contours, -1, (0, 0, 0), 2, cv2.LINE_AA)

        audit = cv2.addWeighted(raster, 0.12, np.full_like(raster, 255), 0.88, 0.0)
        role_colors = {"PERIMETER": (0, 0, 255), "DIVIDER": (0, 140, 0), "REVIEW": (170, 170, 170)}
        for wall in runtime.graph.walls:
            cv2.line(
                audit,
                (int(round(wall.start_px[0])), int(round(wall.start_px[1]))),
                (int(round(wall.end_px[0])), int(round(wall.end_px[1]))),
                role_colors[wall.role],
                2,
                cv2.LINE_AA,
            )
        for gap in runtime.graph.logical_gaps:
            cv2.line(
                audit,
                (int(round(gap.start_px[0])), int(round(gap.start_px[1]))),
                (int(round(gap.end_px[0])), int(round(gap.end_px[1]))),
                (0, 165, 255),
                2,
                cv2.LINE_AA,
            )

        wall_mask = runtime.artifacts["wall_mask"]
        logical_mask = runtime.artifacts["logical_mask"]
        reviewer = WallGraphMultimodalReviewer.__new__(WallGraphMultimodalReviewer)
        call2_composite = reviewer.build_review_image(level_view=level_view, graph=runtime.graph)

        paths = {
            "f03_seed": output_dir / f"{stem}__01_f03_seed.png",
            "hybrid_wall_tracks": output_dir / f"{stem}__02_hybrid_wall_tracks.png",
            "wall_mask": output_dir / f"{stem}__03_wall_mask.png",
            "single_line_wallgraph": output_dir / f"{stem}__04_single_line_wallgraph.png",
            "rooms": output_dir / f"{stem}__05_rooms.png",
            "topology_audit": output_dir / f"{stem}__06_topology_audit.png",
            "logical_mask": output_dir / f"{stem}__07_logical_mask.png",
            "call2_input": output_dir / f"{stem}__08_call2_original_plus_wallgraph.png",
        }
        cv2.imwrite(str(paths["f03_seed"]), seed)
        cv2.imwrite(str(paths["hybrid_wall_tracks"]), hybrid)
        cv2.imwrite(str(paths["wall_mask"]), wall_mask)
        cv2.imwrite(str(paths["single_line_wallgraph"]), clean)
        cv2.imwrite(str(paths["rooms"]), rooms)
        cv2.imwrite(str(paths["topology_audit"]), audit)
        cv2.imwrite(str(paths["logical_mask"]), logical_mask)
        paths["call2_input"].write_bytes(call2_composite)
        return {key: str(value) for key, value in paths.items()}
