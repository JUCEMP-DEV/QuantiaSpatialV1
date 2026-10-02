from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .contracts import SingleLineWallGraph


class AdaptiveDecisionTraceRecorder:
    """Historial append-only de decisiones del prototipo adaptativo.

    Su objetivo es conservar ejemplos para evaluar y, más adelante, entrenar una
    política de selección de módulos. No entrena modelos ni modifica decisiones.
    """

    ENV_PATH = "QUANTIA_ADAPTIVE_DECISION_HISTORY_PATH"

    def __init__(self, *, path: str | Path | None = None) -> None:
        configured = str(os.getenv(self.ENV_PATH, "") or "").strip()
        self.path = Path(path) if path is not None else (Path(configured) if configured else None)

    def append_graph_decision(self, graph: SingleLineWallGraph) -> None:
        self.append({
            "event": "RECONSTRUCTION_DECISION",
            "level_view_id": graph.level_view_id,
            "level_name": graph.level_name,
            "route_plan": graph.route_plan.model_dump(mode="json"),
            "diagnostics": graph.diagnostics.model_dump(mode="json"),
            "wall_count": len(graph.walls),
            "logical_gap_count": len(graph.logical_gaps),
            "interior_space_count": graph.interior_space_count,
        })

    def append_feedback(
        self,
        *,
        level_view_id: str,
        accepted: bool,
        corrections: dict[str, Any] | None = None,
        source: str = "USER_REVIEW",
    ) -> None:
        self.append({
            "event": "RECONSTRUCTION_FEEDBACK",
            "level_view_id": level_view_id,
            "accepted": bool(accepted),
            "source": source,
            "corrections": corrections or {},
        })

    def append(self, payload: dict[str, Any]) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        row = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), **payload}
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
