from __future__ import annotations

import json
from pathlib import Path

from .building_model import QuantiaParametricModel


def write_quantia_parametric_json(
    model: QuantiaParametricModel,
    path: str | Path,
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(model.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return target
