from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass(frozen=True)
class DataParams:
    test_size: float
    random_state: int


@dataclass(frozen=True)
class ModelParams:
    C: float
    max_iter: int


@dataclass(frozen=True)
class Params:
    data: DataParams
    model: ModelParams


def load_params(params_path: str | Path) -> Params:
    path = Path(params_path)
    raw: Dict[str, Any] = yaml.safe_load(path.read_text())

    data = raw.get("data", {})
    model = raw.get("model", {})

    return Params(
        data=DataParams(
            test_size=float(data.get("test_size", 0.2)),
            random_state=int(data.get("random_state", 42)),
        ),
        model=ModelParams(
            C=float(model.get("C", 1.0)),
            max_iter=int(model.get("max_iter", 200)),
        ),
    )
