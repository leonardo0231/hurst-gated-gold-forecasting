from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class Paths(BaseModel):
    model_config = ConfigDict(frozen=True)
    root: Path
    data: Path
    artifacts: Path
    models: Path
    reports: Path


class PipelineConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    project: dict[str, Any]
    data: dict[str, Any]
    targets: dict[str, Any]
    features: dict[str, Any]
    modeling: dict[str, Any]
    evaluation: dict[str, Any]
    governance: dict[str, Any]
    source_path: Path = Field(exclude=True)

    @model_validator(mode="after")
    def validate_locked_rules(self) -> PipelineConfig:
        if self.project.get("trading_mode") not in {"offline", "simulation", "sandbox", "paper"}:
            raise ValueError("Real trading mode is forbidden by the locked protocol")
        if self.targets.get("horizons") != [1, 5, 10, 20]:
            raise ValueError("Forecast horizons must be [1, 5, 10, 20]")
        if self.data.get("same_day_cross_market_allowed") is not False:
            raise ValueError("MVP same-day cross-market data must remain disabled")
        return self

    def paths(self) -> Paths:
        root = self.source_path.resolve().parent.parent
        return Paths(
            root=root,
            data=root / os.getenv("HGE_DATA_DIR", "data"),
            artifacts=root / os.getenv("HGE_ARTIFACT_DIR", "artifacts"),
            models=root / os.getenv("HGE_MODEL_DIR", "models"),
            reports=root / "reports",
        )


def load_config(path: str | Path) -> PipelineConfig:
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    return PipelineConfig(**payload, source_path=source)
