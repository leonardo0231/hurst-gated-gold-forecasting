from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DataConfig:
    source: str = "sample"
    csv_path: str | None = None
    min_rows: int = 700
    decision_hour_utc: int = 22
    symbol: str | None = None
    timeframe: str | None = None
    source_type: str | None = None
    broker: str | None = None
    server: str | None = None
    timezone: str | None = None
    export_date: str | None = None


@dataclass(frozen=True)
class FeatureConfig:
    return_lags: tuple[int, ...] = (1, 2, 3, 5, 10, 20)
    windows: tuple[int, ...] = (5, 10, 20, 63, 126)
    hurst_windows: tuple[int, ...] = (64, 128)
    regime_window: int = 252
    min_feature_coverage: float = 0.70


@dataclass(frozen=True)
class TargetConfig:
    horizons: tuple[int, ...] = (1, 5, 10, 20)
    volatility_window: int = 63
    volatility_min_periods: int = 40
    threshold_k: float = 0.35
    threshold_floor_bps: float = 8.0
    transaction_cost_bps: float = 3.0
    slippage_bps: float = 2.0


@dataclass(frozen=True)
class SplitConfig:
    locked_test_fraction: float = 0.20
    n_walk_forward_folds: int = 5
    min_train_rows: int = 280
    min_validation_rows: int = 45


@dataclass(frozen=True)
class ModelConfig:
    random_seed: int = 42
    fast_mode: bool = False
    probability_threshold_min: float = 0.35
    probability_threshold_max: float = 0.65
    probability_threshold_steps: int = 61
    gate_tolerance: float = 0.005


@dataclass(frozen=True)
class EvaluationConfig:
    primary_metric: str = "balanced_accuracy"
    primary_threshold: float = 0.60
    macro_f1_threshold: float = 0.55
    minimum_class_recall: float = 0.50
    min_test_samples: int = 80
    bootstrap_iterations: int = 400
    bootstrap_block_length: int = 10


@dataclass(frozen=True)
class OutputConfig:
    artifact_dir: str = "artifacts/v2"
    model_dir: str = "models/v2"
    prediction_dir: str = "data/predictions/v2"
    compatibility_outputs: bool = True


@dataclass(frozen=True)
class ThesisConfig:
    project_root: Path
    project_name: str = "HGE Gold Forecasting Thesis V2"
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    targets: TargetConfig = field(default_factory=TargetConfig)
    splits: SplitConfig = field(default_factory=SplitConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    outputs: OutputConfig = field(default_factory=OutputConfig)

    @property
    def artifact_dir(self) -> Path:
        return self.project_root / self.outputs.artifact_dir

    @property
    def model_dir(self) -> Path:
        return self.project_root / self.outputs.model_dir

    @property
    def prediction_dir(self) -> Path:
        return self.project_root / self.outputs.prediction_dir


def _section(payload: dict[str, Any], name: str) -> dict[str, Any]:
    value = payload.get(name, {})
    if not isinstance(value, dict):
        raise ValueError(f"Configuration section '{name}' must be a mapping")
    return value


def _tuple_int(values: Any, default: tuple[int, ...]) -> tuple[int, ...]:
    if values is None:
        return default
    parsed = tuple(int(item) for item in values)
    if not parsed or any(item <= 0 for item in parsed):
        raise ValueError("Window and horizon values must be positive integers")
    return parsed


def load_config(path: Path) -> ThesisConfig:
    path = path.resolve()
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError("Configuration root must be a mapping")
    project_root = path.parent.parent

    data_raw = _section(payload, "data")
    feature_raw = _section(payload, "features")
    target_raw = _section(payload, "targets")
    split_raw = _section(payload, "splits")
    model_raw = _section(payload, "models")
    evaluation_raw = _section(payload, "evaluation")
    output_raw = _section(payload, "outputs")

    config = ThesisConfig(
        project_root=project_root,
        project_name=str(payload.get("project_name", "HGE Gold Forecasting Thesis V2")),
        data=DataConfig(
            source=str(data_raw.get("source", "sample")),
            csv_path=data_raw.get("csv_path"),
            min_rows=int(data_raw.get("min_rows", 700)),
            decision_hour_utc=int(data_raw.get("decision_hour_utc", 22)),
            symbol=_optional_string(data_raw.get("symbol")),
            timeframe=_optional_string(data_raw.get("timeframe")),
            source_type=_optional_string(data_raw.get("source_type")),
            broker=_optional_string(data_raw.get("broker")),
            server=_optional_string(data_raw.get("server")),
            timezone=_optional_string(data_raw.get("timezone")),
            export_date=_optional_string(data_raw.get("export_date")),
        ),
        features=FeatureConfig(
            return_lags=_tuple_int(feature_raw.get("return_lags"), (1, 2, 3, 5, 10, 20)),
            windows=_tuple_int(feature_raw.get("windows"), (5, 10, 20, 63, 126)),
            hurst_windows=_tuple_int(feature_raw.get("hurst_windows"), (64, 128)),
            regime_window=int(feature_raw.get("regime_window", 252)),
            min_feature_coverage=float(feature_raw.get("min_feature_coverage", 0.70)),
        ),
        targets=TargetConfig(
            horizons=_tuple_int(target_raw.get("horizons"), (1, 5, 10, 20)),
            volatility_window=int(target_raw.get("volatility_window", 63)),
            volatility_min_periods=int(target_raw.get("volatility_min_periods", 40)),
            threshold_k=float(target_raw.get("threshold_k", 0.35)),
            threshold_floor_bps=float(target_raw.get("threshold_floor_bps", 8.0)),
            transaction_cost_bps=float(target_raw.get("transaction_cost_bps", 3.0)),
            slippage_bps=float(target_raw.get("slippage_bps", 2.0)),
        ),
        splits=SplitConfig(
            locked_test_fraction=float(split_raw.get("locked_test_fraction", 0.20)),
            n_walk_forward_folds=int(split_raw.get("n_walk_forward_folds", 5)),
            min_train_rows=int(split_raw.get("min_train_rows", 280)),
            min_validation_rows=int(split_raw.get("min_validation_rows", 45)),
        ),
        models=ModelConfig(
            random_seed=int(model_raw.get("random_seed", 42)),
            fast_mode=bool(model_raw.get("fast_mode", False)),
            probability_threshold_min=float(model_raw.get("probability_threshold_min", 0.35)),
            probability_threshold_max=float(model_raw.get("probability_threshold_max", 0.65)),
            probability_threshold_steps=int(model_raw.get("probability_threshold_steps", 61)),
            gate_tolerance=float(model_raw.get("gate_tolerance", 0.005)),
        ),
        evaluation=EvaluationConfig(
            primary_metric=str(evaluation_raw.get("primary_metric", "balanced_accuracy")),
            primary_threshold=float(evaluation_raw.get("primary_threshold", 0.60)),
            macro_f1_threshold=float(evaluation_raw.get("macro_f1_threshold", 0.55)),
            minimum_class_recall=float(evaluation_raw.get("minimum_class_recall", 0.50)),
            min_test_samples=int(evaluation_raw.get("min_test_samples", 80)),
            bootstrap_iterations=int(evaluation_raw.get("bootstrap_iterations", 400)),
            bootstrap_block_length=int(evaluation_raw.get("bootstrap_block_length", 10)),
        ),
        outputs=OutputConfig(
            artifact_dir=str(output_raw.get("artifact_dir", "artifacts/v2")),
            model_dir=str(output_raw.get("model_dir", "models/v2")),
            prediction_dir=str(output_raw.get("prediction_dir", "data/predictions/v2")),
            compatibility_outputs=bool(output_raw.get("compatibility_outputs", True)),
        ),
    )
    _validate_config(config)
    return config


def _optional_string(value: Any) -> str | None:
    return None if value is None else str(value)


def _validate_config(config: ThesisConfig) -> None:
    if not 0.05 <= config.splits.locked_test_fraction <= 0.40:
        raise ValueError("locked_test_fraction must be between 0.05 and 0.40")
    if not 0.0 < config.features.min_feature_coverage <= 1.0:
        raise ValueError("min_feature_coverage must be in (0, 1]")
    if config.evaluation.primary_metric != "balanced_accuracy":
        raise ValueError("V2 currently registers balanced_accuracy as its primary metric")
    if config.models.probability_threshold_min >= config.models.probability_threshold_max:
        raise ValueError("Invalid probability threshold search interval")
