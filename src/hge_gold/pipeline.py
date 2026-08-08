from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from .config import load_config
from .data import load_ohlcv
from .data_audit import write_market_data_quality_audit
from .evaluation import (
    acceptance_status,
    backtest_summary,
    classification_metrics,
    moving_block_bootstrap_ci,
)
from .features import build_feature_matrix
from .modeling import train_and_predict
from .splits import (
    assert_no_label_overlap,
    build_purged_walk_forward_folds,
    split_development_and_locked_test,
)
from .targets import build_horizon_dataset


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    temp.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


_SOURCE_METADATA_FIELDS = (
    "symbol",
    "timeframe",
    "source_type",
    "broker",
    "server",
    "timezone",
    "export_date",
)


def _source_file_metadata(path: Path | None) -> dict[str, str | None]:
    """Read explicitly named provenance columns without loading a large CSV."""
    if path is None:
        return {field: None for field in _SOURCE_METADATA_FIELDS}
    first_row = pd.read_csv(path, nrows=1)
    if first_row.empty:
        return {field: None for field in _SOURCE_METADATA_FIELDS}
    return {
        field: None
        if field not in first_row or pd.isna(first_row[field].iloc[0])
        else str(first_row[field].iloc[0])
        for field in _SOURCE_METADATA_FIELDS
    }


def _source_manifest_metadata(
    source: pd.DataFrame, resolved_source: Path | None, config: Any
) -> dict[str, Any]:
    """Build input provenance from the validated source and declared data metadata."""
    is_market_evidence = resolved_source is not None
    file_metadata = _source_file_metadata(resolved_source)

    def metadata_value(field: str) -> str | None:
        configured_value = getattr(config.data, field)
        return configured_value if configured_value is not None else file_metadata[field]

    source_type = metadata_value("source_type")
    if source_type is None and not is_market_evidence:
        source_type = "synthetic research sample"

    if resolved_source is None:
        source_display = "research_sample"
    else:
        try:
            source_display = resolved_source.relative_to(config.project_root).as_posix()
        except ValueError:
            source_display = str(resolved_source)

    return {
        "source": source_display,
        "source_is_market_evidence": is_market_evidence,
        "source_sha256": None if resolved_source is None else _sha256(resolved_source),
        "source_rows": len(source),
        "source_start": source["date"].iloc[0].isoformat(),
        "source_end": source["date"].iloc[-1].isoformat(),
        "symbol": metadata_value("symbol"),
        "timeframe": metadata_value("timeframe"),
        "source_type": source_type,
        "broker": metadata_value("broker"),
        "server": metadata_value("server"),
        "timezone": metadata_value("timezone"),
        "export_date": metadata_value("export_date"),
    }


def run_thesis_pipeline(config_path: Path, source_csv: Path | None = None) -> dict[str, Path]:
    config = load_config(config_path)
    for directory in (config.artifact_dir, config.model_dir, config.prediction_dir):
        directory.mkdir(parents=True, exist_ok=True)

    resolved_source = None if source_csv is None else source_csv.resolve()
    if resolved_source is None and config.data.csv_path:
        resolved_source = (config.project_root / config.data.csv_path).resolve()
    source = load_ohlcv(resolved_source, config.data.min_rows, config.models.random_seed)
    data_quality_outputs = write_market_data_quality_audit(
        source,
        config.targets.horizons,
        config.features.regime_window,
        config.project_root / "artifacts" / "data_quality",
    )
    features, feature_columns = build_feature_matrix(source, config.features)

    metric_rows: list[dict[str, Any]] = []
    prediction_parts: list[pd.DataFrame] = []
    selection_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    backtests: list[dict[str, Any]] = []

    for horizon in config.targets.horizons:
        dataset = build_horizon_dataset(
            source,
            features,
            feature_columns,
            horizon,
            config.targets,
            config.features,
        )
        eligible = dataset[dataset["is_modeling_eligible"]].reset_index(drop=True)
        development, locked, locked_start_row = split_development_and_locked_test(
            eligible, config.splits
        )
        folds = build_purged_walk_forward_folds(development, config.splits)
        assert_no_label_overlap(development, folds)
        result = train_and_predict(development, locked, feature_columns, folds, config.models)

        y_true = locked["direction_binary"].to_numpy(dtype=int)
        classification = classification_metrics(
            y_true,
            result.locked_probability_up,
            result.threshold,
        )
        bootstrap_ci = moving_block_bootstrap_ci(
            y_true,
            result.locked_probability_up,
            result.threshold,
            config.evaluation.bootstrap_iterations,
            config.evaluation.bootstrap_block_length,
            config.models.random_seed + horizon,
        )
        acceptance = acceptance_status(
            classification,
            config.evaluation,
        )
        metrics: dict[str, Any] = dict(classification)
        metrics.update(bootstrap_ci)
        metric_rows.append(
            {
                "task": "binary_direction",
                "horizon": horizon,
                "target_policy_id": "all_samples_binary_direction_v2_1",
                "feature_set_id": "causal_gold_features_v2",
                "selected_strategy": result.selected_strategy,
                "selected_candidate": result.selected_candidate,
                **{key: value for key, value in metrics.items() if key != "confusion_matrix"},
                "acceptance_status": acceptance["status"],
                "locked_test_start_row": locked_start_row,
            }
        )
        prediction = locked[
            [
                "row_id",
                "date",
                "horizon",
                "forward_log_return",
                "direction_threshold",
                "direction_binary",
                "target_policy_id",
                "feature_set_id",
            ]
        ].copy()
        prediction["probability_up"] = result.locked_probability_up
        prediction["probability_down"] = 1.0 - result.locked_probability_up
        prediction["probability_threshold"] = result.threshold
        prediction["y_pred"] = (result.locked_probability_up >= result.threshold).astype(int)
        prediction["selected_strategy"] = result.selected_strategy
        prediction["selected_candidate"] = result.selected_candidate
        prediction_parts.append(prediction)

        bundle_path = config.model_dir / f"horizon_{horizon}_model_bundle.joblib"
        joblib.dump(result.bundle, bundle_path)
        selection_rows.append(
            {
                "horizon": horizon,
                "selected_strategy": result.selected_strategy,
                "selected_candidate": result.selected_candidate,
                "threshold": result.threshold,
                "validation_metrics": result.validation_metrics,
                "model_path": str(bundle_path.relative_to(config.project_root)),
                "model_sha256": _sha256(bundle_path),
                "selection_uses_locked_test": False,
                "locked_test_start_row": locked_start_row,
                "acceptance": acceptance,
            }
        )
        for row in result.candidate_metrics:
            candidate_rows.append({"horizon": horizon, **row})
        for row in result.fold_metrics:
            fold_rows.append({"horizon": horizon, **row})
        backtests.append(
            {
                "horizon": horizon,
                **backtest_summary(
                    prediction,
                    horizon,
                    config.targets.transaction_cost_bps,
                    config.targets.slippage_bps,
                ),
            }
        )

    metrics_frame = pd.DataFrame(metric_rows)
    predictions_frame = pd.concat(prediction_parts, ignore_index=True)
    candidates_frame = pd.DataFrame(candidate_rows)
    folds_frame = pd.DataFrame(fold_rows)
    backtests_frame = pd.DataFrame(backtests)

    metrics_path = config.artifact_dir / "locked_test_metrics.csv"
    predictions_path = config.prediction_dir / "locked_test_predictions.csv"
    candidates_path = config.artifact_dir / "candidate_selection_metrics.csv"
    folds_path = config.artifact_dir / "walk_forward_fold_metrics.csv"
    backtests_path = config.artifact_dir / "backtest_summary.csv"
    metrics_frame.to_csv(metrics_path, index=False)
    predictions_frame.to_csv(predictions_path, index=False)
    candidates_frame.to_csv(candidates_path, index=False)
    folds_frame.to_csv(folds_path, index=False)
    backtests_frame.to_csv(backtests_path, index=False)

    selection_path = config.artifact_dir / "selected_model_map.json"
    _atomic_json(
        selection_path,
        {
            "pipeline_version": "2.0",
            "project_name": config.project_name,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "selection_uses_locked_test": False,
            "models": selection_rows,
        },
    )
    feature_registry_path = config.artifact_dir / "feature_registry.json"
    _atomic_json(
        feature_registry_path,
        {
            "feature_set_id": "causal_gold_features_v2",
            "features": [
                {
                    "name": column,
                    "causal": True,
                    "uses_future_data": False,
                    "uses_target_data": False,
                }
                for column in feature_columns
            ],
        },
    )

    source_is_market_evidence = resolved_source is not None
    manifest_path = config.artifact_dir / "execution_manifest.json"
    manifest = {
        "pipeline_version": "2.0",
        "created_at_utc": datetime.now(UTC).isoformat(),
        **_source_manifest_metadata(source, resolved_source, config),
        "artifacts": {
            path.relative_to(config.project_root).as_posix(): _sha256(path)
            for path in [
                metrics_path,
                predictions_path,
                candidates_path,
                folds_path,
                backtests_path,
                selection_path,
                feature_registry_path,
                *data_quality_outputs.values(),
            ]
        },
        "acceptance_summary": metrics_frame[
            ["horizon", "acceptance_status", "balanced_accuracy", "macro_f1"]
        ].to_dict(orient="records"),
        "limitations": (
            [
                "A 60% threshold is an empirical acceptance goal, not a guaranteed market result.",
                "Backtest output is research-only and is not investment advice.",
            ]
            if source_is_market_evidence
            else [
                "A 60% threshold is an empirical acceptance goal, not a guaranteed market result.",
                "Synthetic sample results validate software behavior only.",
                "Backtest output is research-only and is not investment advice.",
            ]
        ),
    }
    _atomic_json(manifest_path, manifest)

    return {
        "metrics": metrics_path,
        "predictions": predictions_path,
        "selection": selection_path,
        "manifest": manifest_path,
        "backtests": backtests_path,
        **data_quality_outputs,
    }
