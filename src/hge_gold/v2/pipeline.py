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


def run_thesis_pipeline(config_path: Path, source_csv: Path | None = None) -> dict[str, Path]:
    config = load_config(config_path)
    for directory in (config.artifact_dir, config.model_dir, config.prediction_dir):
        directory.mkdir(parents=True, exist_ok=True)

    resolved_source = source_csv
    if resolved_source is None and config.data.csv_path:
        resolved_source = (config.project_root / config.data.csv_path).resolve()
    source = load_ohlcv(resolved_source, config.data.min_rows, config.models.random_seed)
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
        metrics = classification_metrics(y_true, result.locked_probability_up, result.threshold)
        metrics.update(
            moving_block_bootstrap_ci(
                y_true,
                result.locked_probability_up,
                result.threshold,
                config.evaluation.bootstrap_iterations,
                config.evaluation.bootstrap_block_length,
                config.models.random_seed + horizon,
            )
        )
        acceptance = acceptance_status(metrics, config.evaluation)
        metric_rows.append(
            {
                "task": "binary_direction",
                "horizon": horizon,
                "target_policy_id": "adaptive_actionable_direction_v2_1",
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

    manifest_path = config.artifact_dir / "execution_manifest.json"
    manifest = {
        "pipeline_version": "2.0",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source": "research_sample" if resolved_source is None else str(resolved_source),
        "source_is_market_evidence": resolved_source is not None,
        "artifacts": {
            str(path.relative_to(config.project_root)): _sha256(path)
            for path in [
                metrics_path,
                predictions_path,
                candidates_path,
                folds_path,
                backtests_path,
                selection_path,
                feature_registry_path,
            ]
        },
        "acceptance_summary": metrics_frame[
            ["horizon", "acceptance_status", "balanced_accuracy", "macro_f1"]
        ].to_dict(orient="records"),
        "limitations": [
            "A 60% threshold is an empirical acceptance goal, not a guaranteed market result.",
            "Synthetic sample results validate software behavior only.",
            "Backtest output is research-only and is not investment advice.",
        ],
    }
    _atomic_json(manifest_path, manifest)

    if config.outputs.compatibility_outputs:
        compatibility_metrics = (
            config.project_root
            / "artifacts"
            / "metadata"
            / "phase5_locked_test_metrics_report_v2.csv"
        )
        compatibility_predictions = (
            config.project_root
            / "data"
            / "predictions"
            / "phase4"
            / "phase4_locked_test_predictions_v2.csv"
        )
        compatibility_selection = (
            config.project_root / "artifacts" / "metadata" / "phase4_selected_model_map_v2.json"
        )
        compatibility_metrics.parent.mkdir(parents=True, exist_ok=True)
        compatibility_predictions.parent.mkdir(parents=True, exist_ok=True)
        metrics_frame.to_csv(compatibility_metrics, index=False)
        predictions_frame.to_csv(compatibility_predictions, index=False)
        _atomic_json(compatibility_selection, {"pipeline_version": "2.0", "models": selection_rows})

    return {
        "metrics": metrics_path,
        "predictions": predictions_path,
        "selection": selection_path,
        "manifest": manifest_path,
        "backtests": backtests_path,
    }
