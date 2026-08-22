from __future__ import annotations

import json
import shutil
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
    classification_metrics,
    economic_benchmark_summaries,
    moving_block_bootstrap_ci,
)
from .features import build_feature_matrix
from .modeling import train_and_predict
from .mt5 import build_mt5_replay_signals
from .provenance import (
    code_tree_sha256,
    resolved_config_payload,
    runtime_metadata,
    sha256_file,
    sha256_json,
    source_metadata,
)
from .splits import (
    assert_no_label_overlap,
    build_purged_cpcv_splits,
    build_purged_walk_forward_folds,
    split_development_and_locked_test,
)
from .targets import build_horizon_dataset


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    temp.replace(path)


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temp, index=False)
    temp.replace(path)


def _atomic_joblib(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    joblib.dump(payload, temp)
    temp.replace(path)


_sha256 = sha256_file


def run_thesis_pipeline(config_path: Path, source_csv: Path | None = None) -> dict[str, Path]:
    config = load_config(config_path)
    for directory in (config.artifact_dir, config.model_dir, config.prediction_dir):
        directory.mkdir(parents=True, exist_ok=True)

    resolved_source = None if source_csv is None else source_csv.resolve()
    if resolved_source is None and config.data.csv_path:
        resolved_source = (config.project_root / config.data.csv_path).resolve()
    source = load_ohlcv(resolved_source, config.data.min_rows, config.models.random_seed)
    config_payload = resolved_config_payload(config)
    config_sha256 = sha256_json(config_payload)
    provenance = source_metadata(source, resolved_source, config)
    code_sha256 = code_tree_sha256(config.project_root)
    runtime = runtime_metadata(config.project_root)
    runtime_sha256 = sha256_json(runtime)
    run_identity = {
        "pipeline_version": "3.0",
        "source_sha256": provenance["source_sha256"],
        "source_kind": provenance["source_kind"],
        "config_sha256": config_sha256,
        "code_sha256": code_sha256,
        "runtime_sha256": runtime_sha256,
    }
    run_id = f"{config.data.source_kind}-{sha256_json(run_identity)[:16]}"
    run_dir = config.project_root / "artifacts" / "runs" / run_id
    run_receipt_path = run_dir / "run_receipt.json"
    if run_dir.exists():
        raise FileExistsError(
            f"Immutable run already exists; refusing to overwrite mutable outputs: {run_dir}"
        )
    temporary_run_dir = run_dir.with_name(run_dir.name + ".tmp")
    if temporary_run_dir.exists():
        raise FileExistsError(f"Stale temporary run directory requires review: {temporary_run_dir}")
    data_quality_outputs = write_market_data_quality_audit(
        source,
        config.targets.horizons,
        config.features.regime_window,
        config.artifact_dir / "data_quality",
    )
    features, feature_columns = build_feature_matrix(source, config.features)

    metric_rows: list[dict[str, Any]] = []
    prediction_parts: list[pd.DataFrame] = []
    selection_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    backtests: list[dict[str, Any]] = []
    ablation_rows: list[dict[str, Any]] = []
    cpcv_rows: list[dict[str, Any]] = []
    mt5_signal_paths: list[Path] = []
    model_paths: list[Path] = []

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
        cpcv_splits = build_purged_cpcv_splits(
            development,
            n_groups=8,
            n_test_groups=2,
            embargo_rows=config.splits.embargo_rows,
        )
        for split in cpcv_splits:
            cpcv_rows.append(
                {
                    "horizon": horizon,
                    "split_id": split.split_id,
                    "test_group_ids": "|".join(map(str, split.test_group_ids)),
                    "train_start_row": split.train_start_row,
                    "train_end_row": split.train_end_row,
                    "test_start_row": split.test_start_row,
                    "test_end_row": split.test_end_row,
                    "n_train_raw": split.n_train_raw,
                    "n_train": split.n_train,
                    "n_test": split.n_test,
                    "purged_count": split.purged_count,
                    "embargoed_count": split.embargoed_count,
                }
            )

        hurst_columns = [column for column in feature_columns if "hurst" in column]
        dfa_columns = [
            column for column in feature_columns if "hurst_rs_single_scale_legacy" not in column
        ]
        no_hurst_columns = [column for column in feature_columns if "hurst" not in column]
        variants: list[tuple[str, list[str], bool]] = [
            ("dfa_hurst_with_meta", dfa_columns, True),
        ]
        if config.models.run_ablations:
            variants.extend(
                [
                    ("dfa_hurst_no_meta", dfa_columns, False),
                    ("no_hurst_with_meta", no_hurst_columns, True),
                    ("no_hurst_no_meta", no_hurst_columns, False),
                    (
                        "legacy_hurst_no_meta",
                        [
                            column
                            for column in feature_columns
                            if "hurst" not in column or "hurst_rs_single_scale_legacy" in column
                        ],
                        False,
                    ),
                ]
            )

        results: dict[str, Any] = {}
        for variant, variant_columns, allow_meta in variants:
            drop_hurst = variant.startswith("no_hurst")
            variant_development = (
                development.drop(columns=hurst_columns) if drop_hurst else development
            )
            variant_locked = locked.drop(columns=hurst_columns) if drop_hurst else locked
            variant_result = train_and_predict(
                variant_development,
                variant_locked,
                variant_columns,
                folds,
                config.models,
                allow_meta_model=allow_meta,
            )
            results[variant] = variant_result
            variant_metrics = classification_metrics(
                locked["direction_binary"].to_numpy(dtype=int),
                variant_result.locked_probability_up,
                variant_result.threshold,
            )
            ablation_rows.append(
                {
                    "horizon": horizon,
                    "variant": variant,
                    "feature_count": len(variant_columns),
                    "hurst_features_enabled": not drop_hurst,
                    "meta_model_allowed": allow_meta,
                    "selected_strategy": variant_result.selected_strategy,
                    "selected_candidate": variant_result.selected_candidate,
                    "validation_balanced_accuracy": variant_result.validation_metrics.get(
                        "balanced_accuracy"
                    ),
                    "validation_macro_f1": variant_result.validation_metrics.get("macro_f1"),
                    "historical_holdout_balanced_accuracy": variant_metrics["balanced_accuracy"],
                    "historical_holdout_macro_f1": variant_metrics["macro_f1"],
                    "historical_holdout_roc_auc": variant_metrics["roc_auc"],
                    "historical_holdout_brier_score": variant_metrics["brier_score"],
                    "holdout_status": config.splits.holdout_status,
                    "selection_uses_holdout": False,
                }
            )
            for row in variant_result.candidate_metrics:
                candidate_rows.append({"horizon": horizon, "variant": variant, **row})
            fold_lookup = {fold.fold_id: fold for fold in folds}
            for row in variant_result.fold_metrics:
                fold = fold_lookup[str(row["fold_id"])]
                fold_rows.append(
                    {
                        "horizon": horizon,
                        "variant": variant,
                        **row,
                        "train_start_row": fold.train_start_row,
                        "train_end_row": fold.train_end_row,
                        "validation_start_row": fold.validation_start_row,
                        "validation_end_row": fold.validation_end_row,
                        "n_train_raw": fold.n_train_raw,
                        "n_train": fold.n_train,
                        "purged_count": fold.purged_count,
                        "embargoed_count": fold.embargoed_count,
                    }
                )

        result = results["dfa_hurst_with_meta"]

        bundle_path = config.model_dir / f"horizon_{horizon}_model_bundle.joblib"
        _atomic_joblib(bundle_path, result.bundle)
        model_paths.append(bundle_path)
        bundle_hash = _sha256(bundle_path)
        selection_receipt_path = config.artifact_dir / f"selection_receipt_h{horizon}.json"
        _atomic_json(
            selection_receipt_path,
            {
                "pipeline_version": "3.0",
                "run_id": run_id,
                "horizon": horizon,
                "selection_uses_holdout": False,
                "selected_strategy": result.selected_strategy,
                "selected_candidate": result.selected_candidate,
                "threshold": result.threshold,
                "model_sha256": bundle_hash,
                "source_sha256": provenance["source_sha256"],
                "config_sha256": config_sha256,
                "code_sha256": code_sha256,
                "locked_test_start_row": locked_start_row,
                "holdout_status": config.splits.holdout_status,
            },
        )

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
            max(config.evaluation.bootstrap_block_length, horizon),
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
                "task": "statistical_binary_direction",
                "horizon": horizon,
                "target_policy_id": str(dataset["target_policy_id"].iloc[0]),
                "feature_set_id": "causal_gold_features_v3_dfa",
                "selected_strategy": result.selected_strategy,
                "selected_candidate": result.selected_candidate,
                **{key: value for key, value in metrics.items() if key != "confusion_matrix"},
                "acceptance_status": acceptance["status"],
                "locked_test_start_row": locked_start_row,
                "holdout_status": config.splits.holdout_status,
            }
        )
        prediction = locked[
            [
                "row_id",
                "date",
                "horizon",
                "decision_bar_timestamp",
                "decision_timestamp",
                "entry_timestamp",
                "exit_timestamp",
                "statistical_forward_log_return",
                "statistical_direction_binary",
                "executable_forward_log_return",
                "executable_direction_binary",
                "actionable_direction_three_class",
                "actionable_threshold_bps",
                "is_actionable",
                "direction_threshold",
                "direction_binary",
                "target_policy_id",
                "feature_set_id",
                "return_lag_1",
                "momentum_20",
                "volatility_zscore",
            ]
        ].copy()
        prediction["probability_up"] = result.locked_probability_up
        prediction["probability_down"] = 1.0 - result.locked_probability_up
        prediction["probability_threshold"] = result.threshold
        prediction["y_pred"] = (result.locked_probability_up >= result.threshold).astype(int)
        prediction["selected_strategy"] = result.selected_strategy
        prediction["selected_candidate"] = result.selected_candidate
        prediction_parts.append(prediction)
        economic_prediction = prediction.copy()
        economic_prediction["forward_log_return"] = economic_prediction[
            "executable_forward_log_return"
        ]
        benchmark_summaries = economic_benchmark_summaries(
            economic_prediction,
            horizon,
            config.targets.transaction_cost_bps,
            config.targets.slippage_bps,
            config.models.no_trade_probability_margin,
        )
        for benchmark, summary in benchmark_summaries.items():
            backtests.append(
                {
                    "horizon": horizon,
                    "benchmark": benchmark,
                    "return_definition": "next_open_to_open_after_h_bars",
                    "cost_convention": config.targets.cost_convention,
                    **summary,
                }
            )

        mt5_signals = build_mt5_replay_signals(
            prediction,
            horizon,
            config.models.no_trade_probability_margin,
        )
        mt5_signals_path = config.prediction_dir / f"mt5_replay_signals_h{horizon}.csv"
        _atomic_csv(mt5_signals, mt5_signals_path)
        mt5_signal_paths.append(mt5_signals_path)
        selection_rows.append(
            {
                "horizon": horizon,
                "selected_strategy": result.selected_strategy,
                "selected_candidate": result.selected_candidate,
                "threshold": result.threshold,
                "validation_metrics": result.validation_metrics,
                "model_path": bundle_path.relative_to(config.project_root).as_posix(),
                "model_sha256": bundle_hash,
                "selection_uses_locked_test": False,
                "locked_test_start_row": locked_start_row,
                "locked_test_start_date": locked["date"].iloc[0].isoformat(),
                "holdout_status": config.splits.holdout_status,
                "acceptance": acceptance,
            }
        )

    metrics_frame = pd.DataFrame(metric_rows)
    predictions_frame = pd.concat(prediction_parts, ignore_index=True)
    candidates_frame = pd.DataFrame(candidate_rows)
    folds_frame = pd.DataFrame(fold_rows)
    backtests_frame = pd.DataFrame(backtests)
    ablations_frame = pd.DataFrame(ablation_rows)
    cpcv_frame = pd.DataFrame(cpcv_rows)

    metrics_path = config.artifact_dir / "locked_test_metrics.csv"
    predictions_path = config.prediction_dir / "locked_test_predictions.csv"
    candidates_path = config.artifact_dir / "candidate_selection_metrics.csv"
    folds_path = config.artifact_dir / "walk_forward_fold_metrics.csv"
    backtests_path = config.artifact_dir / "backtest_summary.csv"
    ablations_path = config.artifact_dir / "ablation_metrics.csv"
    cpcv_path = config.artifact_dir / "cpcv_split_manifest.csv"
    _atomic_csv(metrics_frame, metrics_path)
    _atomic_csv(predictions_frame, predictions_path)
    _atomic_csv(candidates_frame, candidates_path)
    _atomic_csv(folds_frame, folds_path)
    _atomic_csv(backtests_frame, backtests_path)
    _atomic_csv(ablations_frame, ablations_path)
    _atomic_csv(cpcv_frame, cpcv_path)

    selection_path = config.artifact_dir / "selected_model_map.json"
    _atomic_json(
        selection_path,
        {
            "pipeline_version": "3.0",
            "run_id": run_id,
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
            "feature_set_id": "causal_gold_features_v3_dfa",
            "features": [
                {
                    "name": column,
                    "causal": True,
                    "uses_future_data": False,
                    "uses_target_data": False,
                    "primary_model_feature": column in dfa_columns,
                    "hurst_family": (
                        "dfa1"
                        if "hurst_dfa1" in column or column == "hurst_regime"
                        else "legacy_single_scale"
                        if "hurst_rs_single_scale_legacy" in column
                        else None
                    ),
                }
                for column in feature_columns
            ],
        },
    )
    validation_diagnostics_path = config.artifact_dir / "validation_diagnostics.json"
    _atomic_json(
        validation_diagnostics_path,
        {
            "walk_forward": "chronological expanding-window with closed-interval event purging",
            "embargo_rows": config.splits.embargo_rows,
            "embargo_rationale": (
                "Zero is appropriate for strictly forward walk-forward folds because no "
                "post-validation "
                "observations enter training; CPCV applies the configured post-test embargo."
            ),
            "cpcv": {
                "scope": "development_only_diagnostic_not_deployment_estimate",
                "groups": 8,
                "test_groups": 2,
                "splits_per_horizon": 28,
            },
            "pbo": {
                "implementation": "src/hge_gold/statistics.py",
                "official_value": None,
                "status": "deferred",
                "reason": (
                    "Per-observation development net-return paths for every registered trial "
                    "are not yet persisted."
                ),
            },
            "deflated_sharpe_ratio": {
                "implementation": "src/hge_gold/statistics.py",
                "official_value": None,
                "status": "deferred",
                "reason": (
                    "Requires the same complete development-only trial return registry as PBO."
                ),
            },
            "holdout_status": config.splits.holdout_status,
            "selection_uses_holdout": False,
        },
    )

    source_is_market_evidence = config.data.source_kind == "market_evidence"
    manifest_path = config.artifact_dir / "execution_manifest.json"
    manifest = {
        "pipeline_version": "3.0",
        "run_id": run_id,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "config_sha256": config_sha256,
        "code_sha256": code_sha256,
        "resolved_config": config_payload,
        "runtime": runtime,
        "runtime_sha256": runtime_sha256,
        **provenance,
        "artifacts": {
            path.relative_to(config.project_root).as_posix(): _sha256(path)
            for path in [
                metrics_path,
                predictions_path,
                candidates_path,
                folds_path,
                backtests_path,
                ablations_path,
                cpcv_path,
                selection_path,
                feature_registry_path,
                validation_diagnostics_path,
                *mt5_signal_paths,
                *model_paths,
                *[
                    config.artifact_dir / f"selection_receipt_h{horizon}.json"
                    for horizon in config.targets.horizons
                ],
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
                (
                    "The 2023-2026 holdout was previously revealed; redesigned results are "
                    "repeated historical OOS evidence, not pristine confirmation."
                ),
                (
                    "PBO and DSR code is implemented and tested, but official values are deferred "
                    "until per-trial development return paths are registered."
                ),
                (
                    "Macro drivers are not integrated; claims are limited to broker-specific "
                    "univariate XAUUSD OHLC tick-volume data."
                ),
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

    files_dir = temporary_run_dir / "files"
    artifact_paths = [
        metrics_path,
        predictions_path,
        candidates_path,
        folds_path,
        backtests_path,
        ablations_path,
        cpcv_path,
        selection_path,
        feature_registry_path,
        validation_diagnostics_path,
        manifest_path,
        *mt5_signal_paths,
        *model_paths,
        *[
            config.artifact_dir / f"selection_receipt_h{horizon}.json"
            for horizon in config.targets.horizons
        ],
        *data_quality_outputs.values(),
    ]
    for path in artifact_paths:
        relative = path.relative_to(config.project_root)
        destination = files_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    receipt = {
        "run_id": run_id,
        "identity": run_identity,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "files": {
            path.relative_to(config.project_root).as_posix(): _sha256(path)
            for path in artifact_paths
        },
    }
    _atomic_json(temporary_run_dir / "run_receipt.json", receipt)
    temporary_run_dir.replace(run_dir)
    trackable_receipt_path = config.artifact_dir / "run_receipt.json"
    _atomic_json(trackable_receipt_path, receipt)

    return {
        "metrics": metrics_path,
        "predictions": predictions_path,
        "selection": selection_path,
        "manifest": manifest_path,
        "backtests": backtests_path,
        "ablations": ablations_path,
        "cpcv": cpcv_path,
        "validation_diagnostics": validation_diagnostics_path,
        "run_receipt": run_receipt_path,
        "trackable_run_receipt": trackable_receipt_path,
        **data_quality_outputs,
    }
