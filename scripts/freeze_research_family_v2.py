"""Freeze the corrected v2 hypothesis family before any metric-producing execution."""

from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from hge_gold.config import FeatureConfig
from hge_gold.features import build_feature_matrix
from hge_gold.partitions import load_frozen_development_partition
from hge_gold.research_experiments import add_robust_hurst_features, select_arm_features
from hge_gold.research_experiments_v2 import ARMS, DECLARED_BUDGET_V2, FAMILY_V2, HORIZONS
from hge_gold.research_protocol import ProtocolViolation, sha256_file


def _write_exclusive(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise ProtocolViolation(f"Refusing to overwrite preregistration: {path}") from exc


def _relative_hash(root: Path, path: Path) -> dict[str, str]:
    return {"path": path.relative_to(root).as_posix(), "sha256": sha256_file(path)}


def freeze_family(root: Path) -> tuple[Path, Path, Path]:
    root = root.resolve()
    config_path = root / "protocol/executable_configs/executable_direction_hurst_ablation_v2.json"
    code_manifest_path = (
        root / "protocol/code_manifests/executable_direction_hurst_ablation_v2.json"
    )
    card_path = root / "protocol/hypothesis_cards/executable_direction_hurst_ablation_v2.json"
    registry_path = (
        root / "artifacts/research/registry/executable_direction_hurst_ablation_v2.jsonl"
    )
    run_glob = root / "artifacts/research/runs"
    if registry_path.exists() or any(run_glob.glob(f"{FAMILY_V2}-*")):
        raise ProtocolViolation(
            "v2 execution evidence already exists; refusing retrospective freeze"
        )
    for path in (config_path, code_manifest_path, card_path):
        if path.exists():
            raise ProtocolViolation(f"v2 preregistration already exists: {path}")

    development_dir = root / "data/partitions/v2/development"
    development_path = development_dir / "prices.csv"
    development_manifest_path = development_dir / "manifest.json"
    availability_path = root / "protocol/data_availability_manifest.json"
    source = load_frozen_development_partition(
        development_path, development_manifest_path, min_rows=3_000
    )
    feature_config = FeatureConfig()
    feature_frame, feature_columns = build_feature_matrix(source, feature_config)
    feature_frame = add_robust_hurst_features(feature_frame, regime_window=252, min_periods=126)
    feature_columns += [name for name in feature_frame.columns if name.startswith("hurst_robust_")]
    features_by_arm = {arm: select_arm_features(feature_columns, arm) for arm in ARMS}

    config: dict[str, Any] = {
        "schema_version": "executable_research_config_v2",
        "hypothesis_family": FAMILY_V2,
        "minimum_development_rows": 3_000,
        "partition": {
            "role": "development_reused_previously_exposed",
            "physical_path": "data/partitions/v2/development/prices.csv",
            "historical_confirmation": "unavailable_already_exposed",
            "audit_access_during_selection": "forbidden",
        },
        "feature_config": {
            "return_lags": [1, 2, 3, 5, 10, 20],
            "windows": [5, 10, 20, 63, 126],
            "hurst_windows": [64, 128],
            "regime_window": 252,
            "min_feature_coverage": 0.70,
        },
        "features_by_arm": features_by_arm,
        "feature_availability": {
            "decision": "after completed D1 bar t",
            "earliest_entry": "next observed broker D1 open t+1",
            "external_sources": "none",
            "future_backfill": "forbidden",
        },
        "target_config": {
            "horizons": list(HORIZONS),
            "volatility_window": 63,
            "volatility_min_periods": 40,
            "threshold_k": 0.35,
            "threshold_floor_bps": 8.0,
            "transaction_cost_bps": 3.0,
            "slippage_bps": 2.0,
            "actionable_cost_buffer_bps": 2.0,
            "execution_lag_bars": 1,
            "cost_convention": "round_trip_total",
            "selection_target": "executable_direction_binary_open_t+1_to_open_t+1+h",
        },
        "robust_hurst": {"regime_window": 252, "min_periods": 126},
        "model": {
            "imputer": "median",
            "imputer_missing_indicator": True,
            "scaler": "standard",
            "estimator": "logistic_regression",
            "logistic_c": 0.2,
            "class_weight": "balanced",
            "max_iter": 2_000,
            "seed": 42,
        },
        "folds": {
            "outer_folds": 5,
            "inner_folds": 3,
            "outer_min_train_rows": 800,
            "inner_min_train_rows": 300,
            "outer_min_validation_rows": 120,
            "inner_min_validation_rows": 60,
            "pre_validation_gap_rows": 0,
            "purge_rule": "train_label_end_index < validation_start_row_id",
            "label_endpoint": "executable_label_end_index",
            "embargo_assessment": {
                "h1": "0 rows; expanding forward split has no post-validation reuse",
                "h5": "0 rows; horizon overlap removed by label-end purging",
                "h10": "0 rows; horizon overlap removed by label-end purging",
                "h20": "0 rows; horizon overlap removed by label-end purging",
            },
        },
        "inner_selection": {
            "calibration_fraction": 0.6666666667,
            "calibration_options": ["none", "sigmoid"],
            "no_trade_margins": [0.0, 0.05],
            "ranking": [
                "balanced_accuracy",
                "macro_f1",
                "cumulative_net_return_after_5bps",
                "prefer_smaller_margin",
                "prefer_no_calibration_on_tie",
            ],
            "schedule": "single_position_non_overlapping_v2",
        },
        "calibration": {"ece_bins": 10},
        "economics": {
            "baseline_cost_bps": 5.0,
            "stress_cost_bps": 10.0,
            "periods_per_year": 252.0,
            "position_policy": "one position; exit processed before same-open next entry",
            "benchmarks": ["cash", "always_long", "always_short", "momentum", "trend"],
            "benchmark_schedule": "same active trade rows as candidate",
        },
        "uncertainty": {
            "method": "joint_moving_block_bootstrap",
            "bootstrap_iterations": 2_000,
            "primary_block_length": 10,
            "sensitivity_block_lengths": [5, 20],
            "confidence": 0.95,
        },
        "multiplicity": {
            "declared_trials": DECLARED_BUDGET_V2,
            "pbo_partitions": 8,
            "pbo_universe": "all 12 trials on common outer-OOF rows",
            "dsr_universe": "annual Sharpe distribution of all 12 trials",
            "expansion_after_budget": "forbidden",
        },
        "promotion_gate": {
            "expected_outer_folds": 5,
            "minimum_median_balanced_accuracy": 0.60,
            "minimum_pooled_macro_f1": 0.55,
            "minimum_pooled_class_recall": 0.50,
            "minimum_folds_at_055": 4,
            "minimum_fold_balanced_accuracy": 0.50,
            "chance_level": 0.50,
            "maximum_brier_increase": 0.005,
            "maximum_ece_increase": 0.01,
            "minimum_non_overlapping_trades": 30,
            "maximum_pbo": 0.20,
            "minimum_dsr_probability": 0.95,
            "qa_flags_all_required": True,
            "economics_baseline_and_stress_must_be_positive": True,
            "all_same_schedule_benchmark_deltas_nonnegative": True,
        },
    }
    _write_exclusive(config_path, config)

    runtime_paths = [
        root / "pyproject.toml",
        root / "uv.lock",
        root / "scripts/run_research_batch_v2.py",
        root / "src/hge_gold/calibration.py",
        root / "src/hge_gold/config.py",
        root / "src/hge_gold/data.py",
        root / "src/hge_gold/execution_v2.py",
        root / "src/hge_gold/features.py",
        root / "src/hge_gold/partitions.py",
        root / "src/hge_gold/research_experiments.py",
        root / "src/hge_gold/research_experiments_v2.py",
        root / "src/hge_gold/research_protocol.py",
        root / "src/hge_gold/research_run.py",
        root / "src/hge_gold/research_validation.py",
        root / "src/hge_gold/statistics.py",
        root / "src/hge_gold/targets.py",
    ]
    code_manifest = {
        "schema_version": "preregistered_runtime_code_manifest_v2",
        "hypothesis_family": FAMILY_V2,
        "files": [_relative_hash(root, path) for path in runtime_paths],
    }
    _write_exclusive(code_manifest_path, code_manifest)

    registered_at = datetime.now(UTC).isoformat()
    card = {
        "schema_version": "hypothesis_card_v2",
        "hypothesis_family": FAMILY_V2,
        "status": "preregistered_before_execution",
        "registered_at_utc": registered_at,
        "scientific_question": (
            "Does either preregistered causal Hurst representation add stable outer-development "
            "value over the identical no-Hurst executable-direction model after corrected "
            "non-overlapping costs?"
        ),
        "arms": list(ARMS),
        "horizons": list(HORIZONS),
        "declared_family_budget": DECLARED_BUDGET_V2,
        "stop_rule": "Run exactly the 12 registered arm-horizon trials once; do not expand v2.",
        "v1_status": "immutable_exhausted_historical_family_not_rerun",
        "partition_policy": config["partition"],
        "audit_prohibition": (
            "No historical_audit_previously_revealed file or metric may be loaded during v2."
        ),
        "confirmation_policy": (
            "No fresh historical confirmation exists; only actual future observations can be "
            "future out of sample after candidate freeze."
        ),
        "executable_config_sha256": sha256_file(config_path),
        "code_manifest_sha256": sha256_file(code_manifest_path),
        "development_manifest_sha256": sha256_file(development_manifest_path),
        "development_data_sha256": sha256_file(development_path),
        "data_availability_manifest_sha256": sha256_file(availability_path),
    }
    _write_exclusive(card_path, card)
    return card_path, config_path, code_manifest_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    for path in freeze_family(args.project_root):
        print(path)


if __name__ == "__main__":
    main()
