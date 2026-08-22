"""Corrected, preregistered v2 development experiment family.

The exhausted v1 family and its artifacts remain immutable. This module uses a physically
separate development CSV, a corrected one-position execution ledger, all-12-trial PBO/DSR,
creation-exclusive staging, and complete output/runtime hashing.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, f1_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .calibration import fit_past_only_sigmoid
from .config import FeatureConfig, TargetConfig
from .execution_v2 import build_non_overlapping_return_ledger
from .features import build_feature_matrix
from .partitions import load_frozen_development_partition
from .research_experiments import add_robust_hurst_features, select_arm_features
from .research_protocol import AppendOnlyExperimentRegistry, ProtocolViolation, sha256_file
from .research_run import (
    ExclusiveFileLock,
    build_run_receipt,
    finalize_staged_run,
    validate_run_receipt,
    write_failed_run_receipt,
)
from .research_validation import (
    OuterFoldMetrics,
    PromotionCriteria,
    PromotionEvidence,
    build_nested_purged_walk_forward_folds,
    evaluate_promotion_gate,
    joint_moving_block_bootstrap,
    nested_split_manifest,
    nested_split_manifest_sha256,
)
from .statistics import deflated_sharpe_ratio, probability_of_backtest_overfitting
from .targets import build_horizon_dataset

FAMILY_V2 = "executable_direction_hurst_ablation_v2"
DECLARED_BUDGET_V2 = 12
ARMS = ("no_hurst", "current_dfa_hurst", "robust_hurst_regime")
HORIZONS = (1, 5, 10, 20)


@dataclass(frozen=True)
class TrialSpecV2:
    arm: str
    horizon: int

    @property
    def experiment_id(self) -> str:
        return f"{FAMILY_V2}-{self.arm}-h{self.horizon}"


def enumerate_preregistered_trials_v2() -> tuple[TrialSpecV2, ...]:
    return tuple(TrialSpecV2(arm=arm, horizon=horizon) for arm in ARMS for horizon in HORIZONS)


def _read_object(path: Path, label: str) -> dict[str, Any]:
    parsed = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(parsed, dict):
        raise ProtocolViolation(f"{label} must be a JSON object")
    return dict(parsed)


def verify_preregistered_bundle(
    *,
    project_root: Path,
    card_path: Path,
    config_path: Path,
    code_manifest_path: Path,
    development_manifest_path: Path,
    development_path: Path,
    data_availability_manifest_path: Path,
) -> dict[str, Any]:
    """Fail before metric access if any preregistered input changed."""

    card = _read_object(card_path, "Hypothesis card")
    if card.get("hypothesis_family") != FAMILY_V2:
        raise ProtocolViolation("Hypothesis card does not declare the v2 family")
    expected = {
        "executable config": (config_path, card.get("executable_config_sha256")),
        "code manifest": (code_manifest_path, card.get("code_manifest_sha256")),
        "development manifest": (
            development_manifest_path,
            card.get("development_manifest_sha256"),
        ),
        "development data": (development_path, card.get("development_data_sha256")),
        "data availability manifest": (
            data_availability_manifest_path,
            card.get("data_availability_manifest_sha256"),
        ),
    }
    for label, (path, digest) in expected.items():
        if not path.is_file() or sha256_file(path) != digest:
            raise ProtocolViolation(f"Preregistered {label} hash mismatch")
    code_manifest = _read_object(code_manifest_path, "Code manifest")
    root = project_root.resolve()
    for entry in code_manifest.get("files", []):
        source = root / str(entry["path"])
        if not source.is_file() or sha256_file(source) != entry.get("sha256"):
            raise ProtocolViolation(f"Preregistered runtime source changed: {entry['path']}")
    config = _read_object(config_path, "Executable config")
    if config.get("hypothesis_family") != FAMILY_V2:
        raise ProtocolViolation("Executable config family mismatch")
    return {"card": card, "config": config, "code_manifest": code_manifest}


def align_trial_return_ledgers(
    ledgers: dict[str, pd.DataFrame],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Align all 12 registered trial paths on their common OOF decision rows."""

    trial_ids = [trial.experiment_id for trial in enumerate_preregistered_trials_v2()]
    if set(ledgers) != set(trial_ids):
        raise ProtocolViolation("Return ledger must contain exactly all 12 registered trials")
    common: set[int] | None = None
    indexed: dict[str, pd.DataFrame] = {}
    for trial_id in trial_ids:
        ledger = ledgers[trial_id]
        required = {"row_id", "candidate_net_log_return"}
        if required.difference(ledger.columns) or ledger["row_id"].duplicated().any():
            raise ProtocolViolation(f"Invalid return ledger for {trial_id}")
        indexed[trial_id] = ledger.set_index("row_id").sort_index()
        rows = set(ledger["row_id"].astype(int))
        common = rows if common is None else common.intersection(rows)
    if common is None or len(common) < 20:
        raise ProtocolViolation("Too few common OOF observations for all-trial PBO/DSR")
    row_ids = np.asarray(sorted(common), dtype=int)
    matrix = np.column_stack(
        [
            indexed[trial_id].loc[row_ids, "candidate_net_log_return"].to_numpy(dtype=float)
            for trial_id in trial_ids
        ]
    )
    if not np.isfinite(matrix).all():
        raise ProtocolViolation("Aligned return matrix contains non-finite values")
    return row_ids, matrix, trial_ids


def _feature_config(payload: dict[str, Any]) -> FeatureConfig:
    return FeatureConfig(
        return_lags=tuple(int(value) for value in payload["return_lags"]),
        windows=tuple(int(value) for value in payload["windows"]),
        hurst_windows=tuple(int(value) for value in payload["hurst_windows"]),
        regime_window=int(payload["regime_window"]),
        min_feature_coverage=float(payload["min_feature_coverage"]),
    )


def _target_config(payload: dict[str, Any]) -> TargetConfig:
    return TargetConfig(
        horizons=tuple(int(value) for value in payload["horizons"]),
        volatility_window=int(payload["volatility_window"]),
        volatility_min_periods=int(payload["volatility_min_periods"]),
        threshold_k=float(payload["threshold_k"]),
        threshold_floor_bps=float(payload["threshold_floor_bps"]),
        transaction_cost_bps=float(payload["transaction_cost_bps"]),
        slippage_bps=float(payload["slippage_bps"]),
        actionable_cost_buffer_bps=float(payload["actionable_cost_buffer_bps"]),
        execution_lag_bars=int(payload["execution_lag_bars"]),
        cost_convention=str(payload["cost_convention"]),
    )


def _model(config: dict[str, Any]) -> Pipeline:
    model = config["model"]
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy=str(model["imputer"]), add_indicator=True)),
            ("scaler", StandardScaler()),
            (
                "logistic",
                LogisticRegression(
                    C=float(model["logistic_c"]),
                    class_weight=str(model["class_weight"]),
                    max_iter=int(model["max_iter"]),
                    random_state=int(model["seed"]),
                ),
            ),
        ]
    )


def _promotion_criteria(config: dict[str, Any]) -> PromotionCriteria:
    payload = config.get("promotion_gate")
    if payload is None:
        return PromotionCriteria()
    return PromotionCriteria(
        expected_outer_folds=int(payload["expected_outer_folds"]),
        minimum_median_balanced_accuracy=float(payload["minimum_median_balanced_accuracy"]),
        minimum_pooled_macro_f1=float(payload["minimum_pooled_macro_f1"]),
        minimum_pooled_class_recall=float(payload["minimum_pooled_class_recall"]),
        minimum_folds_at_055=int(payload["minimum_folds_at_055"]),
        minimum_fold_balanced_accuracy=float(payload["minimum_fold_balanced_accuracy"]),
        chance_level=float(payload["chance_level"]),
        maximum_brier_increase=float(payload["maximum_brier_increase"]),
        maximum_ece_increase=float(payload["maximum_ece_increase"]),
        minimum_non_overlapping_trades=int(payload["minimum_non_overlapping_trades"]),
        maximum_pbo=float(payload["maximum_pbo"]),
        minimum_dsr_probability=float(payload["minimum_dsr_probability"]),
    )


def _metrics(y: np.ndarray, probability: np.ndarray, fold_id: str) -> OuterFoldMetrics:
    prediction = (probability >= 0.5).astype(int)
    return OuterFoldMetrics(
        fold_id=fold_id,
        balanced_accuracy=float(balanced_accuracy_score(y, prediction)),
        macro_f1=float(f1_score(y, prediction, average="macro", zero_division=0)),
        recall_down=float(recall_score(y, prediction, pos_label=0, zero_division=0)),
        recall_up=float(recall_score(y, prediction, pos_label=1, zero_division=0)),
        n_samples=len(y),
    )


def _ece(y: np.ndarray, probability: np.ndarray, bins: int) -> float:
    observed, predicted = calibration_curve(y, probability, n_bins=bins, strategy="uniform")
    return float(np.mean(np.abs(observed - predicted)))


def _calibration_coefficients(y: np.ndarray, probability: np.ndarray) -> tuple[float, float]:
    clipped = np.clip(probability, 1e-6, 1.0 - 1e-6)
    logit = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
    fitted = LogisticRegression(C=1_000_000.0, max_iter=2_000).fit(logit, y)
    return float(fitted.intercept_[0]), float(fitted.coef_[0, 0])


def _prediction_columns() -> list[str]:
    return [
        "row_id",
        "date",
        "entry_row_index",
        "exit_row_index",
        "entry_timestamp",
        "exit_timestamp",
        "executable_forward_log_return",
        "executable_direction_binary",
        "return_lag_1",
        "momentum_20",
    ]


def _inner_choice(
    dataset: pd.DataFrame,
    features: list[str],
    nested: Any,
    config: dict[str, Any],
) -> tuple[str, float, list[dict[str, Any]]]:
    rows: list[pd.DataFrame] = []
    for fold in nested.inner:
        train = dataset.iloc[fold.train_indices]
        validation = dataset.iloc[fold.validation_indices]
        fitted = _model(config).fit(
            train[features], train["executable_direction_binary"].astype(int)
        )
        rows.append(
            validation[_prediction_columns()]
            .assign(
                y=validation["executable_direction_binary"].astype(int).to_numpy(),
                raw_probability=fitted.predict_proba(validation[features])[:, 1],
                inner_fold=fold.fold_id,
            )
            .copy()
        )
    pooled = pd.concat(rows, ignore_index=True).sort_values("row_id")
    fraction = float(config["inner_selection"]["calibration_fraction"])
    split = max(20, int(len(pooled) * fraction))
    calibration = pooled.iloc[:split]
    evaluation = pooled.iloc[split:].copy()
    score_records: list[dict[str, Any]] = []
    choices: list[tuple[float, float, float, str, float]] = []
    for calibration_name in config["inner_selection"]["calibration_options"]:
        probability = evaluation["raw_probability"].to_numpy(dtype=float)
        if calibration_name == "sigmoid":
            calibrator = fit_past_only_sigmoid(
                calibration["raw_probability"].to_numpy(dtype=float),
                calibration["y"].to_numpy(dtype=int),
                calibration_row_ids=calibration["row_id"].to_numpy(dtype=int),
                prediction_row_ids=evaluation["row_id"].to_numpy(dtype=int),
                seed=int(config["model"]["seed"]),
            )
            probability = calibrator.predict(probability)
        metric = _metrics(evaluation["y"].to_numpy(dtype=int), probability, "inner_selection")
        for raw_margin in config["inner_selection"]["no_trade_margins"]:
            margin = float(raw_margin)
            candidate = evaluation.assign(
                probability_up=probability,
                selected_margin=margin,
            )
            ledger = build_non_overlapping_return_ledger(
                candidate,
                cost_bps=float(config["economics"]["baseline_cost_bps"]),
            )
            cumulative = float(ledger["candidate_net_log_return"].sum())
            record = {
                "calibration": calibration_name,
                "margin": margin,
                "balanced_accuracy": metric.balanced_accuracy,
                "macro_f1": metric.macro_f1,
                "cumulative_net_log_return": cumulative,
                "non_overlapping_trades": int(ledger["active"].sum()),
                "calibration_start_row": int(calibration["row_id"].min()),
                "calibration_end_row": int(calibration["row_id"].max()),
                "evaluation_start_row": int(evaluation["row_id"].min()),
                "evaluation_end_row": int(evaluation["row_id"].max()),
            }
            score_records.append(record)
            choices.append(
                (
                    metric.balanced_accuracy,
                    metric.macro_f1,
                    cumulative,
                    str(calibration_name),
                    margin,
                )
            )
    winner = max(
        choices,
        key=lambda item: (item[0], item[1], item[2], -item[4], item[3] == "none"),
    )
    return winner[3], winner[4], score_records


def run_trial_v2(
    dataset: pd.DataFrame,
    features: list[str],
    spec: TrialSpecV2,
    config: dict[str, Any],
) -> dict[str, Any]:
    eligible = dataset.loc[dataset["is_modeling_eligible"]].reset_index(drop=True).copy()
    eligible["label_end_index"] = eligible["executable_label_end_index"].astype(int)
    fold_config = config["folds"]
    folds = build_nested_purged_walk_forward_folds(
        eligible,
        outer_min_train_rows=int(fold_config["outer_min_train_rows"]),
        inner_min_train_rows=int(fold_config["inner_min_train_rows"]),
        outer_min_validation_rows=int(fold_config["outer_min_validation_rows"]),
        inner_min_validation_rows=int(fold_config["inner_min_validation_rows"]),
        pre_validation_gap_rows=int(fold_config["pre_validation_gap_rows"]),
        outer_folds=int(fold_config["outer_folds"]),
        inner_folds=int(fold_config["inner_folds"]),
    )
    split_manifest = nested_split_manifest(
        eligible,
        folds,
        horizon=spec.horizon,
        pre_validation_gap_rows=int(fold_config["pre_validation_gap_rows"]),
    )
    predictions: list[pd.DataFrame] = []
    fold_metrics: list[OuterFoldMetrics] = []
    selections: list[dict[str, Any]] = []
    inner_scores: list[dict[str, Any]] = []
    for nested in folds:
        calibration_name, margin, scores = _inner_choice(eligible, features, nested, config)
        for score in scores:
            score["outer_fold"] = nested.outer.fold_id
        inner_scores.extend(scores)
        train = eligible.iloc[nested.outer.train_indices]
        validation = eligible.iloc[nested.outer.validation_indices]
        fitted = _model(config).fit(
            train[features], train["executable_direction_binary"].astype(int)
        )
        probability = fitted.predict_proba(validation[features])[:, 1]
        if calibration_name == "sigmoid":
            inner_oof: list[pd.DataFrame] = []
            for fold in nested.inner:
                inner_train = eligible.iloc[fold.train_indices]
                inner_validation = eligible.iloc[fold.validation_indices]
                inner_model = _model(config).fit(
                    inner_train[features],
                    inner_train["executable_direction_binary"].astype(int),
                )
                inner_oof.append(
                    pd.DataFrame(
                        {
                            "row_id": inner_validation["row_id"].to_numpy(dtype=int),
                            "y": inner_validation["executable_direction_binary"].to_numpy(
                                dtype=int
                            ),
                            "p": inner_model.predict_proba(inner_validation[features])[:, 1],
                        }
                    )
                )
            calibration_frame = pd.concat(inner_oof, ignore_index=True).sort_values("row_id")
            calibrator = fit_past_only_sigmoid(
                calibration_frame["p"].to_numpy(dtype=float),
                calibration_frame["y"].to_numpy(dtype=int),
                calibration_row_ids=calibration_frame["row_id"].to_numpy(dtype=int),
                prediction_row_ids=validation["row_id"].to_numpy(dtype=int),
                seed=int(config["model"]["seed"]),
            )
            probability = calibrator.predict(probability)
        truth = validation["executable_direction_binary"].to_numpy(dtype=int)
        fold_metrics.append(_metrics(truth, probability, nested.outer.fold_id))
        predictions.append(
            validation[_prediction_columns()]
            .assign(
                probability_up=probability,
                baseline_probability_up=float(
                    train["executable_direction_binary"].astype(int).mean()
                ),
                outer_fold=nested.outer.fold_id,
                selected_calibration=calibration_name,
                selected_margin=margin,
            )
            .copy()
        )
        selections.append(
            {
                "outer_fold": nested.outer.fold_id,
                "calibration": calibration_name,
                "margin": margin,
            }
        )
    return {
        "spec": spec,
        "features": features,
        "split_manifest": split_manifest,
        "split_hash": nested_split_manifest_sha256(split_manifest),
        "predictions": pd.concat(predictions, ignore_index=True)
        .sort_values("row_id")
        .reset_index(drop=True),
        "fold_metrics": tuple(fold_metrics),
        "selections": selections,
        "inner_scores": inner_scores,
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.integer | np.floating):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_exclusive_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(_jsonable(payload), handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise ProtocolViolation(f"Refusing to overwrite finalization receipt: {path}") from exc


def _runtime_sources(project_root: Path, bundle: dict[str, Any], extra: list[Path]) -> list[Path]:
    sources = [project_root / str(entry["path"]) for entry in bundle["code_manifest"]["files"]]
    return sources + extra


def run_preregistered_batch_v2(project_root: Path) -> Path:
    """Execute the one permitted corrected v2 development batch."""

    root = project_root.resolve()
    card_path = root / "protocol/hypothesis_cards/executable_direction_hurst_ablation_v2.json"
    config_path = root / "protocol/executable_configs/executable_direction_hurst_ablation_v2.json"
    code_manifest_path = (
        root / "protocol/code_manifests/executable_direction_hurst_ablation_v2.json"
    )
    development_dir = root / "data/partitions/v2/development"
    development_path = development_dir / "prices.csv"
    development_manifest_path = development_dir / "manifest.json"
    data_availability_manifest_path = root / "protocol/data_availability_manifest.json"
    registry = AppendOnlyExperimentRegistry(
        root / "artifacts/research/registry/executable_direction_hurst_ablation_v2.jsonl"
    )
    lock_path = root / "artifacts/research/locks/executable_direction_hurst_ablation_v2.lock"
    bundle = verify_preregistered_bundle(
        project_root=root,
        card_path=card_path,
        config_path=config_path,
        code_manifest_path=code_manifest_path,
        development_manifest_path=development_manifest_path,
        development_path=development_path,
        data_availability_manifest_path=data_availability_manifest_path,
    )
    config = bundle["config"]
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    batch_id = f"{FAMILY_V2}-{timestamp}"
    staging_dir = root / "artifacts/research/.staging" / batch_id
    final_dir = root / "artifacts/research/runs" / batch_id
    with ExclusiveFileLock(lock_path):
        if registry.read_and_validate():
            raise ProtocolViolation("The corrected v2 family has already been executed")
        staging_dir.mkdir(parents=True, exist_ok=False)
        active_path = staging_dir
        try:
            source = load_frozen_development_partition(
                development_path,
                development_manifest_path,
                min_rows=int(config["minimum_development_rows"]),
            )
            feature_config = _feature_config(config["feature_config"])
            target_config = _target_config(config["target_config"])
            features, feature_columns = build_feature_matrix(source, feature_config)
            robust = config["robust_hurst"]
            features = add_robust_hurst_features(
                features,
                regime_window=int(robust["regime_window"]),
                min_periods=int(robust["min_periods"]),
            )
            feature_columns += [
                name for name in features.columns if name.startswith("hurst_robust_")
            ]
            actual_features = {arm: select_arm_features(feature_columns, arm) for arm in ARMS}
            if actual_features != config["features_by_arm"]:
                raise ProtocolViolation("Actual feature membership differs from preregistration")
            datasets = {
                horizon: build_horizon_dataset(
                    source,
                    features,
                    feature_columns,
                    horizon,
                    target_config,
                    feature_config,
                )
                for horizon in HORIZONS
            }
            results = [
                run_trial_v2(
                    datasets[trial.horizon],
                    actual_features[trial.arm],
                    trial,
                    config,
                )
                for trial in enumerate_preregistered_trials_v2()
            ]
            baseline_cost = float(config["economics"]["baseline_cost_bps"])
            stress_cost = float(config["economics"]["stress_cost_bps"])
            ledgers = {
                item["spec"].experiment_id: build_non_overlapping_return_ledger(
                    item["predictions"], cost_bps=baseline_cost
                )
                for item in results
            }
            stress_ledgers = {
                item["spec"].experiment_id: build_non_overlapping_return_ledger(
                    item["predictions"], cost_bps=stress_cost
                )
                for item in results
            }
            common_rows, return_matrix, trial_ids = align_trial_return_ledgers(ledgers)
            multiplicity = config["multiplicity"]
            pbo_result = probability_of_backtest_overfitting(
                return_matrix,
                n_partitions=int(multiplicity["pbo_partitions"]),
            )
            periods_per_year = float(config["economics"]["periods_per_year"])
            standard_deviation = return_matrix.std(axis=0, ddof=1)
            annual_sharpes = np.divide(
                return_matrix.mean(axis=0),
                standard_deviation,
                out=np.zeros(return_matrix.shape[1]),
                where=standard_deviation > 0,
            ) * np.sqrt(periods_per_year)
            dsr_by_trial: dict[str, float | None] = {}
            for column, trial_id in enumerate(trial_ids):
                try:
                    dsr_by_trial[trial_id] = deflated_sharpe_ratio(
                        return_matrix[:, column],
                        declared_trials=DECLARED_BUDGET_V2,
                        periods_per_year=periods_per_year,
                        trial_sharpe_mean=float(annual_sharpes.mean()),
                        trial_sharpe_std=float(annual_sharpes.std(ddof=1)),
                    ).probability
                except ValueError:
                    dsr_by_trial[trial_id] = None

            registry_payloads: list[dict[str, Any]] = []
            inventory: list[dict[str, Any]] = []
            uncertainty = config["uncertainty"]
            for item in results:
                spec: TrialSpecV2 = item["spec"]
                trial_id = spec.experiment_id
                predictions: pd.DataFrame = item["predictions"]
                ledger = ledgers[trial_id]
                stress = stress_ledgers[trial_id]
                truth = predictions["executable_direction_binary"].to_numpy(dtype=int)
                probability = predictions["probability_up"].to_numpy(dtype=float)
                pooled = _metrics(truth, probability, "pooled")
                bootstrap = joint_moving_block_bootstrap(
                    truth,
                    probability,
                    ledger["candidate_net_log_return"].to_numpy(dtype=float),
                    ledger["trend_net_log_return"].to_numpy(dtype=float),
                    threshold=0.5,
                    primary_block_length=int(uncertainty["primary_block_length"]),
                    sensitivity_block_lengths=tuple(
                        int(value) for value in uncertainty["sensitivity_block_lengths"]
                    ),
                    iterations=int(uncertainty["bootstrap_iterations"]),
                    confidence=float(uncertainty["confidence"]),
                    seed=int(config["model"]["seed"]),
                )
                primary = bootstrap.estimate_for(int(uncertainty["primary_block_length"]))
                intercept, slope = _calibration_coefficients(truth, probability)
                baseline_probability = predictions["baseline_probability_up"].to_numpy(dtype=float)
                bins = int(config["calibration"]["ece_bins"])
                economic_columns = {
                    "cash": "cash_net_log_return",
                    "always_long": "always_long_net_log_return",
                    "always_short": "always_short_net_log_return",
                    "momentum": "momentum_net_log_return",
                    "trend": "trend_net_log_return",
                }
                candidate_net = float(ledger["candidate_net_log_return"].sum())
                evidence = PromotionEvidence(
                    outer_folds=item["fold_metrics"],
                    pooled_balanced_accuracy=pooled.balanced_accuracy,
                    pooled_macro_f1=pooled.macro_f1,
                    pooled_recall_down=pooled.recall_down,
                    pooled_recall_up=pooled.recall_up,
                    balanced_accuracy_ci_low=primary.intervals["balanced_accuracy"].low,
                    balanced_accuracy_ci_high=primary.intervals["balanced_accuracy"].high,
                    balanced_accuracy_sensitivity_lows={
                        str(length): bootstrap.estimate_for(int(length))
                        .intervals["balanced_accuracy"]
                        .low
                        for length in uncertainty["sensitivity_block_lengths"]
                    },
                    candidate_brier=float(brier_score_loss(truth, probability)),
                    baseline_brier=float(brier_score_loss(truth, baseline_probability)),
                    candidate_ece=_ece(truth, probability, bins),
                    baseline_ece=_ece(truth, baseline_probability, bins),
                    calibration_intercept=intercept,
                    calibration_slope=slope,
                    classification_baseline_deltas={
                        "always_up": pooled.balanced_accuracy - 0.5,
                        "always_down": pooled.balanced_accuracy - 0.5,
                    },
                    paired_net_return_ci_low=primary.intervals["paired_mean_net_return"].low,
                    cumulative_net_return_baseline_cost=candidate_net,
                    cumulative_net_return_stress_cost=float(
                        stress["candidate_net_log_return"].sum()
                    ),
                    economic_benchmark_deltas={
                        name: candidate_net - float(ledger[column].sum())
                        for name, column in economic_columns.items()
                    },
                    n_non_overlapping_trades=int(ledger["active"].sum()),
                    trial_return_registry_complete=True,
                    pbo=float(pbo_result.pbo),
                    dsr_probability=dsr_by_trial[trial_id],
                    qa_flags={
                        "leakage_free": True,
                        "audit_isolated": True,
                        "provenance_complete": False,
                        "reproducible": True,
                        "manifest_verified": True,
                    },
                )
                decision = evaluate_promotion_gate(evidence, _promotion_criteria(config))
                experiment_dir = staging_dir / "experiments" / trial_id
                prediction_path = staging_dir / "predictions" / f"{trial_id}.csv"
                ledger_path = staging_dir / "return_ledgers" / f"{trial_id}.csv"
                inner_path = staging_dir / "inner_selection" / f"{trial_id}.csv"
                model_path = staging_dir / "model_specs" / f"{trial_id}.json"
                for directory in (
                    experiment_dir,
                    prediction_path.parent,
                    ledger_path.parent,
                    inner_path.parent,
                    model_path.parent,
                ):
                    directory.mkdir(parents=True, exist_ok=True)
                predictions.to_csv(prediction_path, index=False, lineterminator="\n")
                ledger.to_csv(ledger_path, index=False, lineterminator="\n")
                pd.DataFrame(item["inner_scores"]).to_csv(
                    inner_path, index=False, lineterminator="\n"
                )
                _write_json(experiment_dir / "split_manifest.json", item["split_manifest"])
                _write_json(
                    experiment_dir / "outer_fold_metrics.json",
                    {
                        "folds": [asdict(metric) for metric in item["fold_metrics"]],
                        "pooled": asdict(pooled),
                    },
                )
                _write_json(
                    experiment_dir / "calibration_uncertainty.json",
                    {
                        "candidate_brier": evidence.candidate_brier,
                        "baseline_brier": evidence.baseline_brier,
                        "candidate_ece": evidence.candidate_ece,
                        "baseline_ece": evidence.baseline_ece,
                        "intercept": intercept,
                        "slope": slope,
                        "bootstrap_iterations": int(uncertainty["bootstrap_iterations"]),
                        "balanced_accuracy_ci": [
                            evidence.balanced_accuracy_ci_low,
                            evidence.balanced_accuracy_ci_high,
                        ],
                    },
                )
                _write_json(
                    experiment_dir / "economic_benchmarks.json",
                    {
                        "schedule": "single_position_non_overlapping_v2",
                        "baseline_cost_bps": baseline_cost,
                        "stress_cost_bps": stress_cost,
                        "candidate_net_log_return": candidate_net,
                        "stress_net_log_return": evidence.cumulative_net_return_stress_cost,
                        "benchmark_deltas": evidence.economic_benchmark_deltas,
                        "non_overlapping_trades": evidence.n_non_overlapping_trades,
                        "pbo": evidence.pbo,
                        "pbo_trial_count": len(trial_ids),
                        "pbo_common_observations": len(common_rows),
                        "dsr_probability": evidence.dsr_probability,
                        "declared_multiplicity_trials": DECLARED_BUDGET_V2,
                    },
                )
                _write_json(experiment_dir / "promotion_decision.json", decision.to_dict())
                _write_json(
                    model_path,
                    {
                        "status": "outer_fold_evaluation_specification_only_no_promoted_model",
                        "trial": asdict(spec),
                        "features": item["features"],
                        "outer_selections": item["selections"],
                    },
                )
                registry_payloads.append(
                    {
                        "experiment_id": trial_id,
                        "parent_hypothesis": (
                            "Corrected non-overlapping executable economics for the frozen "
                            "Hurst ablation"
                        ),
                        "hypothesis_family": FAMILY_V2,
                        "timestamp_utc": datetime.now(UTC).isoformat(),
                        "code_sha256": str(bundle["card"]["code_manifest_sha256"]),
                        "config_sha256": sha256_file(config_path),
                        "data_sha256": sha256_file(development_path),
                        "dependency_sha256": sha256_file(root / "uv.lock"),
                        "source_availability_convention": (
                            "Physically isolated development file ending 2023-06-30; "
                            "decision after close t; entry open t+1"
                        ),
                        "feature_list": item["features"],
                        "target": "executable_direction_binary_open_t+1_to_open_t+1+h",
                        "horizon": spec.horizon,
                        "model": "median_imputer_standard_scaler_balanced_logistic_regression",
                        "hyperparameters": {
                            **config["model"],
                            "arm": spec.arm,
                            "horizon": spec.horizon,
                            "inner_calibration": config["inner_selection"]["calibration_options"],
                            "inner_no_trade_margin": config["inner_selection"]["no_trade_margins"],
                        },
                        "fold_definitions": [
                            {
                                "split_manifest_sha256": item["split_hash"],
                                **config["folds"],
                            }
                        ],
                        "train_metrics": {
                            "selection": "inner_purged_walk_forward_only",
                            "outer_selections": item["selections"],
                            "inner_selection_scores": item["inner_scores"],
                            "inner_selection_artifact": inner_path.relative_to(
                                staging_dir
                            ).as_posix(),
                            "inner_selection_sha256": sha256_file(inner_path),
                        },
                        "validation_metrics": {
                            "outer_folds": [asdict(metric) for metric in item["fold_metrics"]],
                            "pooled": asdict(pooled),
                        },
                        "calibration_metrics": {
                            "brier": evidence.candidate_brier,
                            "baseline_brier": evidence.baseline_brier,
                            "ece": evidence.candidate_ece,
                            "baseline_ece": evidence.baseline_ece,
                            "intercept": intercept,
                            "slope": slope,
                        },
                        "economic_metrics": {
                            "schedule": "single_position_non_overlapping_v2",
                            "net_log_return": candidate_net,
                            "stress_net_log_return": evidence.cumulative_net_return_stress_cost,
                            "non_overlapping_trades": evidence.n_non_overlapping_trades,
                            "pbo": evidence.pbo,
                            "pbo_trial_count": len(trial_ids),
                            "dsr_probability": evidence.dsr_probability,
                            "return_ledger_artifact": ledger_path.relative_to(
                                staging_dir
                            ).as_posix(),
                            "return_ledger_sha256": sha256_file(ledger_path),
                        },
                        "decision": "promote" if decision.status == "PASS" else "reject",
                        "decision_reason": list(decision.reasons),
                        "declared_family_budget": DECLARED_BUDGET_V2,
                    }
                )
                inventory.append(
                    {
                        "experiment_id": trial_id,
                        "arm": spec.arm,
                        "horizon": spec.horizon,
                        "decision": "promote" if decision.status == "PASS" else "reject",
                        "pooled_balanced_accuracy": pooled.balanced_accuracy,
                        "macro_f1": pooled.macro_f1,
                        "net_log_return_5bps": candidate_net,
                        "net_log_return_10bps": evidence.cumulative_net_return_stress_cost,
                        "non_overlapping_trades": evidence.n_non_overlapping_trades,
                        "pbo": evidence.pbo,
                        "dsr_probability": evidence.dsr_probability,
                        "failed_gates": list(decision.reasons),
                    }
                )
            _write_json(
                staging_dir / "experiment_inventory.json",
                {
                    "batch_id": batch_id,
                    "hypothesis_family": FAMILY_V2,
                    "experiments": inventory,
                },
            )
            _write_json(
                staging_dir / "multiplicity_report.json",
                {
                    "family": FAMILY_V2,
                    "trial_ids": trial_ids,
                    "trial_count": len(trial_ids),
                    "common_row_ids": common_rows,
                    "common_observations": len(common_rows),
                    "pbo": asdict(pbo_result),
                    "annualized_trial_sharpes": dict(
                        zip(trial_ids, annual_sharpes.tolist(), strict=True)
                    ),
                    "dsr_probability_by_trial": dsr_by_trial,
                },
            )
            receipt = build_run_receipt(
                staging_dir,
                project_root=root,
                runtime_sources=_runtime_sources(
                    root,
                    bundle,
                    [
                        card_path,
                        config_path,
                        code_manifest_path,
                        development_manifest_path,
                        data_availability_manifest_path,
                        root / "uv.lock",
                    ],
                ),
                metadata={
                    "batch_id": batch_id,
                    "hypothesis_family": FAMILY_V2,
                    "partition_role": "development_reused_previously_exposed",
                    "historical_audit_accessed": False,
                    "historical_confirmation_available": False,
                    "registry_path": registry.path.relative_to(root).as_posix(),
                },
            )
            _write_json(staging_dir / "run_receipt.json", receipt)
            validate_run_receipt(staging_dir, project_root=root)
            active_path = finalize_staged_run(staging_dir, final_dir)
            records = [registry.append(payload) for payload in registry_payloads]
            finalization_path = root / "artifacts/research/finalizations" / f"{batch_id}.json"
            _write_exclusive_json(
                finalization_path,
                {
                    "schema_version": "research_batch_finalization_v2",
                    "state": "finalized",
                    "batch_id": batch_id,
                    "finalized_at_utc": datetime.now(UTC).isoformat(),
                    "run_receipt_sha256": sha256_file(final_dir / "run_receipt.json"),
                    "registry_path": registry.path.relative_to(root).as_posix(),
                    "registry_head": asdict(registry.head()),
                    "record_hashes": [record["record_hash"] for record in records],
                },
            )
            return final_dir
        except Exception as exc:
            write_failed_run_receipt(
                root / "artifacts/research/failed_runs",
                batch_id=batch_id,
                error_type=type(exc).__name__,
                error_message=str(exc),
                staging_path=active_path,
            )
            raise
