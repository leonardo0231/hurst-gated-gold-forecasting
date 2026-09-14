"""Isolated, preregistered v3 Hurst ablation with audited secondary calibration.

The v2 family remains immutable historical evidence.  This module defines a new family
whose only modeling change is the preregistered calibration-boundary repair, while also
persisting exact alignment, QA, estimator, and inference evidence for independent review.
"""

from __future__ import annotations

import json
import os
import subprocess
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

from .calibration import assess_sigmoid_eligibility, fit_past_only_sigmoid
from .config import FeatureConfig, TargetConfig
from .execution_v2 import build_non_overlapping_return_ledger
from .features import assert_causal_features, build_feature_matrix
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
    assert_exact_paired_alignment,
    build_nested_purged_walk_forward_folds,
    evaluate_promotion_gate,
    joint_moving_block_bootstrap,
    nested_split_manifest,
    nested_split_manifest_sha256,
    purge_calibration_against_prediction_window,
)
from .statistics import (
    deflated_sharpe_ratio,
    holm_adjust,
    paired_block_bootstrap_metric_difference,
    paired_block_sign_flip_test,
    probability_of_backtest_overfitting,
)
from .targets import build_horizon_dataset

FAMILY_V3 = "executable_direction_hurst_ablation_v3"
FAMILY_V2 = "executable_direction_hurst_ablation_v2"
DECLARED_BUDGET_V3 = 12
ARMS = ("no_hurst", "current_dfa_hurst", "robust_hurst_regime")
HORIZONS = (1, 5, 10, 20)
V2_RUN_ID = "executable_direction_hurst_ablation_v2-20260822T165905Z"


@dataclass(frozen=True)
class TrialSpecV3:
    arm: str
    horizon: int

    @property
    def experiment_id(self) -> str:
        return f"{FAMILY_V3}-{self.arm}-h{self.horizon}"


def enumerate_preregistered_trials_v3() -> tuple[TrialSpecV3, ...]:
    """Return exactly the frozen 3-arm by 4-horizon v3 search space."""

    return tuple(TrialSpecV3(arm=arm, horizon=horizon) for arm in ARMS for horizon in HORIZONS)


def _read_object(path: Path, label: str) -> dict[str, Any]:
    parsed = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(parsed, dict):
        raise ProtocolViolation(f"{label} must be a JSON object")
    return dict(parsed)


def verify_preregistered_bundle_v3(
    *,
    project_root: Path,
    card_path: Path,
    config_path: Path,
    code_manifest_path: Path,
    development_manifest_path: Path,
    development_path: Path,
    data_availability_manifest_path: Path,
) -> dict[str, Any]:
    """Verify all v3 hashes and runtime sources before any metric access."""

    card = _read_object(card_path, "v3 hypothesis card")
    if card.get("hypothesis_family") != FAMILY_V3:
        raise ProtocolViolation("Hypothesis card does not declare the v3 family")
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
    code_manifest = _read_object(code_manifest_path, "v3 code manifest")
    root = project_root.resolve()
    for entry in code_manifest.get("files", []):
        source = root / str(entry["path"])
        if not source.is_file() or sha256_file(source) != entry.get("sha256"):
            raise ProtocolViolation(f"Preregistered v3 runtime source changed: {entry['path']}")
    config = _read_object(config_path, "v3 executable config")
    if config.get("hypothesis_family") != FAMILY_V3:
        raise ProtocolViolation("Executable config family mismatch")
    if int(config.get("declared_family_budget", -1)) != DECLARED_BUDGET_V3:
        raise ProtocolViolation("v3 declared family budget must be exactly 12")
    return {"card": card, "config": config, "code_manifest": code_manifest}


def align_trial_return_ledgers_v3(
    ledgers: dict[str, pd.DataFrame],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Align all 12 v3 return ledgers on exactly their common row IDs."""

    trial_ids = [trial.experiment_id for trial in enumerate_preregistered_trials_v3()]
    if set(ledgers) != set(trial_ids):
        raise ProtocolViolation("v3 return ledger must contain exactly all 12 registered trials")
    common: set[int] | None = None
    indexed: dict[str, pd.DataFrame] = {}
    for trial_id in trial_ids:
        ledger = ledgers[trial_id]
        required = {"row_id", "candidate_net_log_return"}
        if required.difference(ledger.columns) or ledger["row_id"].duplicated().any():
            raise ProtocolViolation(f"Invalid v3 return ledger for {trial_id}")
        indexed[trial_id] = ledger.set_index("row_id").sort_index()
        rows = set(ledger["row_id"].astype(int))
        common = rows if common is None else common.intersection(rows)
    if common is None or len(common) < 20:
        raise ProtocolViolation("Too few common v3 OOF observations for diagnostics")
    row_ids = np.asarray(sorted(common), dtype=int)
    matrix = np.column_stack(
        [
            indexed[trial_id].loc[row_ids, "candidate_net_log_return"].to_numpy(dtype=float)
            for trial_id in trial_ids
        ]
    )
    if not np.isfinite(matrix).all():
        raise ProtocolViolation("Aligned v3 return matrix contains non-finite values")
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
    estimator = model["estimator_parameters"]
    return Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy=str(model["imputer"]),
                    add_indicator=bool(model["imputer_missing_indicator"]),
                    keep_empty_features=bool(model["imputer_keep_empty_features"]),
                ),
            ),
            (
                "scaler",
                StandardScaler(
                    with_mean=bool(model["scaler_with_mean"]),
                    with_std=bool(model["scaler_with_std"]),
                ),
            ),
            (
                "logistic",
                LogisticRegression(
                    C=float(estimator["C"]),
                    penalty=str(estimator["penalty"]),
                    solver=str(estimator["solver"]),
                    fit_intercept=bool(estimator["fit_intercept"]),
                    class_weight=estimator["class_weight"],
                    max_iter=int(estimator["max_iter"]),
                    tol=float(estimator["tol"]),
                    random_state=int(estimator["random_state"]),
                ),
            ),
        ]
    )


def _pipeline_spec(config: dict[str, Any]) -> dict[str, Any]:
    model = config["model"]
    calibration = config["calibration"]["sigmoid"]
    calibrator_parameters = {
        key: value for key, value in calibration.items() if key != "minimum_samples"
    }
    return {
        "imputer": {
            "class": "sklearn.impute.SimpleImputer",
            "strategy": model["imputer"],
            "add_indicator": model["imputer_missing_indicator"],
            "keep_empty_features": model["imputer_keep_empty_features"],
        },
        "scaler": {
            "class": "sklearn.preprocessing.StandardScaler",
            "with_mean": model["scaler_with_mean"],
            "with_std": model["scaler_with_std"],
        },
        "estimator": {
            "class": "sklearn.linear_model.LogisticRegression",
            **model["estimator_parameters"],
        },
        "calibrator": {
            "class": "sklearn.linear_model.LogisticRegression",
            **calibrator_parameters,
        },
    }


def _promotion_criteria(config: dict[str, Any]) -> PromotionCriteria:
    payload = config["promotion_gate"]
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


def _calibration_coefficients(
    y: np.ndarray, probability: np.ndarray, seed: int
) -> tuple[float, float]:
    if np.unique(y).size != 2:
        raise ProtocolViolation("Pooled calibration diagnostic requires both target classes")
    clipped = np.clip(probability, 1e-6, 1.0 - 1e-6)
    logit = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
    fitted = LogisticRegression(
        C=1_000_000.0,
        penalty="l2",
        solver="lbfgs",
        fit_intercept=True,
        max_iter=2_000,
        tol=1e-4,
        random_state=seed,
    ).fit(logit, y)
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
        "executable_label_end_index",
        "return_lag_1",
        "momentum_20",
    ]


def _inner_oof(
    dataset: pd.DataFrame,
    features: list[str],
    nested: Any,
    config: dict[str, Any],
) -> pd.DataFrame:
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
    if not rows:
        raise ProtocolViolation("v3 inner folds produced no OOF predictions")
    return pd.concat(rows, ignore_index=True).sort_values("row_id").reset_index(drop=True)


def _secondary_calibration_context(
    pooled: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, Any]:
    fraction = float(config["inner_selection"]["calibration_fraction"])
    if len(pooled) < 2 or not 0.0 < fraction < 1.0:
        raise ProtocolViolation("Invalid v3 calibration split")
    split = min(max(20, int(len(pooled) * fraction)), len(pooled) - 1)
    raw_calibration = pooled.iloc[:split].copy()
    evaluation = pooled.iloc[split:].copy()
    prediction_start = int(evaluation["row_id"].min())
    calibration, purge = purge_calibration_against_prediction_window(
        raw_calibration,
        label_end_column="executable_label_end_index",
        prediction_start_row=prediction_start,
    )
    eligibility = assess_sigmoid_eligibility(
        calibration["raw_probability"].to_numpy(dtype=float),
        calibration["y"].to_numpy(dtype=int),
        minimum_samples=int(config["calibration"]["sigmoid"]["minimum_samples"]),
    )
    context = {
        "raw_calibration": raw_calibration,
        "calibration": calibration,
        "evaluation": evaluation,
        "purge": purge,
        "eligibility": eligibility,
    }
    return context


def _score_calibration_candidate(
    context: dict[str, Any],
    calibration_name: str,
    margin: float,
    config: dict[str, Any],
) -> tuple[dict[str, Any], tuple[float, float, float, str, float] | None]:
    evaluation: pd.DataFrame = context["evaluation"]
    eligibility = context["eligibility"]
    if calibration_name == "none":
        eligible = True
        reason = "not_applicable"
        probability = evaluation["raw_probability"].to_numpy(dtype=float)
    elif calibration_name == "sigmoid":
        eligible = bool(eligibility.eligible)
        reason = eligibility.reason
        if not eligible:
            probability = np.full(len(evaluation), np.nan, dtype=float)
        else:
            calibration: pd.DataFrame = context["calibration"]
            calibrator = fit_past_only_sigmoid(
                calibration["raw_probability"].to_numpy(dtype=float),
                calibration["y"].to_numpy(dtype=int),
                calibration_row_ids=calibration["row_id"].to_numpy(dtype=int),
                calibration_label_end_ids=calibration["executable_label_end_index"].to_numpy(
                    dtype=int
                ),
                prediction_row_ids=evaluation["row_id"].to_numpy(dtype=int),
                seed=int(config["model"]["estimator_parameters"]["random_state"]),
            )
            probability = calibrator.predict(evaluation["raw_probability"].to_numpy(dtype=float))
    else:
        raise ProtocolViolation(f"Unknown calibration candidate: {calibration_name}")

    purge = context["purge"]
    base: dict[str, Any] = {
        "calibration_candidate": calibration_name,
        "calibration": calibration_name,
        "margin": float(margin),
        "eligible": eligible,
        "eligibility_reason": reason,
        "raw_calibration_count": purge["raw_count"],
        "purged_count": purge["purged_count"],
        "retained_count": purge["retained_count"],
        "max_raw_label_end": purge["max_raw_label_end"],
        "max_retained_label_end": purge["max_retained_label_end"],
        "evaluation_start_row": purge["prediction_start_row"],
        "overlap_count_after_purge": purge["overlap_count_after_purge"],
        "sigmoid_retained_class_count": eligibility.class_count,
        "sigmoid_finite_probability": eligibility.finite_probability,
        "calibration_start_row": int(context["raw_calibration"]["row_id"].min()),
        "calibration_end_row": int(context["raw_calibration"]["row_id"].max()),
        "evaluation_end_row": int(evaluation["row_id"].max()),
    }
    if not eligible:
        base.update(
            {
                "balanced_accuracy": None,
                "macro_f1": None,
                "cumulative_net_log_return": None,
                "non_overlapping_trades": 0,
            }
        )
        return base, None

    metric = _metrics(evaluation["y"].to_numpy(dtype=int), probability, "inner_selection")
    candidate = evaluation.assign(probability_up=probability, selected_margin=float(margin))
    ledger = build_non_overlapping_return_ledger(
        candidate,
        cost_bps=float(config["economics"]["baseline_cost_bps"]),
    )
    cumulative = float(ledger["candidate_net_log_return"].sum())
    base.update(
        {
            "balanced_accuracy": metric.balanced_accuracy,
            "macro_f1": metric.macro_f1,
            "cumulative_net_log_return": cumulative,
            "non_overlapping_trades": int(ledger["active"].sum()),
        }
    )
    choice = (
        metric.balanced_accuracy,
        metric.macro_f1,
        cumulative,
        calibration_name,
        float(margin),
    )
    return base, choice


def _inner_choice(
    dataset: pd.DataFrame,
    features: list[str],
    nested: Any,
    config: dict[str, Any],
) -> tuple[str, float, list[dict[str, Any]], dict[str, Any]]:
    pooled = _inner_oof(dataset, features, nested, config)
    context = _secondary_calibration_context(pooled, config)
    score_records: list[dict[str, Any]] = []
    choices: list[tuple[float, float, float, str, float]] = []
    for calibration_name in config["inner_selection"]["calibration_options"]:
        for raw_margin in config["inner_selection"]["no_trade_margins"]:
            record, choice = _score_calibration_candidate(
                context,
                str(calibration_name),
                float(raw_margin),
                config,
            )
            score_records.append(record)
            if choice is not None:
                choices.append(choice)
    if not choices:
        raise ProtocolViolation("No eligible v3 inner calibration candidate")
    winner = max(
        choices,
        key=lambda item: (item[0], item[1], item[2], -item[4], item[3] == "none"),
    )
    return winner[3], winner[4], score_records, context


def _primary_overlap_count(frame: pd.DataFrame, fold: Any) -> int:
    train = frame.iloc[fold.train_indices]
    validation = frame.iloc[fold.validation_indices]
    return int(
        np.count_nonzero(
            train["executable_label_end_index"].to_numpy(dtype=int)
            >= int(validation["row_id"].min())
        )
    )


def _fold_qa(frame: pd.DataFrame, folds: tuple[Any, ...]) -> dict[str, Any]:
    outer_rows: list[dict[str, Any]] = []
    inner_overlap_count = 0
    for nested in folds:
        outer_overlap = _primary_overlap_count(frame, nested.outer)
        inner_rows: list[dict[str, Any]] = []
        for inner in nested.inner:
            overlap = _primary_overlap_count(frame, inner)
            inner_overlap_count += overlap
            inner_rows.append({"fold_id": inner.fold_id, "overlap_count": overlap})
        outer_rows.append(
            {
                "fold_id": nested.outer.fold_id,
                "overlap_count": outer_overlap,
                "inner": inner_rows,
            }
        )
    outer_overlap_count = int(sum(item["overlap_count"] for item in outer_rows))
    return {
        "primary_purge_verified": outer_overlap_count == 0 and inner_overlap_count == 0,
        "outer_overlap_count": outer_overlap_count,
        "inner_overlap_count": inner_overlap_count,
        "folds": outer_rows,
    }


def run_trial_v3(
    dataset: pd.DataFrame,
    features: list[str],
    spec: TrialSpecV3,
    config: dict[str, Any],
    *,
    folds: tuple[Any, ...] | None = None,
) -> dict[str, Any]:
    """Run one complete v3 arm-horizon trial with all five outer selections."""

    eligible = dataset.loc[dataset["is_modeling_eligible"]].reset_index(drop=True).copy()
    eligible["label_end_index"] = eligible["executable_label_end_index"].astype(int)
    fold_config = config["folds"]
    nested_folds = folds or build_nested_purged_walk_forward_folds(
        eligible,
        outer_min_train_rows=int(fold_config["outer_min_train_rows"]),
        inner_min_train_rows=int(fold_config["inner_min_train_rows"]),
        outer_min_validation_rows=int(fold_config["outer_min_validation_rows"]),
        inner_min_validation_rows=int(fold_config["inner_min_validation_rows"]),
        pre_validation_gap_rows=int(fold_config["pre_validation_gap_rows"]),
        outer_folds=int(fold_config["outer_folds"]),
        inner_folds=int(fold_config["inner_folds"]),
    )
    split_payload = nested_split_manifest(
        eligible,
        nested_folds,
        horizon=spec.horizon,
        pre_validation_gap_rows=int(fold_config["pre_validation_gap_rows"]),
    )
    primary_qa = _fold_qa(eligible, nested_folds)
    predictions: list[pd.DataFrame] = []
    fold_metrics: list[OuterFoldMetrics] = []
    selections: list[dict[str, Any]] = []
    inner_scores: list[dict[str, Any]] = []
    for nested in nested_folds:
        calibration_name, margin, scores, context = _inner_choice(
            eligible, features, nested, config
        )
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
            calibration = context["calibration"]
            evaluation = context["evaluation"]
            calibrator = fit_past_only_sigmoid(
                calibration["raw_probability"].to_numpy(dtype=float),
                calibration["y"].to_numpy(dtype=int),
                calibration_row_ids=calibration["row_id"].to_numpy(dtype=int),
                calibration_label_end_ids=calibration["executable_label_end_index"].to_numpy(
                    dtype=int
                ),
                prediction_row_ids=validation["row_id"].to_numpy(dtype=int),
                seed=int(config["model"]["estimator_parameters"]["random_state"]),
            )
            probability = calibrator.predict(probability)
            if int(evaluation["row_id"].min()) >= int(validation["row_id"].min()):
                raise ProtocolViolation("Inner calibration unexpectedly reaches outer prediction")
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
        purge = context["purge"]
        eligibility = context["eligibility"]
        selections.append(
            {
                "outer_fold": nested.outer.fold_id,
                "calibration": calibration_name,
                "margin": float(margin),
                "sigmoid_eligible": bool(eligibility.eligible),
                "sigmoid_eligibility_reason": eligibility.reason,
                "raw_calibration_count": purge["raw_count"],
                "purged_count": purge["purged_count"],
                "retained_count": purge["retained_count"],
                "max_raw_label_end": purge["max_raw_label_end"],
                "max_retained_label_end": purge["max_retained_label_end"],
                "evaluation_start_row": purge["prediction_start_row"],
                "overlap_count_after_purge": purge["overlap_count_after_purge"],
                "sigmoid_retained_class_count": eligibility.class_count,
                "sigmoid_finite_probability": eligibility.finite_probability,
                "primary_purge_verified": primary_qa["primary_purge_verified"],
                "secondary_calibration_purge_verified": (purge["overlap_count_after_purge"] == 0),
            }
        )
    pooled = pd.concat(predictions, ignore_index=True).sort_values("row_id").reset_index(drop=True)
    secondary_verified = bool(
        selections and all(item["secondary_calibration_purge_verified"] for item in selections)
    )
    return {
        "spec": spec,
        "features": features,
        "folds": nested_folds,
        "split_manifest": split_payload,
        "split_hash": nested_split_manifest_sha256(split_payload),
        "predictions": pooled,
        "fold_metrics": tuple(fold_metrics),
        "selections": selections,
        "inner_scores": inner_scores,
        "primary_qa": primary_qa,
        "secondary_calibration_purge_verified": secondary_verified,
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
        raise ProtocolViolation(f"Refusing to overwrite v3 finalization receipt: {path}") from exc


def _git_sha(project_root: Path) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(project_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ProtocolViolation("Unable to resolve the v3 execution git SHA") from exc
    return completed.stdout.strip()


def _runtime_sources(project_root: Path, bundle: dict[str, Any], extra: list[Path]) -> list[Path]:
    sources = [project_root / str(entry["path"]) for entry in bundle["code_manifest"]["files"]]
    return sources + extra


def _verify_feature_causality(
    source: pd.DataFrame,
    feature_config: FeatureConfig,
    features: pd.DataFrame,
    feature_columns: list[str],
) -> bool:
    mutation_row = max(1, len(source) // 2)
    mutated_source = source.copy()
    mutated_source.loc[mutation_row:, "close"] *= 1.001
    mutated_features, _mutated_columns = build_feature_matrix(mutated_source, feature_config)
    mutated_features = add_robust_hurst_features(
        mutated_features,
        regime_window=feature_config.regime_window,
        min_periods=max(80, feature_config.regime_window // 2),
    )
    all_columns = feature_columns + ["feature_coverage"]
    if set(all_columns) != set(features.columns).difference({"row_id", "date", "close", "volume"}):
        raise ProtocolViolation("Feature causality audit column set differs from v3 features")
    assert_causal_features(
        features,
        mutated_features,
        all_columns,
        inclusive_row=mutation_row - 1,
    )
    return True


def _paired_records(
    results: dict[str, dict[str, Any]],
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    inference = config["inference"]
    bootstrap_config = inference["bootstrap"]
    sign_flip_config = inference["sign_flip"]
    alignment: list[dict[str, Any]] = []
    primary: list[dict[str, Any]] = []
    secondary: list[dict[str, Any]] = []
    contrast_definitions = (
        ("current_dfa_hurst", "dfa1_vs_no_hurst", primary, 0),
        ("robust_hurst_regime", "robust_vs_no_hurst", secondary, 1),
    )
    for horizon in HORIZONS:
        reference = results[f"{FAMILY_V3}-no_hurst-h{horizon}"]["predictions"]
        for arm, contrast, output, contrast_offset in contrast_definitions:
            candidate = results[f"{FAMILY_V3}-{arm}-h{horizon}"]["predictions"]
            alignment_record = assert_exact_paired_alignment(
                reference,
                candidate,
                reference_name=f"no_hurst_h{horizon}",
                candidate_name=f"{arm}_h{horizon}",
            )
            alignment_record["horizon"] = horizon
            alignment_record["contrast"] = contrast
            alignment.append(alignment_record)
            ordered_reference = reference.sort_values("row_id", kind="stable").reset_index(
                drop=True
            )
            ordered_candidate = candidate.sort_values("row_id", kind="stable").reset_index(
                drop=True
            )
            truth = ordered_reference["executable_direction_binary"].to_numpy(dtype=int)
            no_probability = ordered_reference["probability_up"].to_numpy(dtype=float)
            hurst_probability = ordered_candidate["probability_up"].to_numpy(dtype=float)
            seed = int(bootstrap_config["seed"]) + horizon + contrast_offset * 1000
            bootstrap = paired_block_bootstrap_metric_difference(
                truth,
                hurst_probability,
                no_probability,
                int(bootstrap_config["block_length"]),
                int(bootstrap_config["iterations"]),
                seed,
                metric=str(inference["metric"]),
                confidence=float(bootstrap_config["confidence"]),
            )
            sign_flip = paired_block_sign_flip_test(
                truth,
                hurst_probability,
                no_probability,
                int(sign_flip_config["block_length"]),
                int(sign_flip_config["permutations"]),
                int(sign_flip_config["seed"]) + horizon + contrast_offset * 1000,
                metric=str(inference["metric"]),
            )
            output.append(
                {
                    "horizon": horizon,
                    "no_hurst_ba": bootstrap.observed_b,
                    "hurst_ba": bootstrap.observed_a,
                    "delta_ba": bootstrap.observed_delta,
                    "ci_low": bootstrap.ci_low,
                    "ci_high": bootstrap.ci_high,
                    "p_raw": sign_flip.p_raw,
                    "paired_row_count": alignment_record["row_count"],
                    "row_ids_identical": alignment_record["row_ids_identical"],
                    "y_true_identical": alignment_record["y_true_identical"],
                    "bootstrap_method": "moving_block",
                    "bootstrap_block_length": bootstrap.block_length,
                    "bootstrap_iterations": bootstrap.n_resamples,
                    "bootstrap_seed": bootstrap.seed,
                    "sign_flip_method": "non_overlapping_chronological_block_sign_flip",
                    "sign_flip_block_length": sign_flip.block_length,
                    "sign_flip_block_count": sign_flip.block_count,
                    "sign_flip_permutations": sign_flip.n_permutations,
                    "sign_flip_seed": sign_flip.seed,
                }
            )
    primary_p = holm_adjust([float(item["p_raw"]) for item in primary])
    secondary_p = holm_adjust([float(item["p_raw"]) for item in secondary])
    for item, adjusted in zip(primary, primary_p, strict=True):
        item["p_holm"] = adjusted
        item["multiplicity_family"] = "primary_dfa1_vs_no_hurst_across_horizons"
    for item, adjusted in zip(secondary, secondary_p, strict=True):
        item["p_holm"] = adjusted
        item["multiplicity_family"] = "secondary_robust_vs_no_hurst_across_horizons"
    summary = {
        "metric": str(inference["metric"]),
        "paired": True,
        "primary_contrast": "current_dfa_hurst_vs_no_hurst",
        "secondary_contrast": "robust_hurst_regime_vs_no_hurst",
        "bootstrap": {
            "method": str(bootstrap_config["method"]),
            "confidence": float(bootstrap_config["confidence"]),
            "iterations": int(bootstrap_config["iterations"]),
            "block_length": int(bootstrap_config["block_length"]),
            "seed": int(bootstrap_config["seed"]),
        },
        "hypothesis_test": {
            "method": str(sign_flip_config["method"]),
            "permutations": int(sign_flip_config["permutations"]),
            "block_length": int(sign_flip_config["block_length"]),
            "seed": int(sign_flip_config["seed"]),
            "two_sided": True,
        },
        "multiplicity_correction": "holm",
        "multiplicity_handling": (
            "Holm is applied separately across the four preregistered horizons within each "
            "contrast family; the secondary family remains secondary."
        ),
        "alignment": alignment,
        "alignment_verified": bool(
            alignment
            and all(item["row_ids_identical"] and item["y_true_identical"] for item in alignment)
        ),
        "primary": primary,
        "secondary": secondary,
    }
    if not summary["alignment_verified"]:
        raise ProtocolViolation("Exact paired alignment failed for v3 inference")
    return primary, secondary, summary


def _load_v2_registry(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        raise ProtocolViolation("Historical v2 registry is missing for diagnostic comparison")
    records: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parsed = json.loads(line)
        if not isinstance(parsed, dict):
            raise ProtocolViolation("Historical v2 registry contains a non-object record")
        records[str(parsed["experiment_id"])] = parsed
    if len(records) != DECLARED_BUDGET_V3:
        raise ProtocolViolation("Historical v2 registry must contain exactly 12 records")
    return records


def _v2_v3_diagnostic(
    project_root: Path,
    results: dict[str, dict[str, Any]],
    inventory: list[dict[str, Any]],
) -> pd.DataFrame:
    v2 = _load_v2_registry(
        project_root / "artifacts/research/registry/executable_direction_hurst_ablation_v2.jsonl"
    )
    rows: list[dict[str, Any]] = []
    for item in inventory:
        v3_id = str(item["experiment_id"])
        v2_id = v3_id.replace("_v3-", "_v2-", 1)
        v2_record = v2[v2_id]
        v2_selections = v2_record["train_metrics"]["outer_selections"]
        v3_result = results[v3_id]
        v3_selections = v3_result["selections"]
        v2_metrics = v2_record["validation_metrics"]["pooled"]
        rows.append(
            {
                "v2_experiment_id": v2_id,
                "v3_experiment_id": v3_id,
                "arm": item["arm"],
                "horizon": item["horizon"],
                "v2_sigmoid_selected_count": sum(
                    selection["calibration"] == "sigmoid" for selection in v2_selections
                ),
                "v3_sigmoid_selected_count": sum(
                    selection["calibration"] == "sigmoid" for selection in v3_selections
                ),
                "v2_none_selected_count": sum(
                    selection["calibration"] == "none" for selection in v2_selections
                ),
                "v3_none_selected_count": sum(
                    selection["calibration"] == "none" for selection in v3_selections
                ),
                "v2_margin_signature": ",".join(
                    f"{selection['outer_fold']}:{selection['margin']}"
                    for selection in v2_selections
                ),
                "v3_margin_signature": ",".join(
                    f"{selection['outer_fold']}:{selection['margin']}"
                    for selection in v3_selections
                ),
                "v2_balanced_accuracy": v2_metrics["balanced_accuracy"],
                "v3_balanced_accuracy": item["pooled_balanced_accuracy"],
                "delta_balanced_accuracy_v3_minus_v2": item["pooled_balanced_accuracy"]
                - v2_metrics["balanced_accuracy"],
                "v2_macro_f1": v2_metrics["macro_f1"],
                "v3_macro_f1": item["macro_f1"],
                "delta_macro_f1_v3_minus_v2": item["macro_f1"] - v2_metrics["macro_f1"],
                "v2_net_log_return_5bps": v2_record["economic_metrics"]["net_log_return"],
                "v3_net_log_return_5bps": item["net_log_return_5bps"],
                "delta_net_log_return_v3_minus_v2": item["net_log_return_5bps"]
                - v2_record["economic_metrics"]["net_log_return"],
                "v2_decision": v2_record["decision"],
                "v3_decision": item["decision"],
                "status_note": "v2 historical invalid evidence; v3 repaired evidence",
            }
        )
    return pd.DataFrame(rows).sort_values(["arm", "horizon"]).reset_index(drop=True)


def run_preregistered_batch_v3(project_root: Path) -> Path:
    """Execute and finalize the isolated v3 12-trial batch exactly once."""

    root = project_root.resolve()
    card_path = root / "protocol/hypothesis_cards/executable_direction_hurst_ablation_v3.json"
    config_path = root / "protocol/executable_configs/executable_direction_hurst_ablation_v3.json"
    code_manifest_path = (
        root / "protocol/code_manifests/executable_direction_hurst_ablation_v3.json"
    )
    development_dir = root / "data/partitions/v2/development"
    development_path = development_dir / "prices.csv"
    development_manifest_path = development_dir / "manifest.json"
    data_availability_manifest_path = root / "protocol/data_availability_manifest.json"
    registry = AppendOnlyExperimentRegistry(
        root / "artifacts/research/registry/executable_direction_hurst_ablation_v3.jsonl"
    )
    lock_path = root / "artifacts/research/locks/executable_direction_hurst_ablation_v3.lock"
    bundle = verify_preregistered_bundle_v3(
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
    batch_id = f"{FAMILY_V3}-{timestamp}"
    staging_dir = root / "artifacts/research/.staging" / batch_id
    final_dir = root / "artifacts/research/runs" / batch_id
    with ExclusiveFileLock(lock_path):
        if registry.read_and_validate():
            raise ProtocolViolation("The v3 family has already been executed")
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
            robust_columns = [name for name in features.columns if name.startswith("hurst_robust_")]
            feature_columns += robust_columns
            actual_features = {arm: select_arm_features(feature_columns, arm) for arm in ARMS}
            expected_selection = {
                "no_hurst": "all_non_hurst_features",
                "current_dfa_hurst": "base_plus_current_dfa1_features",
                "robust_hurst_regime": "base_plus_robust_regime_features",
            }
            if config.get("feature_selection") != expected_selection:
                raise ProtocolViolation(
                    "v3 feature-selection contract differs from preregistration"
                )
            feature_causality_verified = _verify_feature_causality(
                source,
                feature_config,
                features,
                feature_columns,
            )
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
            folds_by_horizon: dict[int, tuple[Any, ...]] = {}
            for horizon, dataset in datasets.items():
                eligible = (
                    dataset.loc[dataset["is_modeling_eligible"]].reset_index(drop=True).copy()
                )
                eligible["label_end_index"] = eligible["executable_label_end_index"].astype(int)
                folds_by_horizon[horizon] = build_nested_purged_walk_forward_folds(
                    eligible,
                    outer_min_train_rows=int(config["folds"]["outer_min_train_rows"]),
                    inner_min_train_rows=int(config["folds"]["inner_min_train_rows"]),
                    outer_min_validation_rows=int(config["folds"]["outer_min_validation_rows"]),
                    inner_min_validation_rows=int(config["folds"]["inner_min_validation_rows"]),
                    pre_validation_gap_rows=int(config["folds"]["pre_validation_gap_rows"]),
                    outer_folds=int(config["folds"]["outer_folds"]),
                    inner_folds=int(config["folds"]["inner_folds"]),
                )
            results_list = [
                run_trial_v3(
                    datasets[trial.horizon],
                    actual_features[trial.arm],
                    trial,
                    config,
                    folds=folds_by_horizon[trial.horizon],
                )
                for trial in enumerate_preregistered_trials_v3()
            ]
            results = {item["spec"].experiment_id: item for item in results_list}
            if set(results) != {
                trial.experiment_id for trial in enumerate_preregistered_trials_v3()
            }:
                raise ProtocolViolation("v3 did not complete all 12 registered trials")
            primary, secondary, paired_summary = _paired_records(results, config)
            paired_alignment_verified = bool(paired_summary["alignment_verified"])
            secondary_audit_rows: list[dict[str, Any]] = []
            for item in results_list:
                for selection in item["selections"]:
                    secondary_audit_rows.append(
                        {
                            "arm": item["spec"].arm,
                            "horizon": item["spec"].horizon,
                            **selection,
                        }
                    )
            secondary_audit = pd.DataFrame(secondary_audit_rows).sort_values(
                ["arm", "horizon", "outer_fold"]
            )
            if len(secondary_audit) != 60:
                raise ProtocolViolation("v3 secondary calibration audit must contain 60 selections")
            if int(secondary_audit["overlap_count_after_purge"].max()) != 0:
                raise ProtocolViolation("v3 secondary calibration audit contains label overlap")
            all_primary_verified = all(
                item["primary_qa"]["primary_purge_verified"] for item in results_list
            )
            all_secondary_verified = all(
                item["secondary_calibration_purge_verified"] for item in results_list
            )
            if not all_primary_verified or not all_secondary_verified:
                raise ProtocolViolation("v3 temporal QA failed before artifact generation")

            baseline_cost = float(config["economics"]["baseline_cost_bps"])
            stress_cost = float(config["economics"]["stress_cost_bps"])
            ledgers = {
                trial_id: build_non_overlapping_return_ledger(
                    item["predictions"], cost_bps=baseline_cost
                )
                for trial_id, item in results.items()
            }
            stress_ledgers = {
                trial_id: build_non_overlapping_return_ledger(
                    item["predictions"], cost_bps=stress_cost
                )
                for trial_id, item in results.items()
            }
            common_rows, return_matrix, trial_ids = align_trial_return_ledgers_v3(ledgers)
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
                        declared_trials=DECLARED_BUDGET_V3,
                        periods_per_year=periods_per_year,
                        trial_sharpe_mean=float(annual_sharpes.mean()),
                        trial_sharpe_std=float(annual_sharpes.std(ddof=1)),
                    ).probability
                except ValueError:
                    dsr_by_trial[trial_id] = None

            registry_payloads: list[dict[str, Any]] = []
            inventory: list[dict[str, Any]] = []
            uncertainty = config["uncertainty"]
            for item in results_list:
                spec: TrialSpecV3 = item["spec"]
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
                    seed=int(config["model"]["estimator_parameters"]["random_state"]),
                )
                primary_bootstrap = bootstrap.estimate_for(int(uncertainty["primary_block_length"]))
                intercept, slope = _calibration_coefficients(
                    truth,
                    probability,
                    int(config["model"]["estimator_parameters"]["random_state"]),
                )
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
                leakage_free = bool(
                    item["primary_qa"]["primary_purge_verified"]
                    and item["secondary_calibration_purge_verified"]
                    and feature_causality_verified
                )
                qa_flags = {
                    "leakage_free": leakage_free,
                    "primary_purge_verified": bool(item["primary_qa"]["primary_purge_verified"]),
                    "secondary_calibration_purge_verified": bool(
                        item["secondary_calibration_purge_verified"]
                    ),
                    "feature_causality_verified": bool(feature_causality_verified),
                    "paired_alignment_verified": paired_alignment_verified,
                    "audit_isolated": True,
                    "provenance_complete": bool(config["provenance"]["complete"]),
                    "reproducible": True,
                    "manifest_verified": True,
                }
                evidence = PromotionEvidence(
                    outer_folds=item["fold_metrics"],
                    pooled_balanced_accuracy=pooled.balanced_accuracy,
                    pooled_macro_f1=pooled.macro_f1,
                    pooled_recall_down=pooled.recall_down,
                    pooled_recall_up=pooled.recall_up,
                    balanced_accuracy_ci_low=primary_bootstrap.intervals["balanced_accuracy"].low,
                    balanced_accuracy_ci_high=primary_bootstrap.intervals["balanced_accuracy"].high,
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
                    paired_net_return_ci_low=primary_bootstrap.intervals[
                        "paired_mean_net_return"
                    ].low,
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
                    qa_flags=qa_flags,
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
                    experiment_dir / "leakage_audit.json",
                    {
                        "primary": item["primary_qa"],
                        "secondary_calibration": item["selections"],
                        "qa_flags": qa_flags,
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
                        "declared_multiplicity_trials": DECLARED_BUDGET_V3,
                    },
                )
                _write_json(experiment_dir / "promotion_decision.json", decision.to_dict())
                _write_json(
                    model_path,
                    {
                        "status": "outer_fold_evaluation_specification_only_no_promoted_model",
                        "family": FAMILY_V3,
                        "trial": asdict(spec),
                        "features": item["features"],
                        "pipeline": _pipeline_spec(config),
                        "hurst_estimator": config["hurst_estimator"],
                        "regime": config["robust_hurst"],
                        "inner_selection": config["inner_selection"],
                        "outer_selections": item["selections"],
                    },
                )
                registry_payloads.append(
                    {
                        "experiment_id": trial_id,
                        "parent_hypothesis": (
                            "Does causal DFA1 Hurst information add incremental development OOS "
                            "value after secondary label-interval purge?"
                        ),
                        "hypothesis_family": FAMILY_V3,
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
                            "arm": spec.arm,
                            "horizon": spec.horizon,
                            "model": config["model"],
                            "calibration": config["calibration"],
                            "inner_selection": config["inner_selection"],
                            "hurst_estimator": config["hurst_estimator"],
                            "regime": config["robust_hurst"],
                        },
                        "fold_definitions": [
                            {"split_manifest_sha256": item["split_hash"], **config["folds"]}
                        ],
                        "train_metrics": {
                            "selection": (
                                "inner_purged_walk_forward_only_with_secondary_interval_purge"
                            ),
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
                        "qa": qa_flags,
                        "decision": "promote" if decision.status == "PASS" else "reject",
                        "decision_reason": list(decision.reasons),
                        "declared_family_budget": DECLARED_BUDGET_V3,
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
                        "sigmoid_eligible_count": sum(
                            selection["sigmoid_eligible"] for selection in item["selections"]
                        ),
                        "sigmoid_ineligible_count": sum(
                            not selection["sigmoid_eligible"] for selection in item["selections"]
                        ),
                        "sigmoid_selected_count": sum(
                            selection["calibration"] == "sigmoid"
                            for selection in item["selections"]
                        ),
                        "none_selected_count": sum(
                            selection["calibration"] == "none" for selection in item["selections"]
                        ),
                        "failed_gates": list(decision.reasons),
                    }
                )

            _write_json(
                staging_dir / "experiment_inventory.json",
                {
                    "batch_id": batch_id,
                    "hypothesis_family": FAMILY_V3,
                    "trial_count": len(inventory),
                    "experiments": inventory,
                },
            )
            _write_json(staging_dir / "feature_membership.json", actual_features)
            secondary_audit.to_csv(
                staging_dir / "secondary_calibration_audit.csv", index=False, lineterminator="\n"
            )
            paired_dir = staging_dir / "paired_comparisons"
            paired_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(primary).sort_values("horizon").to_csv(
                paired_dir / "dfa1_vs_no_hurst.csv", index=False, lineterminator="\n"
            )
            pd.DataFrame(secondary).sort_values("horizon").to_csv(
                paired_dir / "robust_vs_no_hurst.csv", index=False, lineterminator="\n"
            )
            pd.DataFrame(primary)[
                [
                    "horizon",
                    "no_hurst_ba",
                    "hurst_ba",
                    "delta_ba",
                    "ci_low",
                    "ci_high",
                    "p_raw",
                    "p_holm",
                ]
            ].sort_values("horizon").to_csv(
                paired_dir / "table_5_v3.csv", index=False, lineterminator="\n"
            )
            _write_json(staging_dir / "paired_inference.json", paired_summary)
            diagnostic = _v2_v3_diagnostic(root, results, inventory)
            diagnostic.to_csv(
                staging_dir / "v2_vs_v3_diagnostic.csv", index=False, lineterminator="\n"
            )
            _write_json(
                staging_dir / "multiplicity_report.json",
                {
                    "family": FAMILY_V3,
                    "trial_ids": trial_ids,
                    "trial_count": len(trial_ids),
                    "common_row_ids": common_rows,
                    "common_observations": len(common_rows),
                    "pbo": asdict(pbo_result),
                    "annualized_trial_sharpes": dict(
                        zip(trial_ids, annual_sharpes.tolist(), strict=True)
                    ),
                    "dsr_probability_by_trial": dsr_by_trial,
                    "paired_inference_artifact": "paired_inference.json",
                },
            )
            v2_registry_path = (
                root / "artifacts/research/registry/executable_direction_hurst_ablation_v2.jsonl"
            )
            v2_receipt_path = root / "artifacts/research/runs" / V2_RUN_ID / "run_receipt.json"
            v2_hashes = {
                "registry": sha256_file(v2_registry_path),
                "run_receipt": sha256_file(v2_receipt_path),
            }
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
                    "hypothesis_family": FAMILY_V3,
                    "partition_role": "development_reused_previously_exposed",
                    "historical_audit_accessed": False,
                    "historical_confirmation_available": False,
                    "git_commit_sha": _git_sha(root),
                    "config_sha256": sha256_file(config_path),
                    "code_manifest_sha256": sha256_file(code_manifest_path),
                    "primary_purge_verified": all_primary_verified,
                    "secondary_calibration_purge_verified": all_secondary_verified,
                    "feature_causality_verified": feature_causality_verified,
                    "paired_alignment_verified": paired_alignment_verified,
                    "trial_count": len(inventory),
                    "outer_selection_count": len(secondary_audit),
                    "secondary_overlap_max": int(
                        secondary_audit["overlap_count_after_purge"].max()
                    ),
                    "v2_preservation_hashes": v2_hashes,
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
                    "schema_version": "research_batch_finalization_v3",
                    "state": "finalized",
                    "batch_id": batch_id,
                    "finalized_at_utc": datetime.now(UTC).isoformat(),
                    "run_receipt_sha256": sha256_file(final_dir / "run_receipt.json"),
                    "registry_path": registry.path.relative_to(root).as_posix(),
                    "registry_head": asdict(registry.head()),
                    "record_hashes": [record["record_hash"] for record in records],
                    "v2_preservation_hashes": v2_hashes,
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
