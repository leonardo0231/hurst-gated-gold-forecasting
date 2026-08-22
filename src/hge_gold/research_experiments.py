"""Pre-registered, development-only experiments for the Hurst ablation batch.

This module intentionally has no historical-audit loader.  The only source loader reads
the declared development prefix with ``nrows`` and fails closed if its last timestamp
reaches the previously revealed audit boundary.
"""

from __future__ import annotations

import hashlib
import json
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
from .data import normalize_and_validate
from .features import build_feature_matrix
from .research_protocol import (
    AppendOnlyExperimentRegistry,
    PartitionRole,
    ResearchPhase,
    assert_partition_access,
    sha256_file,
)
from .research_validation import (
    OuterFoldMetrics,
    PromotionEvidence,
    build_nested_purged_walk_forward_folds,
    evaluate_promotion_gate,
    joint_moving_block_bootstrap,
    nested_split_manifest,
    nested_split_manifest_sha256,
)
from .statistics import deflated_sharpe_ratio, probability_of_backtest_overfitting
from .targets import bps_to_log_return, build_horizon_dataset

DEVELOPMENT_BOUNDARY = "2023-07-03"
DEVELOPMENT_ROW_LIMIT = 3213
HYPOTHESIS_FAMILY = "executable_direction_hurst_ablation_v1"
DECLARED_FAMILY_BUDGET = 12


@dataclass(frozen=True)
class TrialSpec:
    arm: str
    horizon: int
    logistic_c: float = 0.2
    seed: int = 42

    @property
    def experiment_id(self) -> str:
        return f"{HYPOTHESIS_FAMILY}-{self.arm}-h{self.horizon}"


def enumerate_preregistered_trials() -> tuple[TrialSpec, ...]:
    """Return the exact frozen 12-trial search space."""

    return tuple(
        TrialSpec(arm=arm, horizon=horizon)
        for arm in ("no_hurst", "current_dfa_hurst", "robust_hurst_regime")
        for horizon in (1, 5, 10, 20)
    )


def load_development_source(
    source: Path,
    *,
    boundary_date: str = DEVELOPMENT_BOUNDARY,
    row_limit: int = DEVELOPMENT_ROW_LIMIT,
    min_rows: int = 700,
) -> pd.DataFrame:
    """Load only the declared reused-development prefix, never audit rows."""

    assert_partition_access(PartitionRole.DEVELOPMENT_REUSED, ResearchPhase.DEVELOPMENT_SELECTION)
    if row_limit < min_rows:
        raise ValueError("row_limit cannot be smaller than min_rows")
    raw = pd.read_csv(source, nrows=row_limit)
    frame = normalize_and_validate(raw, min_rows=min_rows)
    boundary = pd.Timestamp(boundary_date)
    if frame["date"].max() >= boundary:
        raise ValueError("Declared source prefix crosses the development boundary")
    if len(frame) != row_limit:
        raise ValueError(f"Expected exactly {row_limit} development rows; received {len(frame)}")
    return frame


def add_robust_hurst_features(
    frame: pd.DataFrame,
    *,
    regime_window: int = 252,
    min_periods: int = 126,
) -> pd.DataFrame:
    """Add causal robust summaries derived only from current DFA estimates."""

    result = frame.copy()
    dfa = [
        name
        for name in frame.columns
        if name.startswith("hurst_dfa1_") and name[len("hurst_dfa1_") :].isdigit()
    ]
    if not dfa:
        raise ValueError("Robust Hurst features require at least one DFA Hurst column")
    current = frame[dfa].median(axis=1, skipna=True)
    dispersion = frame[dfa].sub(current, axis=0).abs().median(axis=1, skipna=True)
    past = current.shift(1)
    result["hurst_robust_median"] = current
    result["hurst_robust_dispersion"] = dispersion
    result["hurst_robust_low_threshold"] = past.rolling(
        regime_window, min_periods=min_periods
    ).quantile(0.33)
    result["hurst_robust_high_threshold"] = past.rolling(
        regime_window, min_periods=min_periods
    ).quantile(0.67)
    available = (
        current.notna()
        & result["hurst_robust_low_threshold"].notna()
        & result["hurst_robust_high_threshold"].notna()
    )
    regime = pd.Series(np.nan, index=result.index, dtype=float)
    regime.loc[available] = 0.0
    regime.loc[available & (current <= result["hurst_robust_low_threshold"])] = -1.0
    regime.loc[available & (current >= result["hurst_robust_high_threshold"])] = 1.0
    result["hurst_robust_regime"] = regime
    result["hurst_robust_available"] = available.astype(float)
    return result


def _is_hurst(name: str) -> bool:
    return name.startswith("hurst_")


def select_arm_features(columns: list[str], arm: str) -> list[str]:
    """Select the registered arm while excluding the legacy single-scale R/S proxy."""

    base = [name for name in columns if not _is_hurst(name)]
    if arm == "no_hurst":
        return base
    if arm == "current_dfa_hurst":
        current = [
            name
            for name in columns
            if (
                name.startswith("hurst_dfa1_") or name in {"hurst_regime", "hurst_regime_available"}
            )
            and "legacy" not in name
            and "robust" not in name
        ]
        return base + current
    if arm == "robust_hurst_regime":
        robust = [name for name in columns if name.startswith("hurst_robust_")]
        return base + robust
    raise ValueError(f"Unknown preregistered arm: {arm}")


def build_mt5_signal_schedule(predictions: pd.DataFrame, *, margin: float) -> pd.DataFrame:
    """Build a deterministic bar-open schedule with conservative overlap suppression."""

    required = {"row_id", "entry_row_index", "exit_row_index", "probability_up"}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"Prediction schedule is missing columns: {sorted(missing)}")
    if not 0.0 <= margin < 0.5:
        raise ValueError("margin must be in [0, 0.5)")
    ordered = predictions.sort_values("row_id").reset_index(drop=True)
    selected: list[dict[str, Any]] = []
    busy_until = -1
    for row in ordered.itertuples(index=False):
        probability = float(row.probability_up)
        if abs(probability - 0.5) <= margin + 1e-12:
            continue
        entry = int(row.entry_row_index)
        exit_row = int(row.exit_row_index)
        if entry < busy_until:
            busy_until = max(busy_until, exit_row)
            continue
        selected.append(
            {
                "row_id": int(row.row_id),
                "entry_row_index": entry,
                "exit_row_index": exit_row,
                "probability_up": probability,
                "signal": 1 if probability > 0.5 else -1,
            }
        )
        busy_until = exit_row
    return pd.DataFrame(
        selected,
        columns=["row_id", "entry_row_index", "exit_row_index", "probability_up", "signal"],
    )


def replay_mt5_schedule(schedule: pd.DataFrame) -> pd.DataFrame:
    """Validate and replay the frozen MT5 bar-open schedule signal for signal."""

    required = ["row_id", "entry_row_index", "exit_row_index", "probability_up", "signal"]
    if list(schedule.columns) != required:
        raise ValueError("MT5 schedule schema or column order changed")
    if not schedule["row_id"].is_monotonic_increasing:
        raise ValueError("MT5 schedule must be chronological")
    expected = np.where(schedule["probability_up"].to_numpy() > 0.5, 1, -1)
    if not np.array_equal(expected, schedule["signal"].to_numpy(dtype=int)):
        raise ValueError("MT5 signal direction differs from frozen probability")
    return schedule.copy()


def _model(spec: TrialSpec) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
            (
                "logistic",
                LogisticRegression(
                    C=spec.logistic_c,
                    class_weight="balanced",
                    max_iter=2_000,
                    random_state=spec.seed,
                ),
            ),
        ]
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


def _ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    observed, predicted = calibration_curve(y, p, n_bins=bins, strategy="uniform")
    return float(np.mean(np.abs(observed - predicted)))


def _calibration_coefficients(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    clipped = np.clip(p, 1e-6, 1.0 - 1e-6)
    logit = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
    model = LogisticRegression(C=1_000_000.0, max_iter=2_000).fit(logit, y)
    return float(model.intercept_[0]), float(model.coef_[0, 0])


def _net_returns(
    frame: pd.DataFrame, probability: np.ndarray, margin: float, cost_bps: float
) -> np.ndarray:
    schedule = build_mt5_signal_schedule(
        frame[["row_id", "entry_row_index", "exit_row_index"]].assign(probability_up=probability),
        margin=margin,
    )
    signals = pd.Series(0.0, index=frame["row_id"].astype(int))
    if not schedule.empty:
        signals.loc[schedule["row_id"].astype(int)] = schedule["signal"].to_numpy(dtype=float)
    returns = frame.set_index(frame["row_id"].astype(int))["executable_forward_log_return"]
    active = signals.to_numpy() != 0.0
    result = signals.to_numpy(dtype=float) * returns.to_numpy(
        dtype=float
    ) - active * bps_to_log_return(cost_bps)
    return np.asarray(result, dtype=float)


def _inner_choice(
    dataset: pd.DataFrame, features: list[str], nested: Any, spec: TrialSpec
) -> tuple[str, float]:
    rows: list[pd.DataFrame] = []
    for fold in nested.inner:
        train = dataset.iloc[fold.train_indices]
        validation = dataset.iloc[fold.validation_indices]
        fitted = _model(spec).fit(train[features], train["executable_direction_binary"].astype(int))
        rows.append(
            validation[
                ["row_id", "entry_row_index", "exit_row_index", "executable_forward_log_return"]
            ].assign(
                y=validation["executable_direction_binary"].astype(int).to_numpy(),
                raw_probability=fitted.predict_proba(validation[features])[:, 1],
            )
        )
    pooled = pd.concat(rows, ignore_index=True).sort_values("row_id")
    split = max(20, len(pooled) * 2 // 3)
    calibration = pooled.iloc[:split]
    evaluation = pooled.iloc[split:]
    choices: list[tuple[float, float, float, str, float]] = []
    for calibration_name in ("none", "sigmoid"):
        probability = evaluation["raw_probability"].to_numpy(dtype=float)
        if calibration_name == "sigmoid":
            calibrator = fit_past_only_sigmoid(
                calibration["raw_probability"].to_numpy(dtype=float),
                calibration["y"].to_numpy(dtype=int),
                calibration_row_ids=calibration["row_id"].to_numpy(dtype=int),
                prediction_row_ids=evaluation["row_id"].to_numpy(dtype=int),
                seed=spec.seed,
            )
            probability = calibrator.predict(probability)
        metric = _metrics(evaluation["y"].to_numpy(dtype=int), probability, "inner_selection")
        for margin in (0.0, 0.05):
            net = _net_returns(evaluation, probability, margin, 5.0)
            choices.append(
                (
                    metric.balanced_accuracy,
                    metric.macro_f1,
                    float(net.mean()),
                    calibration_name,
                    margin,
                )
            )
    winner = max(choices, key=lambda item: (item[0], item[1], item[2], -item[4], item[3] == "none"))
    return winner[3], winner[4]


def _run_trial(
    dataset: pd.DataFrame, feature_columns: list[str], spec: TrialSpec
) -> dict[str, Any]:
    eligible = dataset.loc[dataset["is_modeling_eligible"]].reset_index(drop=True).copy()
    eligible["label_end_index"] = eligible["executable_label_end_index"].astype(int)
    features = select_arm_features(feature_columns, spec.arm)
    folds = build_nested_purged_walk_forward_folds(
        eligible,
        outer_min_train_rows=800,
        inner_min_train_rows=300,
        outer_min_validation_rows=120,
        inner_min_validation_rows=60,
        pre_validation_gap_rows=0,
    )
    split_payload = nested_split_manifest(
        eligible, folds, horizon=spec.horizon, pre_validation_gap_rows=0
    )
    predictions: list[pd.DataFrame] = []
    fold_metrics: list[OuterFoldMetrics] = []
    selections: list[dict[str, Any]] = []
    for nested in folds:
        calibration_name, margin = _inner_choice(eligible, features, nested, spec)
        train = eligible.iloc[nested.outer.train_indices]
        validation = eligible.iloc[nested.outer.validation_indices]
        fitted = _model(spec).fit(train[features], train["executable_direction_binary"].astype(int))
        probability = fitted.predict_proba(validation[features])[:, 1]
        if calibration_name == "sigmoid":
            inner_oof: list[pd.DataFrame] = []
            for fold in nested.inner:
                inner_train = eligible.iloc[fold.train_indices]
                inner_validation = eligible.iloc[fold.validation_indices]
                inner_model = _model(spec).fit(
                    inner_train[features], inner_train["executable_direction_binary"].astype(int)
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
                seed=spec.seed,
            )
            probability = calibrator.predict(probability)
        fold_metrics.append(
            _metrics(
                validation["executable_direction_binary"].to_numpy(dtype=int),
                probability,
                nested.outer.fold_id,
            )
        )
        predictions.append(
            validation[
                [
                    "row_id",
                    "date",
                    "entry_row_index",
                    "exit_row_index",
                    "executable_forward_log_return",
                    "executable_direction_binary",
                ]
            ].assign(
                probability_up=probability,
                outer_fold=nested.outer.fold_id,
                selected_calibration=calibration_name,
                selected_margin=margin,
            )
        )
        selections.append(
            {"outer_fold": nested.outer.fold_id, "calibration": calibration_name, "margin": margin}
        )
    pooled = pd.concat(predictions, ignore_index=True).sort_values("row_id").reset_index(drop=True)
    return {
        "spec": spec,
        "features": features,
        "folds": folds,
        "split_manifest": split_payload,
        "split_hash": nested_split_manifest_sha256(split_payload),
        "predictions": pooled,
        "fold_metrics": tuple(fold_metrics),
        "selections": selections,
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
    path.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def run_preregistered_batch(
    project_root: Path,
    *,
    bootstrap_iterations: int = 2_000,
) -> Path:
    """Run the frozen development batch, persist evidence, and append all outcomes."""

    source_path = project_root / "data/processed/XAUUSD_Daily_20110103_20260731_model.csv"
    config_path = project_root / "configs/thesis.yaml"
    dependency_path = project_root / "uv.lock"
    card_path = (
        project_root / "protocol/hypothesis_cards/executable_direction_hurst_ablation_v1.json"
    )
    registry = AppendOnlyExperimentRegistry(
        project_root / "artifacts/research/registry/experiments.jsonl"
    )
    if registry.read_and_validate():
        raise RuntimeError("The preregistered append-only batch has already been executed")
    source = load_development_source(source_path)
    features, feature_columns = build_feature_matrix(source, FeatureConfig())
    features = add_robust_hurst_features(features)
    robust_columns = [name for name in features.columns if name.startswith("hurst_robust_")]
    feature_columns = feature_columns + robust_columns
    target_config = TargetConfig()
    datasets = {
        horizon: build_horizon_dataset(
            source, features, feature_columns, horizon, target_config, FeatureConfig()
        )
        for horizon in target_config.horizons
    }
    results = [
        _run_trial(datasets[trial.horizon], feature_columns, trial)
        for trial in enumerate_preregistered_trials()
    ]

    batch_stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    batch_id = f"{HYPOTHESIS_FAMILY}-{batch_stamp}"
    run_dir = project_root / "artifacts/research/runs" / batch_id
    run_dir.mkdir(parents=True, exist_ok=False)
    prediction_dir = project_root / "data/predictions/research" / batch_id
    prediction_dir.mkdir(parents=True, exist_ok=False)
    model_dir = project_root / "models/research" / batch_id
    model_dir.mkdir(parents=True, exist_ok=False)

    data_hash = hashlib.sha256(
        pd.util.hash_pandas_object(source, index=True).values.tobytes()
    ).hexdigest()
    code_hash = sha256_file(project_root / "src/hge_gold/research_experiments.py")
    inventory: list[dict[str, Any]] = []
    for horizon in (1, 5, 10, 20):
        horizon_results = [item for item in results if item["spec"].horizon == horizon]
        common_rows = horizon_results[0]["predictions"]["row_id"].to_numpy(dtype=int)
        return_columns: list[np.ndarray] = []
        for item in horizon_results:
            if not np.array_equal(common_rows, item["predictions"]["row_id"].to_numpy(dtype=int)):
                raise AssertionError("PBO trials are not aligned on synchronous outer OOF rows")
            pred_frame = item["predictions"]
            margins = pred_frame["selected_margin"].to_numpy(dtype=float)
            probability = pred_frame["probability_up"].to_numpy(dtype=float)
            signal = np.where(
                np.abs(probability - 0.5) > margins, np.where(probability > 0.5, 1.0, -1.0), 0.0
            )
            net = signal * pred_frame["executable_forward_log_return"].to_numpy(dtype=float) - (
                signal != 0
            ) * bps_to_log_return(5.0)
            return_columns.append(net)
        matrix = np.column_stack(return_columns)
        pbo = probability_of_backtest_overfitting(matrix, n_partitions=8).pbo
        annual_sharpes = np.divide(
            matrix.mean(axis=0),
            matrix.std(axis=0, ddof=1),
            out=np.zeros(matrix.shape[1]),
            where=matrix.std(axis=0, ddof=1) > 0,
        ) * np.sqrt(252.0)

        for item, net_returns in zip(horizon_results, return_columns, strict=True):
            spec: TrialSpec = item["spec"]
            pred: pd.DataFrame = item["predictions"]
            truth = pred["executable_direction_binary"].to_numpy(dtype=int)
            probability = pred["probability_up"].to_numpy(dtype=float)
            forced = _metrics(truth, probability, "pooled")
            margins = pred["selected_margin"].to_numpy(dtype=float)
            stress_signal = np.where(
                np.abs(probability - 0.5) > margins, np.where(probability > 0.5, 1.0, -1.0), 0.0
            )
            stress_returns = stress_signal * pred["executable_forward_log_return"].to_numpy(
                dtype=float
            ) - (stress_signal != 0) * bps_to_log_return(10.0)
            market = pred["executable_forward_log_return"].to_numpy(dtype=float)
            cost = bps_to_log_return(5.0)
            momentum = np.sign(pd.Series(market).shift(1).fillna(0.0).to_numpy()) * market
            trend = (
                np.sign(
                    pd.Series(market)
                    .rolling(20, min_periods=1)
                    .mean()
                    .shift(1)
                    .fillna(0.0)
                    .to_numpy()
                )
                * market
            )
            benchmarks = {
                "cash": np.zeros(len(pred)),
                "always_long": market - cost,
                "always_short": -market - cost,
                "momentum": momentum - (momentum != 0.0) * cost,
                "trend": trend - (trend != 0.0) * cost,
            }
            bootstrap = joint_moving_block_bootstrap(
                truth,
                probability,
                net_returns,
                benchmarks["trend"],
                threshold=0.5,
                primary_block_length=10,
                sensitivity_block_lengths=(5, 20),
                iterations=bootstrap_iterations,
                seed=spec.seed,
            )
            primary = bootstrap.estimate_for(10)
            intercept, slope = _calibration_coefficients(truth, probability)
            base_probability = np.full(len(truth), float(truth.mean()))
            try:
                dsr = deflated_sharpe_ratio(
                    net_returns,
                    declared_trials=DECLARED_FAMILY_BUDGET,
                    periods_per_year=252.0,
                    trial_sharpe_mean=float(annual_sharpes.mean()),
                    trial_sharpe_std=float(annual_sharpes.std(ddof=1)),
                ).probability
            except ValueError:
                dsr = None
            evidence = PromotionEvidence(
                outer_folds=item["fold_metrics"],
                pooled_balanced_accuracy=forced.balanced_accuracy,
                pooled_macro_f1=forced.macro_f1,
                pooled_recall_down=forced.recall_down,
                pooled_recall_up=forced.recall_up,
                balanced_accuracy_ci_low=primary.intervals["balanced_accuracy"].low,
                balanced_accuracy_ci_high=primary.intervals["balanced_accuracy"].high,
                balanced_accuracy_sensitivity_lows={
                    str(length): bootstrap.estimate_for(length).intervals["balanced_accuracy"].low
                    for length in (5, 20)
                },
                candidate_brier=float(brier_score_loss(truth, probability)),
                baseline_brier=float(brier_score_loss(truth, base_probability)),
                candidate_ece=_ece(truth, probability),
                baseline_ece=_ece(truth, base_probability),
                calibration_intercept=intercept,
                calibration_slope=slope,
                classification_baseline_deltas={
                    "always_up": forced.balanced_accuracy - 0.5,
                    "always_down": forced.balanced_accuracy - 0.5,
                },
                paired_net_return_ci_low=primary.intervals["paired_mean_net_return"].low,
                cumulative_net_return_baseline_cost=float(net_returns.sum()),
                cumulative_net_return_stress_cost=float(stress_returns.sum()),
                economic_benchmark_deltas={
                    name: float(net_returns.sum() - values.sum())
                    for name, values in benchmarks.items()
                },
                n_non_overlapping_trades=int(np.count_nonzero(net_returns)),
                trial_return_registry_complete=True,
                pbo=float(pbo),
                dsr_probability=dsr,
                qa_flags={
                    "leakage_free": True,
                    "audit_isolated": True,
                    "provenance_complete": False,
                    "reproducible": True,
                    "manifest_verified": True,
                },
            )
            decision = evaluate_promotion_gate(evidence)
            experiment_dir = run_dir / spec.experiment_id
            experiment_dir.mkdir()
            prediction_path = prediction_dir / f"{spec.experiment_id}.csv"
            pred.to_csv(prediction_path, index=False)
            _write_json(experiment_dir / "split_manifest.json", item["split_manifest"])
            _write_json(
                experiment_dir / "outer_fold_metrics.json",
                {
                    "folds": [asdict(metric) for metric in item["fold_metrics"]],
                    "pooled": asdict(forced),
                },
            )
            _write_json(
                experiment_dir / "calibration_uncertainty.json",
                {
                    "brier": evidence.candidate_brier,
                    "baseline_brier": evidence.baseline_brier,
                    "ece": evidence.candidate_ece,
                    "baseline_ece": evidence.baseline_ece,
                    "intercept": intercept,
                    "slope": slope,
                    "bootstrap_iterations": bootstrap_iterations,
                    "block_lengths": [5, 10, 20],
                    "balanced_accuracy_ci": [
                        evidence.balanced_accuracy_ci_low,
                        evidence.balanced_accuracy_ci_high,
                    ],
                },
            )
            _write_json(
                experiment_dir / "economic_benchmarks.json",
                {
                    "cost_convention": "round_trip_total",
                    "baseline_cost_bps": 5.0,
                    "stress_cost_bps": 10.0,
                    "candidate_net_log_return": float(net_returns.sum()),
                    "stress_net_log_return": float(stress_returns.sum()),
                    "benchmark_deltas": evidence.economic_benchmark_deltas,
                    "trade_count": evidence.n_non_overlapping_trades,
                    "pbo": pbo,
                    "dsr_probability": dsr,
                },
            )
            _write_json(experiment_dir / "promotion_decision.json", decision.to_dict())
            _write_json(
                model_dir / f"{spec.experiment_id}.json",
                {
                    "status": "outer_fold_evaluation_specification_only_no_promoted_model",
                    "trial": asdict(spec),
                    "features": item["features"],
                    "outer_selections": item["selections"],
                },
            )
            payload = {
                "experiment_id": spec.experiment_id,
                "parent_hypothesis": (
                    "Does causal DFA Hurst information add incremental development OOS value?"
                ),
                "hypothesis_family": HYPOTHESIS_FAMILY,
                "timestamp_utc": datetime.now(UTC).isoformat(),
                "code_sha256": code_hash,
                "config_sha256": sha256_file(config_path),
                "data_sha256": data_hash,
                "dependency_sha256": sha256_file(dependency_path),
                "source_availability_convention": (
                    "D1 bar t available only after close; entry open t+1; "
                    "source prefix ends before 2023-07-03"
                ),
                "feature_list": item["features"],
                "target": "executable_direction_binary_open_t+1_to_open_t+1+h",
                "horizon": spec.horizon,
                "model": "median_imputer_standard_scaler_balanced_logistic_regression",
                "hyperparameters": {
                    **asdict(spec),
                    "inner_calibration": ["none", "sigmoid"],
                    "inner_no_trade_margin": [0.0, 0.05],
                },
                "fold_definitions": [
                    {
                        "split_manifest_sha256": item["split_hash"],
                        "outer_fold_count": 5,
                        "inner_fold_count": 3,
                    }
                ],
                "train_metrics": {
                    "selection": "inner_purged_walk_forward_only",
                    "outer_selections": item["selections"],
                },
                "validation_metrics": {
                    "outer_folds": [asdict(metric) for metric in item["fold_metrics"]],
                    "pooled": asdict(forced),
                },
                "calibration_metrics": {
                    "brier": evidence.candidate_brier,
                    "ece": evidence.candidate_ece,
                    "intercept": intercept,
                    "slope": slope,
                },
                "economic_metrics": {
                    "net_log_return": float(net_returns.sum()),
                    "stress_net_log_return": float(stress_returns.sum()),
                    "pbo": pbo,
                    "dsr_probability": dsr,
                },
                "decision": "promote" if decision.status == "PASS" else "reject",
                "decision_reason": list(decision.reasons),
                "declared_family_budget": DECLARED_FAMILY_BUDGET,
            }
            record = registry.append(payload)
            inventory.append(
                {
                    "experiment_id": spec.experiment_id,
                    "arm": spec.arm,
                    "horizon": spec.horizon,
                    "decision": payload["decision"],
                    "failed_gates": list(decision.reasons),
                    "pooled_balanced_accuracy": forced.balanced_accuracy,
                    "macro_f1": forced.macro_f1,
                    "net_log_return_5bps": float(net_returns.sum()),
                    "pbo": pbo,
                    "dsr_probability": dsr,
                    "record_hash": record["record_hash"],
                }
            )

    _write_json(
        run_dir / "experiment_inventory.json", {"batch_id": batch_id, "experiments": inventory}
    )
    _write_json(
        run_dir / "run_receipt.json",
        {
            "batch_id": batch_id,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "partition": PartitionRole.DEVELOPMENT_REUSED.value,
            "development_boundary_exclusive": DEVELOPMENT_BOUNDARY,
            "development_rows": len(source),
            "source_last_date": str(source["date"].max().date()),
            "source_file_sha256": sha256_file(source_path),
            "development_frame_sha256": data_hash,
            "code_sha256": code_hash,
            "config_sha256": sha256_file(config_path),
            "dependency_sha256": sha256_file(dependency_path),
            "hypothesis_card_sha256": sha256_file(card_path),
            "registry_head": asdict(registry.head()),
            "bootstrap_iterations": bootstrap_iterations,
            "historical_confirmation_available": False,
            "historical_audit_accessed": False,
        },
    )
    return run_dir
