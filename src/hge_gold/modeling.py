from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import f1_score, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import PipelineConfig
from .io import atomic_json, decision_hash, sha256_file, write_csv, write_parquet

TASK_TARGET = {
    "return_regression": "ret_fwd",
    "direction_classification": "direction_label_encoded",
    "trade_action_classification": "trade_label_encoded",
    "volatility_regression": "rv_fwd",
}
CLASS_ORDER = np.array([-1, 0, 1], dtype=int)


@dataclass(frozen=True)
class Fold:
    fold_id: str
    train_end: pd.Timestamp
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp


def build_walk_forward_folds(
    dates: pd.Series, n_folds: int, min_train_rows: int, validation_fraction: float
) -> tuple[list[Fold], pd.Timestamp]:
    unique = pd.Series(pd.to_datetime(dates).dropna().unique()).sort_values().reset_index(drop=True)
    if len(unique) < min_train_rows + 80:
        raise ValueError("Dataset is too small for locked walk-forward protocol")
    locked_start_index = int(len(unique) * 0.8)
    locked_start = pd.Timestamp(unique.iloc[locked_start_index])
    development = unique.iloc[:locked_start_index]
    val_size = max(30, int(len(development) * validation_fraction))
    starts = np.linspace(min_train_rows, len(development) - val_size, n_folds, dtype=int)
    folds: list[Fold] = []
    for number, start in enumerate(starts, start=1):
        if start <= 0 or start + val_size > len(development):
            continue
        folds.append(
            Fold(
                fold_id=f"wf_{number:03d}",
                train_end=pd.Timestamp(development.iloc[start - 1]),
                validation_start=pd.Timestamp(development.iloc[start]),
                validation_end=pd.Timestamp(development.iloc[start + val_size - 1]),
            )
        )
    if not folds:
        raise ValueError("No valid walk-forward folds could be constructed")
    return folds, locked_start


def _feature_columns(registry_path: Path) -> list[str]:
    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    return [
        item["feature_name"]
        for item in payload["features"]
        if item["status"] == "created_validated"
    ]


def _preprocessor(frame: pd.DataFrame, feature_cols: list[str]) -> ColumnTransformer:
    categorical = [
        column for column in feature_cols if pd.api.types.is_string_dtype(frame[column].dtype)
    ]
    numeric = [column for column in feature_cols if column not in categorical]
    return ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            ),
        ],
        remainder="drop",
    )


def _model(family: str, classification: bool, seed: int) -> Any:
    if family == "baseline":
        return (
            DummyClassifier(strategy="prior") if classification else DummyRegressor(strategy="mean")
        )
    if family == "linear_regularized":
        return (
            LogisticRegression(C=0.5, max_iter=1000, random_state=seed)
            if classification
            else Ridge(alpha=1.0)
        )
    if family == "tree_ensemble":
        common = {
            "n_estimators": 40,
            "max_depth": 7,
            "min_samples_leaf": 8,
            "random_state": seed,
            "n_jobs": 1,
        }
        return (
            RandomForestClassifier(class_weight="balanced_subsample", **common)
            if classification
            else RandomForestRegressor(**common)
        )
    raise ValueError(f"Unknown model family: {family}")


def _probabilities(model: Pipeline, frame: pd.DataFrame) -> np.ndarray:
    raw = model.predict_proba(frame)
    classes = np.asarray(model.named_steps["model"].classes_, dtype=int)
    aligned = np.zeros((len(frame), len(CLASS_ORDER)), dtype=float)
    for index, label in enumerate(CLASS_ORDER):
        match = np.flatnonzero(classes == label)
        if len(match):
            aligned[:, index] = raw[:, match[0]]
    totals = aligned.sum(axis=1, keepdims=True)
    if (totals == 0).any():
        raise RuntimeError("Classification probability vector is empty")
    aligned /= totals
    return aligned


def _metric(
    task: str, y_true: np.ndarray, y_pred: np.ndarray, probabilities: np.ndarray | None = None
) -> float:
    if task in {"direction_classification", "trade_action_classification"}:
        return float(f1_score(y_true, y_pred, labels=CLASS_ORDER, average="macro", zero_division=0))
    if task == "volatility_regression":
        variance = np.maximum(y_pred, 1e-12)
        truth = np.maximum(y_true, 1e-12)
        return float(np.mean(np.log(variance) + truth / variance))
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _better(task: str, value: float) -> float:
    return -value if task in {"return_regression", "volatility_regression"} else value


def _prediction_rows(
    source: pd.DataFrame,
    task: str,
    horizon: int,
    candidate_id: str,
    family: str,
    fold_id: str,
    pred: np.ndarray,
    proba: np.ndarray | None,
    include_truth: bool,
) -> pd.DataFrame:
    result = source[["row_id", "date", "date_index", "target_policy_id", "feature_set_id"]].copy()
    result["horizon"] = horizon
    result["task"] = task
    result["target_name"] = TASK_TARGET[task]
    result["candidate_model_id"] = candidate_id
    result["model_family"] = family
    result["fold_id"] = fold_id
    result["y_pred"] = pred
    if include_truth:
        result["y_true"] = source[TASK_TARGET[task]].to_numpy()
    result["y_pred_proba_json"] = (
        [json.dumps(dict(zip(CLASS_ORDER.tolist(), row.tolist(), strict=True))) for row in proba]
        if proba is not None
        else None
    )
    result["class_order_json"] = json.dumps(CLASS_ORDER.tolist()) if proba is not None else None
    result["prediction_unit"] = (
        "class_label"
        if proba is not None
        else ("variance_log_return_squared" if task == "volatility_regression" else "log_return")
    )
    return result


def run_modeling(config: PipelineConfig) -> dict[str, Path]:
    paths = config.paths()
    base_path = (
        paths.data / "processed" / "modeling_base" / "phase3_modeling_base_gold_only.parquet"
    )
    base = pd.read_parquet(base_path)
    base["date"] = pd.to_datetime(base["date"])
    base = base[base["is_modeling_eligible"]].copy()
    feature_cols = _feature_columns(paths.artifacts / "metadata" / "phase3_feature_registry.json")
    for column in feature_cols:
        if not pd.api.types.is_numeric_dtype(base[column].dtype):
            base[column] = base[column].astype(object).where(base[column].notna(), np.nan)
    folds, locked_start = build_walk_forward_folds(
        base["date"],
        int(config.modeling["n_walk_forward_folds"]),
        int(config.modeling["min_train_rows"]),
        float(config.modeling["validation_fraction"]),
    )
    metadata = paths.artifacts / "metadata"
    predictions_dir = paths.data / "predictions" / "phase4"
    model_dir = paths.models / "phase4"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    seed = int(config.project["seed"])
    outer_parts: list[pd.DataFrame] = []
    metric_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    registry_rows: list[dict[str, object]] = []

    for horizon in config.targets["horizons"]:
        horizon_data = base[base["horizon"] == horizon].copy()
        for fold in folds:
            validation = horizon_data[
                horizon_data["date"].between(fold.validation_start, fold.validation_end)
            ].copy()
            train = horizon_data[
                (horizon_data["date"] < fold.validation_start)
                & (pd.to_datetime(horizon_data["label_end_date"]) < fold.validation_start)
            ].copy()
            for _, row in train.iterrows():
                fold_rows.append(
                    {
                        "row_id": row.row_id,
                        "date": row.date,
                        "horizon": horizon,
                        "fold_id": fold.fold_id,
                        "split_role": "train",
                        "is_train": True,
                        "is_validation": False,
                        "is_locked_test": False,
                        "purged_from_train": False,
                    }
                )
            for _, row in validation.iterrows():
                fold_rows.append(
                    {
                        "row_id": row.row_id,
                        "date": row.date,
                        "horizon": horizon,
                        "fold_id": fold.fold_id,
                        "split_role": "validation",
                        "is_train": False,
                        "is_validation": True,
                        "is_locked_test": False,
                        "purged_from_train": False,
                    }
                )
            if len(train) < 80 or len(validation) < 10:
                continue
            for task, target in TASK_TARGET.items():
                classification = task.endswith("classification")
                train_task = train.dropna(subset=[target])
                val_task = validation.dropna(subset=[target])
                for family in config.modeling["model_families"]:
                    candidate_id = f"{task}_h{horizon}_{family}"
                    pipeline = Pipeline(
                        [
                            ("prep", _preprocessor(train_task, feature_cols)),
                            ("model", _model(family, classification, seed)),
                        ]
                    )
                    pipeline.fit(
                        train_task[feature_cols],
                        train_task[target].astype(int) if classification else train_task[target],
                    )
                    proba = (
                        _probabilities(pipeline, val_task[feature_cols]) if classification else None
                    )
                    pred = (
                        CLASS_ORDER[np.argmax(proba, axis=1)]
                        if proba is not None
                        else pipeline.predict(val_task[feature_cols])
                    )
                    value = _metric(task, val_task[target].to_numpy(), pred, proba)
                    outer_parts.append(
                        _prediction_rows(
                            val_task,
                            task,
                            horizon,
                            candidate_id,
                            family,
                            fold.fold_id,
                            pred,
                            proba,
                            True,
                        )
                    )
                    metric_rows.append(
                        {
                            "task": task,
                            "horizon": horizon,
                            "candidate_model_id": candidate_id,
                            "model_family": family,
                            "fold_id": fold.fold_id,
                            "metric_name": "macro_f1"
                            if classification
                            else ("QLIKE" if task == "volatility_regression" else "RMSE"),
                            "metric_value": value,
                            "metric_valid": np.isfinite(value),
                            "n_validation": len(val_task),
                        }
                    )
                    artifact = model_dir / "candidate" / f"{candidate_id}_{fold.fold_id}.joblib"
                    artifact.parent.mkdir(parents=True, exist_ok=True)
                    joblib.dump(pipeline, artifact)
                    registry_rows.append(
                        {
                            "candidate_model_id": candidate_id,
                            "model_instance_id": f"{candidate_id}_{fold.fold_id}",
                            "fold_id": fold.fold_id,
                            "task": task,
                            "horizon": horizon,
                            "model_family": family,
                            "model_status": "candidate_validated",
                            "artifact_role": "candidate_fold_model",
                            "artifact_path": str(artifact.relative_to(paths.root)),
                            "artifact_hash": sha256_file(artifact),
                            "training_random_seed": seed,
                            "library_random_seed": seed,
                            "numpy_random_seed": seed,
                        }
                    )

    outer = pd.concat(outer_parts, ignore_index=True)
    # Rule-based HGE: combines validated linear/tree predictions using only causal regime labels.
    for (task, horizon, fold_id), group in outer.groupby(
        ["task", "horizon", "fold_id"], sort=False
    ):
        families = set(group["model_family"])
        if not {"linear_regularized", "tree_ensemble"}.issubset(families):
            continue
        linear = group[group["model_family"] == "linear_regularized"].sort_values("row_id")
        tree = group[group["model_family"] == "tree_ensemble"].sort_values("row_id")
        if not np.array_equal(linear["row_id"].to_numpy(), tree["row_id"].to_numpy()):
            raise RuntimeError("HGE component prediction alignment failed")
        regimes = (
            base.set_index(["row_id", "horizon"])
            .loc[
                list(zip(linear["row_id"], [horizon] * len(linear), strict=True)),
                "hurst_regime_label",
            ]
            .astype(str)
            .to_numpy()
        )
        tree_weight = np.where(
            regimes == "persistent", 0.7, np.where(regimes == "mean_reverting", 0.3, 0.5)
        )
        if task.endswith("classification"):
            lp = np.vstack(
                [list(json.loads(value).values()) for value in linear["y_pred_proba_json"]]
            )
            tp = np.vstack(
                [list(json.loads(value).values()) for value in tree["y_pred_proba_json"]]
            )
            hp = tree_weight[:, None] * tp + (1 - tree_weight[:, None]) * lp
            pred = CLASS_ORDER[np.argmax(hp, axis=1)]
            proba = hp
        else:
            pred = (
                tree_weight * tree["y_pred"].to_numpy()
                + (1 - tree_weight) * linear["y_pred"].to_numpy()
            )
            proba = None
        candidate_id = f"{task}_h{horizon}_hge_rule_based"
        source = (
            base.set_index(["row_id", "horizon"])
            .loc[list(zip(linear["row_id"], [horizon] * len(linear), strict=True))]
            .reset_index()
        )
        hge_rows = _prediction_rows(
            source, task, horizon, candidate_id, "hge_rule_based", str(fold_id), pred, proba, True
        )
        outer = pd.concat([outer, hge_rows], ignore_index=True)
        value = _metric(task, hge_rows["y_true"].to_numpy(), pred, proba)
        metric_rows.append(
            {
                "task": task,
                "horizon": horizon,
                "candidate_model_id": candidate_id,
                "model_family": "hge_rule_based",
                "fold_id": fold_id,
                "metric_name": "macro_f1"
                if task.endswith("classification")
                else ("QLIKE" if task == "volatility_regression" else "RMSE"),
                "metric_value": value,
                "metric_valid": np.isfinite(value),
                "n_validation": len(hge_rows),
            }
        )

    metrics = pd.DataFrame(metric_rows)
    aggregation = (
        metrics.groupby(
            ["task", "horizon", "candidate_model_id", "model_family", "metric_name"], as_index=False
        )
        .apply(
            lambda group: pd.Series(
                {
                    "metric_value": np.average(
                        group["metric_value"], weights=group["n_validation"]
                    ),
                    "n_validation": group["n_validation"].sum(),
                    "metric_valid": bool(group["metric_valid"].all()),
                }
            ),
            include_groups=False,
        )
        .reset_index(drop=True)
    )
    selected: list[dict[str, object]] = []
    locked_parts: list[pd.DataFrame] = []
    for (task, horizon), candidates in aggregation.groupby(["task", "horizon"], sort=False):
        candidates = candidates[candidates["metric_valid"]].copy()
        candidates["selection_score"] = candidates["metric_value"].map(
            lambda value, selected_task=str(task): _better(selected_task, float(value))
        )
        candidates = candidates.sort_values(
            ["selection_score", "model_family", "candidate_model_id"], ascending=[False, True, True]
        )
        choice = candidates.iloc[0]
        selected_id = str(choice["candidate_model_id"])
        family = str(choice["model_family"])
        target = TASK_TARGET[str(task)]
        horizon_data = base[(base["horizon"] == horizon) & base[target].notna()]
        train = horizon_data[
            (horizon_data["date"] < locked_start)
            & (pd.to_datetime(horizon_data["label_end_date"]) < locked_start)
        ].copy()
        locked = horizon_data[horizon_data["date"] >= locked_start].copy()
        classification = str(task).endswith("classification")
        component_hashes: list[str] = []
        if family == "hge_rule_based":
            component_predictions: dict[str, tuple[np.ndarray, np.ndarray | None]] = {}
            for component_family in ["linear_regularized", "tree_ensemble"]:
                model = Pipeline(
                    [
                        ("prep", _preprocessor(train, feature_cols)),
                        ("model", _model(component_family, classification, seed)),
                    ]
                )
                model.fit(
                    train[feature_cols],
                    train[target].astype(int) if classification else train[target],
                )
                proba = _probabilities(model, locked[feature_cols]) if classification else None
                pred = (
                    CLASS_ORDER[np.argmax(proba, axis=1)]
                    if proba is not None
                    else model.predict(locked[feature_cols])
                )
                component_predictions[component_family] = (pred, proba)
                artifact = model_dir / "selected" / f"{selected_id}_{component_family}.joblib"
                artifact.parent.mkdir(parents=True, exist_ok=True)
                joblib.dump(model, artifact)
                component_hashes.append(sha256_file(artifact))
            regimes = locked["hurst_regime_label"].astype(str).to_numpy()
            weight = np.where(
                regimes == "persistent", 0.7, np.where(regimes == "mean_reverting", 0.3, 0.5)
            )
            if classification:
                component_lp = component_predictions["linear_regularized"][1]
                component_tp = component_predictions["tree_ensemble"][1]
                if component_lp is None or component_tp is None:
                    raise RuntimeError("HGE classification components must expose probabilities")
                proba = weight[:, None] * component_tp + (1 - weight[:, None]) * component_lp
                pred = CLASS_ORDER[np.argmax(proba, axis=1)]
            else:
                proba = None
                pred = (
                    weight * component_predictions["tree_ensemble"][0]
                    + (1 - weight) * component_predictions["linear_regularized"][0]
                )
            hge_artifact = model_dir / "selected" / f"{selected_id}.json"
            atomic_json(
                hge_artifact,
                {
                    "type": "rule_based_hge_v1",
                    "components": ["linear_regularized", "tree_ensemble"],
                    "component_hashes": component_hashes,
                    "uses_locked_test_for_selection": False,
                },
            )
            model_path = hge_artifact
        else:
            model = Pipeline(
                [
                    ("prep", _preprocessor(train, feature_cols)),
                    ("model", _model(family, classification, seed)),
                ]
            )
            model.fit(
                train[feature_cols], train[target].astype(int) if classification else train[target]
            )
            proba = _probabilities(model, locked[feature_cols]) if classification else None
            pred = (
                CLASS_ORDER[np.argmax(proba, axis=1)]
                if proba is not None
                else model.predict(locked[feature_cols])
            )
            model_path = model_dir / "selected" / f"{selected_id}.joblib"
            model_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, model_path)
        locked_frame = _prediction_rows(
            locked, str(task), int(horizon), selected_id, family, "locked_test", pred, proba, False
        )
        locked_frame = locked_frame.rename(columns={"candidate_model_id": "selected_model_id"})
        locked_parts.append(locked_frame)
        selected.append(
            {
                "task": task,
                "horizon": int(horizon),
                "selected_model_id": selected_id,
                "selected_model_family": family,
                "selected_model_artifact_path": str(model_path.relative_to(paths.root)),
                "selected_model_artifact_hash": sha256_file(model_path),
                "candidate_model_status": "candidate_validated",
                "primary_metric_name": choice["metric_name"],
                "primary_metric_value": float(choice["metric_value"]),
                "primary_metric_valid": True,
                "selection_uses_locked_test": False,
                "final_refit_used": True,
            }
        )

    selected_map: dict[str, Any] = {
        "feature_set_id": config.features["feature_set_id"],
        "target_policy_id": config.targets["target_policy_id"],
        "validation_metric_aggregation_policy_id": "metric_specific_aggregation_v1",
        "selected_models": selected,
        "frozen_at_utc": datetime.now(UTC).isoformat(),
        "selection_uses_locked_test": False,
        "immutable_after_frozen": True,
    }
    selected_map["selected_model_map_hash"] = decision_hash(selected_map)
    selected_map_path = metadata / "phase4_selected_model_map.json"
    atomic_json(selected_map_path, selected_map)
    locked_predictions = pd.concat(locked_parts, ignore_index=True)
    locked_predictions["selected_model_map_hash"] = selected_map["selected_model_map_hash"]
    if "y_true" in locked_predictions.columns:
        raise RuntimeError("Locked test prediction artifact must not contain y_true")
    outer_path = predictions_dir / "phase4_outer_validation_predictions.parquet"
    locked_path = predictions_dir / "phase4_locked_test_predictions.parquet"
    write_parquet(outer_path, outer)
    write_parquet(locked_path, locked_predictions)
    metric_path = metadata / "phase4_validation_metrics_report.csv"
    aggregation_path = metadata / "phase4_validation_metric_aggregation_report.csv"
    write_csv(metric_path, metrics)
    write_csv(aggregation_path, aggregation)
    fold_path = metadata / "phase4_fold_assignment.parquet"
    write_parquet(fold_path, pd.DataFrame(fold_rows))
    registry_path = metadata / "phase4_model_registry.json"
    atomic_json(
        registry_path, {"model_record_level": "candidate_model_instance", "models": registry_rows}
    )
    prediction_manifest = {
        "selected_model_map_hash": selected_map["selected_model_map_hash"],
        "locked_test_predictions": {
            "path": str(locked_path.relative_to(paths.root)),
            "hash": sha256_file(locked_path),
            "contains_y_true": False,
            "metrics_computed": False,
            "generated_after_model_selection": True,
        },
        "outer_validation_predictions": {
            "path": str(outer_path.relative_to(paths.root)),
            "hash": sha256_file(outer_path),
        },
    }
    prediction_manifest_path = predictions_dir / "phase4_prediction_manifest.json"
    atomic_json(prediction_manifest_path, prediction_manifest)
    return {
        "selected_map": selected_map_path,
        "locked_predictions": locked_path,
        "outer_predictions": outer_path,
        "metrics": metric_path,
        "folds": fold_path,
        "registry": registry_path,
        "prediction_manifest": prediction_manifest_path,
    }
