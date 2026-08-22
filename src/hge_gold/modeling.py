from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from .config import ModelConfig
from .evaluation import ClassificationMetrics, classification_metrics, tune_probability_threshold
from .splits import WalkForwardFold


@dataclass(frozen=True)
class Candidate:
    name: str
    estimator: Pipeline


@dataclass
class ModelingResult:
    selected_strategy: str
    selected_candidate: str
    threshold: float
    locked_probability_up: np.ndarray
    validation_metrics: dict[str, Any]
    candidate_metrics: list[dict[str, Any]]
    fold_metrics: list[dict[str, Any]]
    bundle: dict[str, Any]


def _candidate_pipelines(config: ModelConfig) -> list[Candidate]:
    seed = config.random_seed
    trees = 80 if config.fast_mode else 320
    return [
        Candidate(
            "logistic_c02",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                    ("scale", StandardScaler()),
                    (
                        "model",
                        LogisticRegression(
                            C=0.2, class_weight="balanced", max_iter=2000, random_state=seed
                        ),
                    ),
                ]
            ),
        ),
        Candidate(
            "logistic_c10",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                    ("scale", StandardScaler()),
                    (
                        "model",
                        LogisticRegression(
                            C=1.0, class_weight="balanced", max_iter=2000, random_state=seed
                        ),
                    ),
                ]
            ),
        ),
        Candidate(
            "random_forest",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                    (
                        "model",
                        RandomForestClassifier(
                            n_estimators=trees,
                            max_depth=9,
                            min_samples_leaf=8,
                            max_features="sqrt",
                            class_weight="balanced_subsample",
                            random_state=seed,
                            n_jobs=1,
                        ),
                    ),
                ]
            ),
        ),
        Candidate(
            "extra_trees",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                    (
                        "model",
                        ExtraTreesClassifier(
                            n_estimators=trees,
                            max_depth=11,
                            min_samples_leaf=6,
                            max_features="sqrt",
                            class_weight="balanced",
                            random_state=seed,
                            n_jobs=1,
                        ),
                    ),
                ]
            ),
        ),
        Candidate(
            "hist_gradient_boosting",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                    (
                        "model",
                        HistGradientBoostingClassifier(
                            learning_rate=0.055,
                            max_iter=90 if config.fast_mode else 220,
                            max_leaf_nodes=15,
                            min_samples_leaf=18,
                            l2_regularization=1.2,
                            random_state=seed,
                        ),
                    ),
                ]
            ),
        ),
    ]


def _positive_probability(model: BaseEstimator, frame: pd.DataFrame) -> np.ndarray:
    probabilities = np.asarray(model.predict_proba(frame), dtype=float)
    classes = np.asarray(model.classes_, dtype=int)
    match = np.flatnonzero(classes == 1)
    if len(match) != 1:
        raise RuntimeError("Binary classifier does not expose class 1 probability")
    return probabilities[:, int(match[0])]


def _fit(model: Pipeline, x: pd.DataFrame, y: np.ndarray) -> Pipeline:
    weights = compute_sample_weight(class_weight="balanced", y=y)
    try:
        model.fit(x, y, model__sample_weight=weights)
    except TypeError:
        model.fit(x, y)
    return model


def _meta_columns(frame: pd.DataFrame) -> list[str]:
    preferred = [
        "hurst_regime",
        "volatility_zscore",
        "trend_efficiency_20",
        "trend_regime",
        "rsi_14",
        "macd_histogram",
        "vol_ratio_20_63",
    ]
    return [column for column in preferred if column in frame.columns]


def _make_meta_frame(
    base_probabilities: dict[str, np.ndarray],
    source: pd.DataFrame,
    candidate_names: list[str],
    meta_columns: list[str],
) -> pd.DataFrame:
    payload: dict[str, Any] = {}
    for name in candidate_names:
        probability = base_probabilities[name]
        payload[f"p_up__{name}"] = probability
        payload[f"confidence__{name}"] = np.abs(probability - 0.5)
    for column in meta_columns:
        payload[f"regime__{column}"] = source[column].to_numpy(dtype=float)
    return pd.DataFrame(payload, index=source.index)


def _candidate_stability_summary(
    fold_metrics: list[dict[str, Any]],
    candidate_name: str,
    fold_ids: set[str],
) -> dict[str, float]:
    rows = [
        row
        for row in fold_metrics
        if row["candidate"] == candidate_name and row["fold_id"] in fold_ids
    ]
    if not rows:
        raise RuntimeError(f"No fold metrics are available for candidate {candidate_name!r}")

    balanced_accuracy = np.asarray(
        [float(row["balanced_accuracy"]) for row in rows],
        dtype=float,
    )
    macro_f1 = np.asarray(
        [float(row["macro_f1"]) for row in rows],
        dtype=float,
    )
    roc_auc = np.asarray(
        [float(row["roc_auc"]) for row in rows],
        dtype=float,
    )

    return {
        "stability_median_balanced_accuracy": float(np.median(balanced_accuracy)),
        "stability_median_macro_f1": float(np.median(macro_f1)),
        "stability_balanced_accuracy_std": float(np.std(balanced_accuracy, ddof=0)),
        "stability_median_roc_auc": float(np.median(roc_auc)),
    }


def _candidate_stability_key(summary: dict[str, float]) -> tuple[float, float, float]:
    return (
        summary["stability_median_balanced_accuracy"],
        summary["stability_median_macro_f1"],
        -summary["stability_balanced_accuracy_std"],
    )


def _purged_meta_train_mask(
    development: pd.DataFrame,
    common_mask: np.ndarray,
    oof_fold: np.ndarray,
    selection_fold: WalkForwardFold,
) -> np.ndarray:
    """Exclude meta labels whose information interval reaches the selection fold."""
    return np.asarray(
        common_mask
        & (oof_fold != selection_fold.fold_id)
        & (
            development["label_end_index"].to_numpy(dtype=int) < selection_fold.validation_start_row
        ),
        dtype=bool,
    )


def train_and_predict(
    development: pd.DataFrame,
    locked: pd.DataFrame,
    feature_columns: list[str],
    folds: list[WalkForwardFold],
    config: ModelConfig,
    allow_meta_model: bool = True,
) -> ModelingResult:
    x_dev = development[feature_columns]
    y_dev = development["direction_binary"].to_numpy(dtype=int)
    x_locked = locked[feature_columns]
    candidates = _candidate_pipelines(config)
    names = [candidate.name for candidate in candidates]
    oof_probabilities = {name: np.full(len(development), np.nan, dtype=float) for name in names}
    oof_fold = np.full(len(development), "", dtype=object)
    fold_metrics: list[dict[str, Any]] = []

    for fold in folds:
        x_train = x_dev.iloc[fold.train_indices]
        y_train = y_dev[fold.train_indices]
        x_validation = x_dev.iloc[fold.validation_indices]
        y_validation = y_dev[fold.validation_indices]
        for candidate in _candidate_pipelines(config):
            model = _fit(candidate.estimator, x_train, y_train)
            probability = _positive_probability(model, x_validation)
            oof_probabilities[candidate.name][fold.validation_indices] = probability
            metrics = classification_metrics(y_validation, probability, 0.5)
            fold_metrics.append(
                {
                    "fold_id": fold.fold_id,
                    "candidate": candidate.name,
                    "n_validation": len(fold.validation_indices),
                    "balanced_accuracy": metrics["balanced_accuracy"],
                    "macro_f1": metrics["macro_f1"],
                    "roc_auc": metrics["roc_auc"],
                }
            )
        oof_fold[fold.validation_indices] = fold.fold_id

    common_mask = np.ones(len(development), dtype=bool)
    for probability in oof_probabilities.values():
        common_mask &= np.isfinite(probability)
    if common_mask.sum() < 100:
        raise RuntimeError("Insufficient out-of-fold predictions for model selection")

    selection_fold = folds[-1]
    last_fold_id = selection_fold.fold_id
    selection_mask = common_mask & (oof_fold == last_fold_id)
    meta_train_mask = _purged_meta_train_mask(
        development,
        common_mask,
        oof_fold,
        selection_fold,
    )

    candidate_selection_fold_ids = {fold.fold_id for fold in folds[:-1]}

    if not candidate_selection_fold_ids:
        raise RuntimeError(
            "At least two walk-forward folds are required for stable model selection"
        )

    if selection_mask.sum() < 30 or (allow_meta_model and meta_train_mask.sum() < 60):
        raise RuntimeError("Insufficient rows for leakage-safe meta-model selection")

    candidate_metrics: list[dict[str, Any]] = []
    best_candidate = names[0]
    best_key = (-np.inf, -np.inf, -np.inf)
    best_candidate_threshold = 0.5
    for name in names:
        stability = _candidate_stability_summary(
            fold_metrics,
            name,
            candidate_selection_fold_ids,
        )

        threshold, metrics = tune_probability_threshold(
            y_dev[selection_mask],
            oof_probabilities[name][selection_mask],
            config.probability_threshold_min,
            config.probability_threshold_max,
            config.probability_threshold_steps,
        )
        row = {
            "candidate": name,
            "selection_fold": last_fold_id,
            "candidate_selection_policy": "median_preselection_folds_v2_1",
            "candidate_selection_fold_count": len(candidate_selection_fold_ids),
            **stability,
            **metrics,
        }
        candidate_metrics.append(row)
        key = _candidate_stability_key(stability)
        if key > best_key:
            best_key = key
            best_candidate = name
            best_candidate_threshold = threshold

    best_metrics = next(row for row in candidate_metrics if row["candidate"] == best_candidate)
    meta_columns = _meta_columns(development)
    all_meta = _make_meta_frame(oof_probabilities, development, names, meta_columns)
    meta_threshold = 0.5
    meta_metrics: ClassificationMetrics | None = None
    use_meta_model = False
    if allow_meta_model:
        selection_meta_model = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        C=0.5,
                        class_weight="balanced",
                        max_iter=2000,
                        random_state=config.random_seed,
                    ),
                ),
            ]
        )
        _fit(
            selection_meta_model,
            all_meta.loc[meta_train_mask],
            y_dev[meta_train_mask],
        )
        meta_probability_selection = _positive_probability(
            selection_meta_model, all_meta.loc[selection_mask]
        )
        meta_threshold, meta_metrics = tune_probability_threshold(
            y_dev[selection_mask],
            meta_probability_selection,
            config.probability_threshold_min,
            config.probability_threshold_max,
            config.probability_threshold_steps,
        )
        use_meta_model = float(meta_metrics["balanced_accuracy"]) + config.gate_tolerance >= float(
            best_metrics["balanced_accuracy"]
        ) and float(meta_metrics["macro_f1"]) + 0.01 >= float(best_metrics["macro_f1"])

    fitted_models: dict[str, Pipeline] = {}
    locked_base_probabilities: dict[str, np.ndarray] = {}
    for candidate in _candidate_pipelines(config):
        fitted = _fit(candidate.estimator, x_dev, y_dev)
        fitted_models[candidate.name] = fitted
        locked_base_probabilities[candidate.name] = _positive_probability(fitted, x_locked)

    final_meta_model: Pipeline | None = None
    validation_metrics: dict[str, Any]

    if use_meta_model:
        final_meta_model = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        C=0.5,
                        class_weight="balanced",
                        max_iter=2000,
                        random_state=config.random_seed,
                    ),
                ),
            ]
        )
        _fit(final_meta_model, all_meta.loc[common_mask], y_dev[common_mask])
        locked_meta = _make_meta_frame(locked_base_probabilities, locked, names, meta_columns)
        locked_probability = _positive_probability(final_meta_model, locked_meta)
        selected_strategy = "stacked_meta_classifier"
        threshold = meta_threshold
        if meta_metrics is None:  # pragma: no cover - guarded by use_meta_model
            raise RuntimeError("Meta-model metrics are unavailable")
        validation_metrics = dict(meta_metrics)
    else:
        locked_probability = locked_base_probabilities[best_candidate]
        selected_strategy = "best_base_model"
        threshold = best_candidate_threshold
        validation_metrics = {
            key: value for key, value in best_metrics.items() if key != "candidate"
        }

    bundle = {
        "version": "2.0",
        "selected_strategy": selected_strategy,
        "selected_candidate": best_candidate,
        "threshold": threshold,
        "feature_columns": feature_columns,
        "meta_columns": meta_columns,
        "candidate_names": names,
        "base_models": fitted_models,
        "meta_model": final_meta_model,
        "meta_model_allowed": allow_meta_model,
        "meta_training_rows": int(meta_train_mask.sum()) if allow_meta_model else 0,
        "meta_training_purge_boundary_row": selection_fold.validation_start_row,
        "meta_training_max_label_end_index": (
            int(development.loc[meta_train_mask, "label_end_index"].max())
            if allow_meta_model and meta_train_mask.any()
            else None
        ),
        "candidate_selection_policy": "median_preselection_folds_v2_1",
        "candidate_selection_fold_ids": [fold.fold_id for fold in folds[:-1]],
        "selection_fold": last_fold_id,
        "strategy_selection_fold": last_fold_id,
        "selection_uses_locked_test": False,
    }
    return ModelingResult(
        selected_strategy=selected_strategy,
        selected_candidate=best_candidate,
        threshold=float(threshold),
        locked_probability_up=locked_probability,
        validation_metrics=validation_metrics,
        candidate_metrics=candidate_metrics,
        fold_metrics=fold_metrics,
        bundle=bundle,
    )


def save_model_bundle(path: str, bundle: dict[str, Any]) -> None:
    joblib.dump(bundle, path)
