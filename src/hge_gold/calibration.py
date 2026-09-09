"""Chronology-safe probability calibration utilities for research experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression

from .research_validation import calibration_label_overlap_mask


@dataclass(frozen=True)
class SigmoidEligibility:
    """Deterministic eligibility decision for a past-only sigmoid candidate."""

    eligible: bool
    reason: str
    retained_count: int
    class_count: int
    finite_probability: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "eligible": self.eligible,
            "reason": self.reason,
            "retained_count": self.retained_count,
            "class_count": self.class_count,
            "finite_probability": self.finite_probability,
        }


def assess_sigmoid_eligibility(
    probability_up: np.ndarray,
    y_true: np.ndarray,
    *,
    minimum_samples: int,
) -> SigmoidEligibility:
    """Apply the preregistered deterministic sigmoid eligibility rule."""

    probability = np.asarray(probability_up, dtype=float)
    truth = np.asarray(y_true, dtype=int)
    if probability.ndim != 1 or truth.ndim != 1 or len(probability) != len(truth):
        raise ValueError("Calibration probability and target arrays must be aligned vectors")
    if minimum_samples < 1:
        raise ValueError("minimum_samples must be positive")
    finite = bool(np.isfinite(probability).all())
    class_count = int(np.unique(truth).size)
    if len(truth) < minimum_samples:
        reason = "insufficient_calibration_samples"
        eligible = False
    elif class_count < 2:
        reason = "single_class_after_purge"
        eligible = False
    elif not finite or np.any((probability < 0.0) | (probability > 1.0)):
        reason = "invalid_probability_data"
        eligible = False
    else:
        reason = "eligible"
        eligible = True
    return SigmoidEligibility(
        eligible=eligible,
        reason=reason,
        retained_count=int(len(truth)),
        class_count=class_count,
        finite_probability=finite,
    )


@dataclass(frozen=True)
class SigmoidCalibrator:
    """Platt-style calibrator fitted on a past-only calibration slice."""

    model: LogisticRegression

    def predict(self, probability_up: np.ndarray) -> np.ndarray:
        values = np.asarray(probability_up, dtype=float).reshape(-1, 1)
        return np.asarray(self.model.predict_proba(values)[:, 1], dtype=float)


def fit_past_only_sigmoid(
    probability_up: np.ndarray,
    y_true: np.ndarray,
    *,
    calibration_row_ids: np.ndarray,
    prediction_row_ids: np.ndarray,
    seed: int,
    calibration_label_end_ids: np.ndarray | None = None,
) -> SigmoidCalibrator:
    """Fit sigmoid calibration and prove its labels cannot reach prediction time.

    ``calibration_label_end_ids`` is optional only for backward compatibility with the
    historical v1/v2 callers.  The repaired v3 caller always supplies it; when supplied,
    the calibrator independently enforces the canonical label-interval invariant.
    """

    probability = np.asarray(probability_up, dtype=float)
    truth = np.asarray(y_true, dtype=int)
    calibration_rows = np.asarray(calibration_row_ids, dtype=int)
    prediction_rows = np.asarray(prediction_row_ids, dtype=int)
    if not (len(probability) == len(truth) == len(calibration_rows)):
        raise ValueError("Calibration arrays must have identical lengths")
    if len(prediction_rows) == 0 or len(calibration_rows) == 0:
        raise ValueError("Calibration and prediction windows cannot be empty")
    if int(calibration_rows.max()) >= int(prediction_rows.min()):
        raise ValueError("Prediction rows must be strictly after calibration rows")
    if calibration_label_end_ids is None:
        label_ends = calibration_rows.copy()
    else:
        label_ends = np.asarray(calibration_label_end_ids, dtype=int)
        if len(label_ends) != len(calibration_rows):
            raise ValueError("Calibration label endpoints must align with calibration rows")
        if np.any(label_ends < calibration_rows):
            raise ValueError("Calibration label endpoints must be at or after calibration rows")
        if calibration_label_overlap_mask(label_ends, prediction_rows).any():
            raise ValueError("Calibration label intervals overlap prediction window")
    if np.unique(truth).size != 2:
        raise ValueError("Sigmoid calibration requires both classes")
    if np.any((probability < 0.0) | (probability > 1.0)):
        raise ValueError("Calibration probabilities must lie in [0, 1]")
    model = LogisticRegression(
        C=1_000_000.0,
        penalty="l2",
        solver="lbfgs",
        fit_intercept=True,
        max_iter=100,
        tol=1e-4,
        random_state=seed,
    )
    model.fit(probability.reshape(-1, 1), truth)
    return SigmoidCalibrator(model=model)
