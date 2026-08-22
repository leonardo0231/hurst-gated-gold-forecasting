"""Chronology-safe probability calibration utilities for research experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression


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
) -> SigmoidCalibrator:
    """Fit sigmoid calibration and prove the prediction window is strictly later."""

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
    if np.unique(truth).size != 2:
        raise ValueError("Sigmoid calibration requires both classes")
    if np.any((probability < 0.0) | (probability > 1.0)):
        raise ValueError("Calibration probabilities must lie in [0, 1]")
    model = LogisticRegression(C=1_000_000.0, solver="lbfgs", random_state=seed)
    model.fit(probability.reshape(-1, 1), truth)
    return SigmoidCalibrator(model=model)
