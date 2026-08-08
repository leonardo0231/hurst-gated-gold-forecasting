from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import SplitConfig


@dataclass(frozen=True)
class WalkForwardFold:
    fold_id: str
    train_indices: np.ndarray
    validation_indices: np.ndarray
    validation_start_row: int
    validation_end_row: int


def split_development_and_locked_test(
    dataset: pd.DataFrame, config: SplitConfig
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    ordered = dataset.sort_values("row_id").reset_index(drop=True)
    cutoff_position = int(len(ordered) * (1.0 - config.locked_test_fraction))
    cutoff_position = min(
        max(cutoff_position, config.min_train_rows + config.min_validation_rows), len(ordered) - 1
    )
    locked_start_row = int(ordered.iloc[cutoff_position]["row_id"])
    development = ordered[
        (ordered["row_id"] < locked_start_row) & (ordered["label_end_index"] < locked_start_row)
    ].reset_index(drop=True)
    locked = ordered[ordered["row_id"] >= locked_start_row].reset_index(drop=True)
    if development.empty or locked.empty:
        raise ValueError("Development/locked-test split produced an empty partition")
    return development, locked, locked_start_row


def build_purged_walk_forward_folds(
    development: pd.DataFrame, config: SplitConfig
) -> list[WalkForwardFold]:
    ordered = development.sort_values("row_id").reset_index(drop=True)
    n_rows = len(ordered)
    if n_rows < config.min_train_rows + config.min_validation_rows:
        raise ValueError("Not enough development rows for walk-forward validation")
    available = n_rows - config.min_train_rows
    validation_size = max(config.min_validation_rows, available // config.n_walk_forward_folds)
    folds: list[WalkForwardFold] = []
    for fold_number in range(config.n_walk_forward_folds):
        validation_start_pos = config.min_train_rows + fold_number * validation_size
        if validation_start_pos >= n_rows:
            break
        validation_end_pos = (
            n_rows
            if fold_number == config.n_walk_forward_folds - 1
            else min(n_rows, validation_start_pos + validation_size)
        )
        validation_positions = np.arange(validation_start_pos, validation_end_pos, dtype=int)
        if len(validation_positions) < config.min_validation_rows:
            continue
        validation_start_row = int(ordered.iloc[validation_start_pos]["row_id"])
        validation_end_row = int(ordered.iloc[validation_end_pos - 1]["row_id"])
        train_mask = (ordered["row_id"] < validation_start_row) & (
            ordered["label_end_index"] < validation_start_row
        )
        train_positions = np.flatnonzero(train_mask.to_numpy())
        if len(train_positions) < config.min_train_rows:
            continue
        train_classes = ordered.iloc[train_positions]["direction_binary"].nunique(dropna=True)
        validation_classes = ordered.iloc[validation_positions]["direction_binary"].nunique(
            dropna=True
        )
        if train_classes < 2 or validation_classes < 2:
            continue
        folds.append(
            WalkForwardFold(
                fold_id=f"wf_{fold_number + 1:02d}",
                train_indices=train_positions,
                validation_indices=validation_positions,
                validation_start_row=validation_start_row,
                validation_end_row=validation_end_row,
            )
        )
    if len(folds) < 2:
        raise ValueError("At least two valid walk-forward folds are required")
    return folds


def assert_no_label_overlap(development: pd.DataFrame, folds: list[WalkForwardFold]) -> None:
    for fold in folds:
        train = development.iloc[fold.train_indices]
        if not (train["label_end_index"] < fold.validation_start_row).all():
            raise AssertionError(f"Label overlap detected in {fold.fold_id}")
