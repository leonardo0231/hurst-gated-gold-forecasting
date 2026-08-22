from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

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
    train_start_row: int | None = None
    train_end_row: int | None = None
    n_train_raw: int = 0
    n_train: int = 0
    n_validation: int = 0
    purged_count: int = 0
    embargoed_count: int = 0


@dataclass(frozen=True)
class CPCVSplit:
    """One development-only combinatorial purged cross-validation split."""

    split_id: str
    train_indices: np.ndarray
    test_indices: np.ndarray
    test_group_ids: tuple[int, ...]
    train_start_row: int | None
    train_end_row: int | None
    test_start_row: int
    test_end_row: int
    n_train_raw: int
    n_train: int
    n_test: int
    purged_count: int
    embargoed_count: int


def _validate_event_frame(frame: pd.DataFrame) -> None:
    required = {"row_id", "label_end_index"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Event frame is missing required columns: {sorted(missing)}")
    if frame["row_id"].duplicated().any():
        raise ValueError("row_id values must be unique")
    if (frame["label_end_index"] < frame["row_id"]).any():
        raise ValueError("Every label_end_index must be at or after its row_id")


def _configured_locked_start_row(ordered: pd.DataFrame, value: str) -> int:
    if "date" not in ordered.columns:
        try:
            boundary_row = int(value)
        except ValueError as exc:
            raise ValueError(
                "A date-based locked_test_start requires a date column; otherwise provide a row id"
            ) from exc
        candidates = ordered.loc[ordered["row_id"] >= boundary_row, "row_id"]
    else:
        dates = pd.to_datetime(ordered["date"], errors="raise")
        boundary = pd.Timestamp(value)
        series_timezone = dates.dt.tz
        if series_timezone is None and boundary.tzinfo is not None:
            boundary = boundary.tz_localize(None)
        elif series_timezone is not None and boundary.tzinfo is None:
            boundary = boundary.tz_localize(series_timezone)
        elif series_timezone is not None:
            boundary = boundary.tz_convert(series_timezone)
        candidates = ordered.loc[dates >= boundary, "row_id"]
    if candidates.empty:
        raise ValueError("locked_test_start is after the final eligible observation")
    return int(candidates.iloc[0])


def split_development_and_locked_test(
    dataset: pd.DataFrame, config: SplitConfig
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Split chronologically and purge development events that enter the holdout.

    A configured ``locked_test_start`` is an immutable calendar (or row-id when no date
    column is present) boundary, so appending observations cannot move the holdout.  The
    historical fraction-based behavior is retained when the value is null.
    """

    ordered = dataset.sort_values("row_id").reset_index(drop=True)
    _validate_event_frame(ordered)
    configured_start = getattr(config, "locked_test_start", None)
    if configured_start is None:
        cutoff_position = int(len(ordered) * (1.0 - config.locked_test_fraction))
        cutoff_position = min(
            max(cutoff_position, config.min_train_rows + config.min_validation_rows),
            len(ordered) - 1,
        )
        locked_start_row = int(ordered.iloc[cutoff_position]["row_id"])
    else:
        locked_start_row = _configured_locked_start_row(ordered, configured_start)

    development = ordered[
        (ordered["row_id"] < locked_start_row) & (ordered["label_end_index"] < locked_start_row)
    ].reset_index(drop=True)
    locked = ordered[ordered["row_id"] >= locked_start_row].reset_index(drop=True)
    if development.empty or locked.empty:
        raise ValueError("Development/locked-test split produced an empty partition")
    return development, locked, locked_start_row


def _training_mask(
    ordered: pd.DataFrame,
    validation_start_pos: int,
    embargo_rows: int,
) -> tuple[np.ndarray, int, int, int]:
    validation_start_row = int(ordered.iloc[validation_start_pos]["row_id"])
    raw_mask = ordered["row_id"].to_numpy(dtype=int) < validation_start_row
    overlap_mask = ordered["label_end_index"].to_numpy(dtype=int) >= validation_start_row
    embargo_mask = np.zeros(len(ordered), dtype=bool)
    if embargo_rows:
        embargo_start = max(0, validation_start_pos - embargo_rows)
        embargo_mask[embargo_start:validation_start_pos] = True
    purged_count = int(np.count_nonzero(raw_mask & overlap_mask))
    embargoed_count = int(np.count_nonzero(raw_mask & ~overlap_mask & embargo_mask))
    train_mask = raw_mask & ~overlap_mask & ~embargo_mask
    return train_mask, int(np.count_nonzero(raw_mask)), purged_count, embargoed_count


def _has_two_classes(frame: pd.DataFrame, positions: np.ndarray) -> bool:
    if "direction_binary" not in frame.columns:
        return True
    return bool(frame.iloc[positions]["direction_binary"].nunique(dropna=True) >= 2)


def build_purged_walk_forward_folds(
    development: pd.DataFrame, config: SplitConfig
) -> list[WalkForwardFold]:
    """Build exactly the requested number of chronological, purged folds.

    The first validation boundary is the earliest boundary that retains the configured
    minimum training count *after* purge and the optional pre-validation safety gap.
    ``embargo_rows`` is recorded as a conservative pre-validation gap here; a conventional
    post-test embargo is applied by :func:`build_purged_cpcv_splits` when future training
    observations are possible.
    """

    ordered = development.sort_values("row_id").reset_index(drop=True)
    _validate_event_frame(ordered)
    n_rows = len(ordered)
    n_folds = int(config.n_walk_forward_folds)
    min_validation = int(config.min_validation_rows)
    embargo_rows = int(getattr(config, "embargo_rows", 0))
    minimum_required = config.min_train_rows + n_folds * min_validation
    if n_rows < minimum_required:
        raise ValueError("Not enough development rows for the requested walk-forward folds")

    latest_start = n_rows - n_folds * min_validation
    first_validation_start: int | None = None
    for position in range(config.min_train_rows, latest_start + 1):
        mask, _, _, _ = _training_mask(ordered, position, embargo_rows)
        if np.count_nonzero(mask) >= config.min_train_rows:
            first_validation_start = position
            break
    if first_validation_start is None:
        raise ValueError(
            "No validation boundary retains the configured minimum purged training rows"
        )

    validation_blocks = np.array_split(
        np.arange(first_validation_start, n_rows, dtype=int),
        n_folds,
    )
    if any(len(block) < min_validation for block in validation_blocks):
        raise ValueError("Unable to construct the requested number of minimum-size folds")

    folds: list[WalkForwardFold] = []
    for fold_number, validation_positions in enumerate(validation_blocks, start=1):
        validation_start_pos = int(validation_positions[0])
        validation_start_row = int(ordered.iloc[validation_start_pos]["row_id"])
        validation_end_row = int(ordered.iloc[int(validation_positions[-1])]["row_id"])
        train_mask, n_train_raw, purged_count, embargoed_count = _training_mask(
            ordered,
            validation_start_pos,
            embargo_rows,
        )
        train_positions = np.flatnonzero(train_mask)
        if len(train_positions) < config.min_train_rows:
            raise ValueError(f"Fold wf_{fold_number:02d} has too few purged training rows")
        if not _has_two_classes(ordered, train_positions):
            raise ValueError(
                f"Fold wf_{fold_number:02d} training partition has fewer than two classes"
            )
        if not _has_two_classes(ordered, validation_positions):
            raise ValueError(
                f"Fold wf_{fold_number:02d} validation partition has fewer than two classes"
            )
        folds.append(
            WalkForwardFold(
                fold_id=f"wf_{fold_number:02d}",
                train_indices=train_positions,
                validation_indices=validation_positions,
                validation_start_row=validation_start_row,
                validation_end_row=validation_end_row,
                train_start_row=int(ordered.iloc[int(train_positions[0])]["row_id"]),
                train_end_row=int(ordered.iloc[int(train_positions[-1])]["row_id"]),
                n_train_raw=n_train_raw,
                n_train=len(train_positions),
                n_validation=len(validation_positions),
                purged_count=purged_count,
                embargoed_count=embargoed_count,
            )
        )
    return folds


def _interval_overlap_mask(
    frame: pd.DataFrame,
    candidate_indices: np.ndarray,
    reference_indices: np.ndarray,
) -> np.ndarray:
    """Return which candidate event intervals overlap any reference interval."""

    candidate_indices = np.asarray(candidate_indices, dtype=int)
    reference_indices = np.asarray(reference_indices, dtype=int)
    if candidate_indices.size == 0 or reference_indices.size == 0:
        return np.zeros(candidate_indices.size, dtype=bool)

    reference = frame.iloc[reference_indices][["row_id", "label_end_index"]].to_numpy(dtype=int)
    reference = reference[np.argsort(reference[:, 0], kind="stable")]
    merged: list[list[int]] = []
    for start, end in reference:
        if not merged or int(start) > merged[-1][1] + 1:
            merged.append([int(start), int(end)])
        else:
            merged[-1][1] = max(merged[-1][1], int(end))
    merged_array = np.asarray(merged, dtype=int)
    starts = merged_array[:, 0]
    ends = merged_array[:, 1]

    candidate = frame.iloc[candidate_indices][["row_id", "label_end_index"]].to_numpy(dtype=int)
    prior_interval = np.searchsorted(starts, candidate[:, 1], side="right") - 1
    valid = prior_interval >= 0
    overlap = np.zeros(candidate_indices.size, dtype=bool)
    overlap[valid] = ends[prior_interval[valid]] >= candidate[valid, 0]
    return overlap


def assert_no_interval_overlap(
    frame: pd.DataFrame,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    *,
    context: str = "split",
) -> None:
    """Assert that no closed training event interval intersects a test event interval."""

    _validate_event_frame(frame)
    overlaps = _interval_overlap_mask(frame, train_indices, test_indices)
    if overlaps.any():
        offending = np.asarray(train_indices, dtype=int)[overlaps]
        raise AssertionError(
            f"Event-interval overlap detected in {context}; training positions "
            f"{offending[:5].tolist()} overlap the test interval"
        )


def assert_no_label_overlap(development: pd.DataFrame, folds: list[WalkForwardFold]) -> None:
    ordered = development.sort_values("row_id").reset_index(drop=True)
    for fold in folds:
        assert_no_interval_overlap(
            ordered,
            fold.train_indices,
            fold.validation_indices,
            context=fold.fold_id,
        )


def build_purged_cpcv_splits(
    development: pd.DataFrame,
    *,
    n_groups: int = 8,
    n_test_groups: int = 2,
    embargo_rows: int = 0,
) -> list[CPCVSplit]:
    """Generate development-only CPCV splits with event purging and post-test embargo.

    Standard CPCV may train on observations later than a test group.  Consequently these
    splits diagnose strategy-selection/backtest-overfitting risk; they are not a substitute
    for chronological walk-forward or a final locked holdout.
    """

    ordered = development.sort_values("row_id").reset_index(drop=True)
    _validate_event_frame(ordered)
    if n_groups < 2 or n_groups > len(ordered):
        raise ValueError("n_groups must be between 2 and the number of development rows")
    if not 0 < n_test_groups < n_groups:
        raise ValueError("n_test_groups must be between 1 and n_groups - 1")
    if embargo_rows < 0:
        raise ValueError("embargo_rows cannot be negative")

    groups = [
        np.asarray(group, dtype=int) for group in np.array_split(np.arange(len(ordered)), n_groups)
    ]
    splits: list[CPCVSplit] = []
    for split_number, test_group_ids in enumerate(
        combinations(range(n_groups), n_test_groups),
        start=1,
    ):
        test_indices = np.sort(np.concatenate([groups[group_id] for group_id in test_group_ids]))
        raw_train_mask = np.ones(len(ordered), dtype=bool)
        raw_train_mask[test_indices] = False
        raw_train_indices = np.flatnonzero(raw_train_mask)
        overlap = _interval_overlap_mask(ordered, raw_train_indices, test_indices)
        purge_mask = np.zeros(len(ordered), dtype=bool)
        purge_mask[raw_train_indices[overlap]] = True

        embargo_mask = np.zeros(len(ordered), dtype=bool)
        if embargo_rows:
            row_ids = ordered["row_id"].to_numpy(dtype=int)
            for group_id in test_group_ids:
                group_event_end = int(ordered.iloc[groups[group_id]]["label_end_index"].max())
                embargo_start = int(np.searchsorted(row_ids, group_event_end, side="right"))
                embargo_mask[embargo_start : min(len(ordered), embargo_start + embargo_rows)] = True
        purged_count = int(np.count_nonzero(raw_train_mask & purge_mask))
        embargoed_count = int(np.count_nonzero(raw_train_mask & ~purge_mask & embargo_mask))
        train_indices = np.flatnonzero(raw_train_mask & ~purge_mask & ~embargo_mask)
        if train_indices.size == 0:
            raise ValueError(f"CPCV split {split_number} has no training observations")
        assert_no_interval_overlap(
            ordered,
            train_indices,
            test_indices,
            context=f"cpcv_{split_number:03d}",
        )
        splits.append(
            CPCVSplit(
                split_id=f"cpcv_{split_number:03d}",
                train_indices=train_indices,
                test_indices=test_indices,
                test_group_ids=tuple(int(group_id) for group_id in test_group_ids),
                train_start_row=int(ordered.iloc[int(train_indices[0])]["row_id"]),
                train_end_row=int(ordered.iloc[int(train_indices[-1])]["row_id"]),
                test_start_row=int(ordered.iloc[int(test_indices[0])]["row_id"]),
                test_end_row=int(ordered.iloc[int(test_indices[-1])]["row_id"]),
                n_train_raw=len(raw_train_indices),
                n_train=len(train_indices),
                n_test=len(test_indices),
                purged_count=purged_count,
                embargoed_count=embargoed_count,
            )
        )
    return splits
