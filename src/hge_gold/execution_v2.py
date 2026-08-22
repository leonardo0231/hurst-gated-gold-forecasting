"""Executable, non-overlapping research schedules and terminal reconciliation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .targets import bps_to_log_return

SCHEDULE_COLUMNS = [
    "row_id",
    "entry_row_index",
    "exit_row_index",
    "probability_up",
    "signal",
    "signal_id",
]


def build_mt5_signal_schedule(predictions: pd.DataFrame, *, margin: float) -> pd.DataFrame:
    """Select the first eligible signal whenever the one-position strategy is flat.

    Rejected overlapping signals never change ``busy_until``. Entry at the exact row where
    the previous position exits is allowed because the exit is processed before the new
    entry at that bar open.
    """

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
        if exit_row <= entry:
            raise ValueError("Every executable signal must exit strictly after entry")
        if entry < busy_until:
            continue
        row_id = int(row.row_id)
        selected.append(
            {
                "row_id": row_id,
                "entry_row_index": entry,
                "exit_row_index": exit_row,
                "probability_up": probability,
                "signal": 1 if probability > 0.5 else -1,
                "signal_id": f"row{row_id}_entry{entry}_exit{exit_row}",
            }
        )
        busy_until = exit_row
    return pd.DataFrame(selected, columns=SCHEDULE_COLUMNS)


def replay_mt5_schedule(schedule: pd.DataFrame) -> pd.DataFrame:
    """Validate a frozen schedule without pretending to be terminal parity."""

    if list(schedule.columns) != SCHEDULE_COLUMNS:
        raise ValueError("MT5 schedule schema or column order changed")
    if not schedule["row_id"].is_monotonic_increasing:
        raise ValueError("MT5 schedule must be chronological")
    expected = np.where(schedule["probability_up"].to_numpy() > 0.5, 1, -1)
    if not np.array_equal(expected, schedule["signal"].to_numpy(dtype=int)):
        raise ValueError("MT5 signal direction differs from frozen probability")
    entries = schedule["entry_row_index"].to_numpy(dtype=int)
    exits = schedule["exit_row_index"].to_numpy(dtype=int)
    if np.any(exits <= entries) or np.any(entries[1:] < exits[:-1]):
        raise ValueError("MT5 schedule contains overlapping or invalid trades")
    return schedule.copy()


def build_non_overlapping_return_ledger(
    predictions: pd.DataFrame,
    *,
    cost_bps: float,
) -> pd.DataFrame:
    """Build a synchronous per-observation ledger on one executable trade schedule."""

    required = {
        "row_id",
        "date",
        "entry_row_index",
        "exit_row_index",
        "entry_timestamp",
        "exit_timestamp",
        "probability_up",
        "selected_margin",
        "executable_forward_log_return",
        "return_lag_1",
        "momentum_20",
    }
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"Return ledger is missing columns: {sorted(missing)}")
    if cost_bps < 0.0:
        raise ValueError("cost_bps cannot be negative")
    ordered = predictions.sort_values("row_id").reset_index(drop=True).copy()
    actionable = ordered.loc[
        (ordered["probability_up"] - 0.5).abs() > ordered["selected_margin"] + 1e-12
    ]
    schedule = build_mt5_signal_schedule(actionable, margin=0.0)
    signal_by_row = (
        schedule.set_index("row_id")["signal"] if not schedule.empty else pd.Series(dtype=int)
    )
    id_by_row = (
        schedule.set_index("row_id")["signal_id"] if not schedule.empty else pd.Series(dtype=str)
    )
    row_ids = ordered["row_id"].astype(int)
    ordered["signal"] = row_ids.map(signal_by_row).fillna(0).astype(int)
    ordered["signal_id"] = row_ids.map(id_by_row).fillna("").astype(str)
    ordered["active"] = ordered["signal"].ne(0)

    forward = ordered["executable_forward_log_return"].to_numpy(dtype=float)
    active = ordered["active"].to_numpy(dtype=bool)
    candidate_signal = ordered["signal"].to_numpy(dtype=float)
    long_signal = np.where(active, 1.0, 0.0)
    short_signal = np.where(active, -1.0, 0.0)
    momentum_signal = np.where(
        active,
        np.where(ordered["return_lag_1"].to_numpy(dtype=float) >= 0.0, 1.0, -1.0),
        0.0,
    )
    trend_signal = np.where(
        active,
        np.where(ordered["momentum_20"].to_numpy(dtype=float) >= 0.0, 1.0, -1.0),
        0.0,
    )
    cost = active.astype(float) * bps_to_log_return(cost_bps)
    ordered["cost_bps"] = float(cost_bps)
    ordered["transaction_cost_log_return"] = cost
    ordered["candidate_gross_log_return"] = candidate_signal * forward
    ordered["candidate_net_log_return"] = candidate_signal * forward - cost
    ordered["cash_net_log_return"] = 0.0
    ordered["always_long_net_log_return"] = long_signal * forward - cost
    ordered["always_short_net_log_return"] = short_signal * forward - cost
    ordered["momentum_net_log_return"] = momentum_signal * forward - cost
    ordered["trend_net_log_return"] = trend_signal * forward - cost
    return ordered


def reconcile_mt5_trades(
    expected_ledger: pd.DataFrame, terminal_trades: pd.DataFrame
) -> dict[str, Any]:
    """Reconcile frozen signals with terminal order/fill/exit/cost evidence row by row."""

    expected_required = {"signal_id", "entry_timestamp", "exit_timestamp", "signal"}
    terminal_required = {
        "signal_id",
        "order_id",
        "entry_deal_id",
        "exit_deal_id",
        "entry_timestamp",
        "exit_timestamp",
        "direction",
        "entry_price",
        "exit_price",
        "spread_bps",
        "commission",
        "swap",
        "slippage_bps",
    }
    missing_expected = expected_required.difference(expected_ledger.columns)
    missing_terminal = terminal_required.difference(terminal_trades.columns)
    if missing_expected or missing_terminal:
        raise ValueError(
            f"Parity evidence missing expected={sorted(missing_expected)} "
            f"terminal={sorted(missing_terminal)}"
        )
    expected = expected_ledger.loc[expected_ledger["signal_id"].astype(str).ne("")].copy()
    terminal = terminal_trades.copy()
    if expected["signal_id"].duplicated().any() or terminal["signal_id"].duplicated().any():
        raise ValueError("signal_id must be unique in both schedules")
    for identifier in ("order_id", "entry_deal_id", "exit_deal_id"):
        values = terminal[identifier].astype(str).str.strip()
        if values.eq("").any() or values.duplicated().any():
            raise ValueError(f"MT5 {identifier} must be non-empty and unique")
    expected_ids = set(expected["signal_id"].astype(str))
    terminal_ids = set(terminal["signal_id"].astype(str))
    if expected_ids != terminal_ids:
        raise ValueError("MT5 signal_id set differs from the frozen Python schedule")
    expected = expected.set_index("signal_id").sort_index()
    terminal = terminal.set_index("signal_id").sort_index()
    for column in ("entry_timestamp", "exit_timestamp"):
        left = pd.to_datetime(expected[column]).to_numpy(dtype="datetime64[ns]")
        right = pd.to_datetime(terminal[column]).to_numpy(dtype="datetime64[ns]")
        if not np.array_equal(left, right):
            raise ValueError(f"MT5 {column} differs from the frozen schedule")
    if not np.array_equal(
        expected["signal"].to_numpy(dtype=int), terminal["direction"].to_numpy(dtype=int)
    ):
        raise ValueError("MT5 direction differs from the frozen schedule")
    if not terminal["direction"].isin((-1, 1)).all():
        raise ValueError("MT5 direction must be either -1 or 1")
    entry_times = pd.to_datetime(terminal["entry_timestamp"])
    exit_times = pd.to_datetime(terminal["exit_timestamp"])
    if not (exit_times > entry_times).all():
        raise ValueError("MT5 exit fill must occur after its entry fill")
    prices = terminal[["entry_price", "exit_price"]].to_numpy(dtype=float)
    if not np.isfinite(prices).all() or np.any(prices <= 0.0):
        raise ValueError("MT5 fill prices must be finite and positive")
    costs = terminal[["spread_bps", "commission", "swap", "slippage_bps"]].to_numpy(dtype=float)
    if not np.isfinite(costs).all():
        raise ValueError("MT5 cost evidence must be finite")
    if (terminal["spread_bps"].to_numpy(dtype=float) < 0.0).any():
        raise ValueError("MT5 spread evidence cannot be negative")
    return {
        "status": "PASS",
        "matched_trades": int(len(expected)),
        "signal_ids": expected.index.astype(str).tolist(),
        "scope": "signal_order_fill_exit_cost_reconciliation",
        "evidence_schema_version": "mt5_trade_chain_v2",
        "native_mql5_inference": False,
    }
