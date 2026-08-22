from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hge_gold.execution_v2 import (
    build_non_overlapping_return_ledger,
    reconcile_mt5_trades,
)


def _predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "row_id": [10, 11, 12, 15],
            "date": pd.to_datetime(["2020-01-10", "2020-01-11", "2020-01-12", "2020-01-15"]),
            "entry_row_index": [11, 12, 13, 16],
            "exit_row_index": [13, 14, 15, 18],
            "entry_timestamp": pd.to_datetime(
                ["2020-01-11", "2020-01-12", "2020-01-13", "2020-01-16"]
            ),
            "exit_timestamp": pd.to_datetime(
                ["2020-01-13", "2020-01-14", "2020-01-15", "2020-01-18"]
            ),
            "probability_up": [0.80, 0.10, 0.90, 0.20],
            "selected_margin": [0.05, 0.05, 0.05, 0.05],
            "executable_forward_log_return": [0.02, -0.01, 0.03, -0.02],
            "return_lag_1": [0.01, -0.01, 0.02, -0.02],
            "momentum_20": [0.05, -0.03, 0.04, -0.06],
        }
    )


def test_return_ledger_uses_only_executable_non_overlapping_trades() -> None:
    ledger = build_non_overlapping_return_ledger(_predictions(), cost_bps=5.0)

    active = ledger.loc[ledger["active"]]
    assert active["row_id"].tolist() == [10, 12, 15]
    assert active["entry_row_index"].tolist() == [11, 13, 16]
    assert active["exit_row_index"].tolist() == [13, 15, 18]
    assert (
        active["entry_row_index"].iloc[1:].to_numpy()
        >= active["exit_row_index"].iloc[:-1].to_numpy()
    ).all()
    assert ledger["candidate_net_log_return"].ne(0.0).sum() == 3
    assert ledger.loc[ledger["row_id"] == 11, "candidate_net_log_return"].item() == 0.0


def test_return_ledger_benchmarks_share_candidate_trade_schedule() -> None:
    ledger = build_non_overlapping_return_ledger(_predictions(), cost_bps=5.0)
    inactive = ~ledger["active"]

    for column in (
        "candidate_net_log_return",
        "always_long_net_log_return",
        "always_short_net_log_return",
        "momentum_net_log_return",
        "trend_net_log_return",
    ):
        assert np.allclose(ledger.loc[inactive, column], 0.0)
    assert np.allclose(ledger["cash_net_log_return"], 0.0)


def test_mt5_reconciliation_is_trade_by_trade_not_a_python_self_copy() -> None:
    ledger = build_non_overlapping_return_ledger(_predictions(), cost_bps=5.0)
    expected = ledger.loc[ledger["active"]].copy()
    terminal = expected[["signal_id", "entry_timestamp", "exit_timestamp", "signal"]].rename(
        columns={"signal": "direction"}
    )
    terminal["commission"] = 0.0
    terminal["swap"] = 0.0
    terminal["slippage_bps"] = 0.0
    terminal["spread_bps"] = 1.5
    terminal["order_id"] = ["order-1", "order-2", "order-3"]
    terminal["entry_deal_id"] = ["entry-1", "entry-2", "entry-3"]
    terminal["exit_deal_id"] = ["exit-1", "exit-2", "exit-3"]
    terminal["entry_price"] = [1_900.0, 1_905.0, 1_910.0]
    terminal["exit_price"] = [1_905.0, 1_910.0, 1_900.0]

    report = reconcile_mt5_trades(expected, terminal)
    assert report["status"] == "PASS"
    assert report["matched_trades"] == 3

    bad = terminal.copy()
    bad.loc[0, "direction"] *= -1
    with pytest.raises(ValueError, match="direction"):
        reconcile_mt5_trades(expected, bad)


def test_mt5_reconciliation_requires_order_fill_exit_and_cost_chain() -> None:
    ledger = build_non_overlapping_return_ledger(_predictions(), cost_bps=5.0)
    expected = ledger.loc[ledger["active"]].copy()
    incomplete = expected[["signal_id", "entry_timestamp", "exit_timestamp", "signal"]].rename(
        columns={"signal": "direction"}
    )
    incomplete["commission"] = 0.0
    incomplete["swap"] = 0.0
    incomplete["slippage_bps"] = 0.0

    with pytest.raises(ValueError, match="entry_deal_id"):
        reconcile_mt5_trades(expected, incomplete)
