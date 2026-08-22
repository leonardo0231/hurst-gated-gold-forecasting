from __future__ import annotations

import pandas as pd
import pytest

from hge_gold.mt5 import MT5_SIGNAL_COLUMNS, build_mt5_replay_signals


def test_mt5_signals_are_next_open_non_overlapping_and_abstain() -> None:
    frame = pd.DataFrame(
        {
            "row_id": [10, 11, 12, 13],
            "entry_timestamp": pd.to_datetime(
                ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
            ),
            "exit_timestamp": pd.to_datetime(
                ["2024-01-04", "2024-01-05", "2024-01-08", "2024-01-09"]
            ),
            "probability_up": [0.60, 0.90, 0.51, 0.30],
            "probability_threshold": [0.55] * 4,
        }
    )
    signals = build_mt5_replay_signals(frame, horizon=2, probability_no_trade_margin=0.05)
    assert signals.columns.tolist() == MT5_SIGNAL_COLUMNS
    assert signals["signal_id"].tolist() == ["h2_row10", "h2_row13"]
    assert signals["direction"].tolist() == [1, -1]
    assert signals.loc[0, "entry_time"] == "2024.01.02 00:00"


def test_mt5_signals_reject_invalid_exit() -> None:
    frame = pd.DataFrame(
        {
            "row_id": [1],
            "entry_timestamp": [pd.Timestamp("2024-01-02")],
            "exit_timestamp": [pd.Timestamp("2024-01-02")],
            "probability_up": [0.9],
            "probability_threshold": [0.5],
        }
    )
    with pytest.raises(ValueError, match="Invalid executable"):
        build_mt5_replay_signals(frame, 1, 0.05)
