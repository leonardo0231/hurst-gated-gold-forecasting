from __future__ import annotations

import pandas as pd

from hge_gold.research_experiments import build_mt5_signal_schedule, replay_mt5_schedule


def test_python_and_mt5_bar_open_schedule_are_signal_by_signal_identical() -> None:
    predictions = pd.DataFrame(
        {
            "row_id": [10, 11, 12, 15],
            "entry_row_index": [11, 12, 13, 16],
            "exit_row_index": [13, 14, 15, 18],
            "probability_up": [0.80, 0.10, 0.90, 0.20],
        }
    )

    python_schedule = build_mt5_signal_schedule(predictions, margin=0.05)
    mt5_replay = replay_mt5_schedule(python_schedule)

    pd.testing.assert_frame_equal(python_schedule, mt5_replay)
    assert python_schedule["entry_row_index"].tolist() == [11, 16]
    assert python_schedule["signal"].tolist() == [1, -1]


def test_schedule_is_deterministic_and_uses_close_before_next_open() -> None:
    predictions = pd.DataFrame(
        {
            "row_id": [5, 6, 8],
            "entry_row_index": [6, 7, 9],
            "exit_row_index": [8, 9, 11],
            "probability_up": [0.55, 0.95, 0.05],
        }
    )

    first = build_mt5_signal_schedule(predictions, margin=0.05)
    second = build_mt5_signal_schedule(predictions, margin=0.05)

    pd.testing.assert_frame_equal(first, second)
    assert first["entry_row_index"].tolist() == [7, 9]
