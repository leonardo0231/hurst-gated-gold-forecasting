from __future__ import annotations

import pandas as pd

from hge_gold.evaluation import backtest_summary


def test_backtest_uses_row_id_gap_for_non_overlapping_trades() -> None:
    predictions = pd.DataFrame(
        {
            "row_id": [0, 2, 5, 7, 10],
            "y_pred": [1, 1, 1, 1, 1],
            "forward_log_return": [0.01, 0.01, 0.01, 0.01, 0.01],
        }
    )

    summary = backtest_summary(
        predictions,
        horizon=5,
        transaction_cost_bps=0.0,
        slippage_bps=0.0,
    )

    assert summary["n_non_overlapping_trades"] == 3
