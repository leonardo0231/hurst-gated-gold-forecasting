from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hge_gold.evaluation import (
    backtest_summary,
    classification_metrics,
    economic_benchmark_summaries,
)


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


def test_classification_metrics_include_calibration_diagnostics() -> None:
    metrics = classification_metrics(
        np.array([0, 0, 1, 1]),
        np.array([0.1, 0.2, 0.8, 0.9]),
        threshold=0.5,
    )

    assert metrics["log_loss"] < 0.3
    assert metrics["expected_calibration_error"] == pytest.approx(0.15)


def test_backtest_no_trade_band_and_initial_equity_drawdown() -> None:
    predictions = pd.DataFrame(
        {
            "row_id": [0, 1],
            "y_pred": [1, 1],
            "probability_up": [0.51, 0.9],
            "forward_log_return": [-0.1, 0.02],
        }
    )

    traded = backtest_summary(predictions, 1, 0.0, 0.0)
    filtered = backtest_summary(
        predictions,
        1,
        0.0,
        0.0,
        probability_no_trade_band=0.05,
    )

    assert traded["max_drawdown"] == pytest.approx(np.exp(-0.1) - 1.0)
    assert filtered["n_non_overlapping_observations"] == 1
    assert filtered["n_non_overlapping_trades"] == 1
    assert filtered["n_decision_opportunities"] == 2
    assert filtered["trade_coverage"] == pytest.approx(0.5)
    assert filtered["cumulative_net_return"] == pytest.approx(np.exp(0.02) - 1.0)


def test_economic_benchmarks_use_identical_observations() -> None:
    predictions = pd.DataFrame(
        {
            "row_id": [0, 1, 2, 3],
            "y_pred": [1, 0, 1, 0],
            "forward_log_return": [0.02, -0.01, 0.03, -0.04],
            "return_lag_1": [0.01, -0.01, 0.01, -0.01],
            "momentum_20": [0.01, 0.01, -0.01, -0.01],
            "volatility_zscore": [-1.0, 1.0, -1.0, 1.0],
        }
    )

    summaries = economic_benchmark_summaries(predictions, 2, 0.0, 0.0)

    assert set(summaries) == {
        "model",
        "always_long",
        "always_short",
        "cash",
        "last_return_momentum",
        "moving_average_20",
        "volatility_filtered_momentum",
        "full_period_buy_and_hold",
    }
    assert {summary["n_non_overlapping_observations"] for summary in summaries.values()} == {2}
    assert summaries["model"]["cumulative_net_return"] == pytest.approx(np.exp(0.05) - 1.0)
    assert summaries["cash"]["cumulative_net_return"] == pytest.approx(0.0)


def test_abstention_precedes_non_overlapping_trade_selection() -> None:
    predictions = pd.DataFrame(
        {
            "row_id": [0, 1, 2, 3, 4, 5],
            "y_pred": [1, 1, 1, 1, 1, 1],
            "probability_up": [0.51, 0.51, 0.90, 0.90, 0.90, 0.90],
            "forward_log_return": [0.01] * 6,
        }
    )

    summary = backtest_summary(
        predictions,
        horizon=3,
        transaction_cost_bps=0.0,
        slippage_bps=0.0,
        probability_no_trade_band=0.05,
    )

    # Rows 0 and 1 abstain. The first position opens at row 2, and the next
    # position may open at row 5 when the first one exits.
    assert summary["n_non_overlapping_trades"] == 2
    assert summary["trade_coverage"] == pytest.approx(2 / 6)
    assert summary["annualization_periods_per_year"] == pytest.approx(84.0)
