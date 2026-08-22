from __future__ import annotations

import pandas as pd

MT5_SIGNAL_COLUMNS = ["entry_time", "exit_time", "direction", "signal_id"]


def build_mt5_replay_signals(
    predictions: pd.DataFrame,
    horizon: int,
    probability_no_trade_margin: float,
) -> pd.DataFrame:
    """Create deterministic, non-overlapping next-open signals for the replay EA."""
    required = {
        "row_id",
        "entry_timestamp",
        "exit_timestamp",
        "probability_up",
        "probability_threshold",
    }
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Missing MT5 signal fields: {sorted(missing)}")
    if not 0.0 <= probability_no_trade_margin < 0.5:
        raise ValueError("probability_no_trade_margin must be in [0, 0.5)")
    ordered = predictions.sort_values("entry_timestamp").reset_index(drop=True)
    selected: list[dict[str, object]] = []
    previous_exit: pd.Timestamp | None = None
    for row in ordered.itertuples(index=False):
        probability = float(row.probability_up)
        if abs(probability - 0.5) < probability_no_trade_margin:
            continue
        entry = pd.Timestamp(row.entry_timestamp)
        exit_ = pd.Timestamp(row.exit_timestamp)
        if pd.isna(entry) or pd.isna(exit_) or exit_ <= entry:
            raise ValueError("Invalid executable entry/exit timestamp")
        if previous_exit is not None and entry < previous_exit:
            continue
        direction = 1 if probability >= float(row.probability_threshold) else -1
        selected.append(
            {
                "entry_time": entry.strftime("%Y.%m.%d %H:%M"),
                "exit_time": exit_.strftime("%Y.%m.%d %H:%M"),
                "direction": direction,
                "signal_id": f"h{int(horizon)}_row{int(row.row_id)}",
            }
        )
        previous_exit = exit_
    return pd.DataFrame(selected, columns=MT5_SIGNAL_COLUMNS)
