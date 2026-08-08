from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from hge_gold.data_audit import write_market_data_quality_audit


def test_market_data_quality_audit_writes_descriptive_artifacts(tmp_path: Path) -> None:
    source = pd.DataFrame(
        {
            "row_id": [0, 1, 2, 3, 4],
            "date": pd.to_datetime(
                ["2024-01-01", "2024-01-02", "2024-01-06", "2024-01-15", "2024-01-16"]
            ),
            "open": [100.0, 101.0, 102.0, 103.0, 104.0],
            "high": [101.0, 102.0, 103.0, 104.0, 105.0],
            "low": [99.0, 100.0, 101.0, 102.0, 103.0],
            "close": [100.0, 101.0, 102.0, 103.0, 104.0],
            "volume": [100.0, 0.0, 100.0, 1_000.0, 1_000.0],
        }
    )

    outputs = write_market_data_quality_audit(
        source,
        horizons=(1, 2),
        regime_window=3,
        output_dir=tmp_path / "artifacts" / "data_quality",
    )

    for path in outputs.values():
        assert path.exists()

    summary = json.loads(outputs["data_quality_summary"].read_text(encoding="utf-8"))
    yearly = pd.read_csv(outputs["yearly_statistics"])
    class_balance = pd.read_csv(outputs["horizon_class_balance"])
    suspicious = pd.read_csv(outputs["suspicious_rows"])

    assert summary["source_rows"] == 5
    assert summary["calendar"]["weekend_row_count"] == 1
    assert summary["calendar"]["long_calendar_gap_count"] == 1
    assert summary["volume"]["zero_volume_count"] == 1
    assert yearly.loc[0, "observations"] == 5
    assert class_balance["horizon"].tolist() == [1, 2]
    assert {"weekend_row", "zero_volume", "potential_missing_weekday"} <= set(
        suspicious["issue_type"]
    )
