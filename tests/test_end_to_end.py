from __future__ import annotations

import json


def test_full_offline_pipeline(full_run) -> None:  # type: ignore[no-untyped-def]
    result = full_run
    report = json.loads(result["report"].read_text())
    assert report["trading_mode"] == "offline"
    assert report["external_submission"] == "NOT_EXECUTED"
    assert report["selected_model_count"] == 16
    assert report["statuses"]["phase11"].startswith("CONDITIONAL")
    assert report["locked_test_y_true_present_in_phase4_predictions"] is False
