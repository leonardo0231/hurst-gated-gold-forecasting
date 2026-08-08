from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from hge_gold.v2.data import generate_research_sample
from hge_gold.v2.pipeline import run_thesis_pipeline

SYNTHETIC_DISCLAIMER = "Synthetic sample results validate software behavior only."


def test_end_to_end_writes_separate_v2_artifacts(tmp_path: Path) -> None:
    config_dir = tmp_path / "configs"
    config_dir.mkdir(parents=True)
    config = config_dir / "thesis_v2.yaml"
    config.write_text(
        """
project_name: Test V2

data:
  source: sample
  min_rows: 700
features:
  return_lags: [1, 2, 3, 5, 10]
  windows: [5, 10, 20, 63]
  hurst_windows: [64, 96]
  regime_window: 160
  min_feature_coverage: 0.65
targets:
  horizons: [5]
  volatility_window: 50
  volatility_min_periods: 30
  threshold_k: 0.25
  threshold_floor_bps: 6
  transaction_cost_bps: 3
  slippage_bps: 2
splits:
  locked_test_fraction: 0.20
  n_walk_forward_folds: 4
  min_train_rows: 240
  min_validation_rows: 40
models:
  random_seed: 11
  fast_mode: true
  probability_threshold_min: 0.35
  probability_threshold_max: 0.65
  probability_threshold_steps: 31
  gate_tolerance: 0.005
evaluation:
  primary_metric: balanced_accuracy
  primary_threshold: 0.60
  macro_f1_threshold: 0.55
  minimum_class_recall: 0.50
  min_test_samples: 60
  bootstrap_iterations: 30
  bootstrap_block_length: 8
outputs:
  artifact_dir: artifacts/v2
  model_dir: models/v2
  prediction_dir: data/predictions/v2
  compatibility_outputs: true
""".strip(),
        encoding="utf-8",
    )
    outputs = run_thesis_pipeline(config)
    for path in outputs.values():
        assert path.exists()
    metrics = pd.read_csv(outputs["metrics"])
    assert metrics.loc[0, "balanced_accuracy"] >= 0.60
    manifest = json.loads(outputs["manifest"].read_text(encoding="utf-8"))
    assert manifest["pipeline_version"] == "2.0"
    assert manifest["source_is_market_evidence"] is False
    assert SYNTHETIC_DISCLAIMER in manifest["limitations"]
    assert manifest["source_sha256"] is None
    assert manifest["source_rows"] == 1500
    assert manifest["symbol"] is None
    legacy_compatible = (
        tmp_path / "artifacts" / "metadata" / "phase5_locked_test_metrics_report_v2.csv"
    )
    assert legacy_compatible.exists()


def test_market_source_manifest_has_fingerprint_and_metadata(tmp_path: Path) -> None:
    source_csv = tmp_path / "XAUUSD_D1.csv"
    source_frame = generate_research_sample(700, seed=11)
    source_frame["server"] = "CSV Test Server"
    source_frame.to_csv(source_csv, index=False)

    config_dir = tmp_path / "configs"
    config_dir.mkdir(parents=True)
    config = config_dir / "thesis_v2.yaml"
    config.write_text(
        """
project_name: Market Evidence Test

data:
  source: market_evidence
  min_rows: 700
  symbol: XAUUSD
  timeframe: D1
  source_type: MT5 broker export
  broker: Test Broker
  timezone: UTC
  export_date: 2026-08-08
features:
  return_lags: [1, 2, 3, 5, 10]
  windows: [5, 10, 20, 63]
  hurst_windows: [64, 96]
  regime_window: 160
  min_feature_coverage: 0.65
targets:
  horizons: [5]
  volatility_window: 50
  volatility_min_periods: 30
  threshold_k: 0.25
  threshold_floor_bps: 6
  transaction_cost_bps: 3
  slippage_bps: 2
splits:
  locked_test_fraction: 0.20
  n_walk_forward_folds: 4
  min_train_rows: 240
  min_validation_rows: 40
models:
  random_seed: 11
  fast_mode: true
  probability_threshold_min: 0.35
  probability_threshold_max: 0.65
  probability_threshold_steps: 31
  gate_tolerance: 0.005
evaluation:
  primary_metric: balanced_accuracy
  primary_threshold: 0.60
  macro_f1_threshold: 0.55
  minimum_class_recall: 0.50
  min_test_samples: 60
  bootstrap_iterations: 30
  bootstrap_block_length: 8
outputs:
  artifact_dir: artifacts/v2
  model_dir: models/v2
  prediction_dir: data/predictions/v2
  compatibility_outputs: false
""".strip(),
        encoding="utf-8",
    )

    outputs = run_thesis_pipeline(config, source_csv)
    manifest = json.loads(outputs["manifest"].read_text(encoding="utf-8"))
    with source_csv.open("rb") as handle:
        expected_hash = hashlib.file_digest(handle, "sha256").hexdigest()

    assert manifest["source_is_market_evidence"] is True
    assert SYNTHETIC_DISCLAIMER not in manifest["limitations"]
    assert manifest["source"] == str(source_csv.relative_to(tmp_path))
    assert manifest["source_sha256"] == expected_hash
    assert manifest["source_rows"] == len(source_frame)
    assert manifest["source_start"] == source_frame["date"].iloc[0].isoformat()
    assert manifest["source_end"] == source_frame["date"].iloc[-1].isoformat()
    assert manifest["symbol"] == "XAUUSD"
    assert manifest["timeframe"] == "D1"
    assert manifest["source_type"] == "MT5 broker export"
    assert manifest["broker"] == "Test Broker"
    assert manifest["server"] == "CSV Test Server"
    assert manifest["timezone"] == "UTC"
    assert manifest["export_date"] == "2026-08-08"
