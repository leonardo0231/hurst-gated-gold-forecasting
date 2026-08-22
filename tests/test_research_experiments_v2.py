from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hge_gold.config import FeatureConfig, TargetConfig
from hge_gold.data import generate_research_sample, normalize_and_validate
from hge_gold.execution_v2 import build_non_overlapping_return_ledger
from hge_gold.features import build_feature_matrix
from hge_gold.research_experiments import add_robust_hurst_features, select_arm_features
from hge_gold.research_experiments_v2 import (
    FAMILY_V2,
    align_trial_return_ledgers,
    enumerate_preregistered_trials_v2,
    run_trial_v2,
    verify_preregistered_bundle,
)
from hge_gold.research_protocol import ProtocolViolation
from hge_gold.targets import build_horizon_dataset


def test_v2_is_a_distinct_fixed_twelve_trial_family() -> None:
    trials = enumerate_preregistered_trials_v2()

    assert FAMILY_V2 == "executable_direction_hurst_ablation_v2"
    assert len(trials) == 12
    assert len({trial.experiment_id for trial in trials}) == 12
    assert all("_v2-" in trial.experiment_id for trial in trials)
    assert {(trial.arm, trial.horizon) for trial in trials} == {
        (arm, horizon)
        for arm in ("no_hurst", "current_dfa_hurst", "robust_hurst_regime")
        for horizon in (1, 5, 10, 20)
    }


def test_pbo_alignment_uses_all_twelve_trials_on_common_rows() -> None:
    ledgers: dict[str, pd.DataFrame] = {}
    for index, trial in enumerate(enumerate_preregistered_trials_v2()):
        start = index % 3
        ledgers[trial.experiment_id] = pd.DataFrame(
            {
                "row_id": np.arange(start, 30),
                "candidate_net_log_return": np.linspace(-0.01, 0.01, 30 - start) + index / 10_000,
            }
        )

    row_ids, matrix, trial_ids = align_trial_return_ledgers(ledgers)

    assert row_ids.tolist() == list(range(2, 30))
    assert matrix.shape == (28, 12)
    assert trial_ids == [trial.experiment_id for trial in enumerate_preregistered_trials_v2()]


def test_preregistered_bundle_fails_if_runtime_code_changes(tmp_path: Path) -> None:
    config = tmp_path / "config.json"
    code = tmp_path / "runner.py"
    manifest = tmp_path / "code_manifest.json"
    card = tmp_path / "card.json"
    development = tmp_path / "prices.csv"
    development_manifest = tmp_path / "development_manifest.json"
    availability_manifest = tmp_path / "availability_manifest.json"
    config.write_text(f'{{"hypothesis_family":"{FAMILY_V2}"}}', encoding="utf-8")
    code.write_text("SEED=42\n", encoding="utf-8")
    development.write_text("date,open,high,low,close,volume\n", encoding="utf-8")
    availability_manifest.write_text('{"sources":[]}', encoding="utf-8")

    from hge_gold.research_protocol import sha256_file

    manifest.write_text(
        '{"files":[{"path":"runner.py","sha256":"' + sha256_file(code) + '"}]}',
        encoding="utf-8",
    )
    development_manifest.write_text(
        '{"partition_role":"development_reused_previously_exposed",'
        '"path":"prices.csv","sha256":"'
        + sha256_file(development)
        + '","row_count":0,"boundary_exclusive":"2023-07-03"}',
        encoding="utf-8",
    )
    card.write_text(
        "{"
        f'"hypothesis_family":"{FAMILY_V2}",'
        f'"executable_config_sha256":"{sha256_file(config)}",'
        f'"code_manifest_sha256":"{sha256_file(manifest)}",'
        f'"development_manifest_sha256":"{sha256_file(development_manifest)}",'
        f'"development_data_sha256":"{sha256_file(development)}",'
        f'"data_availability_manifest_sha256":"{sha256_file(availability_manifest)}"'
        "}",
        encoding="utf-8",
    )

    verify_preregistered_bundle(
        project_root=tmp_path,
        card_path=card,
        config_path=config,
        code_manifest_path=manifest,
        development_manifest_path=development_manifest,
        development_path=development,
        data_availability_manifest_path=availability_manifest,
    )
    code.write_text("SEED=7\n", encoding="utf-8")
    with pytest.raises(ProtocolViolation, match="runtime source"):
        verify_preregistered_bundle(
            project_root=tmp_path,
            card_path=card,
            config_path=config,
            code_manifest_path=manifest,
            development_manifest_path=development_manifest,
            development_path=development,
            data_availability_manifest_path=availability_manifest,
        )

    code.write_text("SEED=42\n", encoding="utf-8")
    availability_manifest.write_text('{"sources":[{"future":true}]}', encoding="utf-8")
    with pytest.raises(ProtocolViolation, match="availability manifest"):
        verify_preregistered_bundle(
            project_root=tmp_path,
            card_path=card,
            config_path=config,
            code_manifest_path=manifest,
            development_manifest_path=development_manifest,
            development_path=development,
            data_availability_manifest_path=availability_manifest,
        )


def test_v2_trial_smoke_persists_inner_scores_and_executable_columns() -> None:
    source = normalize_and_validate(generate_research_sample(1_200, seed=8), min_rows=1_200)
    feature_config = FeatureConfig()
    target_config = TargetConfig(horizons=(5,))
    features, columns = build_feature_matrix(source, feature_config)
    features = add_robust_hurst_features(features)
    columns += [name for name in features.columns if name.startswith("hurst_robust_")]
    dataset = build_horizon_dataset(source, features, columns, 5, target_config, feature_config)
    config = {
        "model": {
            "imputer": "median",
            "logistic_c": 0.2,
            "class_weight": "balanced",
            "max_iter": 2_000,
            "seed": 42,
        },
        "folds": {
            "outer_min_train_rows": 450,
            "inner_min_train_rows": 160,
            "outer_min_validation_rows": 100,
            "inner_min_validation_rows": 50,
            "pre_validation_gap_rows": 0,
            "outer_folds": 2,
            "inner_folds": 2,
        },
        "inner_selection": {
            "calibration_fraction": 0.6666666667,
            "calibration_options": ["none", "sigmoid"],
            "no_trade_margins": [0.0, 0.05],
        },
        "economics": {"baseline_cost_bps": 5.0},
    }
    trial = next(
        item
        for item in enumerate_preregistered_trials_v2()
        if item.arm == "no_hurst" and item.horizon == 5
    )

    result = run_trial_v2(dataset, select_arm_features(columns, trial.arm), trial, config)
    predictions = result["predictions"]
    ledger = build_non_overlapping_return_ledger(predictions, cost_bps=5.0)

    assert len(result["fold_metrics"]) == 2
    assert len(result["inner_scores"]) == 2 * 2 * 2
    assert {"entry_timestamp", "exit_timestamp", "return_lag_1", "momentum_20"}.issubset(
        predictions.columns
    )
    active = ledger.loc[ledger["active"]]
    assert (
        active["entry_row_index"].iloc[1:].to_numpy()
        >= active["exit_row_index"].iloc[:-1].to_numpy()
    ).all()
