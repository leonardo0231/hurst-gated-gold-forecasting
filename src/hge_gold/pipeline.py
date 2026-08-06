from __future__ import annotations

import json
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .artifacts import Artifact, artifact_from_path, write_manifest
from .config import PipelineConfig, load_config
from .data import build_phase1
from .evaluation import run_evaluation
from .features import build_features
from .governance import run_lifecycle_phases, run_phase6, run_phase7
from .io import atomic_json, decision_hash, sha256_file
from .logging import configure_logging
from .modeling import run_modeling
from .targets import build_targets


def _decision(path: Path, payload: dict[str, Any]) -> Path:
    payload["decision_hash"] = decision_hash(payload)
    atomic_json(path, payload)
    return path


def _phase_manifest(
    config: PipelineConfig, phase: int, outputs: dict[str, Path], decision: Path
) -> tuple[Path, Path, Path]:
    unique: dict[Path, Artifact] = {}
    for path in [*outputs.values(), decision]:
        if path.is_file():
            unique[path.resolve()] = artifact_from_path(config.paths().root, path.resolve())
    return write_manifest(config.paths().root, phase, unique.values())


def run_pipeline(
    config_path: str | Path = "configs/pipeline.yaml", source_csv: Path | None = None
) -> dict[str, Any]:
    configure_logging()
    config = load_config(config_path)
    paths = config.paths()
    for directory in [paths.data, paths.artifacts, paths.models, paths.reports]:
        directory.mkdir(parents=True, exist_ok=True)
    metadata = paths.artifacts / "metadata"
    metadata.mkdir(parents=True, exist_ok=True)
    run_started = datetime.now(UTC).isoformat()
    statuses: dict[str, str] = {"phase0": "CLOSED"}

    phase1 = build_phase1(config, source_csv)
    phase1_decision = _decision(
        metadata / "phase1_final_dataset_decision.json",
        {
            "phase1_mvp_data_status": "CLOSED_FOR_MVP",
            "phase1_paper_grade_data_status": "NOT_READY",
            "permission_to_enter_phase2": "CONDITIONAL",
            "permitted_scope": {
                "gold_only_target_construction": "YES",
                "gold_only_technical_features": "YES",
                "hurst_features": "CONDITIONAL",
                "garch_volatility_features": "CONDITIONAL",
                "cross_market_features": "NO",
                "cot_features": "NO",
                "macro_features": "NO",
                "safe_haven_diagnostics": "NO",
            },
            "data_mode": "SIMULATED" if source_csv is None else "USER_PROVIDED",
            "paper_grade_limitation": (
                "Settlement, daily open interest, verified roll metadata, and exchange "
                "timestamps are unavailable."
            ),
            "created_at_utc": datetime.now(UTC).isoformat(),
        },
    )
    _phase_manifest(config, 1, phase1, phase1_decision)
    statuses["phase1"] = "CLOSED_FOR_MVP / PAPER_GRADE_NOT_READY"

    phase2 = build_targets(config)
    phase2_decision = _decision(
        metadata / "phase2_final_target_decision.json",
        {
            "phase2_target_status": "CLOSED",
            "permission_to_enter_phase3": "CONDITIONAL",
            "target_policy_id": "final_mvp_v1",
            "direction_threshold_policy_id": "vol_scaled_k050_floor_5bps",
            "trade_threshold_policy_id": "cost_aware_roundtrip_tc3_slip2_bps",
            "diagnostic_threshold_labels": {
                "allowed_for_analysis_only": True,
                "allowed_for_modeling": False,
            },
            "permitted_phase3_scope": {
                "gold_only_technical_features": "YES",
                "hurst_features": "CONDITIONAL",
                "garch_volatility_features": "CONDITIONAL",
                "cross_market_features": "NO",
                "cot_features": "NO",
                "macro_features": "NO",
                "safe_haven_diagnostics": "NO",
            },
            "created_at_utc": datetime.now(UTC).isoformat(),
        },
    )
    _phase_manifest(config, 2, phase2, phase2_decision)
    statuses["phase2"] = "CLOSED"

    phase3 = build_features(config)
    phase3_decision = _decision(
        metadata / "phase3_final_feature_decision.json",
        {
            "phase3_feature_status": "CONDITIONAL",
            "permission_to_enter_phase4": "CONDITIONAL",
            "feature_set_id": config.features["feature_set_id"],
            "target_policy_id": config.targets["target_policy_id"],
            "modeling_base_path": str(phase3["modeling_base"].relative_to(paths.root)),
            "modeling_base_hash": sha256_file(phase3["modeling_base"]),
            "garch_features_status": "DEFERRED",
            "garch_features_allowed_for_phase4": False,
            "permitted_phase4_scope": {
                "gold_only_modeling": "YES",
                "hurst_gated_modeling": "CONDITIONAL",
                "garch_volatility_modeling": "NO",
                "cross_market_modeling": "NO",
                "cot_modeling": "NO",
                "macro_modeling": "NO",
                "safe_haven_modeling": "NO",
            },
            "created_at_utc": datetime.now(UTC).isoformat(),
        },
    )
    _phase_manifest(config, 3, phase3, phase3_decision)
    statuses["phase3"] = "CONDITIONAL (GARCH_DEFERRED)"

    phase4 = run_modeling(config)
    selected_map = json.loads(phase4["selected_map"].read_text(encoding="utf-8"))
    phase4_decision = _decision(
        metadata / "phase4_final_modeling_decision.json",
        {
            "phase4_modeling_status": "CLOSED",
            "permission_to_enter_phase5": "CONDITIONAL",
            "feature_set_id": config.features["feature_set_id"],
            "target_policy_id": config.targets["target_policy_id"],
            "split_policy_id": "walk_forward_purged_embargoed_v1",
            "selected_model_map_hash": selected_map["selected_model_map_hash"],
            "locked_test_predictions_path": str(
                phase4["locked_predictions"].relative_to(paths.root)
            ),
            "locked_test_predictions_hash": sha256_file(phase4["locked_predictions"]),
            "locked_test_predictions_contain_y_true": False,
            "locked_test_metrics_computed_in_phase4": False,
            "selected_models": selected_map["selected_models"],
            "permitted_phase5_scope": {
                "locked_test_evaluation": "YES",
                "trading_backtest": "CONDITIONAL",
                "statistical_significance_testing": "CONDITIONAL",
                "SHAP_explainability": "NO",
                "HGE_analysis": "CONDITIONAL",
                "safe_haven_evaluation": "NO",
                "paper_grade_claims": "NO",
            },
            "created_at_utc": datetime.now(UTC).isoformat(),
        },
    )
    _phase_manifest(config, 4, phase4, phase4_decision)
    statuses["phase4"] = "CLOSED"

    phase5 = run_evaluation(config)
    phase5_decision = _decision(
        metadata / "phase5_final_evaluation_decision.json",
        {
            "phase5_evaluation_status": "CLOSED",
            "permission_to_enter_phase6": "CONDITIONAL",
            "feature_set_id": config.features["feature_set_id"],
            "target_policy_id": config.targets["target_policy_id"],
            "test_metrics_used_for_model_selection": False,
            "test_metrics_used_for_strategy_optimization": False,
            "model_changed_after_locked_test": False,
            "paper_grade_claims": "NO",
            "permitted_phase6_scope": {
                "manuscript_results_tables": "YES",
                "figures": "YES",
                "methodology_writeup": "YES",
                "claim_writeup": "CONDITIONAL",
                "paper_grade_claims": "NO",
                "cross_market_extension": "NO",
                "safe_haven_claims": "NO",
            },
            "created_at_utc": datetime.now(UTC).isoformat(),
        },
    )
    _phase_manifest(config, 5, phase5, phase5_decision)
    statuses["phase5"] = "CLOSED (SAMPLE_EVIDENCE_ONLY)"

    phase6 = run_phase6(config, phase5_decision)
    statuses["phase6"] = "CLOSED"
    phase7 = run_phase7(config, phase6)
    statuses["phase7"] = "CONDITIONAL (TARGET_JOURNAL_NOT_SELECTED)"
    lifecycle = run_lifecycle_phases(config, phase7)
    for phase in lifecycle:
        statuses[f"phase{phase}"] = "CONDITIONAL (NO_EXTERNAL_SUBMISSION_AUTHORIZED)"

    run_report = {
        "project": config.project["name"],
        "run_started_at_utc": run_started,
        "run_finished_at_utc": datetime.now(UTC).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "trading_mode": config.project["trading_mode"],
        "data_execution": "SIMULATED_NON_CONFIDENTIAL"
        if source_csv is None
        else "USER_PROVIDED_DATA",
        "external_submission": "NOT_EXECUTED",
        "statuses": statuses,
        "selected_model_count": len(selected_map["selected_models"]),
        "locked_test_y_true_present_in_phase4_predictions": False,
    }
    report_path = paths.reports / "execution_report.json"
    atomic_json(report_path, run_report)
    return {
        "report": report_path,
        "statuses": statuses,
        "phase_outputs": {
            1: phase1,
            2: phase2,
            3: phase3,
            4: phase4,
            5: phase5,
            6: phase6,
            7: phase7,
            **lifecycle,
        },
    }
