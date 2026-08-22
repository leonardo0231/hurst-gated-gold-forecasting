from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from hge_gold.research_protocol import (
    AppendOnlyExperimentRegistry,
    PartitionRole,
    ProtocolViolation,
    ResearchPhase,
    assert_partition_access,
    assert_trial_budget,
    claim_partition_access,
    freeze_candidate,
    validate_candidate_freeze,
)


def _experiment(experiment_id: str, *, budget: int = 2) -> dict[str, Any]:
    digest = "a" * 64
    return {
        "experiment_id": experiment_id,
        "parent_hypothesis": "executable_direction_hurst_ablation_v1",
        "hypothesis_family": "hurst_ablation",
        "timestamp_utc": "2026-08-21T00:00:00Z",
        "code_sha256": digest,
        "config_sha256": digest,
        "data_sha256": digest,
        "dependency_sha256": digest,
        "source_availability_convention": "features available no later than decision time",
        "feature_list": ["return_lag_1"],
        "target": "executable_direction",
        "horizon": 1,
        "model": "logistic",
        "hyperparameters": {},
        "fold_definitions": [],
        "train_metrics": {},
        "validation_metrics": {},
        "calibration_metrics": {},
        "economic_metrics": {},
        "decision": "reject",
        "decision_reason": "development gate failed",
        "declared_family_budget": budget,
    }


def _freeze_manifest(candidate_id: str) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "development_promotion_passed": True,
        "experiment_id": "exp-001",
        "code_sha256": "a" * 64,
        "config_sha256": "b" * 64,
        "data_sha256": "c" * 64,
        "dependency_sha256": "d" * 64,
        "promotion_gate": {"passed": True},
    }


def test_registry_is_hash_chained_append_only_and_rejects_duplicate(tmp_path: Path) -> None:
    registry = AppendOnlyExperimentRegistry(tmp_path / "experiments.jsonl")
    first = registry.append(_experiment("exp-001"))
    second = registry.append(_experiment("exp-002"))

    assert first["sequence"] == 0
    assert second["previous_record_hash"] == first["record_hash"]
    assert len(registry.read_and_validate()) == 2
    with pytest.raises(ProtocolViolation, match="Duplicate"):
        registry.append(_experiment("exp-002"))


def test_registry_detects_mutation_and_budget_exhaustion(tmp_path: Path) -> None:
    path = tmp_path / "experiments.jsonl"
    registry = AppendOnlyExperimentRegistry(path)
    registry.append(_experiment("exp-001", budget=1))
    with pytest.raises(ProtocolViolation, match="budget exhausted"):
        registry.append(_experiment("exp-002", budget=1))
    assert_trial_budget(registry.read_and_validate(), "other_family", 1)

    line = json.loads(path.read_text(encoding="utf-8"))
    line["decision_reason"] = "tampered"
    path.write_text(json.dumps(line) + "\n", encoding="utf-8")
    with pytest.raises(ProtocolViolation, match="hash mismatch"):
        registry.read_and_validate()


def test_candidate_freeze_is_creation_exclusive_and_detects_mutation(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    model = project / "model.bin"
    model.write_bytes(b"frozen model")
    root = tmp_path / "candidates"
    candidate = freeze_candidate(
        root,
        _freeze_manifest("cand-001"),
        project_root=project,
        artifact_paths=[model],
    )
    assert validate_candidate_freeze(candidate)["candidate_id"] == "cand-001"
    with pytest.raises(ProtocolViolation, match="already exists"):
        freeze_candidate(root, _freeze_manifest("cand-001"))

    (candidate / "files" / "model.bin").write_bytes(b"changed")
    with pytest.raises(ProtocolViolation, match="mutated"):
        validate_candidate_freeze(candidate)


def test_protected_partition_requires_freeze_and_preload_claim(tmp_path: Path) -> None:
    with pytest.raises(ProtocolViolation, match="cannot be loaded"):
        assert_partition_access(
            PartitionRole.HISTORICAL_AUDIT,
            ResearchPhase.DEVELOPMENT_SELECTION,
        )

    candidate = freeze_candidate(
        tmp_path / "candidates",
        _freeze_manifest("cand-002"),
    )
    with pytest.raises(ProtocolViolation, match="claim"):
        assert_partition_access(
            PartitionRole.HISTORICAL_AUDIT,
            ResearchPhase.HISTORICAL_AUDIT,
            candidate_dir=candidate,
        )
    claim_partition_access(
        candidate,
        PartitionRole.HISTORICAL_AUDIT,
        source_sha256="e" * 64,
    )
    assert_partition_access(
        PartitionRole.HISTORICAL_AUDIT,
        ResearchPhase.HISTORICAL_AUDIT,
        candidate_dir=candidate,
    )
    with pytest.raises(ProtocolViolation, match="overwrite"):
        claim_partition_access(
            candidate,
            PartitionRole.HISTORICAL_AUDIT,
            source_sha256="e" * 64,
        )
