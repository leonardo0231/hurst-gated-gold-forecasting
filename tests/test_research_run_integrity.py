from __future__ import annotations

import json
from pathlib import Path

import pytest

from hge_gold.research_protocol import AppendOnlyExperimentRegistry, ProtocolViolation
from hge_gold.research_run import (
    ExclusiveFileLock,
    build_run_receipt,
    finalize_staged_run,
    validate_run_receipt,
    write_failed_run_receipt,
)


def _experiment(experiment_id: str) -> dict[str, object]:
    digest = "a" * 64
    return {
        "experiment_id": experiment_id,
        "parent_hypothesis": "v2",
        "hypothesis_family": "v2",
        "timestamp_utc": "2026-08-22T00:00:00Z",
        "code_sha256": digest,
        "config_sha256": digest,
        "data_sha256": digest,
        "dependency_sha256": digest,
        "source_availability_convention": "past only",
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
        "decision_reason": ["gate"],
        "declared_family_budget": 12,
    }


def test_exclusive_lock_prevents_a_second_registry_writer(tmp_path: Path) -> None:
    registry = AppendOnlyExperimentRegistry(tmp_path / "registry.jsonl")
    with (
        ExclusiveFileLock(registry.lock_path),
        pytest.raises(ProtocolViolation, match="locked"),
    ):
        registry.append(_experiment("v2-001"))
    registry.append(_experiment("v2-001"))
    assert len(registry.read_and_validate()) == 1


def test_receipt_hashes_every_output_and_runtime_source(tmp_path: Path) -> None:
    project = tmp_path / "project"
    staged = project / "staging" / "run-001"
    staged.mkdir(parents=True)
    (staged / "predictions.csv").write_text("row_id,p\n1,0.6\n", encoding="utf-8")
    (staged / "model.json").write_text('{"model":"logistic"}\n', encoding="utf-8")
    source = project / "runner.py"
    source.write_text("SEED = 42\n", encoding="utf-8")

    receipt = build_run_receipt(
        staged,
        project_root=project,
        runtime_sources=[source],
        metadata={"batch_id": "run-001"},
    )
    (staged / "run_receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
    validate_run_receipt(staged, project_root=project)

    (staged / "predictions.csv").write_text("tampered", encoding="utf-8")
    with pytest.raises(ProtocolViolation, match="mutated"):
        validate_run_receipt(staged, project_root=project)


def test_staged_run_finalizes_exclusively_and_failure_gets_receipt(tmp_path: Path) -> None:
    staged = tmp_path / ".staging" / "run-001"
    staged.mkdir(parents=True)
    (staged / "artifact.txt").write_text("complete", encoding="utf-8")
    destination = tmp_path / "runs" / "run-001"

    finalize_staged_run(staged, destination)
    assert (destination / "artifact.txt").read_text(encoding="utf-8") == "complete"
    with pytest.raises(ProtocolViolation, match="already exists"):
        finalize_staged_run(destination, destination)

    failed = write_failed_run_receipt(
        tmp_path / "failed",
        batch_id="run-002",
        error_type="RuntimeError",
        error_message="synthetic failure",
        staging_path=tmp_path / ".staging" / "run-002",
    )
    assert json.loads(failed.read_text(encoding="utf-8"))["state"] == "failed"
    with pytest.raises(ProtocolViolation, match="overwrite"):
        write_failed_run_receipt(
            tmp_path / "failed",
            batch_id="run-002",
            error_type="RuntimeError",
            error_message="again",
            staging_path=tmp_path / ".staging" / "run-002",
        )
