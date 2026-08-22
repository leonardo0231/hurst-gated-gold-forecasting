from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

CHUNK_SIZE = 1024 * 1024
GENESIS_HASH = "0" * 64


class ProtocolViolation(RuntimeError):
    """Raised when a research-governance invariant would be violated."""


class PartitionRole(StrEnum):
    DEVELOPMENT_REUSED = "development_reused_previously_exposed"
    DEVELOPMENT_CONFIRMATION = "development_confirmation"
    HISTORICAL_AUDIT = "historical_audit_previously_revealed"
    FUTURE_OUT_OF_SAMPLE = "actual_future_out_of_sample"


class ResearchPhase(StrEnum):
    DEVELOPMENT_SELECTION = "development_selection"
    CANDIDATE_FROZEN = "candidate_frozen"
    CONFIRMATION = "confirmation"
    HISTORICAL_AUDIT = "historical_audit"
    CLOSED = "closed"


REQUIRED_EXPERIMENT_FIELDS = frozenset(
    {
        "experiment_id",
        "parent_hypothesis",
        "hypothesis_family",
        "timestamp_utc",
        "code_sha256",
        "config_sha256",
        "data_sha256",
        "dependency_sha256",
        "source_availability_convention",
        "feature_list",
        "target",
        "horizon",
        "model",
        "hyperparameters",
        "fold_definitions",
        "train_metrics",
        "validation_metrics",
        "calibration_metrics",
        "economic_metrics",
        "decision",
        "decision_reason",
        "declared_family_budget",
    }
)


def sha256_file(path: Path, chunk_size: int = CHUNK_SIZE) -> str:
    """Return a streaming SHA-256 without loading the source into memory."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _payload_hash(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(payload)).hexdigest()


def _exclusive_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(
        payload,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    )
    try:
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(serialized)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise ProtocolViolation(f"Refusing to overwrite immutable receipt: {path}") from exc


def _validate_sha256(value: object, field: str) -> None:
    if not isinstance(value, str) or len(value) != 64:
        raise ProtocolViolation(f"{field} must be a 64-character SHA-256")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ProtocolViolation(f"{field} must be hexadecimal") from exc


def validate_experiment_payload(payload: Mapping[str, Any]) -> None:
    missing = REQUIRED_EXPERIMENT_FIELDS.difference(payload)
    if missing:
        raise ProtocolViolation(f"Experiment payload is missing fields: {sorted(missing)}")
    for field in ("code_sha256", "config_sha256", "data_sha256", "dependency_sha256"):
        _validate_sha256(payload[field], field)
    if not isinstance(payload["feature_list"], list):
        raise ProtocolViolation("feature_list must be a list")
    if not isinstance(payload["fold_definitions"], list):
        raise ProtocolViolation("fold_definitions must be a list")
    budget = payload["declared_family_budget"]
    if not isinstance(budget, int) or isinstance(budget, bool) or budget <= 0:
        raise ProtocolViolation("declared_family_budget must be a positive integer")
    if payload["decision"] not in {"promote", "reject", "failed"}:
        raise ProtocolViolation("decision must be promote, reject, or failed")


@dataclass(frozen=True)
class RegistryHead:
    sequence: int
    record_hash: str


class AppendOnlyExperimentRegistry:
    """Hash-chained JSONL registry that never rewrites an existing byte."""

    def __init__(self, path: Path) -> None:
        self.path = path

    @property
    def lock_path(self) -> Path:
        return self.path.with_suffix(self.path.suffix + ".lock")

    def read_and_validate(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        records: list[dict[str, Any]] = []
        previous = GENESIS_HASH
        with self.path.open("r", encoding="utf-8") as handle:
            for sequence, raw_line in enumerate(handle):
                if not raw_line.strip():
                    raise ProtocolViolation("Blank lines are forbidden in the registry")
                parsed = json.loads(raw_line)
                if not isinstance(parsed, dict):
                    raise ProtocolViolation("Every registry line must be a JSON object")
                record = dict(parsed)
                claimed_hash = record.pop("record_hash", None)
                if record.get("sequence") != sequence:
                    raise ProtocolViolation("Registry sequence is not contiguous")
                if record.get("previous_record_hash") != previous:
                    raise ProtocolViolation("Registry hash chain is broken")
                actual_hash = _payload_hash(record)
                if claimed_hash != actual_hash:
                    raise ProtocolViolation("Registry record hash mismatch")
                record["record_hash"] = claimed_hash
                records.append(record)
                previous = actual_hash
        experiment_ids = [str(record["experiment_id"]) for record in records]
        if len(experiment_ids) != len(set(experiment_ids)):
            raise ProtocolViolation("Duplicate experiment_id detected in registry")
        return records

    def head(self) -> RegistryHead:
        records = self.read_and_validate()
        if not records:
            return RegistryHead(sequence=-1, record_hash=GENESIS_HASH)
        last = records[-1]
        return RegistryHead(sequence=int(last["sequence"]), record_hash=str(last["record_hash"]))

    def append(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        from .research_run import ExclusiveFileLock

        validate_experiment_payload(payload)
        with ExclusiveFileLock(self.lock_path):
            existing = self.read_and_validate()
            experiment_id = str(payload["experiment_id"])
            if any(str(record["experiment_id"]) == experiment_id for record in existing):
                raise ProtocolViolation(f"Duplicate experiment_id: {experiment_id}")
            family = str(payload["hypothesis_family"])
            family_records = [r for r in existing if str(r["hypothesis_family"]) == family]
            budget = int(payload["declared_family_budget"])
            if len(family_records) >= budget:
                raise ProtocolViolation(f"Trial budget exhausted for hypothesis family {family!r}")
            if any(int(r["declared_family_budget"]) != budget for r in family_records):
                raise ProtocolViolation("A hypothesis family's declared budget cannot change")

            previous = GENESIS_HASH if not existing else str(existing[-1]["record_hash"])
            record: dict[str, Any] = dict(payload)
            record["sequence"] = len(existing)
            record["previous_record_hash"] = previous
            record_hash = _payload_hash(record)
            record["record_hash"] = record_hash
            self.path.parent.mkdir(parents=True, exist_ok=True)
            encoded = canonical_json(record) + b"\n"
            mode = "xb" if not self.path.exists() else "ab"
            try:
                with self.path.open(mode) as handle:
                    handle.write(encoded)
                    handle.flush()
                    os.fsync(handle.fileno())
            except FileExistsError as exc:
                raise ProtocolViolation(
                    "Concurrent registry creation detected; retry after audit"
                ) from exc
            self.read_and_validate()
            return record


def assert_trial_budget(
    records: Sequence[Mapping[str, Any]], hypothesis_family: str, declared_budget: int
) -> None:
    if declared_budget <= 0:
        raise ValueError("declared_budget must be positive")
    used = sum(1 for record in records if record.get("hypothesis_family") == hypothesis_family)
    if used >= declared_budget:
        raise ProtocolViolation(
            f"Trial budget exhausted for {hypothesis_family!r}: {used}/{declared_budget}"
        )


def assert_partition_access(
    role: PartitionRole,
    phase: ResearchPhase,
    *,
    candidate_dir: Path | None = None,
) -> None:
    """Fail closed before any label or metric file for a protected role is opened."""
    if role == PartitionRole.DEVELOPMENT_REUSED:
        if phase != ResearchPhase.DEVELOPMENT_SELECTION:
            return
        return
    if phase == ResearchPhase.DEVELOPMENT_SELECTION:
        raise ProtocolViolation(f"{role.value} cannot be loaded during development selection")
    if candidate_dir is None:
        raise ProtocolViolation("Protected partition access requires a frozen candidate directory")
    validate_candidate_freeze(candidate_dir)
    receipt = candidate_dir / f"{role.value}_claim.json"
    if not receipt.is_file():
        raise ProtocolViolation(f"Access claim must be written before loading {role.value}")


def claim_partition_access(
    candidate_dir: Path,
    role: PartitionRole,
    *,
    source_sha256: str,
) -> Path:
    if role == PartitionRole.DEVELOPMENT_REUSED:
        raise ProtocolViolation("Development evidence does not use a one-time access claim")
    validate_candidate_freeze(candidate_dir)
    _validate_sha256(source_sha256, "source_sha256")
    receipt = candidate_dir / f"{role.value}_claim.json"
    _exclusive_json(
        receipt,
        {
            "candidate_id": candidate_dir.name,
            "claimed_at_utc": datetime.now(UTC).isoformat(),
            "partition_role": role.value,
            "source_sha256": source_sha256,
            "state": "claimed_before_label_or_metric_load",
        },
    )
    return receipt


def _copy_exclusive(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with source.open("rb") as reader, destination.open("xb") as writer:
            shutil.copyfileobj(reader, writer, length=CHUNK_SIZE)
            writer.flush()
            os.fsync(writer.fileno())
    except FileExistsError as exc:
        raise ProtocolViolation(f"Refusing to overwrite frozen artifact: {destination}") from exc


def freeze_candidate(
    candidate_root: Path,
    manifest: Mapping[str, Any],
    *,
    project_root: Path | None = None,
    artifact_paths: Iterable[Path] = (),
) -> Path:
    """Create a candidate directory once; subsequent calls fail without mutation."""
    if manifest.get("development_promotion_passed") is not True:
        raise ProtocolViolation("Only a candidate that passed development gates may be frozen")
    candidate_id = str(manifest.get("candidate_id", "")).strip()
    if not candidate_id or any(token in candidate_id for token in ("/", "\\", "..")):
        raise ProtocolViolation("candidate_id must be a safe non-empty path component")
    candidate_dir = candidate_root / candidate_id
    try:
        candidate_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise ProtocolViolation(f"Candidate freeze already exists: {candidate_id}") from exc

    frozen_artifacts: list[dict[str, Any]] = []
    for source in artifact_paths:
        source = source.resolve()
        if project_root is not None:
            try:
                relative = source.relative_to(project_root.resolve())
            except ValueError as exc:
                raise ProtocolViolation("Frozen artifact must be within project_root") from exc
        else:
            relative = Path(source.name)
        destination = candidate_dir / "files" / relative
        _copy_exclusive(source, destination)
        frozen_artifacts.append(
            {
                "path": relative.as_posix(),
                "sha256": sha256_file(destination),
                "size_bytes": destination.stat().st_size,
            }
        )

    payload = dict(manifest)
    payload["frozen_at_utc"] = datetime.now(UTC).isoformat()
    payload["frozen_artifacts"] = frozen_artifacts
    payload["historical_confirmation_available"] = False
    payload["historical_audit_role"] = PartitionRole.HISTORICAL_AUDIT.value
    manifest_path = candidate_dir / "freeze_manifest.json"
    _exclusive_json(manifest_path, payload)
    _exclusive_json(
        candidate_dir / "freeze_receipt.json",
        {
            "candidate_id": candidate_id,
            "freeze_manifest_sha256": sha256_file(manifest_path),
            "frozen_artifacts": frozen_artifacts,
            "state": ResearchPhase.CANDIDATE_FROZEN.value,
        },
    )
    return candidate_dir


def validate_candidate_freeze(candidate_dir: Path) -> dict[str, Any]:
    manifest_path = candidate_dir / "freeze_manifest.json"
    receipt_path = candidate_dir / "freeze_receipt.json"
    if not manifest_path.is_file() or not receipt_path.is_file():
        raise ProtocolViolation("Candidate freeze is incomplete")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("freeze_manifest_sha256") != sha256_file(manifest_path):
        raise ProtocolViolation("Candidate freeze manifest was mutated")
    for artifact in receipt.get("frozen_artifacts", []):
        artifact_path = candidate_dir / "files" / str(artifact["path"])
        if not artifact_path.is_file() or sha256_file(artifact_path) != artifact["sha256"]:
            raise ProtocolViolation(f"Frozen artifact was mutated: {artifact['path']}")
    parsed_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(parsed_manifest, dict):
        raise ProtocolViolation("Candidate manifest root must be an object")
    manifest: dict[str, Any] = dict(parsed_manifest)
    if manifest.get("candidate_id") != candidate_dir.name:
        raise ProtocolViolation("Candidate directory and manifest identity differ")
    return manifest


def validate_baseline_freeze(baseline_dir: Path) -> dict[str, Any]:
    manifest_path = baseline_dir / "baseline_manifest.json"
    receipt_path = baseline_dir / "freeze_receipt.json"
    if not manifest_path.is_file() or not receipt_path.is_file():
        raise ProtocolViolation("Baseline freeze is incomplete")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("baseline_manifest_sha256") != sha256_file(manifest_path):
        raise ProtocolViolation("Baseline manifest was mutated")
    parsed_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(parsed_manifest, dict):
        raise ProtocolViolation("Baseline manifest root must be an object")
    return dict(parsed_manifest)
