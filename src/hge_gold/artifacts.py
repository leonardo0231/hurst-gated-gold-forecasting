from __future__ import annotations

import csv
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from .io import atomic_json, sha256_file

ALLOWED_ARTIFACT_STATUS = {"CREATED", "NOT_REQUIRED", "DEFERRED", "MISSING_REQUIRED", "INVALID"}


@dataclass(frozen=True)
class Artifact:
    object_type: str
    artifact_name: str
    required: bool
    required_if: str | None
    enabled_in_config: bool
    artifact_status: str
    path: str | None
    hash: str | None
    validation_status: str

    def validate(self, root: Path) -> None:
        if self.object_type != "artifact":
            raise ValueError(f"{self.artifact_name}: object_type must be artifact")
        if self.artifact_status not in ALLOWED_ARTIFACT_STATUS:
            raise ValueError(f"{self.artifact_name}: invalid artifact_status")
        if self.artifact_status == "CREATED":
            if self.path is None or self.hash is None:
                raise ValueError(f"{self.artifact_name}: created artifact lacks path/hash")
            full = root / self.path
            if not full.exists() or sha256_file(full) != self.hash:
                raise ValueError(f"{self.artifact_name}: filesystem/hash mismatch")
        if self.required and self.artifact_status != "CREATED":
            raise ValueError(f"{self.artifact_name}: required artifact not created")


def artifact_from_path(root: Path, path: Path, *, required: bool = True) -> Artifact:
    relative = str(path.relative_to(root))
    return Artifact(
        "artifact", path.stem, required, None, True, "CREATED", relative, sha256_file(path), "PASS"
    )


def status_artifact(name: str, status: str, reason: str) -> Artifact:
    if status not in {"NOT_REQUIRED", "DEFERRED"}:
        raise ValueError("Status artifact must be NOT_REQUIRED or DEFERRED")
    return Artifact(
        "artifact", f"{name}:{reason}", False, reason, False, status, None, None, "NOT_APPLICABLE"
    )


def write_manifest(
    root: Path, phase: int, artifacts: Iterable[Artifact]
) -> tuple[Path, Path, Path]:
    items = list(artifacts)
    for item in items:
        item.validate(root)
    metadata = root / "artifacts" / "metadata"
    metadata.mkdir(parents=True, exist_ok=True)
    json_path = metadata / f"phase{phase}_artifact_manifest.json"
    csv_path = metadata / f"phase{phase}_artifact_manifest.csv"
    sidecar = metadata / f"phase{phase}_artifact_manifest.sha256"
    payload = {
        "phase": phase,
        "authoritative_format": "json",
        "artifacts": [asdict(item) for item in items],
        "self_hash_excluded": [str(json_path.relative_to(root)), str(csv_path.relative_to(root))],
        "created_at_utc": datetime.now(UTC).isoformat(),
    }
    atomic_json(json_path, payload)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(asdict(items[0]).keys()) if items else ["artifact_name"]
        )
        writer.writeheader()
        for item in items:
            writer.writerow(asdict(item))
    sidecar.write_text(
        f"{sha256_file(json_path)}  {json_path.relative_to(root)}\n"
        f"{sha256_file(csv_path)}  {csv_path.relative_to(root)}\n",
        encoding="utf-8",
    )
    return json_path, csv_path, sidecar
