"""Physical research-partition loading with role and hash enforcement."""

from __future__ import annotations

import json
import os
import secrets
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .data import normalize_and_validate
from .research_protocol import PartitionRole, ProtocolViolation, sha256_file


def freeze_physical_partitions(
    combined_source: Path,
    *,
    development_dir: Path,
    audit_dir: Path,
    boundary_date: str,
) -> dict[str, Path]:
    """Split one legacy combined CSV into disjoint, immutable physical partitions."""

    if development_dir.exists() or audit_dir.exists():
        raise ProtocolViolation("A physical partition destination already exists")
    combined = pd.read_csv(combined_source)
    if "date" not in combined.columns:
        raise ProtocolViolation("Combined source has no date column")
    timestamps = pd.to_datetime(combined["date"], errors="raise")
    if timestamps.duplicated().any() or not timestamps.is_monotonic_increasing:
        raise ProtocolViolation("Combined source timestamps must be unique and chronological")
    boundary = pd.Timestamp(boundary_date)
    development = combined.loc[timestamps < boundary].copy()
    audit = combined.loc[timestamps >= boundary].copy()
    if development.empty or audit.empty or len(development) + len(audit) != len(combined):
        raise ProtocolViolation("Physical partition split is incomplete")

    common_parent = development_dir.parent.resolve()
    if audit_dir.parent.resolve() != common_parent:
        raise ProtocolViolation("Partition directories must share one parent for staged creation")
    staging_root = common_parent / f".partition_staging_{secrets.token_hex(8)}"
    staged_development = staging_root / development_dir.name
    staged_audit = staging_root / audit_dir.name
    staged_development.mkdir(parents=True, exist_ok=False)
    staged_audit.mkdir(parents=True, exist_ok=False)
    parent_hash = sha256_file(combined_source)
    try:
        outputs: list[tuple[pd.DataFrame, Path, PartitionRole]] = [
            (development, staged_development, PartitionRole.DEVELOPMENT_REUSED),
            (audit, staged_audit, PartitionRole.HISTORICAL_AUDIT),
        ]
        for frame, directory, role in outputs:
            prices = directory / "prices.csv"
            frame.to_csv(prices, index=False, lineterminator="\n")
            dates = pd.to_datetime(frame["date"])
            manifest = {
                "schema_version": "physical_research_partition_v1",
                "partition_role": role.value,
                "path": "prices.csv",
                "sha256": sha256_file(prices),
                "row_count": len(frame),
                "start_date": str(dates.min().date()),
                "end_date": str(dates.max().date()),
                "boundary_exclusive": boundary_date,
                "legacy_combined_source": combined_source.resolve().as_posix(),
                "legacy_combined_source_sha256": parent_hash,
                "frozen_at_utc": datetime.now(UTC).isoformat(),
            }
            (directory / "manifest.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        development_dir.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staged_development, development_dir)
        os.replace(staged_audit, audit_dir)
    finally:
        if staging_root.exists():
            shutil.rmtree(staging_root)
    return {
        "development": development_dir / "manifest.json",
        "historical_audit": audit_dir / "manifest.json",
    }


def load_frozen_development_partition(
    source_path: Path,
    manifest_path: Path,
    *,
    min_rows: int = 700,
) -> pd.DataFrame:
    """Load a physically separate, hash-verified development-only CSV."""

    parsed = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(parsed, dict):
        raise ProtocolViolation("Partition manifest must be a JSON object")
    manifest: dict[str, Any] = dict(parsed)
    if manifest.get("partition_role") != PartitionRole.DEVELOPMENT_REUSED.value:
        raise ProtocolViolation("Only a frozen development partition may be loaded for selection")
    expected_path = (manifest_path.parent / str(manifest.get("path", ""))).resolve()
    if source_path.resolve() != expected_path:
        raise ProtocolViolation("Development path differs from its frozen manifest")
    if not source_path.is_file() or sha256_file(source_path) != manifest.get("sha256"):
        raise ProtocolViolation("Development partition hash mismatch")
    frame = normalize_and_validate(pd.read_csv(source_path), min_rows=min_rows)
    if len(frame) != int(manifest.get("row_count", -1)):
        raise ProtocolViolation("Development partition row count mismatch")
    boundary = pd.Timestamp(str(manifest.get("boundary_exclusive")))
    if frame["date"].max() >= boundary:
        raise ProtocolViolation("Development partition crosses the audit boundary")
    return frame
