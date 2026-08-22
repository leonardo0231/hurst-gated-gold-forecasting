from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hge_gold.partitions import freeze_physical_partitions, load_frozen_development_partition
from hge_gold.research_protocol import ProtocolViolation, sha256_file


def _prices(path: Path) -> None:
    close = np.linspace(1_800.0, 1_810.0, 10)
    pd.DataFrame(
        {
            "date": pd.bdate_range("2023-06-01", periods=10),
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": np.arange(10) + 100,
        }
    ).to_csv(path, index=False)


def test_loader_accepts_only_hash_verified_physical_development_file(tmp_path: Path) -> None:
    source = tmp_path / "development.csv"
    _prices(source)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "partition_role": "development_reused_previously_exposed",
                "path": source.name,
                "sha256": sha256_file(source),
                "row_count": 10,
                "boundary_exclusive": "2023-07-03",
            }
        ),
        encoding="utf-8",
    )

    loaded = load_frozen_development_partition(source, manifest, min_rows=10)
    assert len(loaded) == 10
    assert loaded["date"].max() < pd.Timestamp("2023-07-03")

    source.write_text("tampered", encoding="utf-8")
    with pytest.raises(ProtocolViolation, match="hash"):
        load_frozen_development_partition(source, manifest, min_rows=10)


def test_loader_rejects_audit_role_even_if_dates_look_historical(tmp_path: Path) -> None:
    source = tmp_path / "audit.csv"
    _prices(source)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "partition_role": "historical_audit_previously_revealed",
                "path": source.name,
                "sha256": sha256_file(source),
                "row_count": 10,
                "boundary_exclusive": "2023-07-03",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ProtocolViolation, match="development"):
        load_frozen_development_partition(source, manifest, min_rows=10)


def test_partition_freeze_is_complete_disjoint_and_creation_exclusive(tmp_path: Path) -> None:
    combined = tmp_path / "combined.csv"
    close = np.linspace(1_800.0, 1_820.0, 20)
    frame = pd.DataFrame(
        {
            "date": pd.bdate_range("2023-06-19", periods=20),
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": np.arange(20) + 100,
        }
    )
    frame.to_csv(combined, index=False)
    development_dir = tmp_path / "development"
    audit_dir = tmp_path / "audit"

    manifests = freeze_physical_partitions(
        combined,
        development_dir=development_dir,
        audit_dir=audit_dir,
        boundary_date="2023-07-03",
    )

    development = pd.read_csv(development_dir / "prices.csv", parse_dates=["date"])
    audit = pd.read_csv(audit_dir / "prices.csv", parse_dates=["date"])
    assert development["date"].max() < pd.Timestamp("2023-07-03")
    assert audit["date"].min() >= pd.Timestamp("2023-07-03")
    assert len(development) + len(audit) == len(frame)
    assert set(development["date"]).isdisjoint(set(audit["date"]))
    assert manifests["development"].is_file()
    assert manifests["historical_audit"].is_file()

    with pytest.raises(ProtocolViolation, match="already exists"):
        freeze_physical_partitions(
            combined,
            development_dir=development_dir,
            audit_dir=audit_dir,
            boundary_date="2023-07-03",
        )
