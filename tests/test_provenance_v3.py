from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from hge_gold.config import DataConfig, ThesisConfig
from hge_gold.data import generate_research_sample, normalize_and_validate
from hge_gold.provenance import CHUNK_SIZE, code_tree_sha256, sha256_file, source_metadata


def test_streaming_hash_matches_reference_across_multiple_blocks(tmp_path: Path) -> None:
    path = tmp_path / "large.bin"
    payload = b"abc123" * (CHUNK_SIZE // 3)
    path.write_bytes(payload)
    assert len(payload) > CHUNK_SIZE
    assert sha256_file(path) == hashlib.sha256(payload).hexdigest()


def test_csv_path_does_not_imply_market_evidence(tmp_path: Path) -> None:
    path = tmp_path / "synthetic.csv"
    frame = generate_research_sample(700)
    frame.to_csv(path, index=False)
    source = normalize_and_validate(frame, 700)
    config = ThesisConfig(
        project_root=tmp_path,
        data=DataConfig(source_kind="synthetic", source="fixture"),
    )
    metadata = source_metadata(source, path, config)
    assert metadata["source_is_market_evidence"] is False
    assert metadata["source_kind"] == "synthetic"


def test_sidecar_hash_mismatch_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "XAUUSD.csv"
    frame = generate_research_sample(700)
    frame.to_csv(path, index=False)
    sidecar = tmp_path / "manifest.json"
    sidecar.write_text(
        json.dumps(
            {
                "files": [{"file": path.name, "sha256": "0" * 64, "symbol": "XAUUSD"}],
                "timeframe": "D1",
            }
        ),
        encoding="utf-8",
    )
    source = normalize_and_validate(frame, 700)
    config = ThesisConfig(
        project_root=tmp_path,
        data=DataConfig(
            source_kind="market_evidence",
            source_manifest_path=sidecar.name,
        ),
    )
    with pytest.raises(ValueError, match="SHA-256"):
        source_metadata(source, path, config)


def test_code_fingerprint_includes_lock_scripts_and_mt5_source(tmp_path: Path) -> None:
    for directory in ("src/hge_gold", "configs", "scripts", "mt5"):
        (tmp_path / directory).mkdir(parents=True, exist_ok=True)
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    lock = tmp_path / "uv.lock"
    lock.write_text("version=1\n", encoding="utf-8")
    (tmp_path / "src/hge_gold/a.py").write_text("VALUE=1\n", encoding="utf-8")
    (tmp_path / "configs/a.yaml").write_text("value: 1\n", encoding="utf-8")
    (tmp_path / "scripts/run.py").write_text("print('x')\n", encoding="utf-8")
    (tmp_path / "mt5/ea.mq5").write_text("#property strict\n", encoding="utf-8")

    before = code_tree_sha256(tmp_path)
    lock.write_text("version=2\n", encoding="utf-8")

    assert code_tree_sha256(tmp_path) != before
