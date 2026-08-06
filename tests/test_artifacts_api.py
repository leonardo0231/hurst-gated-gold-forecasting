from __future__ import annotations

import asyncio
from pathlib import Path

import httpx

from hge_gold.api import app
from hge_gold.artifacts import Artifact, status_artifact
from hge_gold.io import decision_hash


def request(method: str, path: str, **kwargs: object) -> httpx.Response:
    async def send() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.request(method, path, **kwargs)

    return asyncio.run(send())


def test_artifact_status_vocabulary(tmp_path: Path) -> None:
    status_artifact("optional", "DEFERRED", "garch_deferred").validate(tmp_path)
    invalid = Artifact(
        "artifact", "x", False, None, False, "OPTIONAL", None, None, "NOT_APPLICABLE"
    )
    try:
        invalid.validate(tmp_path)
    except ValueError as exc:
        assert "invalid artifact_status" in str(exc)
    else:
        raise AssertionError("Invalid pseudo-status was accepted")


def test_decision_hash_excludes_self() -> None:
    payload = {"status": "CLOSED", "decision_hash": "old"}
    assert decision_hash(payload) == decision_hash({"status": "CLOSED"})


def test_api_health_and_validation() -> None:
    health = request("GET", "/health")
    assert health.status_code == 200
    assert health.json()["trading_mode"] in {"offline", "simulation", "sandbox", "paper"}
    response = request(
        "POST",
        "/validate-data",
        json={
            "rows": [
                {
                    "date": "2026-01-02",
                    "open": 2000,
                    "high": 2020,
                    "low": 1990,
                    "close": 2010,
                    "volume": 1000,
                },
                {
                    "date": "2026-01-05",
                    "open": 2010,
                    "high": 2030,
                    "low": 2000,
                    "close": 2025,
                    "volume": 1100,
                },
            ]
        },
    )
    assert response.status_code == 200
    assert response.json()["valid"] is True


def test_api_rejects_invalid_ohlc() -> None:
    response = request(
        "POST",
        "/validate-data",
        json={
            "rows": [
                {
                    "date": "2026-01-02",
                    "open": 2000,
                    "high": 1900,
                    "low": 1950,
                    "close": 2010,
                    "volume": 1000,
                },
            ]
        },
    )
    assert response.status_code == 422
