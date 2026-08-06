from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, Field, ValidationError
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from .data import validate_source


class MarketRow(BaseModel):
    date: str
    open: float = Field(gt=0)
    high: float = Field(gt=0)
    low: float = Field(gt=0)
    close: float = Field(gt=0)
    volume: float = Field(ge=0)


class MarketBatch(BaseModel):
    rows: list[MarketRow] = Field(min_length=1, max_length=10_000)


async def health(_: Request) -> JSONResponse:
    return JSONResponse({"status": "ok", "trading_mode": os.getenv("HGE_TRADING_MODE", "offline")})


async def status(_: Request) -> JSONResponse:
    root = Path(__file__).resolve().parents[2]
    report = root / "reports" / "execution_report.json"
    if not report.exists():
        return JSONResponse({"status": "not_executed", "report": None})
    return JSONResponse(json.loads(report.read_text(encoding="utf-8")))


async def validate_data(request: Request) -> JSONResponse:
    try:
        batch = MarketBatch.model_validate(await request.json())
    except (ValidationError, json.JSONDecodeError) as exc:
        return JSONResponse({"detail": str(exc)}, status_code=422)
    frame = pd.DataFrame([row.model_dump() for row in batch.rows])
    try:
        validate_source(frame)
    except ValueError as exc:
        return JSONResponse({"detail": str(exc)}, status_code=422)
    return JSONResponse(
        {
            "valid": True,
            "rows": len(frame),
            "date_min": frame["date"].min(),
            "date_max": frame["date"].max(),
        }
    )


app = Starlette(
    debug=False,
    routes=[
        Route("/health", health, methods=["GET"]),
        Route("/status", status, methods=["GET"]),
        Route("/validate-data", validate_data, methods=["POST"]),
    ],
)
