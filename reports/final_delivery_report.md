# Final delivery report

Generated: 2026-07-11 (Europe/Berlin)

## Outcome

The HGE-Hybrid Gold repository is implemented and executable in offline/simulation mode. The final
standalone run completed all local phases at 2026-07-11T16:25:20Z, trained 16 selected models, wrote
locked predictions without test truth, evaluated them only in Phase 5, ran the costed backtest and
generated the manuscript/governance artifacts.

This is verified software and sample-data evidence. It is not paper-grade market evidence and is not
investment advice.

## Capability status

| Capability | Status | Evidence |
|---|---|---|
| Repository, central configuration, `.env.example`, lockfile | IMPLEMENTED / VERIFIED | `pyproject.toml`, `uv.lock`, `configs/pipeline.yaml` |
| Deterministic non-confidential source and validation | SIMULATED / EXECUTED | `data/raw/sample`, Phase 1 artifacts |
| Real vendor/exchange ingestion and corporate-action/roll audit | BLOCKED | No credentialed source or paper-grade contract metadata supplied |
| Targets for 1/5/10/20 sessions | IMPLEMENTED / TESTED / EXECUTED | Phase 2 target Parquet and leakage report |
| Causal technical, Hurst and volatility features | IMPLEMENTED / TESTED / EXECUTED | Phase 3 matrix and registries |
| GARCH features | BLOCKED / DEFERRED | Explicit Phase 3 decision |
| Walk-forward modeling and locked-test isolation | IMPLEMENTED / TESTED / VERIFIED | 16 selected models; Phase 4 predictions contain no `y_true` |
| Locked evaluation, static baselines, 5000-block bootstrap and FDR | IMPLEMENTED / EXECUTED | 40 metrics, 16 comparisons, 16 significance rows |
| Costed execution-aligned backtest | IMPLEMENTED / TESTED / EXECUTED | 172 observations; 5bps base cost; net return -0.332444 |
| Manuscript tables, figure and evidence map | IMPLEMENTED / EXECUTED | Phase 6 reports |
| Generic submission package | IMPLEMENTED / VERIFIED | Phase 7 ZIP and manifest |
| Real journal submission/editorial lifecycle | BLOCKED / NOT TESTED | No journal, authorization, portal or receipt; no external action taken |
| ASGI health/status/data-validation API | IMPLEMENTED / TESTED / EXECUTED | Three real HTTP requests returned 200 |
| Database and migrations | NOT REQUIRED | Locked documents define artifact storage, not a database |
| Frontend/UI | NOT REQUIRED | Not required by the locked documents |
| Python package build | EXECUTED / VERIFIED | Wheel and sdist under `dist/` |
| Docker definition | IMPLEMENTED / NOT TESTED | Docker executable unavailable in the environment |
| Live trading | BLOCKED | Forbidden by protocol and user instruction; no broker code or order path exists |

## Actual verification results

- Full test suite: **17 passed**, 0 failed, 0 skipped, 272.52 seconds.
- Coverage: **92%** overall with branch coverage enabled; XML and HTML reports included.
- Post-warning API regression: **4 passed** in 0.23 seconds, no warning.
- Ruff formatter/check: all checks passed.
- mypy strict application check: no issues in 14 source files. Untyped third-party import
  diagnostics are disabled; application code remains strict.
- Bandit: 0 findings over 2,758 lines of source.
- pip-audit: 0 known vulnerabilities across 71 installed dependencies. An initial audit found eight
  vulnerabilities; PyArrow and pytest were upgraded, and the thin API moved from FastAPI's
  vulnerable Starlette constraint to Starlette 1.3.1.
- Package build: `hge_hybrid_gold-0.1.0-py3-none-any.whl` and source tarball built offline.
- API runtime: Uvicorn started and returned HTTP 200 for `/health`, `/status` and `/validate-data`.
- Docker build: not executed because `docker` is not installed (`docker: command not found`).

## Backtest interpretation

The base 5bps round-trip scenario produced net return **-0.332444**, annualized Sharpe
**-3.567710**, maximum drawdown **-0.353400**, and turnover **107** on the deterministic fixture.
Cost robustness is monotone from 1bps through 10bps. The negative result is retained; it was not
used to alter model selection or tune the strategy.

## Important files and generated groups

- Source: `src/hge_gold/*.py` (14 modules).
- Tests: `tests/conftest.py` and four test modules.
- Configuration/deployment: `.env.example`, `pyproject.toml`, `uv.lock`, `Dockerfile`,
  `docker-compose.yml`, `configs/pipeline.yaml`, `scripts/run_all.sh`.
- Documentation: `README.md`, `docs/architecture.md`, `docs/api.md`, `docs/data-model.md`,
  `docs/testing.md`, `docs/project-review.md`.
- Phase artifacts: `artifacts/metadata/phase*_*.{json,csv,parquet,sha256}`.
- Data/model outputs: `data/processed`, `data/predictions`, `data/evaluation`, `data/backtests`,
  `models/phase4`.
- Reports: Phase 6 manuscript/table/figure set, Phase 7 generic package, execution report, test,
  coverage and security reports.
- Build products: `dist/*.whl`, `dist/*.tar.gz`.

## Principal commands executed

```text
uv lock
uv sync --extra dev
uv run hge-gold run --config configs/pipeline.yaml
uv run --extra dev pytest ... --cov=hge_gold ...
uv run --extra dev ruff format .
uv run --extra dev ruff check .
uv run --extra dev python -m mypy src/hge_gold
uv run --extra dev python -m bandit -q -r src/hge_gold ...
uv run --extra dev python -m pip_audit --cache-dir /tmp/pip-audit-cache ...
uv run --extra dev uvicorn hge_gold.api:app --host 127.0.0.1 --port 8765
curl /health; curl /status; curl -X POST /validate-data
uv build --offline --out-dir dist
docker --version
```

## Errors reproduced and fixed

1. Pandas `Timestamp.combine(..., tzinfo=...)` incompatibility: replaced by timezone-aware
   `datetime.combine`.
2. Ambiguous `pd.NA` in scikit-learn categorical preprocessing: normalized missing categoricals and
   used constant imputation.
3. HGE component row alignment compared Series indices rather than values: fixed with explicit
   array equality.
4. Empty early-fold numerical feature warning: preserved feature width with
   `keep_empty_features=True`.
5. Bandit B101 on a runtime assertion: replaced with an explicit exception.
6. Eight dependency CVEs: remediated and lockfile regenerated; audit now clean.
7. Starlette TestClient deprecation warning: replaced with direct `httpx.ASGITransport` testing.
8. Writable cache restrictions: all tool caches redirected to `/tmp`; no permissions were widened.

## Remaining limits and technical debt

- Paper-grade readiness is `NOT_READY` until real settlement, open interest, verified roll metadata,
  market calendar/timestamps and source provenance are supplied and audited.
- The sample fixture cannot support financial or publication-grade claims.
- GARCH and all forbidden cross-market/COT/macro/safe-haven branches remain deliberately deferred.
- The ASGI API is operational but intentionally exposes no trading or prediction-order endpoint.
- Docker files are syntax-reviewed only; a Docker daemon is required for build verification.
- Journal-specific formatting and phases 8–11 external events require an explicit journal, manual
  authorization and authentic portal artifacts.

## Reproduction

```bash
uv sync --extra dev
uv run hge-gold run --config configs/pipeline.yaml
uv run --extra dev pytest -q --cov=hge_gold
uv run uvicorn hge_gold.api:app --host 127.0.0.1 --port 8000
```
