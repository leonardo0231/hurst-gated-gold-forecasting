# Project review and implementation order

## Documents read in full

All listed files were read completely. No section was unreadable.

1. `phase0_research_protocol.md`
2. `phase1_data_design.md`
3. `phase1_execution_final_v7.md`
4. `phase2_target_construction_final_v5.md`
5. `phase3_feature_engineering_final_v4.md`
6. `phase4_modeling_final_v5(3).md`
7. `phase5_final_locked.md`
8. `phase6_final_locked_v4.md`
9. `phase7_final_locked_v4.md`
10. `phase8_final_locked_v2.md`
11. `phase9_final_locked_v2.md`
12. `phase10_final_locked_v2.md`
13. `phase11_final_locked_v2.md`

The workspace contained the documents only. There was no existing repository or source code.

## Locked phase map

| Phase | Goal | Main output | Depends on |
|---|---|---|---|
| 0 | Freeze the research protocol | Scope and anti-leakage rules | None |
| 1 | Build and audit the market dataset | Gold table, calendar, interval template, provenance | 0 |
| 2 | Construct multi-task labels | Returns, direction, action, volatility targets | 1 |
| 3 | Create causal features | Gold-only feature matrix and modeling base | 2 |
| 4 | Train and select without touching the test truth | Fold models, selected map, locked predictions | 3 |
| 5 | Evaluate once and backtest realistically | Metrics, bootstrap/FDR, ledger, cost robustness | 4 |
| 6 | Turn evidence into a controlled manuscript | Tables, figure, evidence map, draft | 5 |
| 7 | Build the submission package | Generic package and conditional decision | 6 |
| 8 | Execute a journal submission when authorized | Receipt and execution decision | 7 |
| 9 | Govern editorial workflow | Status/communication/revision registries | 8 |
| 10 | Govern the complete lifecycle | Lifecycle decision and mode registry | 9 |
| 11 | Continue or close without reopening evidence | Continuation decision and closure-safe registries | 10 |

## Required stack and selected implementation

- Python 3.12, typed package under `src/`, `uv` lockfile and Hatch build backend.
- Pandas, NumPy and PyArrow for validated tabular/Parquet artifacts.
- Scikit-learn for deterministic preprocessing, baselines, linear and tree models.
- Matplotlib for evidence-linked static figures.
- Starlette 1.3.1 and Uvicorn for the small ASGI API. Starlette is used directly because the
  current FastAPI dependency constraint resolved to a vulnerable Starlette release.
- Pytest/coverage, Ruff, mypy, Bandit and pip-audit for quality gates.
- Dockerfile and Compose definition for a reproducible API runtime.

No database or frontend is required by the locked documents, so neither was invented. The
artifact filesystem is the authoritative immutable research store for this implementation.

## Necessary technical assumptions

- The default run is `offline` and uses a deterministic, non-confidential 900-row OHLCV fixture.
- The fixture represents a continuous, back-adjusted GC-like series but is not exchange or vendor
  evidence. Settlement, open interest, verified rolls and exchange timestamps remain unavailable.
- A close timestamp of 22:00 UTC is an explicit MVP assumption, not an exchange-certified fact.
- The global final 20% chronological block is the locked test. Development uses three purged,
  embargo-aware walk-forward folds; no random split is used.
- GARCH is `DEFERRED`, as permitted by Phase 3. Cross-market, COT, macro and safe-haven features
  remain forbidden by the Phase 1 decision.
- Phases 8–11 are status-aware governance execution only. No journal was selected and no external
  submission, credential storage, production deployment or live order was authorized.

## Conflicts and resolutions

- Phase 1 permits an MVP data path but separately marks paper-grade readiness as `NOT_READY`.
  Both statuses are preserved; software execution is closed for MVP while scientific claims remain
  sample-only.
- Phase 7 allows a package while the target journal is null. A generic package is produced and
  journal-specific compliance remains `CONDITIONAL`; no submission is simulated as a real event.
- Phase 10/11 closure semantics do not allow an absent external event to be fabricated. Empty,
  typed registries and conditional decisions are created instead of fake receipts or reviews.

## Implementation order used

1. Configuration, safety modes, hashing, atomic IO and artifact vocabulary.
2. Phase 1 source contract, sample fixture, timestamps, calendar and provenance.
3. Phase 2 targets and leakage audit.
4. Phase 3 causal features, registries and modeling base.
5. Phase 4 walk-forward modeling, selection freeze and locked predictions.
6. Phase 5 one-time evaluation, bootstrap/FDR and execution-aligned costed backtest.
7. Phases 6–11 evidence packaging and external-action-safe governance.
8. ASGI API, tests, build/deployment files, security remediation and delivery reports.
