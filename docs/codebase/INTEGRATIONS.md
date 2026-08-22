# External Integrations

## 1) Integration Inventory

| System | Type (API/DB/Queue/etc) | Purpose | Auth model | Criticality | Evidence |
|---|---|---|---|---|---|
| MetaTrader 5 / MetaQuotes-Demo | Desktop terminal, broker data export, tester replay | Market data provenance and frozen-signal execution validation | Terminal installation/account state; exact credential handling is outside this repository | High for market evidence/replay | `configs/thesis.yaml`, `mt5/README.md` |
| Local CSV files | File input | Processed OHLCV market data | Filesystem access | High | `configs/thesis.yaml`, `src/hge_gold/data.py` |
| GitHub Actions | CI | Lint, type checks, tests, Windows MT5 contract tests | GitHub-hosted runner configuration | Medium | `.github/workflows/ci.yml` |

## 2) Data Stores

| Store | Role | Access layer | Key risk | Evidence |
|---|---|---|---|---|
| Local CSV/artifacts/models | Inputs, generated evidence, model bundles | `data.py`, `pipeline.py` | Broker/session-dependent data and mutable canonical output paths | `configs/thesis.yaml`, `src/hge_gold/pipeline.py` |

## 3) Secrets and Credentials Handling

- Credential sources: no environment-variable reads, `.env` template, or secret-manager integration was found.
- Hardcoding checks: no API keys or passwords were found in source/configuration inspected.
- Rotation or lifecycle notes: [TODO] the repository does not document MT5 terminal/account credential lifecycle.

## 4) Reliability and Failure Behavior

- Retry/backoff behavior: none found for integrations.
- Timeout policy: MT5 runner behavior is documented in `mt5/README.md`; no general network timeout policy is present.
- Circuit-breaker or fallback behavior: invalid MT5 signal CSVs fail closed in the replay contract.

## 5) Observability for Integrations

- Logging around external calls: provenance, checksums, and MT5 evidence files are produced; no centralized logging/tracing was found.
- Metrics/tracing coverage: no APM/metrics integration was found.
- Missing visibility gaps: live data download receipt and MT5 credential/connection telemetry are not implemented here.

## 6) Evidence

- `configs/thesis.yaml`
- `src/hge_gold/data.py`
- `src/hge_gold/provenance.py`
- `mt5/README.md`
