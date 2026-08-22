# Codebase Concerns

## 1) Top Risks (Prioritized)

| Severity | Concern | Evidence | Impact | Suggested action |
|---|---|---|---|---|
| high | Historical holdout was previously revealed | `configs/thesis.yaml`, `artifacts/v3/market/validation_diagnostics.json` | It cannot be presented as pristine confirmation | Freeze a new prospective holdout before further selection |
| high | Current market evidence does not meet registered acceptance gates | `artifacts/v3/market/locked_test_metrics.csv` | No supported claim of stable directional skill or profitability | Report negative finding; test new hypotheses only on development/prospective data |
| medium | Data are broker/session dependent and tick volume is non-centralized | `README.md`, `configs/thesis.yaml` | Cross-broker reproducibility is unestablished | Repeat on independently defined broker/session data |
| medium | MT5 layer validates replay rather than native inference | `mt5/README.md` | Model-inference parity is not established | Build parity tests if native inference becomes a requirement |

## 2) Technical Debt

| Debt item | Why it exists | Where | Risk if ignored | Suggested fix |
|---|---|---|---|---|
| Official PBO/DSR values deferred | Complete development trial return paths are not persisted | `artifacts/v3/market/validation_diagnostics.json` | Model-search risk cannot be quantified fully | Persist per-observation development returns for every registered trial |
| No documented formatter | Ruff lint settings exist but no formatting policy/command was found | `pyproject.toml` | Style can drift | Adopt and document `ruff format` if intended |
| MT5 credential lifecycle undocumented | Integration depends on a local terminal/account environment | `mt5/README.md` | Reproduction setup is incomplete | Document secure account/environment setup without committing secrets |

## 3) Security Concerns

| Risk | OWASP category (if applicable) | Evidence | Current mitigation | Gap |
|---|---|---|---|---|
| Untrusted CSV can cause processing failure | N/A (batch data integrity) | `src/hge_gold/data.py` | Schema, ordering, finiteness, and OHLC checks | No cryptographic trust policy for arbitrary supplied source files |
| Local terminal/account configuration is outside codebase controls | N/A | `mt5/README.md` | Signal CSV contract fails closed | No documented credential/storage policy |

## 4) Performance and Scaling Concerns

| Concern | Evidence | Current symptom | Scaling risk | Suggested improvement |
|---|---|---|---|---|
| Repeated model fitting across horizons, variants, and folds | `src/hge_gold/pipeline.py`, `src/hge_gold/modeling.py` | V3 produces several fitted bundles and ablations per run | Longer experiments as feature/model grid grows | Add experiment-level timing and controlled parallelism only after preserving reproducibility |
| Large binary model artifacts duplicated in immutable runs | scan output: model bundles are top largest files | Multiple copies per run | Disk growth | Use retention and storage policy while retaining required evidence |

## 5) Fragile/High-Churn Areas

| Area | Why fragile | Churn signal | Safe change strategy |
|---|---|---|---|
| `src/hge_gold/v2/pipeline.py` in git history / current `src/hge_gold/pipeline.py` workflow | Core selection, split, evaluation, and outputs are tightly orchestrated | Scan lists V2 pipeline as most changed file | Change with end-to-end, provenance, and split tests |
| Targets and evaluation | Target semantics affect every metric and backtest | Scan lists V2 targets/evaluation among high-churn paths | Change contract and tests together; never compare incompatible outputs |
| Persisted artifacts/models | Results are evidence and high-churn historical files | Scan lists metrics, predictions, and model bundles | Preserve manifests and use new run identities instead of overwriting evidence |

## 6) `[ASK USER]` Questions

1. [ASK USER] Should `docs/project_documentation_fa.md` become the canonical thesis-facing project document, replacing any older V2-only Persian documentation?
2. [ASK USER] Which academic citation style and chapter structure should be applied once the thesis file is provided?

## 7) Evidence

- `docs/codebase/.codebase-scan.txt`
- `artifacts/v3/market/locked_test_metrics.csv`
- `artifacts/v3/market/validation_diagnostics.json`
- `src/hge_gold/pipeline.py`
- `mt5/README.md`
