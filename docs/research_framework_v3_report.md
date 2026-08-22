# Research Framework V3 — Audit, Implementation, and Evidence Report

Date: 2026-08-20  
Authoritative problem statement: `docs/project_problems_mapped_to_literature_fa.md`  
Immutable research run: `market_evidence-98f632731a9bdea9`

## Executive conclusion

The redesigned framework does **not** demonstrate 60–62% directional accuracy, robust incremental value from Hurst features, robust value from the stacked meta-classifier, or reliable profitability. This is the scientifically correct outcome of the current evidence.

All four historical holdout acceptance gates fail. Balanced accuracy is approximately 50–52%, and every 95% moving-block-bootstrap interval includes 0.50. The Hurst and stacker ablations are inconsistent across horizons. The simplified Python execution model is negative at H5, H10, and H20; the actual MT5 fixed-lot replay is materially negative at H10 and H20. The small MT5 H5 gain has profit factor 1.01 and Sharpe 0.01 and is not evidence of an economically useful strategy.

The 2023-07-03 onward period was already revealed in earlier project work. It is therefore labeled `historical_holdout_v1_previously_revealed`, not presented as a pristine confirmatory test, and was not used for V3 model, feature, threshold, or no-trade selection.

## Data and immutable identity

- Processed source: 4,017 XAUUSD D1 rows, 2011-01-03 through 2026-07-31.
- Processed SHA-256: `0c6d0b9ee76377d2d7048b53459fa6fc285af25fc21d76fad569454239f79a16`.
- Raw SHA-256: `36eac4639864f8f9dec7abc3d3b32e301d5f8a02414374585d8bfc22c6393fd8`.
- Source: MetaQuotes-Demo, MetaQuotes Ltd., XAUUSD, D1, UTC.
- Volume: broker tick volume, not real centralized exchange volume.
- Export date: 2026-08-07, derived from the raw file record; the sidecar itself was reconstructed on 2026-08-20.
- Metadata evidence status: reconstructed from file metadata, an exact live-MT5 bar comparison, and configuration—not claimed as a contemporaneous downloader receipt.
- Configuration SHA-256: `dbcaf6957aae2100a2ac4d6494838709782f6de88ae338f9944354204fd79daa`.
- Research code/dependency/config SHA-256: `846592c4661a5822f41fc1d894160819432eb661c670f87644e3ff5d237f7321`.
- Runtime SHA-256: `025eda872c1cc1502056010f7943b9ee8c7d52d10a26ee15086b2c662d13ae00`.

The execution manifest records source rows/range, symbol, timeframe, source type, broker, server, timezone, candle/session convention, export date, decision convention, raw/processed hashes, resolved configuration, runtime versions, git state, artifact hashes, and explicit limitations. The registered transformation `mt5_tab_export_to_canonical_ohlcv_v1` was executed and reproduced the canonical 4,017 rows. SHA-256 reads are chunked. Run identity fingerprints dependencies, runtime, scripts, and MQL5 source; an existing identity is rejected before mutable outputs are touched. Both an immutable copy and a trackable canonical receipt are written.

Daily XAUUSD candles are broker/session dependent. A different broker server, timezone, holiday calendar, or session close can change OHLC bars, tick volume, features, targets, signals, and reported performance.

## Statistical holdout evidence

| Horizon | N | Balanced accuracy | 95% block-bootstrap CI | Macro-F1 | Selected development strategy | Status |
|---:|---:|---:|---:|---:|---|---|
| 1 | 794 | 0.5039 | [0.4661, 0.5384] | 0.5004 | stacked meta / random forest | FAIL |
| 5 | 790 | 0.5219 | [0.4613, 0.5721] | 0.4582 | Extra Trees base | FAIL |
| 10 | 785 | 0.5047 | [0.4612, 0.5500] | 0.4035 | Extra Trees base | FAIL |
| 20 | 775 | 0.5038 | [0.4216, 0.5802] | 0.4820 | stacked meta / Extra Trees | FAIL |

The acceptance target was not changed after observing these results. Accuracy, class recalls, MCC, ROC-AUC, Brier score, log loss, expected calibration error, threshold, and confidence limits are retained in `locked_test_metrics.csv`.

## Target and execution redesign

Three distinct target families are now retained:

1. Statistical label: close at decision bar `t` to close at `t + H`.
2. Executable label/return: entry at the open/first tick of `t + 1`, exit at the open after `H` held bars.
3. Cost-aware `down / no-trade / up` actionability label using an adaptive movement threshold plus a cost buffer.

The actionable label is diagnostic only. No sample is selected or discarded because of a future realized move. Decision timestamp, feature-availability timestamp, entry/exit row, and entry/exit timestamp are explicit. This prevents the former same-close execution ambiguity and future-actionability selection bias.

## Chronological validation and multiple testing

- No random split exists.
- All horizons use the common locked boundary 2023-07-03.
- Walk-forward folds are expanding, chronological, and return the exact configured fold count.
- Label information intervals are purged from training when they overlap validation or the locked boundary.
- Meta-classifier training labels are separately purged before its selection fold, fixing the critical second-level overlap leak.
- An embargo of zero is documented for strict forward folds because no post-validation observations enter training. The development-only CPCV splitter supports a configured post-test embargo.
- CPCV split manifests are generated on development data only (8 groups, choose 2: 28 splits per horizon).
- Probabilistic Sharpe Ratio, Deflated Sharpe Ratio, and PBO implementations have unit tests.

Official PBO and Deflated Sharpe values are intentionally deferred: the pipeline does not yet persist the complete per-observation development return path for every registered trial. Computing them from incomplete trials or the revealed holdout would create false precision.

## Hurst and model ablations

The former one-scale rescaled-range implementation is explicitly retained as `legacy`; it is no longer presented as a robust estimator. The primary estimator is a more defensible multiscale DFA1 on increments with causal windows, scale checks, missing availability states, and no clipping. Formal heavy-tail, outlier, window-length, and alternative-estimator sensitivity remains deferred. Hurst is used only as a possible regime descriptor, never as a direct directional law.

Each horizon evaluates a preregistered matrix: DFA-Hurst with/without stacked meta, no-Hurst with/without stacked meta, and legacy-Hurst without meta. The primary configuration is chosen from development validation only; holdout ablation results are descriptive.

Key result: no stable incremental contribution is supported. For example, H1 holdout balanced accuracy is higher for DFA without meta (0.5572) and legacy without meta (0.5585) than for the primary stacked configuration (0.5039), while H20 reverses that ordering. This instability is evidence against a general Hurst or stacker claim, not a reason to select the best holdout variant.

The component previously called a “gate” is now accurately named a stacked meta-classifier. H5 and H10 select a horizon-specific base model; H1 and H20 select the stacker on development evidence. Every saved model has a pre-holdout selection receipt and hash.

## Economic analysis

The Python backtest first applies the frozen no-trade probability margin, then opens the next eligible signal whenever flat. This exactly matches the one-position MT5 scheduling rule. It uses next-open-to-open-after-H returns and a registered 5 bps round-trip cost assumption.

| Horizon | Trades | Python model cumulative return | Same-schedule always-long | Interpretation |
|---:|---:|---:|---:|---|
| 1 | 23 | +0.0032 | -0.0559 | Very sparse; not robust |
| 5 | 115 | -0.0624 | +0.2863 | Model underperforms |
| 10 | 63 | -0.3293 | +0.6190 | Model materially underperforms |
| 20 | 33 | -0.0081 | +0.6748 | Model underperforms |

Same-schedule long, short, cash, last-return momentum, 20-day trend, and volatility-filtered trend are reported, plus a full-period buy-and-hold schedule. Sparse-strategy Sharpe is annualized from the actual selected-trade frequency over the decision span, not `sqrt(252/H)`. The Python calculation is a constant-proportional log-return abstraction, while MT5 uses fixed 0.01-lot nominal exposure; their return magnitudes are therefore not directly comparable.

## Actual MetaTrader 5 Strategy Tester replay

`mt5/HGE_SignalReplay.mq5` was compiled with MetaEditor (0 errors, 0 warnings) and executed by MetaTrader 5 build 6120 against MetaQuotes-Demo XAUUSD. Each run used the frozen signal CSV, D1, fixed 0.01 lots, no optimization, 2023-07-03 to 2026-08-01, and `Model=2` (Open prices only). Open-prices mode is appropriate because the EA only acts once at a new D1 bar open; it does not use intrabar state. A real-tick attempt was not completed because 2023–2025 tick archives were absent locally.

| Horizon | Frozen signals / MT5 trades | History quality | Net profit (USD) | Profit factor | Sharpe | Max equity drawdown |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 23 / 23 | 99% | -5.30 | 0.99 | -0.03 | 338.11 (3.29%) |
| 5 | 115 / 115 | 99% | +35.73 | 1.01 | 0.01 | 1,797.68 (17.74%) |
| 10 | 63 / 63 | 99% | -1,045.37 | 0.70 | -0.32 | 2,468.40 (24.43%) |
| 20 | 33 / 33 | 99% | -526.12 | 0.78 | -0.17 | 1,984.02 (18.73%) |

The H5 nominal gain is economically negligible relative to its drawdown and has no uncertainty-adjusted evidence. MT5 validates frozen-signal execution, not native MQL5 model inference, live trading, or future profitability. Each `artifacts/mt5/h*/mt5_evidence.json` records hashes, environment, tester parameters, and parsed report statistics; the HTML report, compiled EX5, preset, redacted tester configuration, and clean compiler receipt are archived beside it. `artifacts/mt5/mt5_evidence_index.json` hashes the complete layer and links it to the pipeline run ID.

## Multi-agent audit consolidation

Five specialized reviewers worked under non-overlapping file ownership before integration:

- Data/target audit: identified same-close ambiguity, future-actionability risks, incomplete raw lineage, and broker tick-volume semantics. Implemented the separated target contract, explicit timestamps, volume mapping, source metadata, and tests.
- Validation/statistics audit: identified a critical meta-label overlap leak and the non-pristine holdout. Implemented second-level purging, common date boundary, exact chronological folds, CPCV split support, PSR/DSR/PBO functions, and tests.
- Hurst/regime audit: identified finite-sample and one-scale R/S weaknesses. Implemented causal DFA1, explicit legacy naming, missing regimes, and formal no-Hurst ablations.
- Model/backtest audit: identified that the “gate” was a stacker, same-close economic assumptions, weak no-trade semantics, and missing benchmarks. Implemented accurate naming, horizon-specific selection, calibration metrics, execution scheduling, benchmarks, and MT5 replay export.
- Reproducibility/QA audit: identified false market-evidence paths, incomplete provenance, mutable outputs, absent MT5 evidence, and CI gaps. Its independent final review then found timestamp, identity, sparse-Sharpe, and MT5-archive gaps; all release blockers were fixed, and its closure review found no remaining scientific or implementation release blocker.

## Implemented priorities and deferred work

Implemented now:

- Priority 0 provenance and reproducibility metadata.
- Priority 1 separated statistical, executable, and actionability targets.
- Common chronological holdout, interval purging, meta-label purging, CPCV split audit, and embargo rationale.
- Robust primary Hurst estimator plus formal no-Hurst/no-meta/legacy ablations.
- Horizon-specific development selection, calibration diagnostics, no-trade execution, costs, sparse-frequency annualization, and expanded causal economic benchmarks.
- MT5 EA, compiler/tester runner, frozen signals, genuine four-horizon Strategy Tester evidence, and Windows CI contract tests.
- Automated causality, alignment, split, statistics, provenance, model, evaluation, MT5-contract, and end-to-end tests.

Explicitly deferred:

- A genuinely untouched prospective confirmation period.
- Full development trial return registry needed for official PBO and Deflated Sharpe values.
- Native MQL5 feature/model inference parity; current MT5 layer replays frozen Python signals.
- A broker-calibrated spread/commission/swap model in Python and matched proportional-risk sizing across Python/MT5.
- Licensed, point-in-time macro/fundamental inputs. Current claims remain limited to broker-specific univariate OHLC and tick volume.
- Cross-broker/session replication and tick-history-complete real-tick MT5 replay.
- Hurst finite-sample, heavy-tail, outlier, window, and alternative-estimator sensitivity.
- Random-signal benchmark distributions and row-level Python-versus-MT5 fill reconciliation.

## Verification

- `ruff check .`: pass.
- `mypy src/hge_gold`: pass.
- `pytest --cov=src/hge_gold --cov-report=term-missing`: 41 passed, total coverage 86%.
- Full market pipeline: pass; immutable receipt written.
- MetaEditor: 0 errors, 0 warnings.
- MT5 Strategy Tester: four successful runs; reports and evidence manifests archived.
- Receipt verification: 27 pipeline files and 24 MT5 files matched their registered SHA-256 values.
- Duplicate-run guard: repeated identical execution failed before changing the canonical manifest.

## Skills Used

- `github:github`: used to inspect repository/branch/CI context and keep repository actions read-only. It contributed CI-scope and worktree-safety guidance; no remote issue, PR, commit, or push was performed.
- Official web research (not a plugin skill): used only for MetaQuotes/MetaTrader primary documentation on command-line tester configuration, tick models, UTC bar handling, and tester callbacks.

Not used because they were not needed: Hugging Face datasets (no HF data sourced), Trackio (local immutable artifacts already provide experiment tracking), Jupyter (no notebook-dependent analysis), and Vercel verification (not a web/Vercel application).

## Thesis-safe claim

The defensible conclusion is: **under the recorded MetaQuotes-Demo daily dataset, purged chronological protocol, and previously revealed 2023–2026 historical holdout, the tested models do not establish directional skill above chance, do not establish incremental Hurst or stacked-meta value, and do not establish robust profitability.** The project now provides a substantially stronger framework for future prospective testing, but future evidence must be reported without changing this conclusion retroactively.
