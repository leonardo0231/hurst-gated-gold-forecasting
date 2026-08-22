# Research-improvement loop: implementation and results

Date: 2026-08-22  
Canonical development batch: `executable_direction_hurst_ablation_v1-20260822T151315Z`  
Scientific outcome: **all 12 registered experiments rejected; no candidate frozen**

Independent release recommendation: **reject**. The negative development conclusion is
supported with limitations, but the software protocol is not release-ready and no model
candidate is eligible for release.

## 1. Protocol implemented

The project now has a formal development-only workflow with a frozen baseline, a
hash-chained append-only experiment registry, preregistered hypothesis cards and trial
budgets, exact nested purged expanding walk-forward manifests (five outer and three inner
folds), moving-block uncertainty, promotion gates, protected-partition access claims,
creation-exclusive candidate freezes, point-in-time feature availability contracts,
deferred-item records, deterministic signal schedules, and independent QA artifacts.

All preprocessing, calibration, and inner selection are fitted using outer-training data
only. Purging uses the executable event interval through `open[t+1+h]`; the older
close-to-close label endpoint is not used by the new splitter. In a strictly expanding
scheme, future observations never return to training, so a post-validation embargo is not
applicable. An optional conservative pre-validation gap remains supported and tested.

Principal implementation paths:

- `docs/research_protocol.md`
- `src/hge_gold/research_protocol.py`
- `src/hge_gold/research_validation.py`
- `src/hge_gold/availability.py`
- `src/hge_gold/research_experiments.py`
- `src/hge_gold/calibration.py`
- `scripts/run_research_batch.py`
- `protocol/schemas/`
- `protocol/hypothesis_cards/executable_direction_hurst_ablation_v1.json`
- `artifacts/research/registry/experiments.jsonl`

## 2. Data partitions and exposure

The requested 2022-01 through 2023-06 interval is **not fresh confirmation data**. The
frozen baseline proves that the prior fifth development fold already included:

| Horizon | Previously used fold-5 interval |
|---:|---|
| 1 | 2021-04-05 through 2023-06-29 |
| 5 | 2021-03-30 through 2023-06-23 |
| 10 | 2021-03-25 through 2023-06-16 |
| 20 | 2021-03-17 through 2023-06-02 |

The authoritative roles are therefore:

- Source rows through 2023-06: `development_reused_previously_exposed`.
- No historical `development_confirmation` partition is available.
- From 2023-07-03: `historical_audit_previously_revealed`.
- Only observations acquired after a frozen candidate and never examined during selection
  can become `actual_future_out_of_sample` evidence.

The new batch loaded exactly the first 3,213 source rows with `pandas.read_csv(nrows=...)`;
the last loaded bar was 2023-06-20. The run receipt states
`historical_audit_accessed: false`. Source and availability manifests record the broker,
session reconstruction, decision convention, tick-volume semantics, coverage, revision
policy, and unresolved provenance limitations.

## 3. Experiment inventory

The preregistered budget was exactly 12 trials: three arms × four horizons, fixed
class-balanced logistic regression (`C=0.2`, seed 42), with only `none`/`sigmoid`
calibration and 0/0.05 no-trade margin selected inside inner folds.

| Arm | H | Pooled BA | Macro-F1 | Registered 5 bps net log return | Decision |
|---|---:|---:|---:|---:|---|
| no Hurst | 1 | 0.5025 | 0.4853 | -0.4228 | reject |
| current DFA | 1 | 0.4995 | 0.4907 | -0.9438 | reject |
| robust Hurst | 1 | 0.4983 | 0.4981 | -0.8923 | reject |
| no Hurst | 5 | 0.4804 | 0.4790 | -0.6475 | reject |
| current DFA | 5 | 0.4865 | 0.4848 | -0.2751 | reject |
| robust Hurst | 5 | 0.5086 | 0.5086 | 1.4854 | reject |
| no Hurst | 10 | 0.5001 | 0.4947 | -0.1194 | reject |
| current DFA | 10 | 0.4833 | 0.4796 | -2.0350 | reject |
| robust Hurst | 10 | 0.4882 | 0.4871 | -1.0179 | reject |
| no Hurst | 20 | 0.4800 | 0.4341 | -9.2737 | reject |
| current DFA | 20 | 0.4832 | 0.4253 | -6.6747 | reject |
| robust Hurst | 20 | 0.5113 | 0.4585 | -3.3449 | reject |

Failed attempts are retained with the same metrics, manifests, decisions, and hash-chain
protection as the stronger attempts. The registry validates at 12 records with head hash
`2e2adf78501ac0c844b5bd2354d3b3bc1b97a7147e941ffea8a72b7222671e3e`.

## 4. Best development candidate

No development candidate passed. The highest pooled balanced accuracy was robust Hurst at
H20 (0.5113), with macro-F1 0.4585, recall-up 0.2121, and a 95% moving-block interval of
[0.4725, 0.5451]. Its five outer-fold balanced accuracies were 0.5000, 0.5194, 0.5183,
0.5000, and 0.5143. This is well below the registered promotion requirements and is not
distinguishable from chance.

Robust Hurst H5 was the only arm with a positive registered baseline-cost return, but it
had BA 0.5086 and macro-F1 0.5086 and failed stability, uncertainty, calibration,
same-schedule benchmark, DSR, and QA gates. It was correctly rejected.

## 5. Confirmation and historical audit

There was no one-time confirmation run because no unexposed historical confirmation period
exists and no candidate passed development gates. The new batch did not run the
`historical_audit_previously_revealed` partition. Existing V3 historical-audit results are
retained as prior evidence only: BA 0.5039 (H1), 0.5219 (H5), 0.5047 (H10), and 0.5038
(H20), all marked FAIL and with intervals including chance. They were not used to select
the new batch.

There is no actual future out-of-sample result in this project.

## 6. Hurst ablation

Neither current DFA Hurst nor the registered robust causal Hurst regime establishes
incremental development value over the no-Hurst arm. Robust Hurst is numerically better at
H5 and H20 but all values remain near chance and all promotion decisions fail. Current DFA
is worse than no-Hurst at H1, H10, and H20. No claim that Hurst adds predictive value is
supported.

## 7. Economic and MT5 evidence

The independent post-run overlap audit found that the completed batch's outer H5/H10/H20
economic paths realized overlapping horizon returns. Consequently, the registered
economic benchmark deltas, PBO, DSR, and `n_non_overlapping_trades` for those horizons are
not valid promotion evidence. They remain immutable in the registry and are explicitly
superseded for interpretation by
`artifacts/research/qa/post_run_economic_overlap_audit.json`; no decision was rewritten.

The corrected audit-only paths leave robust H5 positive (+0.5319 at 5 bps and +0.3341 at
10 bps across 396 non-overlapping trades), but this is not proof of profitability: the
classification and uncertainty gates fail, corrected same-schedule benchmark intervals,
PBO, and DSR are absent, and the result is post-run QA evidence rather than selection
evidence.

No new candidate MT5 backtest was authorized. Existing MT5 evidence is a compiled frozen
signal-replay EA using MetaQuotes-Demo XAUUSD, fixed 0.01 lots, no optimization, and Open
Prices Only. It is not real-tick evidence or native MQL5 inference parity. The new unit
parity test proves deterministic Python schedule/replay identity only; it does not prove
trade-by-trade terminal parity for a new candidate.

## 8. Leakage and reproducibility checks

Automated coverage includes chronological integrity, closed-interval purging, optional
pre-validation gap/legacy CPCV embargo, no label overlap, timestamp-aware backward joins,
no future fill, source vintage/revision policies, past-only calibration, train-only
transform fitting, protected audit access, registry mutation detection, trial-budget
enforcement, candidate-freeze mutation detection, stable promotion boundaries,
deterministic seeds, manifest hashing, and Python/MT5 schedule parity.

The canonical run receipt preserves source, development-frame, code, configuration,
dependency, hypothesis-card, and registry-head hashes. The source file hash is
`0c6d0b9ee76377d2d7048b53459fa6fc285af25fc21d76fad569454239f79a16`.

## 9. Limitations

- The 2023–2026 audit was previously revealed; it is not a pristine locked test.
- The proposed 2022–June 2023 confirmation period was already used in model selection.
- No fresh historical confirmation set or actual future OOS evidence exists.
- MT5 source timestamps/session boundaries are reconstructed from the available export;
  provenance is not sufficiently complete for an unconditional release claim.
- The completed batch's multi-day economic/PBO/DSR paths used overlapping outcomes.
- PBO was computed within three synchronized Hurst arms per horizon while DSR used the
  declared 12-trial budget; the invalid economic paths prevent interpreting either as a
  valid gate for this batch.
- No licensed timestamp-audited point-in-time DXY, rates, volatility, or COMEX dataset was
  integrated; arbitrary external features were not added.
- A tooling timeout caused an unreceipted duplicate run directory ending `151330Z` before
  the registry duplicate guard stopped it. It is noncanonical and retained for QA rather
  than silently deleted.

## 10. Deferred items

P0: preregister a separate v2 family with corrected non-overlapping economic, benchmark,
PBO, DSR, and trade-count paths. P0: acquire genuinely future data for prospective
confirmation. P1: licensed point-in-time macro/COMEX sources with release/vintage
timestamps. P1: real-tick and cross-broker MT5 validation. P2: native MQL5 inference
parity. P2: disposition of the transparent incomplete duplicate run after review.

## 11. Commands and verification outcomes

```text
uv run ruff check .
  All checks passed.

uv run mypy src/hge_gold
  Success: no issues found in 20 source files.

uv run pytest -q
  68 passed in 55.18s.

uv run pytest -q tests/test_research_experiments.py
  8 passed in 2.10s (includes train-only transform and deterministic-seed tests).

uv run python scripts/run_research_batch.py --project-root . --bootstrap-iterations 2000
  Canonical background child completed; 12 registry records and canonical run receipt.

AppendOnlyExperimentRegistry(...).read_and_validate()
  12 records; hash chain valid; head sequence 11.

validate_baseline_freeze(...)
  baseline-market-evidence-98f632731a9bdea9 validated.
```

The initial short-timeout shell left its child running. A second invocation was stopped by
`ProtocolViolation: Duplicate experiment_id`; this is evidence that the duplicate guard
worked, not a second valid batch.

## 12. Multi-agent consolidation

| Agent | Evidence-backed finding | Affected ownership | Priority / risk | Action |
|---|---|---|---|---|
| research protocol | 2022–June 2023 exposed; V3 was not nested; audit already revealed | protocol, registry, freeze artifacts | P0 selection contamination | Implemented protocol and baseline freeze |
| data/feature | broker/session provenance reconstructed; tick volume; no valid point-in-time macro source frozen | availability, source manifest, DQ tests | P0 leakage/provenance | Implemented contracts; macro deferred |
| validation/statistics | executable endpoint purging and nested outer/inner separation required; uncertainty and multiple testing must fail closed | splitter, statistics, gate tests | P0 false discovery | Implemented |
| model/experiment | fixed 12-trial time-safe Hurst ablation; no audit selection; MT5 replay scope only | runner, calibration, predictions, model specs, parity tests | P0 oversearch/execution mismatch | Implemented; all rejected |
| independent QA | 12/12 rejected; all CIs include chance; multi-day economics invalid; receipt, concurrency, audit-byte isolation, preregistration, and terminal-parity gaps | QA reports only | P0/P1 release control | Final recommendation: reject |

The signed independent reports are `docs/independent_qa_report.md` and
`artifacts/research/independent_qa_report.json`. Independent checks matched all 27 frozen
baseline hashes, found zero chronology or closed-interval purge violations in 12 split
manifests, confirmed every OOF prediction precedes the audit boundary, and validated the
12-record registry chain. QA additionally found that the receipt does not hash all outputs
or imported code, the registry is not a complete immutable return ledger, the development
run hashes the combined through-2026 source despite row-limited modeling, preregistration
did not freeze every executable degree of freedom, and the JSONL registry lacks an
inter-process lock.

## 13. Skills Used

No optional skill was used. GitHub state/CI, Hugging Face datasets/Trackio, Jupyter, and
Vercel verification were not needed. The local repository, its existing immutable artifact
system, and mandatory multi-agent audits were sufficient and avoided adding an irrelevant
tracking or hosting dependency.

## Evidence classification

- **Development evidence:** the new 12-trial nested purged batch on previously reused
  pre-2023-07 data; valid for rejection, not proof of future profitability.
- **One-time confirmation evidence:** none; no eligible historical partition exists and no
  candidate passed.
- **Previously revealed historical-audit evidence:** existing V3 2023-07 onward FAIL
  results only; not used in the new selection loop.
- **Actual future out-of-sample evidence:** none.
