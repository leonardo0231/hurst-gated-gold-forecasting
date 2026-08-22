# Independent QA report

Date: 2026-08-22  
Canonical batch: `executable_direction_hurst_ablation_v1-20260822T151315Z`  
Final release recommendation: **reject**

## Release disposition

- **Software protocol release: reject.** The chronological/purged statistical core and
  append-only ledger are useful, but multi-day economic execution, immutable run receipts,
  concurrent execution, and protected-partition isolation need correction before the loop
  is release-ready.
- **Model candidate release: reject.** All 12 registered experiments failed. No candidate
  freeze exists, so confirmation, new historical audit, and candidate MT5 testing were not
  authorized.
- **Negative-result interpretation: supported with limitations.** It is defensible to
  report that this development batch produced no promotable candidate and no stable Hurst
  advantage. It is not defensible to use its registered multi-day economics, PBO, or DSR as
  promotion evidence.

## Evidence classification

| Evidence class | QA disposition |
|---|---|
| Development | Twelve nested purged experiments on `development_reused_previously_exposed`; usable for rejection, not future-performance claims |
| One-time confirmation | None; no eligible historical partition exists |
| Previously revealed historical audit | Existing V3 evidence from 2023-07-03 onward only; all horizons failed and it was not rerun for this batch |
| Actual future out-of-sample | None |

The 2022 through June 2023 period was already used in V3 fold five and selection. Its
classification as `development_reused_previously_exposed` is correct. The project also
correctly labels 2023-07-03 onward `historical_audit_previously_revealed`.

## Findings

### QA-P0-001 — Multi-day economic evidence is invalid for promotion

Priority: P0. The final batch loop builds H5/H10/H20 return paths from every action row,
not from the non-overlapping bar-open schedule. It then calls the number of nonzero rows
`n_non_overlapping_trades`. For example, no-Hurst reports 1,089/2,351/2,331 H5/H10/H20
trades, while a chronological one-position schedule contains 264/236/117.

The scheduler used during inner margin selection has a second defect: a rejected overlapping
signal extends `busy_until`, incorrectly suppressing later otherwise eligible trades. The
current parity test encodes this behavior.

Affected files include `src/hge_gold/research_experiments.py`,
`tests/test_mt5_parity.py`, the registry economic fields, and each completed run's economic
and promotion artifacts. The risk is material: registered economic benchmark deltas,
economic confidence intervals, PBO, DSR, and multi-day trade counts cannot support promotion
or profitability.

Recommended action: preserve v1 unchanged, correct scheduling and return construction, add
independent trade fixtures, and preregister a separate v2 family. The post-run overlap audit
is implemented and appropriately says that corrected values are diagnostic only; the actual
fix is deferred.

### QA-P0-002 — No candidate can be released

Priority: P0. The registry has 12 unique valid records and all are rejected. The highest
pooled balanced accuracy is robust-Hurst H20 at 0.5113, macro-F1 0.4585, with a 95% block
interval of [0.4725, 0.5451]. Every trial's interval includes 0.50. There is no candidate
freeze directory.

Recommended action: keep all trials closed and do not run confirmation, historical audit,
or MT5 for this family. This recommendation is implemented by the current rejection records.

### QA-P1-003 — Run receipts are not fully immutable or reproducible

Priority: P1. The canonical receipt hashes source/config/dependency/card, the registry head,
and only the main experiment module. It does not hash the 62 run files, 12 prediction files,
12 model-spec files, or imported helpers such as validation, statistics, calibration,
features, and targets. Therefore output mutation and helper-code drift cannot be detected
from the receipt.

Recommended action: next-family runs should stage outputs, hash every output and the entire
relevant code/runtime set, finalize atomically, and provide a read-only replay mode. Deferred.

### QA-P1-004 — Concurrent execution is unsafe

Priority: P1. The interrupted `151330Z` run contains one trial's artifacts, prediction, and
model spec but no inventory or receipt. Both processes could pass the empty-registry guard;
the registry uses read-then-append without an inter-process lock. The canonical chain is
intact, but this incident demonstrates ambiguous partial-output risk.

Recommended action: add batch/registry locking, staged finalization, and failed-run receipts;
quarantine or explicitly index the duplicate without silently deleting it. Deferred and
listed in the deferred register.

### QA-P1-005 — Audit isolation is not literal byte isolation

Priority: P1. Modeling reads exactly 3,213 rows with `nrows`, all saved OOF dates predate
2023-07-03, and independent inspection found no split/purge violation. However, the runner
uses one CSV containing data through 2026 and hashes the entire file while asserting
`historical_audit_accessed: false`. There is no evidence the audit outcomes influenced the
models, but the literal no-load assertion is stronger than the filesystem boundary proves.

Recommended action: create a development-only frozen source object and keep audit bytes
behind a separate claimed loader. Deferred.

### QA-P1-006 — Registry content is not a complete return ledger

Priority: P1. The 12-line hash chain, unique IDs, and family budget validate. But numeric
inner-choice scores and synchronized trial return paths are absent from the registry;
prediction files that could reconstruct them are not receipt-hashed. PBO uses three arms
within each horizon, while DSR declares 12 trials but derives its Sharpe distribution from
only three same-horizon arms. The overlapping paths independently invalidate both.

Recommended action: receipt-hash numeric inner evidence and aligned per-trial paths and
predefine whether the multiplicity family is per horizon or all 12 trials. Deferred.

### QA-P1-007 — Preregistration is partial

Priority: P1. The card freezes arms, horizons, budget, calibration, and margin choices, but
not the exact robust-window parameters, feature list, logistic settings, fold geometry,
bootstrap settings, or economic formulas. The executable code hash is recorded after the
run rather than in a separately time-anchored pre-run freeze.

Recommended action: the next hypothesis card should include or hash the complete executable
configuration before evaluation. Deferred.

### QA-P1-008 — MT5 parity scope is narrow

Priority: P1. No new candidate MT5 test was authorized. Existing MT5 evidence belongs to
the previously revealed V3 signal replay, uses Open Prices Only, and is neither real-tick
evidence nor native MQL5 inference. The new unit test validates a Python schedule against a
function that returns a validated copy; it does not reconcile terminal orders or fills.

Recommended action: retain only the deterministic Python schedule-contract claim. A future
frozen candidate must reconcile signals, orders, fills, exits, costs, and timestamps row by
row. The limitation is implemented; terminal parity remains deferred.

### QA-P0-009 — Exposure status is correctly handled

Priority: P0. The baseline freeze and availability manifest substantiate prior use of the
proposed confirmation interval. No new confirmation, audit claim, candidate freeze, or
future-OOS artifact exists. Preserve these labels. Implemented.

## Hurst conclusion

No stable incremental value is demonstrated. No-Hurst is numerically best at H1 and H10;
robust Hurst is numerically best at H5 and H20. All results remain near chance, every
confidence interval includes 0.50, and every arm-horizon trial is rejected.

## Economic and MT5 conclusion

The post-run audit gives corrected diagnostic returns, including a positive robust-Hurst H5
path, but those values are not preregistered selection evidence and corrected same-schedule
benchmarks, uncertainty, PBO, and DSR are absent. No profitability claim is supported.

No new candidate MT5 result exists. The archived V3 Open Prices Only replay is previously
revealed historical-audit evidence and cannot validate this batch.

## Verification performed

| Command/check | Outcome |
|---|---|
| `uv run ruff check .` | Pass: all checks passed |
| `uv run mypy src/hge_gold` | Pass: no issues in 20 source files |
| `uv run pytest -q` | Pass: 66 tests in 63.53 seconds |
| Independent registry chain recomputation | Pass: 12 unique records, sequence 0-11, head `2e2adf...671e3e` |
| Baseline artifact SHA-256 verification | Pass: all 27 registered files match |
| Split chronology/closed-interval purge audit | Pass: 12 manifests, zero outer/inner violations |
| Prediction-boundary audit | Pass: all 12 OOF files end before 2023-07-03 |
| Registry-to-split and metric linkage audit | Pass for all 12 experiments |
| Canonical versus duplicate first-trial byte comparison | Identical, but only partial reproduction and not a valid second run |
| `uv run python scripts/run_research_batch.py --bootstrap-iterations 1` | Expected fail-closed: batch already executed; no write |

The test suite is healthy, but passing tests do not cover the discovered executable-path,
artifact-receipt, concurrency, or real MT5 parity gaps.

## Unsupported claims

Do not claim that a candidate passed, Hurst adds stable value, the batch is profitable, the
batch has MT5 trade parity, 2022-June 2023 is confirmation, 2023-2026 is pristine locked
test evidence, or any existing result is actual future OOS.

## Final recommendation

**reject**

The negative development result itself is credible, but the protocol is not release-ready
and no model candidate is eligible for release.
