# Controlled Research-Improvement Protocol

Protocol version: `research_protocol_v1`  
Baseline: `baseline-market-evidence-98f632731a9bdea9`

## Scientific status

This protocol controls model improvement; it does not promise a target accuracy. A failed hypothesis or the absence of a promotable candidate is a valid research outcome.

The period approximately January 2022 through June 2023 is already exposed. Current V3 fold five covers March/April 2021 through June 2023 and was used for probability-threshold and stacked-meta/base-strategy selection. A separate development experiment registry also records repeated evaluation of this window. It must therefore be labeled `development_reused_previously_exposed`, never confirmation.

The period beginning `2023-07-03` was previously revealed and is labeled exactly `historical_audit_previously_revealed`. There is no fresh historical confirmation period. Only data acquired after a candidate freeze can provide `actual_future_out_of_sample` evidence.

## Evidence roles

| Evidence | Allowed use | Prohibited claim |
|---|---|---|
| `development_reused_previously_exposed` | Nested inner selection and outer development evaluation | Confirmation or future OOS |
| `development_confirmation` | One-time use only if an exposure audit proves it untouched | Not available in this project |
| `historical_audit_previously_revealed` | Descriptive frozen-candidate audit after a pre-load claim | Locked, pristine, or confirmatory test |
| `actual_future_out_of_sample` | One-time prospective evidence after candidate freeze | Retrospective development evidence |

Code must call the partition-access guard before opening any protected label, prediction, return, or metric file. Historical-audit and future-OOS access requires a valid candidate freeze and a creation-exclusive claim receipt written before the load.

## Phase 0: immutable baseline

The baseline manifest records the pre-improvement identity, source lineage, result/model/signal hashes, MT5 evidence index, known exposure, and failed outcome. Existing V2, V3, MT5, model-experiment, and immutable run directories are evidence and must not be overwritten. New work writes only to new content-addressed run and candidate directories.

The authoritative pre-improvement identity is:

- code tree: `846592c4661a5822f41fc1d894160819432eb661c670f87644e3ff5d237f7321`;
- git commit: `b61658f4ac0d43933d9ef171f4122d3a26460aa6` with a dirty research worktree recorded;
- processed source: `0c6d0b9ee76377d2d7048b53459fa6fc285af25fc21d76fad569454239f79a16`;
- baseline run: `market_evidence-98f632731a9bdea9`.

## Phase 1: append-only experiment registry

Every attempted arm-horizon experiment consumes its declared family budget whether it succeeds, fails, or is rejected. The JSONL registry is hash chained: each line includes its sequence, previous-record hash, and canonical record hash. Appends validate the complete existing chain, reject duplicate experiment IDs, reject a changed family budget, and stop when the budget is exhausted. Existing bytes are never rewritten.

Every record contains hypothesis identity, timestamp, source/code/config/dependency hashes, availability convention, feature list, target/horizon, model and hyperparameters, fold definitions, training and validation metrics, calibration and economic metrics, and the promotion/rejection/failure decision with reason. Failed runs receive the same provenance standard as successful runs.

## Phase 2: preregistered batch

The initial card is `executable_direction_hurst_ablation_v1`:

- arms: no Hurst, current causal DFA-Hurst, robust Hurst/regime alternative;
- fixed horizons: 1, 5, 10, 20;
- budget: exactly 12 arm-horizon experiments;
- inner-only choices: calibration `none`/`sigmoid` and no-trade margin `0.00`/`0.05`;
- no post-registration feature, horizon, threshold, model, or budget expansion.

The four inner choices are selection alternatives inside each arm-horizon experiment, not additional outer trials. A failed run still consumes its arm-horizon slot.

## Phase 3: nested purged walk-forward

Use five common-calendar outer development blocks. For every outer fold, all model, feature, hyperparameter, calibration, threshold, and no-trade selection occurs only in purged chronological inner folds within that outer-training prefix. The selected inner specification is then evaluated once on the outer fold.

All transformations fit only on the applicable training partition. Training label intervals must end before validation begins. Horizon-specific final usable rows must reflect label and execution endpoints. Embargo is recorded per horizon with its feature/execution-dependence rationale. Report per-fold metrics, pooled outer OOF metrics, and time-dependence-aware moving/block-bootstrap intervals.

Outer labels never select the candidate evaluated on the same fold. The revealed historical audit is inaccessible to development code.

## Phase 4: promotion and rejection

A candidate can be frozen only when all preregistered development gates pass:

- median outer balanced accuracy at least `0.60`;
- pooled outer macro-F1 at least `0.55`;
- pooled recall for every actionable class at least `0.50`;
- at least four of five outer folds have balanced accuracy at least `0.55`;
- no material fold below `0.50`; any regime exception must have been defined before the batch and remains visible in sensitivity results;
- pooled block-bootstrap uncertainty distinguishes performance from chance;
- calibration does not materially worsen against the preregistered comparator;
- no inferiority to simple statistical and same-schedule economic benchmarks;
- positive evidence under baseline and higher-cost stress assumptions;
- complete provenance, reproducibility, leakage, registry, and test-isolation checks.

The exact statistical tolerances and economic uncertainty calculations belong in the frozen hypothesis/config evidence. A post-hoc regime story cannot turn a failed gate into a pass. Any failed mandatory gate produces `reject`, with its reason appended to the registry.

## Phase 5: freeze and protected evaluation

Candidate directories are creation exclusive. A freeze is allowed only after development promotion passes. The freeze manifest and copied model/prediction/config evidence are hashed by a separate receipt. Validation rejects any mutation. The manifest states that no fresh historical confirmation exists.

The freeze manifest is never edited. Subsequent one-time actions create new receipts. A partition claim must be written before labels or metrics are loaded. Confirmation or audit outcomes cannot change the frozen candidate. A failed candidate is closed; a new effort requires a separately preregistered batch.

## Phase 6: MT5 evidence

Only frozen signals/settings may be replayed. No MT5 optimization is allowed. Evidence must identify spread, commission, swap, slippage, symbol specification, lot sizing, sessions, tester model, and tick availability. Signal and trade parity must be reconciled row by row. Open-prices testing is acceptable only when the EA contract proves it acts exclusively at daily opens; the lack of complete real-tick history remains a limitation.

## Independent QA and release

Independent QA verifies registry continuity, budgets, hashes, exposure roles, partition-load claims, split manifests, reproducibility, unsupported claims, and Python/MT5 parity. It does not tune models. Its final disposition is `approve`, `approve_with_limitations`, or `reject`.

Development evidence, one-time confirmation evidence, revealed historical-audit evidence, and actual future OOS evidence must be reported in separate sections. In this project, the historical-confirmation section must state that none exists.

## Deferred work

The append-only deferred register records the lack of a fresh historical confirmation period, official PBO/DSR pending complete trial return paths, point-in-time macro sources, cross-broker/real-tick MT5 replication, and native MQL5 inference parity.
