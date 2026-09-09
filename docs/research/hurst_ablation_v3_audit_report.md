# Hurst ablation v3 repair audit

## Scope and baseline

- Base branch: `main`
- Base commit: `0ba1fef471f026d614cb03a33dc643b8f603c694`
- Historical family: `executable_direction_hurst_ablation_v2`
- Historical run: `executable_direction_hurst_ablation_v2-20260822T165905Z`
- Repair family: `executable_direction_hurst_ablation_v3`
- Audit status at initialization: v2 is immutable historical evidence and is not valid for final leakage-free inference.

The working tree was clean after `git fetch --all --prune`, and all remote references were inspected before the repair branch was created.

## Repository paths inspected

The audit traced the following source and evidence paths:

`src/hge_gold/research_experiments_v2.py`, `src/hge_gold/research_validation.py`, `src/hge_gold/calibration.py`, `src/hge_gold/features.py`, `src/hge_gold/research_experiments.py`, `src/hge_gold/statistics.py`, `src/hge_gold/research_run.py`, `src/hge_gold/research_protocol.py`, the v2 protocol card/config/code manifest, the v2 registry, the immutable v2 run receipt, inner-selection artifacts, split manifests, and the research test suite.

## Reproduced root cause

The v2 call path is:

1. `run_trial_v2()` creates executable labels and an executable label endpoint at lines 345–346.
2. `_inner_choice()` fits each inner fold and appends only `_prediction_columns()` plus `y`, `raw_probability` at lines 265–279.
3. `_prediction_columns()` does not include `executable_label_end_index`.
4. The pooled frame is sorted by `row_id` and split at lines 280–284.
5. `fit_past_only_sigmoid()` is called with only `calibration_row_ids` and `prediction_row_ids` at lines 289–297. The calibrator checks row ordering, not the end of each calibration label interval.
6. The selected sigmoid is refitted on pooled inner OOF rows at lines 380–406 using the same incomplete boundary check.

The historical artifact makes the invalid ordering observable. For example, v2 H1 inner selection records a calibration row range ending at `673` and an evaluation range starting at `674`. The executable target endpoint is `row_id + execution_lag_bars + horizon`, so the calibration event at row `673` reaches row `675`; its closed label interval overlaps the evaluation window even though `673 < 674` is true. The corresponding inferred overlap counts are 2 rows for H1, 6 for H5, 11 for H10, and 21 for H20 at every outer fold. This is the interval defect reported by the existing independent QA evidence; the v2 artifacts remain unchanged.

## v2 search space and estimator

The historical v2 family declares exactly 12 trials:

- arms: `no_hurst`, `current_dfa_hurst`, `robust_hurst_regime`;
- horizons: H1, H5, H10, H20;
- base estimator: median imputation, standardization, balanced Logistic Regression with `C=0.2`, `max_iter=2000`, and seed 42;
- inner choices: no calibration/sigmoid calibration and no-trade margins 0.00/0.05;
- five outer folds and three inner folds.

v3 will preserve this search space and estimator semantics while making all reproducibility-critical defaults explicit in a new family specification.

## DFA1 source truth

The current implementation in `features.py` computes a causal rolling DFA1 estimate from log prices: first differences are demeaned and cumulatively integrated, scales are powers of two from 4 through `n_increments // 3`, forward and reverse non-overlapping segments are linearly detrended, RMS fluctuations are regressed in log-log space, and at least three usable scales are required. The source also requires at least 32 finite observations and rejects near-constant increments. v3 will document these values rather than redesigning the estimator.

## Required v3 controls

The repaired family will carry the executable label endpoint through inner predictions, purge calibration rows using the canonical closed-interval rule `calibration_label_end_index < evaluation_start_row_id`, and pass the same endpoint arrays into an independent fail-closed sigmoid guard. Sigmoid eligibility will be deterministic and recorded before any outer result interpretation. v3 will also prove exact common-calendar row and target alignment across the two preregistered contrasts and generate block-aware paired inference with Holm correction.
