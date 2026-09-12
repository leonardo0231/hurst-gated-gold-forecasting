# Hurst Ablation v3: Final Results

## Executive summary

The finalized `executable_direction_hurst_ablation_v3` batch contains the preregistered 12 trials: three feature arms (`no_hurst`, `current_dfa_hurst`, and `robust_hurst_regime`) crossed with horizons H1, H5, H10, and H20. The primary paired contrast, current DFA1-Hurst versus no-Hurst, is negative at all four horizons. Every primary 95% block-bootstrap confidence interval includes zero, and every Holm-adjusted p-value is `0.44995500449955006`. The secondary robust-Hurst contrast is not consistent across horizons; its H1 raw p-value is `0.0143985601439856`, but the within-family Holm-adjusted p-value is `0.0575942405759424`.

All 12 trial-level promotion decisions are `reject`. The repaired primary and secondary purge checks, feature-causality checks, paired-alignment checks, and leakage-overlap checks pass in the final artifacts. However, each per-trial leakage audit records `provenance_complete=false`; consequently, the promotion gate `all_qa_flags_pass` is false for every trial. These results therefore support a bounded development-run conclusion, not a production or clean-unseen-test claim.

## 1. Run provenance

| Field | Final artifact value |
|---|---|
| Hypothesis family | `executable_direction_hurst_ablation_v3` |
| Batch ID | `executable_direction_hurst_ablation_v3-20260912T082354Z` |
| Finalization state | `finalized` |
| Finalized at UTC | `2026-09-12T08:34:43.827049+00:00` |
| Run receipt created at UTC | `2026-09-12T08:34:43.677102+00:00` |
| Run receipt state before atomic finalization | `prepared_for_atomic_finalization` |
| Git commit SHA | `c3377374ef62b0e2e6e8b91b985707a11de1af10` |
| Code-manifest SHA256 | `21ade5e704fc3344b8d52fdf203e0a33ec518b16f8ffdd940cb107e96c145292` |
| Executable-config SHA256 | `897cdcabd8174a7f65b93be9a871b1a8816f02f3750cf6e7000ce9f4a31bad83` |
| Partition role | `development_reused_previously_exposed` |
| Trial count | `12` |
| Outer selections | `60` |
| Primary purge verified | `true` |
| Secondary calibration purge verified | `true` |
| Maximum secondary overlap after purge | `0` |
| Paired alignment verified | `true` |
| Registry head | sequence `11`, record hash `8b63d5aa5071e7b1423120d6d13b784a06fbcba36e78d9abe274090c0fb66711` |
| Finalization receipt SHA256 | `82d23b97deb53d9839bf8eb7ebc6ec74ca21fed4838f440188e830844f819c9a` |

The run receipt records `historical_audit_accessed=false` and `historical_confirmation_available=false`. The v2 preservation hashes carried by the final receipt are registry `2340ab37fa44e548d02f03c45718225a7630aa361493bb608065d82d0a460dc0` and run receipt `bd6bf37c85cd495cf2c3114b043d6733557feecdb765d78d538aa5f2b5b02482`.

## 2. Calibration-purge repair evidence and QA

### 2.1 Purge rules

The primary split artifact records protocol `nested_purged_expanding_walk_forward_v1` with the closed interval convention `closed_[row_id,label_end_index]`. The secondary calibration guard records the strict rule:

```text
calibration_label_end_index < evaluation_start_row_id
```

The split manifests contain five outer folds and three inner folds per outer fold (`[3, 3, 3, 3, 3]`). They record zero pre-validation gap rows and zero post-validation embargo rows; the stated rationale is that strictly forward expanding folds do not train on future rows.

Across the 12 final leakage-audit JSON files:

| Check | Count / result |
|---|---:|
| Primary outer-fold overlap checks | `60`; maximum overlap `0` |
| Primary inner-fold overlap checks | `180`; maximum overlap `0` |
| Secondary calibration audit rows | `60`; maximum post-purge overlap `0` |
| Distinct per-horizon purge counts | `2`, `6`, `11`, `21` |

The secondary audit records the following purge count for every arm and every outer fold at each horizon:

| Horizon | Purged calibration rows per outer selection | Retained-label condition | Post-purge overlap |
|---:|---:|---|---:|
| H1 | `2` | `max_retained_label_end < evaluation_start_row` | `0` |
| H5 | `6` | `max_retained_label_end < evaluation_start_row` | `0` |
| H10 | `11` | `max_retained_label_end < evaluation_start_row` | `0` |
| H20 | `21` | `max_retained_label_end < evaluation_start_row` | `0` |

All 60 secondary-audit rows report `sigmoid_eligible=true`, `sigmoid_finite_probability=true`, and two retained sigmoid classes. The audit therefore documents the repaired calibration boundary rather than relying only on row-order checks.

### 2.2 QA status

The common QA signature in all 12 `leakage_audit.json` files is:

| QA flag | Final value |
|---|---:|
| `audit_isolated` | `true` |
| `feature_causality_verified` | `true` |
| `leakage_free` | `true` |
| `manifest_verified` | `true` |
| `paired_alignment_verified` | `true` |
| `primary_purge_verified` | `true` |
| `secondary_calibration_purge_verified` | `true` |
| `reproducible` | `true` |
| `provenance_complete` | `false` |

The final inventory records `reject` for all 12 trials, and each promotion decision includes `all_qa_flags_pass` among its failed gates. The false provenance-completeness flag is retained as a result limitation; it is not silently converted into a pass.

## 3. Exact DFA1 specification and hyperparameters

### 3.1 Feature arms

The final model specifications contain 65 features for `no_hurst`, 70 for `current_dfa_hurst`, and 71 for `robust_hurst_regime`. Relative to the common feature set, the Hurst-specific additions are:

| Arm | Hurst-specific features recorded in `feature_membership.json` |
|---|---|
| `no_hurst` | None |
| `current_dfa_hurst` | `hurst_dfa1_64`, `hurst_dfa1_128`, `hurst_dfa1_available`, `hurst_regime`, `hurst_regime_available` |
| `robust_hurst_regime` | `hurst_robust_median`, `hurst_robust_dispersion`, `hurst_robust_low_threshold`, `hurst_robust_high_threshold`, `hurst_robust_regime`, `hurst_robust_available` |

The exact DFA1 estimator specification recorded in the final model-spec JSONs is:

| Parameter | Value |
|---|---|
| Method | `DFA1` |
| Input | `log_price_first_difference` |
| Increment order | `1` |
| Detrend order | `1` |
| Windows | `[64, 128]` |
| Minimum observations | `32` |
| Minimum usable scales | `3` |
| Minimum scale | `4` |
| Scale growth | `2` |
| Maximum scale | `floor(n_increments/3)` |
| Segment scheme | `forward_backward_nonoverlapping` |
| Near-constant increment standard-deviation floor | `1E-12` |

For the regime arm, the recorded regime construction is causal: the input is `causal_median_of_dfa1_windows`, the threshold input is `one-period-lagged_hurst_summary`, the low and high quantiles are `0.33` and `0.67`, `min_periods` is `126`, and `regime_window` is `252`.

### 3.2 Model, calibration, and selection hyperparameters

The common pipeline recorded in the final model specifications is:

| Component | Exact specification |
|---|---|
| Imputer | `sklearn.impute.SimpleImputer(strategy=median, add_indicator=true, keep_empty_features=false)` |
| Scaler | `sklearn.preprocessing.StandardScaler(with_mean=true, with_std=true)` |
| Classifier | `sklearn.linear_model.LogisticRegression(C=0.2, class_weight=balanced, fit_intercept=true, max_iter=2000, penalty=l2, random_state=42, solver=lbfgs, tol=0.0001)` |
| Sigmoid calibrator | `sklearn.linear_model.LogisticRegression(C=1000000.0, fit_intercept=true, max_iter=100, penalty=l2, random_state=42, solver=lbfgs, tol=0.0001)` |

Inner selection uses `calibration_fraction=0.6666666667`, calibration options `none` and `sigmoid`, and no-trade margins `0.0` and `0.05`. The exact ranking order is balanced accuracy, macro-F1, cumulative net return after 5 bps, preference for the smaller margin, and preference for no calibration on a tie. The final specifications record five outer folds and three inner folds per outer fold.

## 4. Trial-level results: all 12 trials

Values below are taken from `experiment_inventory.json`. `Sigmoid / none` is the number of outer selections using each calibration choice; net return is the recorded net log return after 5 bps.

| Trial | Arm | Horizon | Pooled BA | Macro-F1 | Net log return @5bps | Non-overlap trades | Sigmoid / none | PBO | DSR probability | Decision |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `no_hurst-h1` | `no_hurst` | 1 | 0.5230686368307289 | 0.5155247865444473 | -0.5901086470116947 | 1062 | 1 / 4 | 0.34285714285714286 | 0.0010740015368670974 | reject |
| `no_hurst-h5` | `no_hurst` | 5 | 0.49715636695155396 | 0.4779333661741335 | -0.1998122324953891 | 381 | 2 / 3 | 0.34285714285714286 | 0.05679586182080054 | reject |
| `no_hurst-h10` | `no_hurst` | 10 | 0.5074394513521692 | 0.5019478453082503 | -0.06979894085319788 | 186 | 2 / 3 | 0.34285714285714286 | 0.1372952495004114 | reject |
| `no_hurst-h20` | `no_hurst` | 20 | 0.4993443937617896 | 0.43010139987750073 | -0.237140571919342 | 101 | 4 / 1 | 0.34285714285714286 | 0.057755386037802114 | reject |
| `current_dfa_hurst-h1` | `current_dfa_hurst` | 1 | 0.5081999220438681 | 0.46835419940952344 | -0.7623542455662888 | 1035 | 2 / 3 | 0.34285714285714286 | 0.00020371054448381498 | reject |
| `current_dfa_hurst-h5` | `current_dfa_hurst` | 5 | 0.4878106026072829 | 0.46287091661530333 | -0.11883350027802647 | 369 | 2 / 3 | 0.34285714285714286 | 0.08918347633602086 | reject |
| `current_dfa_hurst-h10` | `current_dfa_hurst` | 10 | 0.4751797369141441 | 0.40987694302427247 | 0.04826188027765481 | 101 | 4 / 1 | 0.34285714285714286 | 0.23778708622968536 | reject |
| `current_dfa_hurst-h20` | `current_dfa_hurst` | 20 | 0.4731490465173114 | 0.42157945123673657 | -0.28216230060376823 | 98 | 3 / 2 | 0.34285714285714286 | 0.04173336215572859 | reject |
| `robust_hurst_regime-h1` | `robust_hurst_regime` | 1 | 0.49490096027780733 | 0.49473701766936934 | -0.9327993960852679 | 1717 | 0 / 5 | 0.34285714285714286 | 0.0002531525115763711 | reject |
| `robust_hurst_regime-h5` | `robust_hurst_regime` | 5 | 0.5101738474355145 | 0.5100880252129705 | 0.3206197698290237 | 469 | 1 / 4 | 0.34285714285714286 | 0.3873817117809997 | reject |
| `robust_hurst_regime-h10` | `robust_hurst_regime` | 10 | 0.47641088074689875 | 0.4761778902125142 | 0.0014336410099453356 | 235 | 2 / 3 | 0.34285714285714286 | 0.16877484820123 | reject |
| `robust_hurst_regime-h20` | `robust_hurst_regime` | 20 | 0.5078061776253884 | 0.45343455118965637 | -0.335072041245559 | 86 | 3 / 2 | 0.34285714285714286 | 0.026452482653712583 | reject |

The final model specifications explicitly state `outer_fold_evaluation_specification_only_no_promoted_model` for the evaluated specifications.

## 5. Paired Hurst-versus-no-Hurst inference

The paired metric is `balanced_accuracy`. The final inference artifact verifies identical row IDs and identical `y_true` for every paired comparison. Paired row counts are 2,377 (H1), 2,369 (H5), 2,359 (H10), and 2,339 (H20).

Inference settings are exact as recorded in `paired_inference.json` and the paired CSVs:

- 95% moving-block bootstrap, block length `10`, `10000` iterations.
- Two-sided non-overlapping chronological block sign-flip test, block length `10`, `10000` permutations.
- Top-level seed `42`; per-horizon primary seeds are `43`, `47`, `52`, and `62`, and per-horizon secondary seeds are `1043`, `1047`, `1052`, and `1062`.
- Holm correction is applied separately across the four preregistered horizons within each contrast family; the robust-Hurst family remains secondary.

### Table 5. Primary paired current DFA1-Hurst versus no-Hurst results

These are the values in `paired_comparisons/table_5_v3.csv`; `n` is the paired row count recorded in the corresponding paired-inference CSV.

| Horizon | No-Hurst BA | Current DFA1-Hurst BA | Delta (Hurst − no-Hurst) | 95% block-bootstrap CI | Raw p | Holm p | n |
|---:|---:|---:|---:|---|---:|---:|---:|
| 1 | 0.5230686368307289 | 0.5081999220438681 | -0.014868714786860804 | [-0.030894262459083548, 0.0017156333532695473] | 0.12168783121687832 | 0.44995500449955006 | 2377 |
| 5 | 0.49715636695155396 | 0.4878106026072829 | -0.00934576434427109 | [-0.02209783541433406, 0.0026886734973247594] | 0.1743825617438256 | 0.44995500449955006 | 2369 |
| 10 | 0.5074394513521692 | 0.4751797369141441 | -0.032259714438025056 | [-0.06917885108149889, 0.006890040915916365] | 0.15058494150584942 | 0.44995500449955006 | 2359 |
| 20 | 0.4993443937617896 | 0.4731490465173114 | -0.026195347244478195 | [-0.05673857009392553, 0.006336179532205586] | 0.11248875112488751 | 0.44995500449955006 | 2339 |

### Secondary paired robust-Hurst-regime versus no-Hurst results

| Horizon | No-Hurst BA | Robust-Hurst BA | Delta (Hurst − no-Hurst) | 95% block-bootstrap CI | Raw p | Holm p | n |
|---:|---:|---:|---:|---|---:|---:|---:|
| 1 | 0.5230686368307289 | 0.49490096027780733 | -0.028167676552921572 | [-0.04813965960520689, -0.00700838998064057] | 0.0143985601439856 | 0.0575942405759424 | 2377 |
| 5 | 0.49715636695155396 | 0.5101738474355145 | 0.013017480483960564 | [-0.01766093306417382, 0.04032216721918955] | 0.45465453454654536 | 0.9093090690930907 | 2369 |
| 10 | 0.5074394513521692 | 0.47641088074689875 | -0.031028570605270422 | [-0.07028439743490097, 0.009687654301938969] | 0.1466853314668533 | 0.4400559944005599 | 2359 |
| 20 | 0.4993443937617896 | 0.5078061776253884 | 0.008461783863598804 | [-0.022250547139137204, 0.03937284056877043] | 0.607039296070393 | 0.9093090690930907 | 2339 |

## 6. Discussion

The primary result is directionally consistent but not positive: current DFA1-Hurst has a negative paired delta at H1, H5, H10, and H20. The intervals all cross zero, and the raw and Holm-adjusted p-values do not support an incremental balanced-accuracy advantage over the no-Hurst baseline. The evidence therefore does not establish that adding the current causal DFA1 features improves development out-of-sample classification.

The robust-Hurst regime arm is mixed rather than consistently beneficial. At H1, its delta is negative and its 95% interval is below zero, while the raw p-value is `0.0143985601439856`; after the preregistered within-family Holm correction, the p-value is `0.0575942405759424`. H5 and H20 have positive point estimates but intervals crossing zero, while H10 is negative with an interval crossing zero. The H1 raw signal is therefore not a multiplicity-confirmed, horizon-wide result.

The repair itself is supported by the interval evidence: calibration label endpoints were carried into the secondary guard, the strict endpoint rule was applied, and all recorded primary and secondary overlap counts are zero. This establishes that the specific calibration-boundary defect targeted by v3 is absent in the final audit artifacts. It does not, by itself, turn the development partition into an unseen test set or resolve the recorded provenance-completeness caveat.

## 7. Limitations of interpretation

1. The final receipt labels the partition `development_reused_previously_exposed`; these values are not a clean, newly locked test estimate.
2. `provenance_complete=false` is present in all 12 leakage audits, and `all_qa_flags_pass=false` is consequently present in all 12 promotion decisions. This is a material QA limitation, not a missing presentation detail.
3. The paired inference corrects for serial dependence using the recorded block length and chronological sign-flip design, but the result remains conditional on those choices and on the finite paired samples.
4. Holm correction is applied within each contrast family across four horizons. The robust-Hurst family remains secondary and is not pooled with the primary family for multiplicity correction.
5. No trial is promoted. The model-spec artifacts describe outer-fold evaluation specifications only, so the table should not be read as evidence of a deployable selected model.
6. The paired inference concerns balanced accuracy. The trial table's net-return, calibration, DSR, and PBO fields are separate recorded decision/economic diagnostics and should not be substituted for the paired balanced-accuracy estimand.

## 8. Conclusion

The final, repaired v3 batch does not provide evidence that current DFA1-Hurst features add incremental predictive value over the no-Hurst baseline across H1, H5, H10, and H20. The robust-Hurst-regime arm shows one unadjusted H1 signal, but it does not survive the preregistered Holm adjustment and is not consistent across horizons. The appropriate conclusion is therefore non-confirmation of incremental Hurst value in this development run, with no model promotion. The purge repair passes its recorded overlap and causality checks, while the unresolved provenance-completeness flag limits the strength of any broader generalization.

## Source artifacts

All reported values and QA statements were read from the finalized v3 JSON/CSV artifacts below; no source code, DOCX, JSON, or CSV was modified.

- `artifacts/research/finalizations/executable_direction_hurst_ablation_v3-20260912T082354Z.json`
- `artifacts/research/runs/executable_direction_hurst_ablation_v3-20260912T082354Z/run_receipt.json`
- `artifacts/research/runs/executable_direction_hurst_ablation_v3-20260912T082354Z/experiment_inventory.json`
- `artifacts/research/runs/executable_direction_hurst_ablation_v3-20260912T082354Z/feature_membership.json`
- `artifacts/research/runs/executable_direction_hurst_ablation_v3-20260912T082354Z/model_specs/*.json`
- `artifacts/research/runs/executable_direction_hurst_ablation_v3-20260912T082354Z/experiments/*/split_manifest.json`
- `artifacts/research/runs/executable_direction_hurst_ablation_v3-20260912T082354Z/experiments/*/leakage_audit.json`
- `artifacts/research/runs/executable_direction_hurst_ablation_v3-20260912T082354Z/paired_inference.json`
- `artifacts/research/runs/executable_direction_hurst_ablation_v3-20260912T082354Z/paired_comparisons/dfa1_vs_no_hurst.csv`
- `artifacts/research/runs/executable_direction_hurst_ablation_v3-20260912T082354Z/paired_comparisons/robust_vs_no_hurst.csv`
- `artifacts/research/runs/executable_direction_hurst_ablation_v3-20260912T082354Z/paired_comparisons/table_5_v3.csv`
- `artifacts/research/runs/executable_direction_hurst_ablation_v3-20260912T082354Z/secondary_calibration_audit.csv`
- `artifacts/research/runs/executable_direction_hurst_ablation_v3-20260912T082354Z/multiplicity_report.json`
- `artifacts/research/runs/executable_direction_hurst_ablation_v3-20260912T082354Z/v2_vs_v3_diagnostic.csv`

The run receipt also fingerprints the data/protocol inputs, including `data/partitions/v2/development/manifest.json`, `protocol/data_availability_manifest.json`, `protocol/hypothesis_cards/executable_direction_hurst_ablation_v3.json`, `protocol/executable_configs/executable_direction_hurst_ablation_v3.json`, and `protocol/code_manifests/executable_direction_hurst_ablation_v3.json`.
