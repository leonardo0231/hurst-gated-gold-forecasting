# Data and model contract

The validated source contract requires ordered, unique dates; positive OHLC prices; nonnegative volume; finite values; and OHLC invariants. Phase 1 assigns causal availability and decision timestamps, builds an empirical gold calendar, and records the limitations of close-based simulated continuous data.

Targets are forward log returns, three-class direction labels, three-class trade labels, future realized variance, and future realized volatility for 1, 5, 10, and 20 trading days. Statistical direction thresholds are causal 63-day volatility scaled with a 5 bps floor. Economic action uses a fixed 5 bps round-trip threshold.

Features use only data at or before date `t`. No full-sample normalization, winsorization, imputation, ranking, or target-informed selection is performed. Fold-local preprocessing is fit during Phase 4. Feature matrices contain no targets; the modeling base joins only the registered primary target policy.

The final chronological block is globally locked. Model selection uses aggregated outer-validation metrics only. Phase 4 writes locked-test predictions without true targets. Phase 5 performs the one permitted join and evaluates the frozen selected map.

