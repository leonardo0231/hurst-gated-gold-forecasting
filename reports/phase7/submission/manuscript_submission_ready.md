# HGE-Hybrid Gold Forecasting Framework

This reproducible MVP evaluates a leakage-safe, gold-only forecasting pipeline on non-confidential deterministic sample data. Results are implementation evidence, not paper-grade market evidence.


# Introduction

The project tests a multi-horizon, Hurst-aware forecasting architecture under strict temporal validation and artifact governance.


# Data, targets, and features

Targets use forward log returns for 1, 5, 10, and 20 trading days. Direction thresholds are volatility-scaled and trade labels use a five-basis-point round-trip economic threshold. Features are causal and gold-only.


# Methods

Models are trained with purged walk-forward validation. The final chronological block is locked until model selection is frozen. Preprocessing is fit only on each training partition.


# Results

The locked evaluation contains 16 task-horizon comparisons: 5 supported, 3 conditionally supported, and 8 not supported against preregistered baselines. All negative results remain in the tables.


# Discussion

The execution verifies the software and research controls. Statistical results from the deterministic fixture must not be generalized to financial markets.


# Limitations

The run uses deterministic sample data, close-based targets, gold-only features, deferred GARCH, and no paper-grade vendor provenance.


# Conclusion

The implementation demonstrates a leakage-safe end-to-end research workflow. It does not establish paper-grade predictive or trading claims.


# Reproducibility

Configuration, lockfile, manifests, model hashes, deterministic seeds, tests, and reports are included. Raw proprietary data is not included.


# Appendix

See artifact manifests and machine-readable decisions for exact provenance and status semantics.
