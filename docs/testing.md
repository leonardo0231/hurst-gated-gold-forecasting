# Testing and verification

The automated suite covers data schema and edge cases, threshold conversion, target formulas, target/feature leakage guards, Hurst bounds, feature-matrix separation, temporal splits, locked-test isolation, probability vectors, selected-model coverage, Phase 5 target join, cost accounting, artifact status rules, decision self-hashes, API behavior, and an end-to-end smoke run.

Security scans are expected to report their raw findings. A failing test is not skipped or deleted to make the build green. External network, real broker, journal portal, and paid data-source tests are intentionally out of scope for the default offline execution and are reported as simulated or blocked—not passed as real integrations.

