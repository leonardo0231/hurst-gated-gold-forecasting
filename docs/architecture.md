# Architecture

The dependency direction is strictly forward:

```text
audited data → targets → causal features → purged modeling → locked evaluation
             → manuscript → submission package → lifecycle governance
```

Core scientific modules live under `src/hge_gold`: `data`, `targets`, `features`, `modeling`, and `evaluation`. Shared hash, atomic I/O, configuration, artifact, and decision helpers prevent phase-specific implementations from inventing incompatible conventions.

Every generated decision uses canonical JSON hashing with its own `decision_hash` excluded. Artifact manifests use the fixed object schema and the allowed statuses `CREATED`, `NOT_REQUIRED`, `DEFERRED`, `MISSING_REQUIRED`, and `INVALID`. JSON is authoritative; CSV is the human-readable export; the SHA-256 sidecar is not a manifest member.

The default run uses deterministic sample data. It exercises the complete computational path while correctly keeping paper-grade status `NOT_READY`. Governance phases remain conditional because target-journal selection and submission authorization are human/external actions.

