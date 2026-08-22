# Testing Patterns

## 1) Test Stack and Commands

- Primary test framework: pytest 9.1.1.
- Assertion/mocking tools: native `assert`, pytest fixtures and temporary paths; [TODO] a dedicated mock library was not found.
- Commands:

```bash
uv run pytest
uv run pytest tests/test_end_to_end.py
uv run pytest --cov=hge_gold --cov-fail-under=80
```

## 2) Test Layout

- Test file placement pattern: `tests/test_*.py`.
- Naming convention: test functions begin with `test_`.
- Setup files: no repository-level `conftest.py` was found in the scanned test list.

## 3) Test Scope Matrix

| Scope | Covered? | Typical target | Notes |
|---|---|---|---|
| Unit | Yes | data, features, targets, splits, evaluation, modeling | Dedicated module-focused files exist |
| Integration | Yes | end-to-end pipeline and provenance | `test_end_to_end.py`, `test_provenance_v3.py` |
| E2E | Partial | Synthetic full pipeline and MT5 signal contract | No live broker/trading test is asserted |

## 4) Mocking and Isolation Strategy

- Main isolation approach: deterministic synthetic OHLCV data and pytest temporary directories.
- Isolation guarantees: end-to-end tests provide paths/configuration under `tmp_path`; production immutable runs reject overwrites.
- Common failure mode: market-evidence runs depend on local data files and are not suitable for CI as market-result tests.

## 5) Coverage and Quality Signals

- Coverage tool + threshold: pytest-cov, CI threshold 80%.
- Current reported coverage: [TODO] not measured in this documentation pass.
- Known gaps/flaky areas: real-tick MT5 replay is not covered because required tick archives are unavailable locally.

## 6) Evidence

- `pyproject.toml`
- `.github/workflows/ci.yml`
- `tests/test_end_to_end.py`
- `tests/test_mt5_contract.py`
