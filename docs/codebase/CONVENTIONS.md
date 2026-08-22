# Coding Conventions

## 1) Naming Rules

| Item | Rule | Example | Evidence |
|---|---|---|---|
| Files | `snake_case.py` | `research_validation.py` | `src/hge_gold/` |
| Functions/methods | `snake_case` | `build_feature_matrix` | `src/hge_gold/features.py` |
| Types/interfaces | `PascalCase` dataclass/TypedDict | `ThesisConfig`, `ClassificationMetrics` | `src/hge_gold/config.py`, `src/hge_gold/evaluation.py` |
| Constants/env vars | Uppercase module constants; no env-var convention found | `REQUIRED_COLUMNS` | `src/hge_gold/data.py` |

## 2) Formatting and Linting

- Formatter: [TODO] no dedicated formatter command/config was found.
- Linter: Ruff, line length 100, target Python 3.11.
- Most relevant enforced rules: E, F, I, B, UP, SIM.
- Run commands: `uv run ruff check .`; `uv run mypy src/hge_gold`.

## 3) Import and Module Conventions

- Imports are standard-library, third-party, then local-relative imports, as represented in source files and checked by Ruff's import rules.
- Internal modules use relative imports such as `from .pipeline import run_thesis_pipeline`.
- No public barrel-export policy was found.

## 4) Error and Logging Conventions

- Error strategy: validation and invalid states raise `ValueError`, `RuntimeError`, or `FileExistsError`; no structured error envelope is present because this is a CLI/batch application.
- Logging style: no logging framework was found; the CLI prints resulting artifact paths through Typer.
- Sensitive-data redaction rules: [TODO] no credential-bearing configuration or explicit redaction policy was found.

## 5) Testing Conventions

- Tests are in `tests/` and follow `test_*.py` / `test_*` naming.
- Isolation uses pytest temporary paths and deterministic synthetic fixtures where appropriate.
- CI requires 80% coverage; the local README command reports coverage but does not specify a threshold.

## 6) Evidence

- `pyproject.toml`
- `src/hge_gold/cli.py`
- `src/hge_gold/data.py`
- `.github/workflows/ci.yml`
