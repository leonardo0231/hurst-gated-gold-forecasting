from __future__ import annotations

import argparse
from pathlib import Path

from hge_gold.research_experiments import run_preregistered_batch


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the frozen development-only research batch")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--bootstrap-iterations", type=int, default=2_000)
    args = parser.parse_args()
    print(
        run_preregistered_batch(
            args.project_root.resolve(), bootstrap_iterations=args.bootstrap_iterations
        )
    )


if __name__ == "__main__":
    main()
