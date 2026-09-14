from __future__ import annotations

import argparse
from pathlib import Path

from hge_gold.research_experiments_v3 import run_preregistered_batch_v3


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the isolated repaired v3 Hurst ablation batch"
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    print(run_preregistered_batch_v3(args.project_root.resolve()))


if __name__ == "__main__":
    main()
