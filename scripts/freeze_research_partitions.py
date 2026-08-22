from __future__ import annotations

import argparse
from pathlib import Path

from hge_gold.partitions import freeze_physical_partitions


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze disjoint development/audit CSV partitions")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--boundary", default="2023-07-03")
    args = parser.parse_args()
    root = args.project_root.resolve()
    manifests = freeze_physical_partitions(
        root / "data/processed/XAUUSD_Daily_20110103_20260731_model.csv",
        development_dir=root / "data/partitions/v2/development",
        audit_dir=root / "data/partitions/v2/historical_audit_previously_revealed",
        boundary_date=args.boundary,
    )
    for role, path in manifests.items():
        print(f"{role}: {path}")


if __name__ == "__main__":
    main()
