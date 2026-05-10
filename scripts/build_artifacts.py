"""Build Streamlit artifacts from the adapted notebook pipeline.

Run from the repository root:
    python scripts/build_artifacts.py

This downloads the Kaggle basketball SQLite dataset and uses nba_api to fetch league game
logs. It can take several minutes because it reconstructs player Elo and team rolling features.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Allow `python scripts/build_artifacts.py` from repo root.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline import build_all_artifacts, save_artifacts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-year", type=int, default=1996, help="First NBA season start year, e.g. 1996 for 1996-97.")
    parser.add_argument("--end-year", type=int, default=2024, help="Last NBA season start year, e.g. 2024 for 2024-25.")
    parser.add_argument("--split-date", default="2022-10-01", help="Temporal evaluation split date.")
    parser.add_argument("--artifacts-dir", default="artifacts")
    args = parser.parse_args()

    outputs = build_all_artifacts(
        start_year=args.start_year,
        end_year=args.end_year,
        evaluation_split_date=args.split_date,
    )
    save_artifacts(outputs, args.artifacts_dir)

    print("\nArtifacts written to:")
    print(f"  {Path(args.artifacts_dir) / 'model_bundle.joblib'}")
    print(f"  {Path(args.artifacts_dir) / 'team_snapshot.csv'}")
    print(f"  {Path(args.artifacts_dir) / 'model_training_rows.parquet'}")
    print("\nEvaluation metrics:")
    for k, v in outputs.metrics.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
