"""Score every candidate pair of one tenant with the baseline scorer as the score-pairs
Prefect flow: nothing is written to the stores, the scores are cached as Parquet beside the
feature matrix and the run is logged to MLflow.

uv run --env-file .env python scripts/score_pairs.py --tenant <tenant> [--cache-dir .cache/features]
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from linking_engine.discovery.scoring import summarise_scores
from linking_engine.pipeline.features import CACHE_DIR
from linking_engine.pipeline.flows import score_pairs_flow


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tenant", required=True)
    ap.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    args = ap.parse_args()
    report, path, mlflow_run = asyncio.run(score_pairs_flow(args.tenant, args.cache_dir))
    print(summarise_scores(report))
    print(f"scores {path}")
    print(f"mlflow run {mlflow_run}")


if __name__ == "__main__":
    main()
