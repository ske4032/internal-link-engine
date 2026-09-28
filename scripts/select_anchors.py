"""Choose every pair's anchor for one tenant as the anchor-selection Prefect flow: nothing is
written to the stores, the choices and the pairs left without one are written as Parquet under the cache
directory and the run is logged to MLflow. Without a Voyage key the semantic rung is skipped.

uv run --env-file .env python scripts/select_anchors.py --tenant <tenant> [--cache-dir .cache/features]
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from linking_engine.anchor.scoring import summarise_selection
from linking_engine.pipeline.features import CACHE_DIR
from linking_engine.pipeline.flows import anchor_selection_flow


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tenant", required=True)
    ap.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    args = ap.parse_args()
    report, path, mlflow_run = asyncio.run(anchor_selection_flow(args.tenant, args.cache_dir))
    print(summarise_selection(report))
    print(f"anchor choices {path}")
    print(f"mlflow run {mlflow_run}")


if __name__ == "__main__":
    main()
