"""Find the bridge links that keep one tenant's hubs connected, as the hub-bridges Prefect flow:
nothing is written to the stores, the bridges and hub pairs are written as Parquet under the
cache directory and the run is logged to MLflow.

uv run --env-file .env python scripts/hub_bridges.py --tenant <tenant> [--cache-dir .cache/features]
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from linking_engine.discovery.bridges import summarise_bridges
from linking_engine.pipeline.features import CACHE_DIR
from linking_engine.pipeline.flows import hub_bridges_flow


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tenant", required=True)
    ap.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    args = ap.parse_args()
    report, path, mlflow_run = asyncio.run(hub_bridges_flow(args.tenant, args.cache_dir))
    print(summarise_bridges(report))
    print(f"bridges {path}")
    print(f"mlflow run {mlflow_run}")


if __name__ == "__main__":
    main()
