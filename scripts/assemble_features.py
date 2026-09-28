"""Assemble the feature matrix of one tenant's candidate pairs as the feature-assembly Prefect
flow: nothing is written to the stores, the matrix is cached as Parquet under the cache
directory and the run is logged to MLflow.

uv run --env-file .env python scripts/assemble_features.py --tenant <tenant> [--cache-dir .cache/features]
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from linking_engine.discovery.features import CHUNK_PAIRS, summarise_features
from linking_engine.pipeline.features import CACHE_DIR
from linking_engine.pipeline.flows import feature_assembly_flow


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tenant", required=True)
    ap.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    ap.add_argument("--chunk-pairs", type=int, default=CHUNK_PAIRS)
    args = ap.parse_args()
    report, path, mlflow_run = asyncio.run(
        feature_assembly_flow(args.tenant, args.cache_dir, args.chunk_pairs)
    )
    print(summarise_features(report))
    print(f"feature matrix {path}")
    print(f"mlflow run {mlflow_run}")


if __name__ == "__main__":
    main()
