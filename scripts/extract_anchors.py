"""Extract anchor phrases for one tenant's candidate pairs and hub bridges as the
anchor-extraction Prefect flow: nothing is written to the stores, the anchors are written as
Parquet under the cache directory and the run is logged to MLflow.

uv run --env-file .env python scripts/extract_anchors.py --tenant <tenant> [--cache-dir .cache/features]
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from linking_engine.anchor.extraction import summarise_anchors
from linking_engine.pipeline.features import CACHE_DIR
from linking_engine.pipeline.flows import anchor_extraction_flow


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tenant", required=True)
    ap.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    args = ap.parse_args()
    report, path, mlflow_run = asyncio.run(anchor_extraction_flow(args.tenant, args.cache_dir))
    print(summarise_anchors(report))
    print(f"anchors {path}")
    print(f"mlflow run {mlflow_run}")


if __name__ == "__main__":
    main()
