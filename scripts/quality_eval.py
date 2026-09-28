"""Evaluate one tenant's quality as the quality-eval Prefect flow: read-only against both
stores, logged to MLflow with alerts against the tenant's previous quality run. Keyword
relevance needs VOYAGE_API_KEY; without it that check is reported as not applicable.

uv run --env-file .env python scripts/quality_eval.py --tenant <tenant> [--cache-dir .cache/features]
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from linking_engine.pipeline.features import CACHE_DIR
from linking_engine.pipeline.flows import quality_eval_flow
from linking_engine.pipeline.quality import summarise_quality


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tenant", required=True)
    ap.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    args = ap.parse_args()
    report, mlflow_run = asyncio.run(quality_eval_flow(args.tenant, args.cache_dir))
    print(summarise_quality(report))
    print(f"mlflow run {mlflow_run}")


if __name__ == "__main__":
    main()
