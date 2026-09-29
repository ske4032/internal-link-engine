"""Run one tenant's pipeline as the tenant-pipeline Prefect flow: every stage flow started once
the stages it needs have finished, independent stages side by side, and one MLflow run with
each stage's status, seconds, peak memory and run. prepare-corpus reads the crawl source; with
--from-stage only that stage and those after it run, once the outputs before it are found.
--reports adds the report stages, --retrain trains the ranker before rank-pairs.

uv run --env-file .env python scripts/run_pipeline.py --tenant <tenant> --source-db <db> --source-collection <c> [--retrain] [--reports]
uv run --env-file .env python scripts/run_pipeline.py --tenant <tenant> --from-stage <stage> [--retrain] [--reports]
"""

from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path

from linking_engine.pipeline.features import CACHE_DIR
from linking_engine.pipeline.flows import tenant_pipeline_flow
from linking_engine.pipeline.tenant_pipeline import NEEDS, summarise_pipeline


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--tenant", required=True)
    ap.add_argument("--source-db", help="crawl database (read-only)")
    ap.add_argument("--source-collection", help="crawl collection")
    ap.add_argument("--retrain", action="store_true", help="train the ranker before rank-pairs")
    ap.add_argument("--from-stage", choices=tuple(NEEDS))
    ap.add_argument("--reports", action="store_true", help="add the report stages")
    ap.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    args = ap.parse_args()
    if (args.source_db is None) != (args.source_collection is None):
        ap.error("--source-db and --source-collection go together")
    if args.source_db is not None and args.source_db == os.environ["MONGO_DB"]:
        ap.error("source and target database must differ: the source is read-only")
    report = asyncio.run(
        tenant_pipeline_flow(
            args.tenant,
            args.source_db,
            args.source_collection,
            retrain=args.retrain,
            from_stage=args.from_stage,
            reports=args.reports,
            cache_dir=args.cache_dir,
        )
    )
    print(summarise_pipeline(report))


if __name__ == "__main__":
    main()
