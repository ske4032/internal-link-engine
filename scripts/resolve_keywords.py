"""Resolve every crawled page's target keyword as the resolve-keywords Prefect flow: the
keyword edges are replaced in Neo4j and the run is logged to MLflow. Run after load-graph and
before graph-analytics.

uv run --env-file .env python scripts/resolve_keywords.py --tenant <tenant>
"""

from __future__ import annotations

import argparse
import asyncio

from linking_engine.pipeline.flows import resolve_keywords_flow
from linking_engine.pipeline.keywords import summarise_keywords


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tenant", required=True)
    args = ap.parse_args()
    report, mlflow_run = asyncio.run(resolve_keywords_flow(args.tenant))
    print(summarise_keywords(report))
    print(f"mlflow run {mlflow_run}")


if __name__ == "__main__":
    main()
