"""Retrieve the candidate sources of every indexable page of one tenant as the
candidate-retrieval Prefect flow: nothing is written to Neo4j, the run is logged to MLflow.

uv run --env-file .env python scripts/candidates.py --tenant <tenant> [--index page_gnn]
"""

from __future__ import annotations

import argparse
import asyncio
from typing import get_args

from linking_engine.discovery.candidates import summarise_candidates
from linking_engine.models import VectorIndex
from linking_engine.pipeline.flows import candidate_retrieval_flow


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tenant", required=True)
    ap.add_argument("--index", choices=get_args(VectorIndex), default="page_content")
    args = ap.parse_args()
    found, mlflow_run = asyncio.run(candidate_retrieval_flow(args.tenant, args.index))
    print(summarise_candidates(found.report))
    print(f"mlflow run {mlflow_run}")


if __name__ == "__main__":
    main()
