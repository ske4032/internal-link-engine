"""Apply Neo4j migrations and Mongo indexes, then load one tenant from MONGO_DB into Neo4j and
group its exact duplicate pages, as the load-graph Prefect flow; the groups are logged to MLflow.

uv run --env-file .env python scripts/load_graph.py --tenant <tenant>
"""

from __future__ import annotations

import argparse
import asyncio

from linking_engine.pipeline.duplicates import summarise_duplicates
from linking_engine.pipeline.flows import load_graph_flow


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tenant", required=True)
    args = ap.parse_args()
    report, counts, duplicates, mlflow_run = asyncio.run(load_graph_flow(args.tenant))
    print(report.model_dump_json(indent=2))
    print(counts.model_dump_json())
    print(summarise_duplicates(duplicates))
    print(f"mlflow run {mlflow_run}")


if __name__ == "__main__":
    main()
