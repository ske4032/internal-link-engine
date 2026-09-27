"""Score and cluster one tenant's crawled pages as the graph-analytics Prefect flow:
results are written to Neo4j and the run is logged to MLflow.

uv run --env-file .env python scripts/graph_analytics.py --tenant <tenant>
"""

from __future__ import annotations

import argparse
import asyncio

from linking_engine.pipeline.analytics import summarise
from linking_engine.pipeline.flows import graph_analytics_flow


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tenant", required=True)
    args = ap.parse_args()
    centrality, communities, hubs, mlflow_run = asyncio.run(graph_analytics_flow(args.tenant))
    print(summarise(centrality, communities, hubs))
    print(f"mlflow run {mlflow_run}")


if __name__ == "__main__":
    main()
