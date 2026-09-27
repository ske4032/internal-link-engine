"""Score one tenant's crawled pages with PageRank and exact betweenness, and write them to Neo4j.

uv run --env-file .env python scripts/graph_centrality.py --tenant <tenant>
"""

from __future__ import annotations

import argparse
import asyncio
import os

from linking_engine.graph.repo import GraphRepo
from linking_engine.pipeline.analytics import compute_centrality


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tenant", required=True)
    args = ap.parse_args()

    async with await GraphRepo.connect(
        os.environ["NEO4J_URI"], os.environ["NEO4J_USER"], os.environ["NEO4J_PASSWORD"]
    ) as graph:
        await graph.check_server()
        report = await compute_centrality(graph, args.tenant)
        print(report.model_dump_json(indent=2))


if __name__ == "__main__":
    asyncio.run(main())
