"""Apply Neo4j migrations and Mongo indexes, then load one tenant from MONGO_DB into Neo4j.

uv run --env-file .env python scripts/load_graph.py --tenant action1
"""

from __future__ import annotations

import argparse
import asyncio
import os

from linking_engine.graph.repo import GraphRepo
from linking_engine.ingest.graph_load import load_tenant_graph
from linking_engine.ingest.mongo_repo import MongoRepo


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tenant", required=True)
    args = ap.parse_args()

    async with (
        await GraphRepo.connect(
            os.environ["NEO4J_URI"], os.environ["NEO4J_USER"], os.environ["NEO4J_PASSWORD"]
        ) as graph,
        await MongoRepo.connect(os.environ["MONGO_URI"], os.environ["MONGO_DB"]) as mongo,
    ):
        await graph.check_server()
        applied = await graph.migrate()
        print(f"neo4j migrations applied: {list(applied) or 'none pending'}")
        await mongo.ensure_indexes()
        print("mongo indexes ensured")
        report = await load_tenant_graph(mongo, graph, args.tenant)
        print(report.model_dump_json(indent=2))
        print((await graph.counts(args.tenant)).model_dump_json())


if __name__ == "__main__":
    asyncio.run(main())
