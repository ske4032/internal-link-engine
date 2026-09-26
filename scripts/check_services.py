"""Check that Neo4j, MongoDB, Prefect and MLflow are reachable and compatible.

uv run --env-file .env python scripts/check_services.py
"""

from __future__ import annotations

import asyncio
import os

from linking_engine.graph.repo import GraphRepo
from linking_engine.ingest.mongo_repo import MongoRepo
from linking_engine.pipeline.health import check_service_versions


async def main() -> None:
    async with await GraphRepo.connect(
        os.environ["NEO4J_URI"], os.environ["NEO4J_USER"], os.environ["NEO4J_PASSWORD"]
    ) as graph:
        await graph.check_server()
        print(f"neo4j ok, vector indexes {await graph.vector_index_dimensions()}")
    async with await MongoRepo.connect(os.environ["MONGO_URI"], os.environ["MONGO_DB"]):
        print("mongodb ok")
    user = os.environ.get("MLFLOW_TRACKING_USERNAME")
    password = os.environ.get("MLFLOW_TRACKING_PASSWORD")
    versions = await check_service_versions(
        os.environ["PREFECT_API_URL"],
        os.environ["MLFLOW_TRACKING_URI"],
        (user, password) if user and password else None,
    )
    for name, (client, server) in versions.items():
        print(f"{name} ok, client {client} server {server}")


if __name__ == "__main__":
    asyncio.run(main())
