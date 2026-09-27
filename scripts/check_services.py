"""Check that Neo4j, MongoDB, Prefect and MLflow are reachable and compatible, and warn when
MLflow artifact transfers would bypass the tracking server.

uv run --env-file .env python scripts/check_services.py
"""

from __future__ import annotations

import asyncio
import os

import httpx

from linking_engine.graph.repo import GraphRepo
from linking_engine.ingest.mongo_repo import MongoRepo
from linking_engine.pipeline.health import (
    check_service_versions,
    mlflow_server_info,
    presigned_transfer_warnings,
)


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
    auth = (user, password) if user and password else None
    versions = await check_service_versions(
        os.environ["PREFECT_API_URL"], os.environ["MLFLOW_TRACKING_URI"], auth
    )
    for name, (client, server) in versions.items():
        print(f"{name} ok, client {client} server {server}")
    async with httpx.AsyncClient(timeout=10.0, auth=auth) as http:
        info = await mlflow_server_info(http, os.environ["MLFLOW_TRACKING_URI"])
    warnings = presigned_transfer_warnings(info, os.environ)
    for warning in warnings:
        print(f"WARNING: {warning}")
    if not warnings:
        print("mlflow artifacts ok, transfers go through the tracking server")


if __name__ == "__main__":
    asyncio.run(main())
