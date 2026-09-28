"""Everything a tenant has in both stores, so a read-only stage can be shown to write nothing."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from linking_engine.graph.repo import GraphRepo
    from linking_engine.ingest.mongo_repo import MongoRepo


async def graph_state(graph: GraphRepo, tenant: str) -> tuple[list[object], list[object]]:
    """Every node of the tenant with its labels and properties, and every relationship
    between them with its type and properties, in a stable order."""
    nodes = await graph._read(
        "MATCH (n {tenantId: $t}) "
        "RETURN labels(n) AS labels, properties(n) AS props, elementId(n) AS id ORDER BY id",
        t=tenant,
    )
    relationships = await graph._read(
        "MATCH (a {tenantId: $t})-[r]->(b) "
        "RETURN elementId(a) AS a, type(r) AS type, properties(r) AS props, elementId(b) AS b, "
        "elementId(r) AS id ORDER BY id",
        t=tenant,
    )
    return [dict(row) for row in nodes], [dict(row) for row in relationships]


async def mongo_state(mongo: MongoRepo) -> dict[str, list[dict[str, object]]]:
    """Every document of every collection in the test database, in a stable order."""
    state: dict[str, list[dict[str, object]]] = {}
    for name in sorted(await mongo._db.list_collection_names()):
        state[name] = [doc async for doc in mongo._db[name].find({}).sort("_id", 1)]
    return state
