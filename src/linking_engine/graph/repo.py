"""Neo4j access. The only module that talks to Neo4j; every method is tenant-scoped."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from importlib.resources import files
from itertools import batched
from typing import TYPE_CHECKING, Final, LiteralString, Self

from neo4j import READ_ACCESS, WRITE_ACCESS, AsyncGraphDatabase
from neo4j.exceptions import (
    AuthError,
    ConfigurationError,
    DriverError,
    Neo4jError,
    ServiceUnavailable,
    SessionExpired,
)
from pydantic import AnyUrl, ValidationError

from linking_engine.errors import (
    DatabaseAuthError,
    DatabaseError,
    DatabaseReadError,
    DatabaseUnavailableError,
    DatabaseWriteError,
    SchemaError,
)
from linking_engine.models import IssueFlag, Link, Page, TenantGraphCounts

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Mapping, Sequence
    from types import TracebackType

    from neo4j import AsyncDriver, AsyncManagedTransaction

VECTOR_DIMENSIONS: Final = 2048
VECTOR_INDEXES: Final = ("page_content", "page_gnn")
PAGE_BATCH: Final = 500
LINK_BATCH: Final = 1000

Row = dict[str, object]

# Crawl fields owned by ingestion. None removes the property; computed fields are never written here.
_CRAWL_PROPERTIES: Final = {
    "status_code": "statusCode",
    "content_hash": "contentHash",
    "word_count": "wordCount",
    "page_type": "pageType",
    "is_indexable": "isIndexable",
    "crawl_depth": "crawlDepth",
    "language": "language",
    "freshness": "freshness",
    "published_at": "publishedAt",
    "lifecycle_stage": "lifecycleStage",
}
PAGE_PROPERTIES: Final = {
    "url": "url",
    "is_placeholder": "isPlaceholder",
    **_CRAWL_PROPERTIES,
    "page_rank": "pageRank",
    "betweenness": "betweenness",
    "link_community_id": "linkCommunityId",
    "keyword_community_id": "keywordCommunityId",
    "hub_id": "hubId",
    "is_chunked": "isChunked",
    "embedding_model": "embeddingModel",
    "embedding_dimensions": "embeddingDimensions",
    "embedded_content_hash": "embeddedContentHash",
    "content_embedding": "content_embedding",
    "gnn_embedding": "gnn_embedding",
}
_INGESTED_LINK_PROPERTIES: Final = {
    "anchor_text": "anchorText",
    "link_position": "linkPosition",
    "is_follow": "isFollow",
    "surrounding_text": "surroundingText",
}
LINK_PROPERTIES: Final = {
    "position": "position",
    **_INGESTED_LINK_PROPERTIES,
    "anchor_type": "anchorType",
    "weight": "weight",
    "surrounding_embedding": "surroundingEmbedding",
    "target_status_code": "targetStatusCode",
    "issue_flags": "issueFlags",
    "verdict": "verdict",
}

# Status flags and the FIX verdict on edge r, derived from its target t. Other audit flags are kept.
_EDGE_STATUS: Final = """
    r.targetStatusCode = t.statusCode,
    r.issueFlags = [f IN coalesce(r.issueFlags, []) WHERE NOT f IN ['BROKEN', 'REDIRECTED']]
        + CASE WHEN t.statusIssue IS NULL THEN [] ELSE [t.statusIssue] END,
    r.verdict = CASE
        WHEN t.statusIssue IS NOT NULL THEN 'FIX'
        WHEN r.verdict = 'FIX' THEN null
        ELSE r.verdict
    END"""

# Inbound edges are re-flagged in the same statement, so a status change never leaves stale verdicts.
_UPSERT_PAGES: Final = (
    """
UNWIND $rows AS row
MERGE (p:Page {tenantId: $tenant, url: row.url})
SET p += row.props, p.isPlaceholder = false, p.statusIssue = row.statusIssue
WITH p
CALL (p) {
  MATCH (:Page)-[r:LINKS_TO]->(p)
  WITH r, p AS t
  SET"""
    + _EDGE_STATUS
    + """
}
RETURN count(p) AS n
"""
)
_UPSERT_PLACEHOLDERS: Final = """
UNWIND $urls AS url
MERGE (p:Page {tenantId: $tenant, url: url})
ON CREATE SET p.isPlaceholder = true
RETURN count(p) AS n
"""
_UPSERT_LINKS: Final = (
    """
UNWIND $rows AS row
MATCH (s:Page {tenantId: $tenant, url: row.source})
MATCH (t:Page {tenantId: $tenant, url: row.target})
MERGE (s)-[r:LINKS_TO {position: row.position}]->(t)
SET r += row.props,"""
    + _EDGE_STATUS
    + """
RETURN count(r) AS n
"""
)
_PRUNE_LINKS: Final = """
UNWIND $sources AS src
MATCH (s:Page {tenantId: $tenant, url: src.url})-[r:LINKS_TO]->(t:Page)
WHERE NOT [r.position, t.url] IN src.keep
DELETE r
RETURN count(r) AS n
"""
_GET_PAGES: Final = """
UNWIND $urls AS url
MATCH (p:Page {tenantId: $tenant, url: url})
RETURN p {.*} AS page
"""
_GET_PAGES_NO_VECTORS: Final = """
UNWIND $urls AS url
MATCH (p:Page {tenantId: $tenant, url: url})
RETURN p {.*, content_embedding: null, gnn_embedding: null} AS page
"""
_PAGE_AFTER: Final = """
MATCH (p:Page {tenantId: $tenant}) WHERE p.url > $after
RETURN p {.*} AS page ORDER BY p.url LIMIT $limit
"""
_PAGE_AFTER_NO_VECTORS: Final = """
MATCH (p:Page {tenantId: $tenant}) WHERE p.url > $after
RETURN p {.*, content_embedding: null, gnn_embedding: null} AS page ORDER BY p.url LIMIT $limit
"""
_LINKS_FROM: Final = """
UNWIND $urls AS url
MATCH (s:Page {tenantId: $tenant, url: url})-[r:LINKS_TO]->(t:Page)
RETURN s.url AS source, t.url AS target, properties(r) AS props
ORDER BY source, props.position
"""
_COUNTS: Final = """
MATCH (p:Page {tenantId: $tenant})
WITH count(p) AS total,
     sum(CASE WHEN coalesce(p.isPlaceholder, false) THEN 1 ELSE 0 END) AS ph,
     sum(CASE WHEN p.statusIssue = 'REDIRECTED' THEN 1 ELSE 0 END) AS redirected,
     sum(CASE WHEN p.statusIssue = 'BROKEN' THEN 1 ELSE 0 END) AS broken
OPTIONAL MATCH (:Page {tenantId: $tenant})-[r:LINKS_TO]->()
RETURN total - ph AS pages, ph AS placeholders, redirected, broken, count(r) AS links,
       sum(CASE WHEN r.verdict = 'FIX' THEN 1 ELSE 0 END) AS fix
"""
_DELETE_TENANT: Final = """
MATCH (n) WHERE n.tenantId = $tenant
CALL (n) { DETACH DELETE n } IN TRANSACTIONS OF $batch ROWS
RETURN count(*) AS n
"""
_MIGRATION_CONSTRAINT: Final = (
    "CREATE CONSTRAINT migration_name IF NOT EXISTS FOR (m:_Migration) REQUIRE m.name IS UNIQUE"
)
_APPLIED_MIGRATIONS: Final = "MATCH (m:_Migration) RETURN m.name AS name"
_RECORD_MIGRATION: Final = (
    "MERGE (m:_Migration {name: $name}) ON CREATE SET m.appliedAt = datetime() RETURN count(m) AS n"
)
_VECTOR_INDEXES: Final = """
SHOW VECTOR INDEXES YIELD name, options
RETURN name, options.indexConfig.`vector.dimensions` AS dimensions
"""
_COSINE_PROBE: Final = "RETURN vector.similarity.cosine([1.0, 0.0], [1.0, 0.0]) AS similarity"


@dataclass(frozen=True)
class Migration:
    name: str
    statements: tuple[str, ...]


def split_statements(text: str) -> tuple[str, ...]:
    # Statements must not contain a literal ';'.
    lines = [line for line in text.splitlines() if not line.lstrip().startswith("//")]
    return tuple(s.strip() for s in "\n".join(lines).split(";") if s.strip())


def load_migrations() -> tuple[Migration, ...]:
    folder = files("linking_engine.graph").joinpath("migrations")
    entries = sorted(
        (entry for entry in folder.iterdir() if entry.name.endswith(".cypher")),
        key=lambda entry: entry.name,
    )
    return tuple(
        Migration(entry.name, split_statements(entry.read_text(encoding="utf-8")))
        for entry in entries
    )


class GraphRepo:
    """Tenant-scoped Neo4j access. Create one per process and share it (pooled driver)."""

    def __init__(self, driver: AsyncDriver) -> None:
        self._driver = driver

    @classmethod
    async def connect(
        cls,
        uri: str,
        user: str,
        password: str,
        *,
        max_connection_pool_size: int = 50,
        connection_timeout: float = 10.0,
    ) -> Self:
        try:
            driver = AsyncGraphDatabase.driver(
                uri,
                auth=(user, password),
                max_connection_pool_size=max_connection_pool_size,
                connection_timeout=connection_timeout,
            )
        except (ConfigurationError, ValueError) as error:
            raise DatabaseUnavailableError(
                "neo4j", f"invalid connection settings: {error}"
            ) from error
        try:
            await driver.verify_connectivity()
        except AuthError as error:
            await driver.close()
            raise DatabaseAuthError("neo4j", f"credentials rejected at {uri}") from error
        except (Neo4jError, DriverError, OSError) as error:
            await driver.close()
            raise DatabaseUnavailableError("neo4j", f"cannot connect to {uri}: {error}") from error
        return cls(driver)

    async def close(self) -> None:
        await self._driver.close()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.close()

    # ── server and schema ────────────────────────────────────────────────────

    async def check_server(self) -> None:
        try:
            rows = await self._read(_COSINE_PROBE)
        except DatabaseReadError as error:
            raise SchemaError(
                "neo4j", "vector.similarity.cosine is unavailable; Neo4j 5.18+ is required"
            ) from error
        if _int_or_float(rows[0]["similarity"]) != 1.0:
            raise SchemaError("neo4j", "vector.similarity.cosine returned a wrong result")

    async def migrate(self) -> tuple[str, ...]:
        """Apply unrecorded migrations in filename order; returns the names applied.

        Schema statements can't share a transaction with writes, so each runs on
        its own and the migration is recorded last. A partly applied migration is
        re-run whole on the next call, which is why statements must be idempotent.
        """
        await self._auto(_MIGRATION_CONSTRAINT)
        applied = {str(row["name"]) for row in await self._read(_APPLIED_MIGRATIONS)}
        done: list[str] = []
        for migration in load_migrations():
            if migration.name not in applied:
                await self.apply_migration(migration)
                done.append(migration.name)
        await self.verify_vector_indexes()
        return tuple(done)

    async def apply_migration(self, migration: Migration) -> None:
        for statement in migration.statements:
            try:
                await self._auto(statement)
            except DatabaseError as error:
                raise SchemaError(
                    "neo4j", f"migration {migration.name} failed at: {statement[:120]}"
                ) from error
        await self._write(_RECORD_MIGRATION, name=migration.name)

    async def vector_index_dimensions(self) -> dict[str, int]:
        return {
            str(row["name"]): int(_int_or_float(row["dimensions"]))
            for row in await self._auto(_VECTOR_INDEXES)
        }

    async def verify_vector_indexes(self) -> None:
        found = await self.vector_index_dimensions()
        for name in VECTOR_INDEXES:
            if found.get(name) != VECTOR_DIMENSIONS:
                raise SchemaError(
                    "neo4j",
                    f"vector index {name} has dimensions {found.get(name)}, "
                    f"expected {VECTOR_DIMENSIONS}",
                )

    # ── writes ───────────────────────────────────────────────────────────────

    async def upsert_pages(
        self, tenant_id: str, pages: Sequence[Page], *, batch_size: int = PAGE_BATCH
    ) -> int:
        """Upsert crawled pages by (tenantId, url). A placeholder with the same url becomes crawled."""
        _require_tenant(tenant_id)
        if any(page.is_placeholder for page in pages):
            raise ValueError("placeholders are written with upsert_placeholders")
        written = 0
        for chunk in batched(pages, batch_size):
            rows = [
                {
                    "url": str(page.url),
                    "statusIssue": _to_property(status_issue(page.status_code)),
                    "props": {
                        prop: _to_property(getattr(page, field))
                        for field, prop in _CRAWL_PROPERTIES.items()
                    },
                }
                for page in chunk
            ]
            written += await self._write_all(_UPSERT_PAGES, len(rows), tenant=tenant_id, rows=rows)
        return written

    async def upsert_placeholders(
        self, tenant_id: str, urls: Sequence[str], *, batch_size: int = PAGE_BATCH
    ) -> int:
        """Create missing nodes for uncrawled link targets; existing nodes are untouched."""
        _require_tenant(tenant_id)
        written = 0
        for chunk in batched(urls, batch_size):
            written += await self._write_all(
                _UPSERT_PLACEHOLDERS, len(chunk), tenant=tenant_id, urls=list(chunk)
            )
        return written

    async def replace_links(
        self,
        tenant_id: str,
        sources: Sequence[str],
        links: Sequence[Link],
        *,
        batch_size: int = LINK_BATCH,
    ) -> tuple[int, int]:
        """Make ``links`` the complete outgoing set of each url in ``sources``;
        other edges leaving those sources are deleted. Both endpoints must exist.
        Returns (written, deleted)."""
        _require_tenant(tenant_id)
        source_set = set(sources)
        stray = {str(link.source_url) for link in links} - source_set
        if stray:
            raise ValueError(f"links from urls not listed in sources: {sorted(stray)[:3]}")

        written = 0
        for chunk in batched(links, batch_size):
            rows = [
                {
                    "source": str(link.source_url),
                    "target": str(link.target_url),
                    "position": link.position,
                    "props": {
                        prop: _to_property(getattr(link, field))
                        for field, prop in _INGESTED_LINK_PROPERTIES.items()
                    },
                }
                for link in chunk
            ]
            written += await self._write_all(_UPSERT_LINKS, len(rows), tenant=tenant_id, rows=rows)

        keep: dict[str, list[list[object]]] = {url: [] for url in sources}
        for link in links:
            keep[str(link.source_url)].append([link.position, str(link.target_url)])
        deleted = 0
        for group in batched(keep.items(), batch_size):
            rows = [{"url": url, "keep": pairs} for url, pairs in group]
            deleted += _int(await self._write(_PRUNE_LINKS, tenant=tenant_id, sources=rows))
        return written, deleted

    async def delete_tenant(self, tenant_id: str, *, batch_size: int = 1000) -> int:
        _require_tenant(tenant_id)
        return _int(await self._auto(_DELETE_TENANT, tenant=tenant_id, batch=batch_size))

    # ── reads ────────────────────────────────────────────────────────────────

    async def get_pages(
        self,
        tenant_id: str,
        urls: Sequence[str],
        *,
        include_vectors: bool = False,
        batch_size: int = PAGE_BATCH,
    ) -> list[Page]:
        _require_tenant(tenant_id)
        query = _GET_PAGES if include_vectors else _GET_PAGES_NO_VECTORS
        pages: list[Page] = []
        for chunk in batched(urls, batch_size):
            rows = await self._read(query, tenant=tenant_id, urls=list(chunk))
            pages.extend(_page_from(row["page"]) for row in rows)
        return pages

    async def iter_pages(
        self, tenant_id: str, *, include_vectors: bool = False, batch_size: int = PAGE_BATCH
    ) -> AsyncIterator[list[Page]]:
        _require_tenant(tenant_id)
        query = _PAGE_AFTER if include_vectors else _PAGE_AFTER_NO_VECTORS
        after = ""
        while True:
            rows = await self._read(query, tenant=tenant_id, after=after, limit=batch_size)
            if not rows:
                return
            pages = [_page_from(row["page"]) for row in rows]
            yield pages
            after = str(pages[-1].url)

    async def links_from(
        self, tenant_id: str, source_urls: Sequence[str], *, batch_size: int = PAGE_BATCH
    ) -> list[Link]:
        _require_tenant(tenant_id)
        links: list[Link] = []
        for chunk in batched(source_urls, batch_size):
            rows = await self._read(_LINKS_FROM, tenant=tenant_id, urls=list(chunk))
            links.extend(_link_from(row) for row in rows)
        return links

    async def counts(self, tenant_id: str) -> TenantGraphCounts:
        _require_tenant(tenant_id)
        row = (await self._read(_COUNTS, tenant=tenant_id))[0]
        return TenantGraphCounts(
            pages=_int_row(row, "pages"),
            placeholders=_int_row(row, "placeholders"),
            links=_int_row(row, "links"),
            redirected_pages=_int_row(row, "redirected"),
            broken_pages=_int_row(row, "broken"),
            fix_links=_int_row(row, "fix"),
        )

    # ── transport ────────────────────────────────────────────────────────────

    async def _read(self, query: LiteralString, **params: object) -> list[Row]:
        try:
            async with self._driver.session(default_access_mode=READ_ACCESS) as session:
                return await session.execute_read(_collect, query, params)
        except (Neo4jError, DriverError) as error:
            raise _translate(error, write=False) from error

    async def _write(self, query: LiteralString, **params: object) -> list[Row]:
        try:
            async with self._driver.session(default_access_mode=WRITE_ACCESS) as session:
                return await session.execute_write(_collect, query, params)
        except (Neo4jError, DriverError) as error:
            raise _translate(error, write=True) from error

    async def _auto(self, query: str, **params: object) -> list[Row]:
        # Auto-commit: required by schema statements and CALL ... IN TRANSACTIONS.
        try:
            async with self._driver.session(default_access_mode=WRITE_ACCESS) as session:
                result = await session.run(query, params)
                return [record.data() async for record in result]
        except (Neo4jError, DriverError) as error:
            raise _translate(error, write=True) from error

    async def _write_all(self, query: LiteralString, expected: int, **params: object) -> int:
        written = _int(await self._write(query, **params))
        if written != expected:
            raise DatabaseWriteError(
                "neo4j",
                f"wrote {written} of {expected} rows; rows with missing endpoint pages are dropped",
            )
        return written


async def _collect(
    tx: AsyncManagedTransaction, query: LiteralString, params: Mapping[str, object]
) -> list[Row]:
    result = await tx.run(query, dict(params))
    return [record.data() async for record in result]


def _translate(error: Neo4jError | DriverError, *, write: bool) -> DatabaseError:
    if isinstance(error, AuthError):
        return DatabaseAuthError("neo4j", "credentials rejected")
    if isinstance(error, ServiceUnavailable | SessionExpired):
        return DatabaseUnavailableError("neo4j", f"server unavailable: {error}")
    kind = DatabaseWriteError if write else DatabaseReadError
    return kind("neo4j", f"{type(error).__name__}: {error}")


def status_issue(status_code: int | None) -> IssueFlag | None:
    """3xx is REDIRECTED, 4xx and 5xx are BROKEN; 2xx and unknown (placeholders) are None."""
    if status_code is None or status_code < 300:
        return None
    return IssueFlag.REDIRECTED if status_code < 400 else IssueFlag.BROKEN


def _require_tenant(tenant_id: str) -> None:
    if not tenant_id.strip():
        raise ValueError("tenant_id must be a non-empty string")


def _to_property(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, AnyUrl):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    return value


def _from_property(value: object) -> object:
    to_native = getattr(value, "to_native", None)
    return to_native() if callable(to_native) else value


def _page_from(node: object) -> Page:
    if not isinstance(node, dict):
        raise DatabaseReadError("neo4j", f"expected a page map, got {type(node).__name__}")
    data = {
        field: _from_property(node[prop])
        for field, prop in PAGE_PROPERTIES.items()
        if node.get(prop) is not None
    }
    try:
        return Page.model_validate(data)
    except ValidationError as error:
        raise DatabaseReadError(
            "neo4j", f"page {node.get('url')!r} does not fit the Page model: {error}"
        ) from error


def _link_from(row: Row) -> Link:
    props = row["props"]
    if not isinstance(props, dict):
        raise DatabaseReadError("neo4j", f"expected link properties, got {type(props).__name__}")
    data: dict[str, object] = {
        field: _from_property(props[prop])
        for field, prop in LINK_PROPERTIES.items()
        if props.get(prop) is not None
    }
    data["source_url"] = row["source"]
    data["target_url"] = row["target"]
    try:
        return Link.model_validate(data)
    except ValidationError as error:
        raise DatabaseReadError(
            "neo4j", f"link {row['source']!r} -> {row['target']!r} does not fit the Link model"
        ) from error


def _int(rows: list[Row]) -> int:
    return sum(_int_row(row, "n") for row in rows)


def _int_row(row: Row, key: str) -> int:
    return int(_int_or_float(row[key]))


def _int_or_float(value: object) -> int | float:
    if isinstance(value, int | float):
        return value
    raise DatabaseReadError("neo4j", f"expected a number, got {value!r}")
