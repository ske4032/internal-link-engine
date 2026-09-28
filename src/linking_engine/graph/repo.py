"""Neo4j access. The only module that talks to Neo4j; every method is tenant-scoped."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from enum import Enum
from importlib.resources import files
from itertools import batched
from typing import TYPE_CHECKING, Final, LiteralString, Self

import numpy as np
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
from linking_engine.models import (
    CommunityContext,
    EmbeddingModelCount,
    EmbeddingSelection,
    EmbeddingTarget,
    IssueFlag,
    KeywordSource,
    Link,
    LinkGraphSnapshot,
    LinkText,
    Page,
    PageStructure,
    TargetSelection,
    TenantGraphCounts,
)
from linking_engine.urls import normalise_url

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, AsyncIterator, Iterable, Mapping, Sequence
    from types import TracebackType

    import numpy.typing as npt
    from neo4j import AsyncDriver, AsyncManagedTransaction

    from linking_engine.models import (
        AnchorKeyUpdate,
        EdgeRef,
        HubCentroid,
        KeywordTarget,
        PageCentrality,
        PageCommunities,
        PageHub,
        SentenceTarget,
        VectorIndex,
    )

VECTOR_DIMENSIONS: Final = 2048
# Vector index name -> the Page property it indexes.
VECTOR_PROPERTIES: Final = {"page_content": "content_embedding", "page_gnn": "gnn_embedding"}
VECTOR_INDEXES: Final = tuple(VECTOR_PROPERTIES)
PAGE_BATCH: Final = 500
LINK_BATCH: Final = 1000
CENTRALITY_BATCH: Final = 5000
# anchor_tenant_text rejects index entries over ~8 KB (an 8149-byte text failed at 8150 on 5.26);
# half of that leaves room for tenantId. A longer key would fail its flush on every run.
ANCHOR_KEY_MAX_BYTES: Final = 4096

Row = dict[str, object]

# Crawl fields owned by ingestion. None removes the property; computed fields are never written here.
_CRAWL_PROPERTIES: Final = {
    "status_code": "statusCode",
    "content_hash": "contentHash",
    "body_hash": "bodyHash",
    "word_count": "wordCount",
    "page_type": "pageType",
    "is_indexable": "isIndexable",
    "crawl_depth": "crawlDepth",
    "language": "language",
    "freshness": "freshness",
    "published_at": "publishedAt",
    "lifecycle_stage": "lifecycleStage",
    "menu_inlinks": "menuInlinks",
    "footer_inlinks": "footerInlinks",
}
PAGE_PROPERTIES: Final = {
    "url": "url",
    "is_placeholder": "isPlaceholder",
    **_CRAWL_PROPERTIES,
    "page_rank": "pageRank",
    "page_rank_percentile": "pageRankPercentile",
    "betweenness": "betweenness",
    "betweenness_percentile": "betweennessPercentile",
    "link_community_id": "linkCommunityId",
    "keyword_community_id": "keywordCommunityId",
    "content_community_id": "contentCommunityId",
    "is_link_pillar": "isLinkPillar",
    "is_keyword_pillar": "isKeywordPillar",
    "is_content_pillar": "isContentPillar",
    "is_hub_pillar": "isHubPillar",
    "is_orphan": "isOrphan",
    "is_dead_end": "isDeadEnd",
    "orphan_label": "orphanLabel",
    "hub_id": "hubId",
    "is_chunked": "isChunked",
    "embedding_model": "embeddingModel",
    "embedding_dimensions": "embeddingDimensions",
    "embedded_body_hash": "embeddedBodyHash",
    "embedded_at": "embeddedAt",
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
    "anchor_key": "anchorKey",
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
# A target is a crawled 2xx page whose vector is missing or was computed from another body.
_EMBEDDING_SELECTION: Final = """
MATCH (p:Page {tenantId: $tenant})
WITH p, CASE
    WHEN coalesce(p.isPlaceholder, false) THEN 'placeholder'
    WHEN p.statusCode IS NULL OR p.statusCode < 200 OR p.statusCode > 299 THEN 'non_2xx'
    WHEN p.embeddedBodyHash IS NOT NULL AND p.embeddedBodyHash = p.bodyHash
         AND p.content_embedding IS NOT NULL THEN 'up_to_date'
    ELSE 'target'
  END AS state
ORDER BY p.url
RETURN count(CASE state WHEN 'placeholder' THEN 1 END) AS placeholders,
       count(CASE state WHEN 'non_2xx' THEN 1 END) AS non_2xx,
       count(CASE state WHEN 'up_to_date' THEN 1 END) AS up_to_date,
       collect(CASE state WHEN 'target' THEN {url: p.url, body_hash: p.bodyHash} END) AS targets
"""
_EMBEDDING_MODELS: Final = """
MATCH (p:Page {tenantId: $tenant})
WHERE p.content_embedding IS NOT NULL
RETURN p.embeddingModel AS embedding_model, count(p) AS vectors
ORDER BY embedding_model
"""
# setNodeVectorProperty stores float32, half the size of a plain SET of the same list.
_WRITE_EMBEDDINGS: Final = """
UNWIND $rows AS row
MATCH (p:Page {tenantId: $tenant, url: row.url})
WHERE p.bodyHash = row.hash AND NOT coalesce(p.isPlaceholder, false)
CALL db.create.setNodeVectorProperty(p, 'content_embedding', row.vec)
SET p.embeddedBodyHash = row.hash,
    p.embeddingModel = $model,
    p.embeddingDimensions = $dimensions,
    p.embeddedAt = datetime()
RETURN count(p) AS n
"""
_DROPPED_EMBEDDINGS: Final = (
    "rows whose page is missing, is a placeholder or has a changed bodyHash are dropped"
)
# Keyset page over (source url, position). Ordering by the full (tenantId, url) index key lets the
# planner read the index in order with a PartialTop; ORDER BY s.url alone sorts every remaining edge.
_LINK_TEXTS_AFTER: Final = """
MATCH (s:Page {tenantId: $tenant}) WHERE s.url >= $after_url
MATCH (s)-[r:LINKS_TO]->()
WHERE s.url > $after_url OR r.position > $after_position
RETURN s.url AS source_url,
       r.position AS position,
       r.anchorText AS anchor_text,
       r.surroundingText AS surrounding_text,
       r.anchorKey AS anchor_key,
       r.anchorGeneric AS anchor_generic,
       r.surroundingEmbeddedHash AS surrounding_embedded_hash,
       r.surroundingEmbeddingModel AS surrounding_embedding_model
ORDER BY s.tenantId, s.url, r.position
LIMIT $limit
"""
# A null key removes anchorKey.
_SET_ANCHOR_KEYS: Final = """
UNWIND $rows AS row
MATCH (:Page {tenantId: $tenant, url: row.source})-[r:LINKS_TO {position: row.position}]->()
SET r.anchorKey = row.key, r.anchorGeneric = row.generic
RETURN count(r) AS n
"""
_DROPPED_EDGES: Final = "each (source url, position) must match exactly one LINKS_TO edge"
_ANCHOR_KEYS_TO_EMBED: Final = """
UNWIND $keys AS key
WITH key
WHERE NOT EXISTS {
  MATCH (a:Anchor {tenantId: $tenant, text: key})
  WHERE a.embeddingModel = $model AND a.embedding IS NOT NULL
}
RETURN key
"""
_WRITE_ANCHOR_EMBEDDINGS: Final = """
UNWIND $rows AS row
MERGE (a:Anchor {tenantId: $tenant, text: row.key})
WITH a, row
CALL db.create.setNodeVectorProperty(a, 'embedding', row.vec)
SET a.embeddingModel = $model,
    a.embeddingDimensions = $dimensions,
    a.embeddedAt = datetime()
RETURN count(a) AS n
"""
_DROPPED_ANCHORS: Final = "every key must merge exactly one Anchor"
_WRITE_CENTRALITY: Final = """
UNWIND $rows AS row
MATCH (p:Page {tenantId: $tenant, url: row.url})
WHERE NOT coalesce(p.isPlaceholder, false)
SET p.pageRank = row.pageRank,
    p.pageRankPercentile = row.pageRankPercentile,
    p.betweenness = row.betweenness,
    p.betweennessPercentile = row.betweennessPercentile
RETURN count(p) AS n
"""
# A page that was crawled before and is now only a link target keeps no stale scores.
_CLEAR_PLACEHOLDER_CENTRALITY: Final = """
MATCH (p:Page {tenantId: $tenant, isPlaceholder: true})
WHERE p.pageRank IS NOT NULL OR p.betweenness IS NOT NULL
REMOVE p.pageRank, p.pageRankPercentile, p.betweenness, p.betweennessPercentile
RETURN count(p) AS n
"""
# Null values remove the property: a page that lost its community keeps no stale id.
_WRITE_COMMUNITIES: Final = """
UNWIND $rows AS row
MATCH (p:Page {tenantId: $tenant, url: row.url})
WHERE NOT coalesce(p.isPlaceholder, false)
SET p.linkCommunityId = row.linkCommunityId,
    p.keywordCommunityId = row.keywordCommunityId,
    p.contentCommunityId = row.contentCommunityId,
    p.isLinkPillar = row.isLinkPillar,
    p.isKeywordPillar = row.isKeywordPillar,
    p.isContentPillar = row.isContentPillar,
    p.isOrphan = row.isOrphan,
    p.isDeadEnd = row.isDeadEnd,
    p.orphanLabel = row.orphanLabel
RETURN count(p) AS n
"""
_CLEAR_PLACEHOLDER_COMMUNITIES: Final = """
MATCH (p:Page {tenantId: $tenant, isPlaceholder: true})
WHERE p.isOrphan IS NOT NULL OR p.linkCommunityId IS NOT NULL
   OR p.keywordCommunityId IS NOT NULL OR p.contentCommunityId IS NOT NULL
REMOVE p.linkCommunityId, p.keywordCommunityId, p.contentCommunityId, p.isLinkPillar,
       p.isKeywordPillar, p.isContentPillar, p.isOrphan, p.isDeadEnd, p.orphanLabel
RETURN count(p) AS n
"""
_KEYWORD_TARGETS: Final = """
MATCH (p:Page {tenantId: $tenant})-[:TARGETS_KEYWORD]->(k:Keyword {tenantId: $tenant})
WHERE NOT coalesce(p.isPlaceholder, false)
RETURN p.url AS url, k.text AS text, k.language AS language
"""
# Keyset page over url: a (tenantId, url) seek read in index order and stopped at $limit.
# Without the hint, or ordered by p.url alone, the planner has scanned every tenant's pages or
# sorted the rest of this tenant's, loading every remaining vector for each page.
_PAGE_VECTORS: Final = """
MATCH (p:Page {tenantId: $tenant})
USING INDEX SEEK p:Page(tenantId, url)
WHERE p.url > $after AND NOT coalesce(p.isPlaceholder, false) AND p[$property] IS NOT NULL
RETURN p.url AS url, p[$property] AS vec
ORDER BY p.tenantId, p.url
LIMIT $limit
"""
_CANDIDATE_TARGETS: Final = """
MATCH (p:Page {tenantId: $tenant})
WHERE NOT coalesce(p.isPlaceholder, false)
WITH p, CASE
    WHEN NOT coalesce(p.isIndexable, 200 <= p.statusCode <= 299, false) THEN 'not_indexable'
    WHEN p[$property] IS NULL THEN 'without_vector'
    ELSE 'target'
  END AS state
ORDER BY p.url
RETURN count(p) AS crawled_pages,
       count(CASE state WHEN 'not_indexable' THEN 1 END) AS not_indexable,
       count(CASE state WHEN 'without_vector' THEN 1 END) AS without_vector,
       collect(CASE state WHEN 'target'
         THEN {url: p.url, indexable_assumed: p.isIndexable IS NULL} END) AS targets
"""
# Inbound and outbound count distinct crawled pages, so repeated links between a pair count once.
_PAGE_STRUCTURE: Final = """
MATCH (p:Page {tenantId: $tenant})
WHERE NOT coalesce(p.isPlaceholder, false)
RETURN p.url AS url,
       p.language AS language,
       coalesce(p.wordCount, 0) AS word_count,
       COUNT {
         MATCH (s:Page {tenantId: $tenant})-[:LINKS_TO]->(p)
         WHERE s <> p AND NOT coalesce(s.isPlaceholder, false)
         RETURN DISTINCT s
       } AS inbound,
       COUNT {
         MATCH (p)-[:LINKS_TO]->(t:Page {tenantId: $tenant})
         WHERE t <> p AND NOT coalesce(t.isPlaceholder, false)
         RETURN DISTINCT t
       } AS outbound,
       p.isOrphan AS is_orphan,
       p.pageRankPercentile AS page_rank_percentile,
       p.crawlDepth AS crawl_depth,
       p.linkCommunityId AS link_community_id,
       p.keywordCommunityId AS keyword_community_id,
       p.contentCommunityId AS content_community_id,
       p.hubId AS hub_id,
       coalesce(p.isHubPillar, false) AS is_hub_pillar
ORDER BY url
"""
_PAGE_LANGUAGES: Final = """
MATCH (p:Page {tenantId: $tenant})
WHERE NOT coalesce(p.isPlaceholder, false)
RETURN p.url AS url, p.language AS language
"""
_KEYWORD_EDGE_PAGES: Final = """
MATCH (p:Page {tenantId: $tenant})-[:TARGETS_KEYWORD {source: $source}]->()
RETURN DISTINCT p.url AS url
"""
# Everything of the source on a page goes except its kept (text, language) pairs: all of it on a
# placeholder, and any edge to another tenant's keyword.
_PRUNE_KEYWORD_EDGES: Final = """
UNWIND $pages AS page
MATCH (p:Page {tenantId: $tenant, url: page.url})-[r:TARGETS_KEYWORD {source: $source}]->(k)
WHERE coalesce(p.isPlaceholder, false)
   OR NOT coalesce(k.tenantId = $tenant AND [k.text, k.language] IN page.keep, false)
DELETE r
RETURN count(r) AS n
"""
# Rows whose url is not a crawled page match nothing, so they create no Keyword either.
_WRITE_KEYWORD_TARGETS: Final = """
UNWIND $rows AS row
MATCH (p:Page {tenantId: $tenant, url: row.url})
WHERE NOT coalesce(p.isPlaceholder, false)
MERGE (k:Keyword {tenantId: $tenant, text: row.text, language: row.language})
MERGE (p)-[r:TARGETS_KEYWORD {source: $source}]->(k)
SET r.priority = row.priority,
    r.isPrimary = row.isPrimary,
    r.rung = row.rung,
    r.rank = row.rank,
    r.resolved = CASE WHEN row.rung IS NULL THEN null ELSE true END
RETURN count(DISTINCT row) AS n
"""
_MARK_STRATEGIC_KEYWORDS: Final = """
MATCH (k:Keyword {tenantId: $tenant})
WITH k, EXISTS {
  (:Page {tenantId: $tenant})-[:TARGETS_KEYWORD {source: $strategic}]->(k)
} AS strategic
WHERE k.isStrategic IS NULL OR k.isStrategic <> strategic
SET k.isStrategic = strategic
RETURN count(k) AS n
"""
_COMMUNITY_CONTEXT: Final = """
MATCH (p:Page {tenantId: $tenant})
WHERE NOT coalesce(p.isPlaceholder, false)
RETURN p.url AS url, coalesce(p.menuInlinks, 0) AS menu, coalesce(p.footerInlinks, 0) AS footer,
       p.linkCommunityId AS link, p.keywordCommunityId AS keyword, p.contentCommunityId AS content,
       p.hubId AS hub
"""
_WRITE_PAGE_HUBS: Final = """
UNWIND $rows AS row
MATCH (p:Page {tenantId: $tenant, url: row.url})
WHERE NOT coalesce(p.isPlaceholder, false)
SET p.hubId = row.hubId, p.isHubPillar = row.isHubPillar
RETURN count(p) AS n
"""
_CLEAR_PLACEHOLDER_HUBS: Final = """
MATCH (p:Page {tenantId: $tenant, isPlaceholder: true})
WHERE p.hubId IS NOT NULL OR p.isHubPillar IS NOT NULL
REMOVE p.hubId, p.isHubPillar
RETURN count(p) AS n
"""
_UPSERT_HUBS: Final = """
UNWIND $hubs AS hub
MERGE (h:Hub {tenantId: $tenant, hubId: hub.hubId})
SET h.size = hub.size, h.pillarUrl = hub.pillarUrl, h.active = true, h.updatedAt = datetime()
WITH h, hub
CALL db.create.setNodeVectorProperty(h, 'centroid', hub.centroid)
RETURN count(h) AS n
"""
# Retired hubs keep their node and centroid, so their ids are never handed out again.
_RETIRE_HUBS: Final = """
MATCH (h:Hub {tenantId: $tenant})
WHERE coalesce(h.active, false) AND NOT h.hubId IN $ids
SET h.active = false, h.size = 0, h.pillarUrl = null, h.retiredAt = datetime()
RETURN count(h) AS n
"""
_STORED_HUBS: Final = """
MATCH (h:Hub {tenantId: $tenant})
RETURN h.hubId AS hub, coalesce(h.active, false) AS active, h.centroid AS centroid
"""
_SNAPSHOT_PAGES: Final = """
MATCH (p:Page {tenantId: $tenant})
RETURN p.url AS url, coalesce(p.isPlaceholder, false) AS placeholder
ORDER BY url
"""
_SNAPSHOT_LINKS: Final = """
MATCH (a:Page {tenantId: $tenant})-[:LINKS_TO]->(b:Page {tenantId: $tenant})
RETURN a.url AS source, b.url AS target
"""
# Each vector is sent once per unique sentence and fanned out to its edges server-side.
# Tenant-scoped: a vector is only ever reused within the tenant that paid for it.
_SURROUNDING_VECTORS: Final = """
UNWIND $hashes AS h
MATCH (:Page {tenantId: $tenant})-[r:LINKS_TO {surroundingEmbeddedHash: h}]->()
WHERE r.surroundingEmbeddingModel = $model AND r.surroundingEmbedding IS NOT NULL
WITH h, head(collect(r.surroundingEmbedding)) AS vec
RETURN h AS hash, vec
"""
_WRITE_SURROUNDING_EMBEDDINGS: Final = """
UNWIND $rows AS row
UNWIND row.edges AS edge
MATCH (:Page {tenantId: $tenant, url: edge.source})-[r:LINKS_TO {position: edge.position}]->()
CALL db.create.setRelationshipVectorProperty(r, 'surroundingEmbedding', row.vec)
SET r.surroundingEmbeddedHash = row.hash,
    r.surroundingEmbeddingModel = $model
RETURN count(r) AS n
"""
_CLEAR_SURROUNDING_EMBEDDINGS: Final = """
UNWIND $rows AS row
MATCH (:Page {tenantId: $tenant, url: row.source})-[r:LINKS_TO {position: row.position}]->()
REMOVE r.surroundingEmbedding, r.surroundingEmbeddedHash, r.surroundingEmbeddingModel
RETURN count(r) AS n
"""
_ANCHOR_EMBEDDING_MODELS: Final = """
MATCH (a:Anchor {tenantId: $tenant})
WHERE a.embedding IS NOT NULL
RETURN a.embeddingModel AS embedding_model, count(a) AS vectors
ORDER BY embedding_model
"""
_SURROUNDING_EMBEDDING_MODELS: Final = """
MATCH (:Page {tenantId: $tenant})-[r:LINKS_TO]->()
WHERE r.surroundingEmbedding IS NOT NULL
RETURN r.surroundingEmbeddingModel AS embedding_model, count(r) AS vectors
ORDER BY embedding_model
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
        urls = [normalise_url(url) for url in urls]
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
        sources = [normalise_url(url) for url in sources]
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

    async def write_embeddings(
        self,
        tenant_id: str,
        urls: Sequence[str],
        body_hashes: Sequence[str],
        vectors: npt.NDArray[np.float32],
        *,
        model: str,
        dimensions: int,
    ) -> int:
        """Write one flush of content vectors in one transaction: every row is written or none."""
        _require_tenant(tenant_id)
        urls = [normalise_url(url) for url in urls]
        _check_embeddings(
            urls,
            vectors,
            what="urls",
            model=model,
            dimensions=dimensions,
            counts=(("body hashes", len(body_hashes)),),
        )
        rows = [
            {"url": url, "hash": body_hash, "vec": vector}
            for url, body_hash, vector in zip(urls, body_hashes, vectors.tolist(), strict=True)
        ]
        return await self._write_exactly(
            _WRITE_EMBEDDINGS,
            len(rows),
            _DROPPED_EMBEDDINGS,
            tenant=tenant_id,
            rows=rows,
            model=model,
            dimensions=dimensions,
        )

    async def set_anchor_keys(
        self,
        tenant_id: str,
        updates: Sequence[AnchorKeyUpdate],
        *,
        batch_size: int = LINK_BATCH,
    ) -> int:
        """Set anchorKey and anchorGeneric on the given edges; a None key removes anchorKey."""
        _require_tenant(tenant_id)
        _check_unique_edges((u.source_url, u.position) for u in updates)
        if any(u.anchor_key is not None and not u.anchor_key.strip() for u in updates):
            raise ValueError("anchor_key must be None or a non-blank string")
        written = 0
        for chunk in batched(updates, batch_size):
            rows = [
                {
                    "source": u.source_url,
                    "position": u.position,
                    "key": u.anchor_key,
                    "generic": u.anchor_generic,
                }
                for u in chunk
            ]
            written += await self._write_all(
                _SET_ANCHOR_KEYS, len(rows), dropped=_DROPPED_EDGES, tenant=tenant_id, rows=rows
            )
        return written

    async def write_anchor_embeddings(
        self,
        tenant_id: str,
        keys: Sequence[str],
        vectors: npt.NDArray[np.float32],
        *,
        model: str,
        dimensions: int,
    ) -> int:
        """Write one flush of anchor vectors, one Anchor per key, in one transaction."""
        _require_tenant(tenant_id)
        _check_embeddings(keys, vectors, what="keys", model=model, dimensions=dimensions)
        oversize = sum(len(key.encode()) > ANCHOR_KEY_MAX_BYTES for key in keys)
        if oversize:
            raise ValueError(
                f"{oversize} keys exceed {ANCHOR_KEY_MAX_BYTES} UTF-8 bytes, the Anchor index limit"
            )
        rows = [
            {"key": key, "vec": vector} for key, vector in zip(keys, vectors.tolist(), strict=True)
        ]
        return await self._write_exactly(
            _WRITE_ANCHOR_EMBEDDINGS,
            len(rows),
            _DROPPED_ANCHORS,
            tenant=tenant_id,
            rows=rows,
            model=model,
            dimensions=dimensions,
        )

    async def write_surrounding_embeddings(
        self,
        tenant_id: str,
        targets: Sequence[SentenceTarget],
        vectors: npt.NDArray[np.float32],
        *,
        model: str,
        dimensions: int,
    ) -> int:
        """Write one flush of sentence vectors to every target edge; all edges or none."""
        _require_tenant(tenant_id)
        _check_embeddings(
            [t.sentence_hash for t in targets],
            vectors,
            what="sentence hashes",
            model=model,
            dimensions=dimensions,
        )
        _check_unique_edges((e.source_url, e.position) for t in targets for e in t.edges)
        rows = [
            {
                "hash": target.sentence_hash,
                "vec": vector,
                "edges": [{"source": e.source_url, "position": e.position} for e in target.edges],
            }
            for target, vector in zip(targets, vectors.tolist(), strict=True)
        ]
        return await self._write_exactly(
            _WRITE_SURROUNDING_EMBEDDINGS,
            sum(len(target.edges) for target in targets),
            _DROPPED_EDGES,
            tenant=tenant_id,
            rows=rows,
            model=model,
        )

    async def surrounding_vectors(
        self, tenant_id: str, hashes: Sequence[str], *, model: str, batch_size: int = LINK_BATCH
    ) -> dict[str, npt.NDArray[np.float32]]:
        """Stored vectors of this tenant for sentence hashes, from any edge with the same model."""
        _require_tenant(tenant_id)
        found: dict[str, npt.NDArray[np.float32]] = {}
        for chunk in batched(hashes, batch_size):
            rows = await self._read(
                _SURROUNDING_VECTORS, tenant=tenant_id, hashes=list(chunk), model=model
            )
            for row in rows:
                vector = row["vec"]
                if not isinstance(vector, list):
                    raise DatabaseReadError(
                        "neo4j", f"sentence {row['hash']!r} has no stored vector"
                    )
                found[str(row["hash"])] = np.asarray(vector, dtype=np.float32)
        return found

    async def clear_surrounding_embeddings(
        self, tenant_id: str, edges: Sequence[EdgeRef], *, batch_size: int = LINK_BATCH
    ) -> int:
        """Remove the surrounding vector, hash and model from each edge; returns edges cleared."""
        _require_tenant(tenant_id)
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        _check_unique_edges((e.source_url, e.position) for e in edges)
        cleared = 0
        for chunk in batched(edges, batch_size):
            rows = [{"source": e.source_url, "position": e.position} for e in chunk]
            cleared += await self._write_all(
                _CLEAR_SURROUNDING_EMBEDDINGS,
                len(rows),
                dropped=_DROPPED_EDGES,
                tenant=tenant_id,
                rows=rows,
            )
        return cleared

    async def write_centrality(
        self,
        tenant_id: str,
        scores: Sequence[PageCentrality],
        *,
        batch_size: int = CENTRALITY_BATCH,
    ) -> int:
        """Write every crawled page's scores in one transaction; a row that matches no crawled
        page rolls the whole write back. Placeholders lose any scores they had."""
        _require_tenant(tenant_id)
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        urls = [score.url for score in scores]
        if len(set(urls)) != len(urls):
            raise ValueError("one score per page: duplicate urls")
        chunks = [
            [
                {
                    "url": score.url,
                    "pageRank": score.page_rank,
                    "pageRankPercentile": score.page_rank_percentile,
                    "betweenness": score.betweenness,
                    "betweennessPercentile": score.betweenness_percentile,
                }
                for score in chunk
            ]
            for chunk in batched(scores, batch_size)
        ]
        return await self._write_pages(
            _WRITE_CENTRALITY, _CLEAR_PLACEHOLDER_CENTRALITY, tenant_id, chunks
        )

    async def write_communities(
        self,
        tenant_id: str,
        rows: Sequence[PageCommunities],
        *,
        batch_size: int = CENTRALITY_BATCH,
    ) -> int:
        """Write every crawled page's communities, pillar flags and link state in one
        transaction; a row that matches no crawled page rolls the whole write back."""
        _require_tenant(tenant_id)
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        urls = [row.url for row in rows]
        if len(set(urls)) != len(urls):
            raise ValueError("one row per page: duplicate urls")
        chunks = [
            [
                {
                    "url": row.url,
                    "linkCommunityId": row.link_community_id,
                    "keywordCommunityId": row.keyword_community_id,
                    "contentCommunityId": row.content_community_id,
                    "isLinkPillar": row.is_link_pillar,
                    "isKeywordPillar": row.is_keyword_pillar,
                    "isContentPillar": row.is_content_pillar,
                    "isOrphan": row.is_orphan,
                    "isDeadEnd": row.is_dead_end,
                    "orphanLabel": _to_property(row.orphan_label),
                }
                for row in chunk
            ]
            for chunk in batched(rows, batch_size)
        ]
        return await self._write_pages(
            _WRITE_COMMUNITIES, _CLEAR_PLACEHOLDER_COMMUNITIES, tenant_id, chunks
        )

    async def _write_pages(
        self,
        write: LiteralString,
        clear: LiteralString,
        tenant_id: str,
        chunks: list[list[Row]],
    ) -> int:
        try:
            async with self._driver.session(default_access_mode=WRITE_ACCESS) as session:
                return await session.execute_write(
                    _write_page_rows, write, clear, tenant_id, chunks
                )
        except (Neo4jError, DriverError) as error:
            raise _translate(error, write=True) from error

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
        urls = [normalise_url(url) for url in urls]
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
        source_urls = [normalise_url(url) for url in source_urls]
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

    async def embedding_selection(self, tenant_id: str) -> EmbeddingSelection:
        """Classify the tenant's pages for embedding; targets are ordered by url."""
        _require_tenant(tenant_id)
        row = (await self._read(_EMBEDDING_SELECTION, tenant=tenant_id))[0]
        targets = row["targets"]
        if not isinstance(targets, list):
            raise DatabaseReadError(
                "neo4j", f"expected a target list, got {type(targets).__name__}"
            )
        try:
            return EmbeddingSelection(
                targets=tuple(EmbeddingTarget.model_validate(target) for target in targets),
                up_to_date=_int_row(row, "up_to_date"),
                placeholders=_int_row(row, "placeholders"),
                non_2xx=_int_row(row, "non_2xx"),
            )
        except ValidationError as error:
            raise DatabaseReadError(
                "neo4j", f"embedding targets do not fit the model: {error}"
            ) from error

    async def embedding_models(self, tenant_id: str) -> tuple[EmbeddingModelCount, ...]:
        """Stored content vectors per embeddingModel, ordered by model with None last."""
        _require_tenant(tenant_id)
        return _model_counts(await self._read(_EMBEDDING_MODELS, tenant=tenant_id))

    async def anchor_embedding_models(self, tenant_id: str) -> tuple[EmbeddingModelCount, ...]:
        """Stored Anchor vectors per embeddingModel, ordered by model with None last."""
        _require_tenant(tenant_id)
        return _model_counts(await self._read(_ANCHOR_EMBEDDING_MODELS, tenant=tenant_id))

    async def surrounding_embedding_models(self, tenant_id: str) -> tuple[EmbeddingModelCount, ...]:
        """Stored surroundingEmbedding per surroundingEmbeddingModel, ordered with None last."""
        _require_tenant(tenant_id)
        return _model_counts(await self._read(_SURROUNDING_EMBEDDING_MODELS, tenant=tenant_id))

    async def iter_link_texts(
        self, tenant_id: str, *, batch_size: int = LINK_BATCH
    ) -> AsyncGenerator[list[LinkText], None]:
        """Every LINKS_TO edge's texts and markers, ordered by (source url, position); no vectors."""
        _require_tenant(tenant_id)
        # LIMIT 0 would end the scan at once and read as a tenant without edges.
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        after_url, after_position = "", -1
        while True:
            rows = await self._read(
                _LINK_TEXTS_AFTER,
                tenant=tenant_id,
                after_url=after_url,
                after_position=after_position,
                limit=batch_size,
            )
            if not rows:
                return
            try:
                texts = [LinkText.model_validate(row) for row in rows]
            except ValidationError as error:
                raise DatabaseReadError(
                    "neo4j", f"link texts do not fit the LinkText model: {error}"
                ) from error
            yield texts
            after_url, after_position = texts[-1].source_url, texts[-1].position

    async def anchor_keys_to_embed(
        self,
        tenant_id: str,
        keys: Sequence[str],
        *,
        model: str,
        batch_size: int = LINK_BATCH,
    ) -> tuple[str, ...]:
        """The keys, in input order, whose Anchor has no vector from ``model``."""
        _require_tenant(tenant_id)
        if not model.strip():
            raise ValueError("model must be a non-empty string")
        pending: set[str] = set()
        for chunk in batched(keys, batch_size):
            rows = await self._read(
                _ANCHOR_KEYS_TO_EMBED, tenant=tenant_id, keys=list(chunk), model=model
            )
            pending.update(str(row["key"]) for row in rows)
        return tuple(key for key in keys if key in pending)

    async def link_graph(self, tenant_id: str) -> LinkGraphSnapshot:
        """Every page (placeholders and orphans included) and body link of a tenant, consistently."""
        _require_tenant(tenant_id)
        try:
            async with self._driver.session(default_access_mode=READ_ACCESS) as session:
                pages, links = await session.execute_read(_snapshot, tenant_id)
        except (Neo4jError, DriverError) as error:
            raise _translate(error, write=False) from error
        try:
            return LinkGraphSnapshot(
                tenant_id=tenant_id,
                pages=tuple(str(url) for url, _ in pages),
                placeholders=tuple(bool(flag) for _, flag in pages),
                links=tuple((str(source), str(target)) for source, target in links),
            )
        except ValidationError as error:
            raise DatabaseReadError("neo4j", f"link graph of {tenant_id!r}: {error}") from error

    async def keyword_targets(self, tenant_id: str) -> list[tuple[str, str, str]]:
        """(page url, keyword text, keyword language) for every crawled page's target keyword."""
        _require_tenant(tenant_id)
        rows = await self._read(_KEYWORD_TARGETS, tenant=tenant_id)
        return [(str(r["url"]), str(r["text"]), str(r["language"])) for r in rows]

    async def content_vectors(
        self, tenant_id: str, *, batch_size: int = PAGE_BATCH
    ) -> dict[str, npt.NDArray[np.float32]]:
        """Content embeddings of the tenant's crawled pages, paged by url."""
        return await self.page_vectors(tenant_id, index="page_content", batch_size=batch_size)

    async def page_vectors(
        self, tenant_id: str, *, index: VectorIndex = "page_content", batch_size: int = PAGE_BATCH
    ) -> dict[str, npt.NDArray[np.float32]]:
        """The vectors in ``index``'s property of the tenant's crawled pages, paged by url."""
        _require_tenant(tenant_id)
        vector_property = _vector_property(index)
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        vectors: dict[str, npt.NDArray[np.float32]] = {}
        after = ""
        while True:
            rows = await self._read_values(
                _PAGE_VECTORS,
                tenant=tenant_id,
                property=vector_property,
                after=after,
                limit=batch_size,
            )
            for url, vector in rows:
                if not isinstance(vector, list):
                    raise DatabaseReadError("neo4j", f"page {url!r} has no stored vector")
                vectors[str(url)] = np.asarray(vector, dtype=np.float32)
            if len(rows) < batch_size:
                return vectors
            after = str(rows[-1][0])

    async def candidate_targets(
        self, tenant_id: str, *, index: VectorIndex = "page_content"
    ) -> TargetSelection:
        """The tenant's crawled pages that can be link targets, ordered by url, and how many
        of the rest are not indexable or have no vector in ``index``."""
        _require_tenant(tenant_id)
        rows = await self._read(
            _CANDIDATE_TARGETS, tenant=tenant_id, property=_vector_property(index)
        )
        try:
            return TargetSelection.model_validate(rows[0])
        except ValidationError as error:
            raise DatabaseReadError(
                "neo4j", f"candidate targets of {tenant_id!r}: {error}"
            ) from error

    async def page_structure(self, tenant_id: str) -> list[PageStructure]:
        """Language, size, body link counts, depth and cluster labels of every crawled page,
        ordered by url."""
        _require_tenant(tenant_id)
        rows = await self._read(_PAGE_STRUCTURE, tenant=tenant_id)
        try:
            return [PageStructure.model_validate(row) for row in rows]
        except ValidationError as error:
            raise DatabaseReadError("neo4j", f"page structure of {tenant_id!r}: {error}") from error

    async def page_languages(self, tenant_id: str) -> dict[str, str | None]:
        """The language of every crawled page; None when ingestion assigned none."""
        _require_tenant(tenant_id)
        languages: dict[str, str | None] = {}
        for row in await self._read(_PAGE_LANGUAGES, tenant=tenant_id):
            url, language = row["url"], row["language"]
            if not isinstance(url, str) or not (language is None or isinstance(language, str)):
                raise DatabaseReadError(
                    "neo4j", f"page {url!r} of {tenant_id!r} has language {language!r}"
                )
            languages[url] = language
        return languages

    async def community_context(self, tenant_id: str) -> list[CommunityContext]:
        """Template inlink counts and the previous run's community ids of every crawled page."""
        _require_tenant(tenant_id)
        rows = await self._read(_COMMUNITY_CONTEXT, tenant=tenant_id)
        try:
            return [
                CommunityContext(
                    url=str(r["url"]),
                    menu_inlinks=_int_row(r, "menu"),
                    footer_inlinks=_int_row(r, "footer"),
                    link_community_id=_optional_int(r["link"]),
                    keyword_community_id=_optional_int(r["keyword"]),
                    content_community_id=_optional_int(r["content"]),
                    hub_id=_optional_int(r["hub"]),
                )
                for r in rows
            ]
        except ValidationError as error:
            raise DatabaseReadError(
                "neo4j", f"community context of {tenant_id!r}: {error}"
            ) from error

    async def stored_hubs(self, tenant_id: str) -> tuple[dict[int, npt.NDArray[np.float32]], int]:
        """Centroids of the tenant's active hubs, and the first id no hub has ever used."""
        _require_tenant(tenant_id)
        active: dict[int, npt.NDArray[np.float32]] = {}
        next_id = 0
        for row in await self._read(_STORED_HUBS, tenant=tenant_id):
            hub = _int_row(row, "hub")
            next_id = max(next_id, hub + 1)
            if row["active"]:
                centroid = row["centroid"]
                if not isinstance(centroid, list):
                    raise DatabaseReadError("neo4j", f"hub {hub} of {tenant_id!r} has no centroid")
                active[hub] = np.asarray(centroid, dtype=np.float32)
        return active, next_id

    async def write_hubs(
        self,
        tenant_id: str,
        pages: Sequence[PageHub],
        hubs: Sequence[HubCentroid],
        *,
        batch_size: int = CENTRALITY_BATCH,
    ) -> int:
        """Page hubs, active Hub nodes and the retirement of hubs no longer found, in one
        transaction; a row that matches no crawled page rolls it all back."""
        _require_tenant(tenant_id)
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        urls = [page.url for page in pages]
        if len(set(urls)) != len(urls):
            raise ValueError("one row per page: duplicate urls")
        ids = [hub.hub_id for hub in hubs]
        if len(set(ids)) != len(ids):
            raise ValueError("one row per hub: duplicate hub ids")
        for hub in hubs:
            if len(hub.centroid) != VECTOR_DIMENSIONS:
                raise ValueError(
                    f"hub {hub.hub_id} centroid has {len(hub.centroid)} dimensions, "
                    f"expected {VECTOR_DIMENSIONS}"
                )
        chunks: list[list[Row]] = [
            [
                {"url": page.url, "hubId": page.hub_id, "isHubPillar": page.is_hub_pillar}
                for page in chunk
            ]
            for chunk in batched(pages, batch_size)
        ]
        rows: list[Row] = [
            {
                "hubId": hub.hub_id,
                "size": hub.size,
                "pillarUrl": hub.pillar_url,
                "centroid": list(hub.centroid),
            }
            for hub in hubs
        ]
        try:
            async with self._driver.session(default_access_mode=WRITE_ACCESS) as session:
                return await session.execute_write(_write_hub_rows, tenant_id, chunks, rows)
        except (Neo4jError, DriverError) as error:
            raise _translate(error, write=True) from error

    async def replace_keyword_targets(
        self,
        tenant_id: str,
        source: KeywordSource,
        targets: Sequence[KeywordTarget],
        *,
        batch_size: int = LINK_BATCH,
    ) -> tuple[int, int]:
        """Make the tenant's ``source`` keyword edges exactly ``targets`` on crawled pages, in
        one transaction. Rows on other urls are not written. Returns (written, stale deleted)."""
        _require_tenant(tenant_id)
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if any(target.source is not source for target in targets):
            raise ValueError(f"every target must come from {source.value}")
        keys = [(t.url, t.text, t.language) for t in targets]
        if len(set(keys)) != len(keys):
            raise ValueError("duplicate targets: one per (url, text, language)")
        resolved = Counter(t.url for t in targets if t.rung is not None)
        if any(count > 1 for count in resolved.values()):
            raise ValueError("more than one resolved keyword on one page")
        rows: list[Row] = [
            {
                "url": t.url,
                "text": t.text,
                "language": t.language,
                "priority": t.priority,
                "isPrimary": t.is_primary,
                "rung": _to_property(t.rung),
                "rank": t.rank,
            }
            for t in targets
        ]
        try:
            async with self._driver.session(default_access_mode=WRITE_ACCESS) as session:
                return await session.execute_write(
                    _replace_keyword_rows, tenant_id, source, rows, batch_size
                )
        except (Neo4jError, DriverError) as error:
            raise _translate(error, write=True) from error

    # ── transport ────────────────────────────────────────────────────────────

    async def _read(self, query: LiteralString, **params: object) -> list[Row]:
        try:
            async with self._driver.session(default_access_mode=READ_ACCESS) as session:
                return await session.execute_read(_collect, query, params)
        except (Neo4jError, DriverError) as error:
            raise _translate(error, write=False) from error

    async def _read_values(self, query: LiteralString, **params: object) -> list[list[object]]:
        # Rows as plain value lists. Record.data() walks every element of every list, which
        # dominates reads of 2048-float vectors.
        try:
            async with self._driver.session(default_access_mode=READ_ACCESS) as session:
                return await session.execute_read(_values, query, params)
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

    async def _write_all(
        self,
        query: LiteralString,
        expected: int,
        *,
        dropped: str = "rows with missing endpoint pages are dropped",
        **params: object,
    ) -> int:
        written = _int(await self._write(query, **params))
        if written != expected:
            raise DatabaseWriteError("neo4j", f"wrote {written} of {expected} rows; {dropped}")
        return written

    async def _write_exactly(
        self, query: LiteralString, expected: int, dropped: str, **params: object
    ) -> int:
        # The count is checked before commit, so a short write rolls back whole.
        try:
            async with self._driver.session(default_access_mode=WRITE_ACCESS) as session:
                return await session.execute_write(_count_exactly, query, params, expected, dropped)
        except (Neo4jError, DriverError) as error:
            raise _translate(error, write=True) from error


async def _snapshot(
    tx: AsyncManagedTransaction, tenant_id: str
) -> tuple[list[list[object]], list[list[object]]]:
    # Both reads in one transaction: no link can point at a page missing from the list.
    pages = await (await tx.run(_SNAPSHOT_PAGES, tenant=tenant_id)).values()
    links = await (await tx.run(_SNAPSHOT_LINKS, tenant=tenant_id)).values()
    return pages, links


async def _write_page_rows(
    tx: AsyncManagedTransaction,
    write: LiteralString,
    clear: LiteralString,
    tenant_id: str,
    chunks: list[list[Row]],
) -> int:
    written = 0
    for rows in chunks:
        written += _int(await _collect(tx, write, {"tenant": tenant_id, "rows": rows}))
    expected = sum(len(rows) for rows in chunks)
    if written != expected:
        raise DatabaseWriteError(
            "neo4j",
            f"wrote {written} of {expected} rows, rolled back; "
            "rows whose page is missing or a placeholder are dropped",
        )
    await _collect(tx, clear, {"tenant": tenant_id})
    return written


async def _write_hub_rows(
    tx: AsyncManagedTransaction, tenant_id: str, chunks: list[list[Row]], hubs: list[Row]
) -> int:
    written = await _write_page_rows(
        tx, _WRITE_PAGE_HUBS, _CLEAR_PLACEHOLDER_HUBS, tenant_id, chunks
    )
    upserted = _int(await _collect(tx, _UPSERT_HUBS, {"tenant": tenant_id, "hubs": hubs}))
    if upserted != len(hubs):
        raise DatabaseWriteError("neo4j", f"wrote {upserted} of {len(hubs)} hubs, rolled back")
    ids = [hub["hubId"] for hub in hubs]
    await _collect(tx, _RETIRE_HUBS, {"tenant": tenant_id, "ids": ids})
    return written


async def _replace_keyword_rows(
    tx: AsyncManagedTransaction,
    tenant_id: str,
    source: KeywordSource,
    rows: list[Row],
    batch_size: int,
) -> tuple[int, int]:
    params: Row = {"tenant": tenant_id, "source": source.value}
    keep: dict[str, list[list[object]]] = {}
    for row in rows:
        keep.setdefault(str(row["url"]), []).append([row["text"], row["language"]])
    existing = [str(row["url"]) for row in await _collect(tx, _KEYWORD_EDGE_PAGES, params)]
    deleted = 0
    for urls in batched(existing, batch_size):
        pages = [{"url": url, "keep": keep.get(url, [])} for url in urls]
        deleted += _int(await _collect(tx, _PRUNE_KEYWORD_EDGES, {**params, "pages": pages}))
    written = 0
    for chunk in batched(rows, batch_size):
        written += _int(await _collect(tx, _WRITE_KEYWORD_TARGETS, {**params, "rows": list(chunk)}))
    if source is KeywordSource.CLIENT_STRATEGIC:
        await _collect(
            tx, _MARK_STRATEGIC_KEYWORDS, {"tenant": tenant_id, "strategic": source.value}
        )
    return written, deleted


async def _collect(
    tx: AsyncManagedTransaction, query: LiteralString, params: Mapping[str, object]
) -> list[Row]:
    result = await tx.run(query, dict(params))
    return [record.data() async for record in result]


async def _values(
    tx: AsyncManagedTransaction, query: LiteralString, params: Mapping[str, object]
) -> list[list[object]]:
    return await (await tx.run(query, dict(params))).values()


async def _count_exactly(
    tx: AsyncManagedTransaction,
    query: LiteralString,
    params: Mapping[str, object],
    expected: int,
    dropped: str,
) -> int:
    written = _int(await _collect(tx, query, params))
    if written != expected:
        raise DatabaseWriteError(
            "neo4j", f"wrote {written} of {expected} rows, rolled back; {dropped}"
        )
    return written


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


def _vector_property(index: str) -> str:
    try:
        return VECTOR_PROPERTIES[index]
    except KeyError:
        raise ValueError(f"unknown vector index {index!r}") from None


def _check_embeddings(
    ids: Sequence[str],
    vectors: npt.NDArray[np.float32],
    *,
    what: str,
    model: str,
    dimensions: int,
    counts: Sequence[tuple[str, int]] = (),
) -> None:
    """Validate one flush: ``ids`` name its rows; ``counts`` are other per-row inputs."""
    if not model.strip():
        raise ValueError("model must be a non-empty string")
    # Neo4j accepts a vector of another size but leaves it out of the index.
    if dimensions != VECTOR_DIMENSIONS:
        raise ValueError(
            f"dimensions {dimensions} does not match the {VECTOR_DIMENSIONS}d vector index"
        )
    if not ids:
        raise ValueError("no embeddings to write")
    if vectors.dtype != np.float32 or vectors.ndim != 2 or vectors.shape[1] != dimensions:
        raise ValueError(
            f"vectors must be a float32 matrix with {dimensions} columns, "
            f"got {vectors.dtype} {vectors.shape}"
        )
    sizes = ((what, len(ids)), *counts)
    if any(size != vectors.shape[0] for _, size in sizes):
        listed = ", ".join(f"{size} {name}" for name, size in sizes)
        raise ValueError(f"got {listed} and {vectors.shape[0]} vectors")
    if any(not i.strip() for i in ids):
        raise ValueError(f"blank {what} in one flush")
    if len(set(ids)) != len(ids):
        raise ValueError(f"duplicate {what} in one flush")
    if not np.isfinite(vectors).all():
        raise ValueError("vectors contain NaN or infinite values")


def _check_unique_edges(edges: Iterable[tuple[str, int]]) -> None:
    seen: set[tuple[str, int]] = set()
    for edge in edges:
        if edge in seen:
            raise ValueError(f"duplicate edge {edge[0]!r} position {edge[1]} in one call")
        seen.add(edge)


def _model_counts(rows: list[Row]) -> tuple[EmbeddingModelCount, ...]:
    try:
        return tuple(EmbeddingModelCount.model_validate(row) for row in rows)
    except ValidationError as error:
        raise DatabaseReadError(
            "neo4j", f"embedding model counts do not fit the model: {error}"
        ) from error


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


def _optional_int(value: object) -> int | None:
    return None if value is None else int(_int_or_float(value))


def _int_or_float(value: object) -> int | float:
    if isinstance(value, int | float):
        return value
    raise DatabaseReadError("neo4j", f"expected a number, got {value!r}")
