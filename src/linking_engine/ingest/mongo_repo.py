"""MongoDB access: the project database and the read-only crawl source."""

from __future__ import annotations

from datetime import UTC, datetime
from functools import partial
from itertools import batched
from typing import TYPE_CHECKING, Final, Self

from pydantic import AnyUrl, BaseModel, ValidationError
from pydantic.alias_generators import to_camel
from pymongo import ASCENDING, DESCENDING, AsyncMongoClient, DeleteMany, IndexModel, UpdateOne
from pymongo.errors import (
    AutoReconnect,
    BulkWriteError,
    ConfigurationError,
    ConnectionFailure,
    OperationFailure,
    PyMongoError,
)
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from linking_engine.errors import (
    DatabaseAuthError,
    DatabaseError,
    DatabaseReadError,
    DatabaseUnavailableError,
    DatabaseWriteError,
    SchemaError,
)
from linking_engine.models import (
    AnchorRules,
    CrawlPage,
    GscMetrics,
    GscQuery,
    GscQueryStats,
    LanguageRules,
    LinkRecord,
    PageRecord,
    PageSummary,
    StrategicKeyword,
)
from linking_engine.urls import UrlRules, normalise_url

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
    from types import TracebackType

    from pymongo.asynchronous.collection import AsyncCollection

Document = dict[str, object]

WRITE_BATCH: Final = 1000
READ_BATCH: Final = 1000
_ATTEMPTS: Final = 3
_AUTH_CODES: Final = frozenset({13, 18})  # Unauthorized, AuthenticationFailed
_THIRTY_DAYS: Final = 30 * 24 * 3600
# The GSC rollup stores its fields under their snake_case names, not camelCase.
_GSC_METRICS_KEYS: Final = {field: field for field in GscMetrics.model_fields}

# All project indexes. Changing options of an existing index needs a manual drop.
INDEXES: Final[dict[str, tuple[IndexModel, ...]]] = {
    "pages": (
        IndexModel([("tenantId", ASCENDING), ("url", ASCENDING)], unique=True, name="tenant_url"),
        IndexModel([("tenantId", ASCENDING), ("scrapedAt", DESCENDING)], name="tenant_scraped"),
        IndexModel([("tenantId", ASCENDING), ("contentHash", ASCENDING)], name="tenant_hash"),
    ),
    "links": (
        IndexModel(
            [("tenantId", ASCENDING), ("sourceUrl", ASCENDING), ("position", ASCENDING)],
            unique=True,
            name="tenant_source_position",
        ),
        IndexModel([("tenantId", ASCENDING), ("targetUrl", ASCENDING)], name="tenant_target"),
    ),
    "gsc_metrics": (
        IndexModel([("tenantId", ASCENDING), ("url", ASCENDING)], unique=True, name="tenant_url"),
    ),
    "gsc_queries": (
        IndexModel(
            [("tenantId", ASCENDING), ("url", ASCENDING), ("impressions", DESCENDING)],
            name="tenant_url_impressions",
        ),
        IndexModel([("tenantId", ASCENDING), ("query", ASCENDING)], name="tenant_query"),
    ),
    "strategic_keywords": (
        IndexModel([("tenantId", ASCENDING), ("url", ASCENDING)], name="tenant_url"),
        IndexModel([("tenantId", ASCENDING), ("keyword", ASCENDING)], name="tenant_keyword"),
        IndexModel(
            [("tenantId", ASCENDING), ("url", ASCENDING), ("isPrimary", ASCENDING)],
            partialFilterExpression={"isPrimary": True},
            name="tenant_url_primary",
        ),
    ),
    "link_audit": (
        IndexModel(
            [("tenantId", ASCENDING), ("sourceUrl", ASCENDING), ("targetUrl", ASCENDING)],
            name="tenant_source_target",
        ),
        IndexModel([("tenantId", ASCENDING), ("auditedAt", DESCENDING)], name="tenant_audited"),
    ),
    "recommendations": (
        IndexModel(
            [("tenantId", ASCENDING), ("fromUrl", ASCENDING), ("score", DESCENDING)],
            name="tenant_from_score",
        ),
        IndexModel(
            [("tenantId", ASCENDING), ("actionType", ASCENDING), ("score", DESCENDING)],
            name="tenant_action_score",
        ),
        IndexModel([("tenantId", ASCENDING), ("createdAt", ASCENDING)], name="tenant_created"),
        # MongoDB allows TTL only on a single-field index, so expiry is the same for every tenant.
        IndexModel([("createdAt", ASCENDING)], expireAfterSeconds=_THIRTY_DAYS, name="created_ttl"),
    ),
    "anchor_feedback": (
        IndexModel([("tenantId", ASCENDING), ("createdAt", DESCENDING)], name="tenant_created"),
        IndexModel(
            [("tenantId", ASCENDING), ("actionType", ASCENDING), ("accepted", ASCENDING)],
            name="tenant_action_accepted",
        ),
    ),
    "tenant_config": (IndexModel([("tenantId", ASCENDING)], unique=True, name="tenant"),),
    "ctr_curves": (IndexModel([("tenantId", ASCENDING)], unique=True, name="tenant"),),
}


class MongoRepo:
    """Tenant-scoped access to the project database. Writes are keyed upserts."""

    def __init__(self, client: AsyncMongoClient[Document], database: str) -> None:
        self._client = client
        self._db = client[database]

    @classmethod
    async def connect(cls, uri: str, database: str, *, timeout_ms: int = 5000) -> Self:
        return cls(await _open(uri, database, timeout_ms=timeout_ms), database)

    async def close(self) -> None:
        await self._client.close()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.close()

    async def ensure_indexes(self) -> None:
        for name, models in INDEXES.items():
            try:
                await _retrying(
                    partial(self._db[name].create_indexes, list(models)),
                    write=True,
                    what=f"indexes on {name}",
                )
            except DatabaseWriteError as error:
                raise SchemaError("mongodb", f"cannot create indexes on {name}") from error

    async def write_pages(
        self,
        tenant_id: str,
        pages: Sequence[PageRecord],
        links: Sequence[LinkRecord],
        *,
        batch_size: int = WRITE_BATCH,
    ) -> tuple[int, int, int]:
        """Upsert pages with their complete link sets and delete link positions a
        page no longer has. Returns (pages, links, stale links deleted)."""
        _require_tenant(tenant_id)
        per_page = {str(page.url): 0 for page in pages}
        for link in links:
            source = str(link.source_url)
            if source not in per_page:
                raise ValueError(f"link from {source} whose page is not being written")
            per_page[source] += 1
        for page in pages:
            if per_page[str(page.url)] != page.link_count:
                raise ValueError(
                    f"{page.url}: link_count {page.link_count}, {per_page[str(page.url)]} links given"
                )

        now = datetime.now(UTC)
        page_ops = [
            UpdateOne(
                {"tenantId": tenant_id, "url": str(page.url)},
                {"$set": {**_to_document(page), "tenantId": tenant_id, "preparedAt": now}},
                upsert=True,
            )
            for page in pages
        ]
        link_ops = [
            UpdateOne(
                {
                    "tenantId": tenant_id,
                    "sourceUrl": str(link.source_url),
                    "position": link.position,
                },
                {"$set": {**_to_document(link), "tenantId": tenant_id, "preparedAt": now}},
                upsert=True,
            )
            for link in links
        ]
        stale_ops = [
            DeleteMany(
                {
                    "tenantId": tenant_id,
                    "sourceUrl": str(page.url),
                    "position": {"$gte": page.link_count},
                }
            )
            for page in pages
        ]
        written_pages, _ = await self._bulk("pages", page_ops, batch_size)
        written_links, _ = await self._bulk("links", link_ops, batch_size)
        _, deleted = await self._bulk("links", stale_ops, batch_size)
        return written_pages, written_links, deleted

    async def get_url_rules(self, tenant_id: str) -> UrlRules:
        """The tenant's query parameter overrides; none stored means the built-in rules."""
        _require_tenant(tenant_id)
        document = await _retrying(
            partial(
                self._db["tenant_config"].find_one,
                {"tenantId": tenant_id},
                {"_id": 0, "urlKeepParams": 1, "urlDropParams": 1},
            ),
            write=False,
            what="read url rules",
        )
        if not document:
            return UrlRules()
        return UrlRules.model_validate(
            {
                "keep_params": document.get("urlKeepParams") or [],
                "drop_params": document.get("urlDropParams") or [],
            }
        )

    async def set_url_rules(self, tenant_id: str, rules: UrlRules) -> None:
        _require_tenant(tenant_id)
        await _retrying(
            partial(
                self._db["tenant_config"].update_one,
                {"tenantId": tenant_id},
                {
                    "$set": {
                        "urlKeepParams": sorted(rules.keep_params),
                        "urlDropParams": sorted(rules.drop_params),
                        "urlRulesUpdatedAt": datetime.now(UTC),
                    }
                },
                upsert=True,
            ),
            write=True,
            what="write url rules",
        )

    async def get_anchor_rules(self, tenant_id: str) -> AnchorRules:
        """The tenant's generic-anchor overrides; none stored means the built-in dictionary."""
        _require_tenant(tenant_id)
        document = await _retrying(
            partial(
                self._db["tenant_config"].find_one,
                {"tenantId": tenant_id},
                {"_id": 0, "genericAnchorsAdd": 1, "genericAnchorsRemove": 1},
            ),
            write=False,
            what="read anchor rules",
        )
        if not document:
            return AnchorRules()
        return AnchorRules.model_validate(
            {
                "generic_add": document.get("genericAnchorsAdd") or [],
                "generic_remove": document.get("genericAnchorsRemove") or [],
            }
        )

    async def set_anchor_rules(self, tenant_id: str, rules: AnchorRules) -> None:
        _require_tenant(tenant_id)
        await _retrying(
            partial(
                self._db["tenant_config"].update_one,
                {"tenantId": tenant_id},
                {
                    "$set": {
                        "genericAnchorsAdd": sorted(rules.generic_add),
                        "genericAnchorsRemove": sorted(rules.generic_remove),
                        "anchorRulesUpdatedAt": datetime.now(UTC),
                    }
                },
                upsert=True,
            ),
            write=True,
            what="write anchor rules",
        )

    async def get_language_rules(self, tenant_id: str) -> LanguageRules:
        """The tenant's default language and url prefix languages; none stored means English."""
        _require_tenant(tenant_id)
        document = await _retrying(
            partial(
                self._db["tenant_config"].find_one,
                {"tenantId": tenant_id},
                {"_id": 0, "defaultLanguage": 1, "languagePrefixes": 1},
            ),
            write=False,
            what="read language rules",
        )
        if not document:
            return LanguageRules()
        data: Document = {}
        if document.get("defaultLanguage") is not None:
            data["default_language"] = document["defaultLanguage"]
        prefixes = document.get("languagePrefixes") or []
        if isinstance(prefixes, list) and all(isinstance(p, dict) for p in prefixes):
            prefixes = [(p.get("path"), p.get("language")) for p in prefixes]
        data["prefixes"] = prefixes
        try:
            return LanguageRules.model_validate(data)
        except ValidationError as error:
            raise DatabaseReadError(
                "mongodb", f"language rules of {tenant_id!r} do not fit LanguageRules: {error}"
            ) from error

    async def set_language_rules(self, tenant_id: str, rules: LanguageRules) -> None:
        _require_tenant(tenant_id)
        await _retrying(
            partial(
                self._db["tenant_config"].update_one,
                {"tenantId": tenant_id},
                {
                    "$set": {
                        "defaultLanguage": rules.default_language,
                        "languagePrefixes": [
                            {"path": path, "language": language}
                            for path, language in sorted(rules.prefixes)
                        ],
                        "languageRulesUpdatedAt": datetime.now(UTC),
                    }
                },
                upsert=True,
            ),
            write=True,
            what="write language rules",
        )

    async def delete_tenant(self, tenant_id: str) -> int:
        _require_tenant(tenant_id)
        deleted = 0
        for name in INDEXES:
            result = await _retrying(
                partial(self._db[name].delete_many, {"tenantId": tenant_id}),
                write=True,
                what=f"delete tenant from {name}",
            )
            deleted += result.deleted_count
        return deleted

    async def iter_page_summaries(
        self, tenant_id: str, *, batch_size: int = READ_BATCH
    ) -> AsyncIterator[list[PageSummary]]:
        _require_tenant(tenant_id)
        async for documents in _find_batches(
            self._db["pages"], {"tenantId": tenant_id}, PageSummary, batch_size
        ):
            yield [_from_document(PageSummary, document) for document in documents]

    async def iter_page_records(
        self, tenant_id: str, *, batch_size: int = READ_BATCH
    ) -> AsyncIterator[list[PageRecord]]:
        """Every stored page of the tenant with its text, in storage order."""
        _require_tenant(tenant_id)
        async for documents in _find_batches(
            self._db["pages"], {"tenantId": tenant_id}, PageRecord, batch_size
        ):
            yield [_from_document(PageRecord, document) for document in documents]

    async def gsc_query_stats(
        self, tenant_id: str, *, batch_size: int = READ_BATCH
    ) -> list[GscQueryStats]:
        """Every stored GSC query row of the tenant with its metrics, ordered by url then query."""
        _require_tenant(tenant_id)
        rows: list[GscQueryStats] = []
        async for documents in _find_batches(
            self._db["gsc_queries"], {"tenantId": tenant_id}, GscQueryStats, batch_size
        ):
            rows.extend(_from_document(GscQueryStats, document) for document in documents)
        return sorted(rows, key=lambda row: (row.url, row.query))

    async def gsc_metrics(
        self, tenant_id: str, *, batch_size: int = READ_BATCH
    ) -> list[GscMetrics]:
        """The tenant's 28-day GSC totals per page, ordered by url."""
        _require_tenant(tenant_id)
        rows: list[GscMetrics] = []
        async for documents in _find_batches(
            self._db["gsc_metrics"],
            {"tenantId": tenant_id},
            GscMetrics,
            batch_size,
            keys=_GSC_METRICS_KEYS,
        ):
            rows.extend(
                _from_document(GscMetrics, document, keys=_GSC_METRICS_KEYS)
                for document in documents
            )
        return sorted(rows, key=lambda row: row.url)

    async def gsc_queries(
        self, tenant_id: str, *, batch_size: int = READ_BATCH
    ) -> list[tuple[str, str]]:
        """(url, query) of every stored GSC query row of the tenant, ordered by url then query."""
        _require_tenant(tenant_id)
        rows: list[tuple[str, str]] = []
        async for documents in _find_batches(
            self._db["gsc_queries"], {"tenantId": tenant_id}, GscQuery, batch_size
        ):
            for document in documents:
                row = _from_document(GscQuery, document)
                rows.append((row.url, row.query))
        return sorted(rows)

    async def strategic_keywords(
        self, tenant_id: str, *, batch_size: int = READ_BATCH
    ) -> list[StrategicKeyword]:
        """The client's keywords per page, ordered by url, keyword and language."""
        _require_tenant(tenant_id)
        rows: list[StrategicKeyword] = []
        async for documents in _find_batches(
            self._db["strategic_keywords"], {"tenantId": tenant_id}, StrategicKeyword, batch_size
        ):
            rows.extend(_from_document(StrategicKeyword, document) for document in documents)
        return sorted(rows, key=lambda row: (row.url, row.keyword, row.language))

    async def get_pages(
        self, tenant_id: str, urls: Sequence[str], *, batch_size: int = READ_BATCH
    ) -> list[PageRecord]:
        _require_tenant(tenant_id)
        urls = [normalise_url(url) for url in urls]
        return await self._find_in("pages", tenant_id, "url", urls, PageRecord, batch_size)

    async def links_for(
        self, tenant_id: str, source_urls: Sequence[str], *, batch_size: int = READ_BATCH
    ) -> list[LinkRecord]:
        _require_tenant(tenant_id)
        source_urls = [normalise_url(url) for url in source_urls]
        return await self._find_in(
            "links", tenant_id, "sourceUrl", source_urls, LinkRecord, batch_size
        )

    async def _find_in[M: BaseModel](
        self,
        name: str,
        tenant_id: str,
        key: str,
        values: Sequence[str],
        model: type[M],
        batch_size: int,
    ) -> list[M]:
        found: list[M] = []
        for chunk in batched(values, batch_size):
            query: Document = {"tenantId": tenant_id, key: {"$in": list(chunk)}}
            async for documents in _find_batches(self._db[name], query, model, batch_size):
                found.extend(_from_document(model, document) for document in documents)
        return found

    async def _bulk(
        self, name: str, ops: Sequence[UpdateOne | DeleteMany], batch_size: int
    ) -> tuple[int, int]:
        written = deleted = 0
        for chunk in batched(ops, batch_size):
            result = await _retrying(
                partial(self._db[name].bulk_write, list(chunk), ordered=False),
                write=True,
                what=f"bulk write to {name}",
            )
            written += result.upserted_count + result.matched_count
            deleted += result.deleted_count
        return written, deleted


class CrawlSource:
    """Read-only access to the crawler's collection. Has no write methods."""

    def __init__(self, client: AsyncMongoClient[Document], database: str, collection: str) -> None:
        self._client = client
        self._collection: AsyncCollection[Document] = client[database][collection]

    @classmethod
    async def connect(
        cls, uri: str, database: str, collection: str, *, timeout_ms: int = 5000
    ) -> Self:
        return cls(await _open(uri, database, timeout_ms=timeout_ms), database, collection)

    async def close(self) -> None:
        await self._client.close()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.close()

    async def iter_pages(self, *, batch_size: int = READ_BATCH) -> AsyncIterator[list[CrawlPage]]:
        async for documents in _find_batches(self._collection, {}, CrawlPage, batch_size):
            yield [_from_document(CrawlPage, _parse_scraped_at(document)) for document in documents]


def _keys(model: type[BaseModel]) -> dict[str, str]:
    """Stored camelCase key -> model field."""
    return {to_camel(field): field for field in model.model_fields}


def _to_document(model: BaseModel) -> Document:
    return {to_camel(field): _bson(value) for field, value in model.model_dump().items()}


def _bson(value: object) -> object:
    if isinstance(value, AnyUrl):
        return str(value)
    if isinstance(value, tuple | list):
        return [_bson(item) for item in value]
    if isinstance(value, dict):
        return {to_camel(str(key)): _bson(item) for key, item in value.items()}
    return value


def _from_document[M: BaseModel](
    model: type[M], document: Mapping[str, object], *, keys: Mapping[str, str] | None = None
) -> M:
    """``keys`` maps stored keys to fields when they are not the camelCase field names."""
    stored = _keys(model) if keys is None else keys
    data = {field: document[key] for key, field in stored.items() if key in document}
    try:
        return model.model_validate(data)
    except ValidationError as error:
        key = document.get("url") or document.get("sourceUrl")
        raise DatabaseReadError(
            "mongodb", f"document {key!r} does not fit {model.__name__}: {error}"
        ) from error


def _parse_scraped_at(document: Document) -> Document:
    # The crawler stores scrapedAt as an ISO string.
    scraped = document.get("scrapedAt")
    if isinstance(scraped, str):
        return {**document, "scrapedAt": datetime.fromisoformat(scraped)}
    return document


async def _find_batches(
    collection: AsyncCollection[Document],
    query: Document,
    model: type[BaseModel],
    batch_size: int,
    *,
    keys: Mapping[str, str] | None = None,
) -> AsyncIterator[list[Document]]:
    """Keyset pagination on _id, so a retried batch never repeats or skips documents."""
    projection = dict.fromkeys(_keys(model) if keys is None else keys, 1)
    last_id: object = None
    while True:
        page_query: Document = (
            query if last_id is None else {"$and": [query, {"_id": {"$gt": last_id}}]}
        )

        async def read(q: Document = page_query) -> list[Document]:
            cursor = collection.find(q, projection).sort("_id", ASCENDING).limit(batch_size)
            return await cursor.to_list()

        documents = await _retrying(read, write=False, what=f"read {collection.name}")
        if not documents:
            return
        last_id = documents[-1]["_id"]
        yield documents
        if len(documents) < batch_size:
            return


async def _open(uri: str, database: str, *, timeout_ms: int) -> AsyncMongoClient[Document]:
    try:
        client: AsyncMongoClient[Document] = AsyncMongoClient(
            uri, serverSelectionTimeoutMS=timeout_ms, tz_aware=True
        )
    except (ConfigurationError, ValueError) as error:
        raise DatabaseUnavailableError(
            "mongodb", f"invalid connection settings: {error}"
        ) from error
    try:
        await _retrying(partial(client[database].command, "ping"), write=False, what="connect")
    except DatabaseError:
        await client.close()
        raise
    return client


async def _retrying[T](call: Callable[[], Awaitable[T]], *, write: bool, what: str) -> T:
    # Standalone mongod does not retry writes; this is safe because every write is idempotent.
    try:
        async for attempt in AsyncRetrying(
            retry=retry_if_exception_type(AutoReconnect),
            stop=stop_after_attempt(_ATTEMPTS),
            wait=wait_exponential(multiplier=0.5, max=4),
            reraise=True,
        ):
            with attempt:
                return await call()
    except PyMongoError as error:
        raise _translate(error, write=write, what=what) from error
    raise AssertionError("unreachable")


def _translate(error: PyMongoError, *, write: bool, what: str) -> DatabaseError:
    if isinstance(error, BulkWriteError):
        errors = error.details.get("writeErrors") or []
        first = errors[0].get("errmsg") if errors else str(error)
        return DatabaseWriteError("mongodb", f"{what}: {len(errors)} write errors, first: {first}")
    if isinstance(error, OperationFailure) and error.code in _AUTH_CODES:
        return DatabaseAuthError("mongodb", f"{what}: {error}")
    if isinstance(error, ConnectionFailure):
        return DatabaseUnavailableError("mongodb", f"{what}: server unavailable: {error}")
    kind = DatabaseWriteError if write else DatabaseReadError
    return kind("mongodb", f"{what}: {error}")


def _require_tenant(tenant_id: str) -> None:
    if not tenant_id.strip():
        raise ValueError("tenant_id must be a non-empty string")
