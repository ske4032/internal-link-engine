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
    AnchorTypeProfile,
    CrawlPage,
    ExcludedPage,
    ExportedPair,
    ExtractionSettings,
    GscMetrics,
    GscQuery,
    GscQueryStats,
    LabelEvent,
    LabelExport,
    LanguageRules,
    LinkAuditResult,
    LinkRecord,
    PageRecord,
    PageSummary,
    ScorerWeights,
    StrategicKeyword,
)
from linking_engine.urls import UrlRules, normalise_path, normalise_url

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable, Iterable, Mapping, Sequence
    from types import TracebackType

    from pymongo.asynchronous.collection import AsyncCollection

Document = dict[str, object]

WRITE_BATCH: Final = 1000
READ_BATCH: Final = 1000
_ATTEMPTS: Final = 3
_AUTH_CODES: Final = frozenset({13, 18})  # Unauthorized, AuthenticationFailed
# The GSC rollup stores its fields under their snake_case names, not camelCase.
_GSC_METRICS_KEYS: Final = {field: field for field in GscMetrics.model_fields}

# All project indexes but the served output's (output.collections.INDEXES). Changing options of
# an existing index needs a manual drop.
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
        IndexModel(
            [
                ("tenantId", ASCENDING),
                ("runId", ASCENDING),
                ("sourceUrl", ASCENDING),
                ("position", ASCENDING),
            ],
            unique=True,
            name="tenant_run_source_position",
        ),
    ),
    # One marker per completed audit run; a run without one is never read as the latest.
    "link_audit_runs": (
        IndexModel([("tenantId", ASCENDING), ("runId", ASCENDING)], unique=True, name="tenant_run"),
        IndexModel(
            [("tenantId", ASCENDING), ("auditedAt", DESCENDING), ("runId", DESCENDING)],
            name="tenant_audited_run",
        ),
    ),
    "anchor_feedback": (
        IndexModel([("tenantId", ASCENDING), ("createdAt", DESCENDING)], name="tenant_created"),
        IndexModel(
            [("tenantId", ASCENDING), ("actionType", ASCENDING), ("accepted", ASCENDING)],
            name="tenant_action_accepted",
        ),
        # Hand labels only: production feedback carries no import.
        IndexModel(
            [("tenantId", ASCENDING), ("importId", ASCENDING), ("pairId", ASCENDING)],
            unique=True,
            partialFilterExpression={"importId": {"$exists": True}},
            name="tenant_import_pair",
        ),
    ),
    "label_pairs": (
        IndexModel(
            [("tenantId", ASCENDING), ("pairId", ASCENDING)], unique=True, name="tenant_pair"
        ),
        IndexModel([("tenantId", ASCENDING), ("exportId", ASCENDING)], name="tenant_export"),
    ),
    "label_exports": (
        IndexModel(
            [("tenantId", ASCENDING), ("exportId", ASCENDING)], unique=True, name="tenant_export"
        ),
    ),
    "label_imports": (
        IndexModel(
            [("tenantId", ASCENDING), ("importId", ASCENDING)], unique=True, name="tenant_import"
        ),
        IndexModel([("tenantId", ASCENDING), ("completedAt", ASCENDING)], name="tenant_completed"),
    ),
    "excluded_pages": (
        IndexModel([("tenantId", ASCENDING), ("url", ASCENDING)], unique=True, name="tenant_url"),
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

    async def get_excluded_paths(self, tenant_id: str) -> frozenset[str]:
        """Paths the tenant keeps out of the pipeline, besides its sitemaps; normalised."""
        _require_tenant(tenant_id)
        document = await _retrying(
            partial(
                self._db["tenant_config"].find_one,
                {"tenantId": tenant_id},
                {"_id": 0, "excludedPaths": 1},
            ),
            write=False,
            what="read excluded paths",
        )
        stored = (document or {}).get("excludedPaths") or []
        if not isinstance(stored, list) or not all(isinstance(item, str) for item in stored):
            raise DatabaseReadError("mongodb", "excluded paths are not a list of strings")
        return frozenset(normalise_path(item) for item in stored)

    async def set_excluded_paths(self, tenant_id: str, paths: Iterable[str]) -> frozenset[str]:
        """Replace the tenant's excluded paths; the root is refused, since it would exclude the
        whole site. Returns the stored, normalised set."""
        _require_tenant(tenant_id)
        normalised = frozenset(normalise_path(path) for path in paths if path.strip())
        if "/" in normalised:
            raise ValueError("the root path cannot be excluded: it would exclude every page")
        await _retrying(
            partial(
                self._db["tenant_config"].update_one,
                {"tenantId": tenant_id},
                {
                    "$set": {
                        "excludedPaths": sorted(normalised),
                        "excludedPathsUpdatedAt": datetime.now(UTC),
                    }
                },
                upsert=True,
            ),
            write=True,
            what="write excluded paths",
        )
        return normalised

    async def replace_excluded_pages(self, tenant_id: str, pages: Sequence[ExcludedPage]) -> int:
        """Make ``pages`` the tenant's excluded pages, with their reasons and labels, as the
        latest preparation found them."""
        _require_tenant(tenant_id)
        now = datetime.now(UTC)
        await _retrying(
            partial(self._db["excluded_pages"].delete_many, {"tenantId": tenant_id}),
            write=True,
            what="clear excluded pages",
        )
        if not pages:
            return 0
        result = await _retrying(
            partial(
                self._db["excluded_pages"].insert_many,
                [
                    {**_to_document(page), "tenantId": tenant_id, "excludedAt": now}
                    for page in pages
                ],
            ),
            write=True,
            what="write excluded pages",
        )
        return len(result.inserted_ids)

    async def excluded_pages(self, tenant_id: str) -> tuple[ExcludedPage, ...]:
        """The tenant's excluded pages, by url."""
        _require_tenant(tenant_id)
        collection = self._db["excluded_pages"]
        projection = dict.fromkeys(_keys(ExcludedPage), 1)

        async def read() -> list[Document]:
            cursor = collection.find({"tenantId": tenant_id}, projection).sort("url", ASCENDING)
            return await cursor.to_list()

        documents = await _retrying(read, write=False, what="read excluded pages")
        return tuple(_from_document(ExcludedPage, document) for document in documents)

    async def delete_pages(self, tenant_id: str, urls: Sequence[str]) -> tuple[int, int]:
        """Delete these pages of the tenant and the links stored from them. Returns (pages,
        links) deleted."""
        _require_tenant(tenant_id)
        if not urls:
            return 0, 0
        keys = list(urls)
        pages = await _retrying(
            partial(self._db["pages"].delete_many, {"tenantId": tenant_id, "url": {"$in": keys}}),
            write=True,
            what="delete excluded pages",
        )
        links = await _retrying(
            partial(
                self._db["links"].delete_many, {"tenantId": tenant_id, "sourceUrl": {"$in": keys}}
            ),
            write=True,
            what="delete excluded pages' links",
        )
        return pages.deleted_count, links.deleted_count

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

    async def get_scorer_weights(self, tenant_id: str) -> ScorerWeights | None:
        """The tenant's scorer weights; None means the packaged default."""
        _require_tenant(tenant_id)
        document = await _retrying(
            partial(
                self._db["tenant_config"].find_one,
                {"tenantId": tenant_id},
                {"_id": 0, "scorerWeights": 1},
            ),
            write=False,
            what="read scorer weights",
        )
        stored = document.get("scorerWeights") if document else None
        return _stored_config(ScorerWeights, stored, f"scorer weights of {tenant_id!r}")

    async def set_scorer_weights(self, tenant_id: str, weights: ScorerWeights | None) -> None:
        """Store the tenant's scorer weights; None removes them, back to the default."""
        _require_tenant(tenant_id)
        now = datetime.now(UTC)
        update = (
            {"$set": {"scorerWeights": _to_document(weights), "scorerWeightsUpdatedAt": now}}
            if weights is not None
            else {"$unset": {"scorerWeights": ""}, "$set": {"scorerWeightsUpdatedAt": now}}
        )
        await _retrying(
            partial(
                self._db["tenant_config"].update_one, {"tenantId": tenant_id}, update, upsert=True
            ),
            write=True,
            what="write scorer weights",
        )

    async def get_extraction_settings(self, tenant_id: str) -> ExtractionSettings | None:
        """The tenant's anchor extraction settings; None means the defaults."""
        _require_tenant(tenant_id)
        document = await _retrying(
            partial(
                self._db["tenant_config"].find_one,
                {"tenantId": tenant_id},
                {"_id": 0, "extractionSettings": 1},
            ),
            write=False,
            what="read extraction settings",
        )
        stored = document.get("extractionSettings") if document else None
        return _stored_config(ExtractionSettings, stored, f"extraction settings of {tenant_id!r}")

    async def set_extraction_settings(
        self, tenant_id: str, settings: ExtractionSettings | None
    ) -> None:
        """Store the tenant's extraction settings; None removes them, back to the defaults."""
        _require_tenant(tenant_id)
        now = datetime.now(UTC)
        update = (
            {
                "$set": {
                    "extractionSettings": _to_document(settings),
                    "extractionSettingsUpdatedAt": now,
                }
            }
            if settings is not None
            else {
                "$unset": {"extractionSettings": ""},
                "$set": {"extractionSettingsUpdatedAt": now},
            }
        )
        await _retrying(
            partial(
                self._db["tenant_config"].update_one, {"tenantId": tenant_id}, update, upsert=True
            ),
            write=True,
            what="write extraction settings",
        )

    async def get_anchor_type_profile(self, tenant_id: str) -> AnchorTypeProfile | None:
        """The tenant's anchor type profile; None means the default."""
        _require_tenant(tenant_id)
        document = await _retrying(
            partial(
                self._db["tenant_config"].find_one,
                {"tenantId": tenant_id},
                {"_id": 0, "anchorTypeProfile": 1},
            ),
            write=False,
            what="read anchor type profile",
        )
        stored = document.get("anchorTypeProfile") if document else None
        return _stored_config(AnchorTypeProfile, stored, f"anchor type profile of {tenant_id!r}")

    async def set_anchor_type_profile(
        self, tenant_id: str, profile: AnchorTypeProfile | None
    ) -> None:
        """Store the tenant's anchor type profile; None removes it, back to the default."""
        _require_tenant(tenant_id)
        now = datetime.now(UTC)
        update = (
            {
                "$set": {
                    "anchorTypeProfile": _to_document(profile),
                    "anchorTypeProfileUpdatedAt": now,
                }
            }
            if profile is not None
            else {
                "$unset": {"anchorTypeProfile": ""},
                "$set": {"anchorTypeProfileUpdatedAt": now},
            }
        )
        await _retrying(
            partial(
                self._db["tenant_config"].update_one, {"tenantId": tenant_id}, update, upsert=True
            ),
            write=True,
            what="write anchor type profile",
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
        self,
        tenant_id: str,
        *,
        batch_size: int = READ_BATCH,
        urls: Iterable[str] | None = None,
    ) -> AsyncIterator[list[PageRecord]]:
        """Every stored page of the tenant with its text, in storage order; with ``urls``, only
        those pages, looked up ``batch_size`` urls at a time."""
        _require_tenant(tenant_id)
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        queries: Iterable[Document] = (
            [{"tenantId": tenant_id}]
            if urls is None
            else (
                {"tenantId": tenant_id, "url": {"$in": list(chunk)}}
                for chunk in batched(sorted(set(urls)), batch_size)
            )
        )
        for query in queries:
            async for documents in _find_batches(self._db["pages"], query, PageRecord, batch_size):
                yield [_from_document(PageRecord, document) for document in documents]

    async def page_titles(
        self, tenant_id: str, *, batch_size: int = READ_BATCH
    ) -> list[str | None]:
        """The meta title of every stored page of the tenant, and nothing else: what brand affix
        detection reads."""
        _require_tenant(tenant_id)
        titles: list[str | None] = []
        async for documents in _find_batches(
            self._db["pages"],
            {"tenantId": tenant_id},
            PageRecord,
            batch_size,
            keys={"metaTitle": "meta_title"},
        ):
            for document in documents:
                title = document.get("metaTitle")
                if title is not None and not isinstance(title, str):
                    raise DatabaseReadError(
                        "mongodb", f"a page of {tenant_id!r} has a non-text title {title!r}"
                    )
                titles.append(title)
        return titles

    async def page_titles_by_url(
        self, tenant_id: str, *, batch_size: int = READ_BATCH
    ) -> dict[str, str | None]:
        """The meta title of every stored page of the tenant, by url."""
        _require_tenant(tenant_id)
        titles: dict[str, str | None] = {}
        async for documents in _find_batches(
            self._db["pages"],
            {"tenantId": tenant_id},
            PageRecord,
            batch_size,
            keys={"url": "url", "metaTitle": "meta_title"},
        ):
            for document in documents:
                url, title = document.get("url"), document.get("metaTitle")
                if not isinstance(url, str) or not (title is None or isinstance(title, str)):
                    raise DatabaseReadError(
                        "mongodb", f"a page of {tenant_id!r} has url {url!r} and title {title!r}"
                    )
                titles[url] = title
        return titles

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

    async def insert_link_audit(
        self,
        tenant_id: str,
        run_id: str,
        docs: Sequence[LinkAuditResult],
        *,
        batch_size: int = WRITE_BATCH,
    ) -> int:
        """Add one audit run of the tenant, one document per edge. A retried run
        rewrites its own documents and never another run's. Returns the documents written."""
        _require_tenant(tenant_id)
        if not run_id.strip():
            raise ValueError("run_id must be a non-empty string")
        if any(doc.run_id != run_id for doc in docs):
            raise ValueError(f"every document must belong to run {run_id!r}")
        edges = [(doc.source_url, doc.position) for doc in docs]
        if len(set(edges)) != len(edges):
            raise ValueError("one document per edge: duplicate (source_url, position)")
        ops = [
            UpdateOne(
                {
                    "tenantId": tenant_id,
                    "runId": run_id,
                    "sourceUrl": doc.source_url,
                    "position": doc.position,
                },
                {"$set": {**_to_document(doc), "tenantId": tenant_id}},
                upsert=True,
            )
            for doc in docs
        ]
        written, _ = await self._bulk("link_audit", ops, batch_size)
        return written

    async def complete_link_audit(
        self,
        tenant_id: str,
        run_id: str,
        *,
        audited_at: datetime,
        documents: int,
        edges: int,
    ) -> None:
        """Mark a run complete once its last link_audit and edge batches are written; only a
        marked run is ever read as the latest. ``documents`` must equal the run's stored
        documents, ``edges`` is the edges written back. Marking a run again rewrites its marker."""
        _require_tenant(tenant_id)
        if not run_id.strip():
            raise ValueError("run_id must be a non-empty string")
        if documents < 0 or edges < 0:
            raise ValueError("documents and edges cannot be negative")
        stored = await _retrying(
            partial(
                self._db["link_audit"].count_documents, {"tenantId": tenant_id, "runId": run_id}
            ),
            write=False,
            what="count link audit documents",
        )
        if stored != documents:
            raise DatabaseWriteError(
                "mongodb",
                f"run {run_id!r} of {tenant_id!r} has {stored} of {documents} link audit "
                "documents; not marked complete",
            )
        await _retrying(
            partial(
                self._db["link_audit_runs"].update_one,
                {"tenantId": tenant_id, "runId": run_id},
                {
                    "$set": {
                        "auditedAt": audited_at,
                        "completedAt": datetime.now(UTC),
                        "documents": documents,
                        "edges": edges,
                    }
                },
                upsert=True,
            ),
            write=True,
            what="mark link audit run complete",
        )

    async def prune_link_audit(self, tenant_id: str, keep: str) -> tuple[int, int]:
        """Delete the tenant's link_audit documents and run markers of every run but ``keep``,
        once ``keep`` is complete. Returns (documents, markers) deleted."""
        _require_tenant(tenant_id)
        if not keep.strip():
            raise ValueError("keep must be a non-empty run id")
        marker = await _retrying(
            partial(
                self._db["link_audit_runs"].find_one,
                {"tenantId": tenant_id, "runId": keep},
                {"_id": 1},
            ),
            write=False,
            what="read link audit run",
        )
        if marker is None:
            raise DatabaseWriteError(
                "mongodb", f"run {keep!r} of {tenant_id!r} is not complete; nothing pruned"
            )
        others: Document = {"tenantId": tenant_id, "runId": {"$ne": keep}}
        documents = await _retrying(
            partial(self._db["link_audit"].delete_many, others),
            write=True,
            what="prune link audit documents",
        )
        markers = await _retrying(
            partial(self._db["link_audit_runs"].delete_many, others),
            write=True,
            what="prune link audit runs",
        )
        return documents.deleted_count, markers.deleted_count

    async def latest_link_audit_run(self, tenant_id: str) -> tuple[str, datetime] | None:
        """The run id and completion time of the tenant's most recent completed audit run, as
        `latest_link_audit` picks it; None when none was completed."""
        _require_tenant(tenant_id)
        marker = await self._latest_link_audit_marker(tenant_id)
        if marker is None:
            return None
        run_id, completed = marker.get("runId"), marker.get("completedAt")
        if not isinstance(run_id, str) or not run_id or not isinstance(completed, datetime):
            raise DatabaseReadError(
                "mongodb", f"the latest link audit of {tenant_id!r} has no run or completion time"
            )
        return run_id, completed

    async def _latest_link_audit_marker(self, tenant_id: str) -> Document | None:
        marker: Document | None = await _retrying(
            partial(
                self._db["link_audit_runs"].find_one,
                {"tenantId": tenant_id},
                {"_id": 0, "runId": 1, "completedAt": 1},
                sort=[("auditedAt", DESCENDING), ("runId", DESCENDING)],
            ),
            write=False,
            what="read latest link audit run",
        )
        return marker

    async def latest_link_audit(self, tenant_id: str) -> tuple[LinkAuditResult, ...]:
        """The tenant's most recent completed audit run, ordered by source url and position;
        empty when none was completed. The latest auditedAt picks the run, the greatest runId
        breaks a tie."""
        _require_tenant(tenant_id)
        marker = await self._latest_link_audit_marker(tenant_id)
        if marker is None:
            return ()
        run_id = marker.get("runId")
        if not isinstance(run_id, str) or not run_id:
            raise DatabaseReadError("mongodb", f"the latest link audit of {tenant_id!r} has no run")
        collection = self._db["link_audit"]
        query: Document = {"tenantId": tenant_id, "runId": run_id}
        projection = dict.fromkeys(_keys(LinkAuditResult), 1)

        # One index-ordered cursor; a retry re-reads the run from the start.
        async def read() -> list[Document]:
            cursor = collection.find(query, projection).sort(
                [("sourceUrl", ASCENDING), ("position", ASCENDING)]
            )
            return await cursor.to_list()

        documents = await _retrying(read, write=False, what="read link audit")
        return tuple(_from_document(LinkAuditResult, document) for document in documents)

    async def insert_label_pairs(
        self, tenant_id: str, export_id: str, pairs: Sequence[ExportedPair]
    ) -> int:
        """Add the pairs of one label export. A retried export rewrites its own pairs; a pair id
        of another export is refused. Returns the pairs written."""
        _require_tenant(tenant_id)
        if not export_id.strip():
            raise ValueError("export_id must be a non-empty string")
        ids = [pair.pair_id for pair in pairs]
        if len(set(ids)) != len(ids):
            raise ValueError("one document per pair: duplicate pair_id")
        ops = [
            UpdateOne(
                {"tenantId": tenant_id, "pairId": pair.pair_id, "exportId": export_id},
                {"$set": {**_to_document(pair), "tenantId": tenant_id, "exportId": export_id}},
                upsert=True,
            )
            for pair in pairs
        ]
        written, _ = await self._bulk("label_pairs", ops, WRITE_BATCH)
        return written

    async def complete_label_export(self, tenant_id: str, export: LabelExport) -> None:
        """Mark an export complete once all its pairs are written; only a marked export's pairs
        can be imported."""
        _require_tenant(tenant_id)
        stored = await self._count(
            "label_pairs", {"tenantId": tenant_id, "exportId": export.export_id}, "label pairs"
        )
        if stored != export.pairs:
            raise DatabaseWriteError(
                "mongodb",
                f"export {export.export_id!r} of {tenant_id!r} has {stored} of {export.pairs} "
                "pairs; not marked complete",
            )
        await _retrying(
            partial(
                self._db["label_exports"].update_one,
                {"tenantId": tenant_id, "exportId": export.export_id},
                {
                    "$set": {
                        **_to_document(export),
                        "tenantId": tenant_id,
                        "completedAt": datetime.now(UTC),
                    }
                },
                upsert=True,
            ),
            write=True,
            what="mark label export complete",
        )

    async def label_export_ids(self, tenant_id: str, pair_ids: Sequence[str]) -> frozenset[str]:
        """The tenant's complete exports holding any of ``pair_ids``."""
        _require_tenant(tenant_id)
        found: set[str] = set()
        for chunk in batched(pair_ids, READ_BATCH):
            exports = await _retrying(
                partial(
                    self._db["label_pairs"].distinct,
                    "exportId",
                    {"tenantId": tenant_id, "pairId": {"$in": list(chunk)}},
                ),
                write=False,
                what="read label export ids",
            )
            found.update(str(export) for export in exports)
        if not found:
            return frozenset()
        complete = await _retrying(
            partial(
                self._db["label_exports"].distinct,
                "exportId",
                {"tenantId": tenant_id, "exportId": {"$in": sorted(found)}},
            ),
            write=False,
            what="read complete label exports",
        )
        return frozenset(str(export) for export in complete)

    async def label_export(
        self, tenant_id: str, export_id: str
    ) -> tuple[LabelExport, tuple[ExportedPair, ...]] | None:
        """A complete export of the tenant and its pairs; None when there is no such export."""
        _require_tenant(tenant_id)
        marker: Document | None = await _retrying(
            partial(
                self._db["label_exports"].find_one,
                {"tenantId": tenant_id, "exportId": export_id},
                dict.fromkeys(_keys(LabelExport), 1),
            ),
            write=False,
            what="read label export",
        )
        if marker is None:
            return None
        export = _from_document(LabelExport, marker)
        pairs: list[ExportedPair] = []
        async for documents in _find_batches(
            self._db["label_pairs"],
            {"tenantId": tenant_id, "exportId": export_id},
            ExportedPair,
            READ_BATCH,
        ):
            pairs.extend(_from_document(ExportedPair, document) for document in documents)
        if len(pairs) != export.pairs:
            raise DatabaseReadError(
                "mongodb",
                f"export {export_id!r} of {tenant_id!r} has {len(pairs)} of {export.pairs} pairs",
            )
        return export, tuple(sorted(pairs, key=lambda pair: pair.pair_id))

    async def insert_label_events(
        self, tenant_id: str, import_id: str, events: Sequence[LabelEvent]
    ) -> int:
        """Add the label events of one import to anchor_feedback. A retried import rewrites its
        own events and never another import's. Returns the events written."""
        _require_tenant(tenant_id)
        if not import_id.strip():
            raise ValueError("import_id must be a non-empty string")
        if any(event.import_id != import_id for event in events):
            raise ValueError(f"every event must belong to import {import_id!r}")
        ids = [event.pair_id for event in events]
        if len(set(ids)) != len(ids):
            raise ValueError("one event per pair: duplicate pair_id")
        ops = [
            UpdateOne(
                {"tenantId": tenant_id, "importId": import_id, "pairId": event.pair_id},
                {"$set": {**_to_document(event), "tenantId": tenant_id}},
                upsert=True,
            )
            for event in events
        ]
        written, _ = await self._bulk("anchor_feedback", ops, WRITE_BATCH)
        return written

    async def complete_label_import(
        self, tenant_id: str, import_id: str, *, export_id: str, events: int
    ) -> None:
        """Mark an import complete once all its events are written; only a marked import's
        labels are ever read."""
        _require_tenant(tenant_id)
        if not import_id.strip() or not export_id.strip():
            raise ValueError("import_id and export_id must be non-empty strings")
        stored = await self._count(
            "anchor_feedback", {"tenantId": tenant_id, "importId": import_id}, "label events"
        )
        if stored != events:
            raise DatabaseWriteError(
                "mongodb",
                f"import {import_id!r} of {tenant_id!r} has {stored} of {events} label events; "
                "not marked complete",
            )
        await _retrying(
            partial(
                self._db["label_imports"].update_one,
                {"tenantId": tenant_id, "importId": import_id},
                {
                    "$set": {
                        "exportId": export_id,
                        "events": events,
                        "completedAt": datetime.now(UTC),
                    }
                },
                upsert=True,
            ),
            write=True,
            what="mark label import complete",
        )

    async def hand_labels(self, tenant_id: str) -> tuple[LabelEvent, ...]:
        """Each pair's hand label from the tenant's latest complete import that labels it,
        ordered by source and target url. Earlier labels stay stored as its history."""
        _require_tenant(tenant_id)

        async def read() -> list[Document]:
            cursor = (
                self._db["label_imports"]
                .find({"tenantId": tenant_id}, {"_id": 0, "importId": 1})
                .sort([("completedAt", ASCENDING), ("importId", ASCENDING)])
            )
            return await cursor.to_list()

        markers = await _retrying(read, write=False, what="read label imports")
        order = {str(marker["importId"]): index for index, marker in enumerate(markers)}
        latest: dict[tuple[str, str], LabelEvent] = {}
        for chunk in batched(sorted(order), READ_BATCH):
            query: Document = {"tenantId": tenant_id, "importId": {"$in": list(chunk)}}
            async for documents in _find_batches(
                self._db["anchor_feedback"], query, LabelEvent, READ_BATCH
            ):
                for document in documents:
                    event = _from_document(LabelEvent, document)
                    key = (event.source_url, event.target_url)
                    held = latest.get(key)
                    if held is None or order[event.import_id] > order[held.import_id]:
                        latest[key] = event
        return tuple(latest[key] for key in sorted(latest))

    async def _count(self, name: str, query: Document, what: str) -> int:
        counted: int = await _retrying(
            partial(self._db[name].count_documents, query),
            write=False,
            what=f"count {what}",
        )
        return counted

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


def _stored_config[M: BaseModel](model: type[M], stored: object, what: str) -> M | None:
    """A tenant config sub-document as ``model``; None when nothing is stored. One stored form,
    camelCase: a snake_case key copied from a packaged JSON would otherwise be dropped
    silently, so any unknown key fails."""
    if stored is None:
        return None
    if not isinstance(stored, dict):
        raise DatabaseReadError("mongodb", f"{what} are not a document")
    keys = _keys(model)
    unknown = sorted(str(key) for key in stored if key not in keys)
    if unknown:
        raise DatabaseReadError(
            "mongodb",
            f"{what} have unknown keys {', '.join(unknown)}; expected {', '.join(keys)}",
        )
    try:
        return model.model_validate(
            {field: stored[key] for key, field in keys.items() if key in stored}
        )
    except ValidationError as error:
        raise DatabaseReadError(
            "mongodb", f"{what} do not fit {model.__name__}: {error}"
        ) from error


def _to_document(model: BaseModel) -> Document:
    return {to_camel(field): _bson(value) for field, value in model.model_dump().items()}


def _bson(value: object) -> object:
    if isinstance(value, AnyUrl):
        return str(value)
    if isinstance(value, tuple | list):
        return [_bson(item) for item in value]
    if isinstance(value, set | frozenset):
        return [_bson(item) for item in sorted(value, key=str)]
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
