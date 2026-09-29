"""Reads a tenant's served output: the latest complete run and its listings, paged in their
stable order."""

import re
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Final

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from pydantic.alias_generators import to_camel
from pymongo import ASCENDING, DESCENDING
from pymongo.asynchronous.database import AsyncDatabase
from pymongo.errors import ConnectionFailure, OperationFailure, PyMongoError

from linking_engine.errors import (
    DatabaseAuthError,
    DatabaseReadError,
    DatabaseUnavailableError,
    DatabaseWriteError,
)
from linking_engine.models import (
    NEW_LINK_ACTIONS,
    ActionType,
    BridgePair,
    DuplicateGroup,
    ExcludedPage,
    ExclusionReason,
    HubSummary,
    Listing,
    OrphanLabel,
    PageDetail,
    PageProfile,
    Recommendation,
    RunInfo,
    TargetFix,
    UnanchoredOut,
    UnanchoredReason,
)
from linking_engine.output.collections import (
    BRIDGES,
    DUPLICATES,
    EXCLUDED_PAGES,
    HUBS,
    PAGES,
    RECOMMENDATIONS,
    RUNS,
    TARGET_FIXES,
    UNANCHORED,
    Document,
    from_document,
)
from linking_engine.urls import UrlKey

# A cursor is the last item's ordinal: a canonical non-negative integer that fits int64.
_ORDINAL: Final = re.compile(r"0|[1-9][0-9]{0,17}")
_EXCLUDED_KEYS: Final = {to_camel(field): field for field in ExcludedPage.model_fields}
_AUTH_CODES: Final = frozenset({13, 18})  # Unauthorized, AuthenticationFailed


class InvalidCursorError(ValueError):
    """An ``after`` cursor that no listing handed out."""


class RecommendationFilter(BaseModel):
    """Exact-match filters on a run's recommendations; several action types match any."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source: UrlKey | None = None
    target: UrlKey | None = None
    action_types: tuple[ActionType, ...] = ()
    tier: int | None = Field(default=None, ge=1)


class PageFilter(BaseModel):
    """Exact-match filters on a run's page profiles."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    hub: int | None = Field(default=None, ge=0)
    orphan: bool | None = None
    orphan_label: OrphanLabel | None = None
    dead_end: bool | None = None
    # In a duplicate group, as canonical or copy.
    duplicate: bool | None = None


class BridgeFilter(BaseModel):
    """A hub on either side of a bridge pair."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    hub: int | None = Field(default=None, ge=0)


class UnanchoredFilter(BaseModel):
    """Exact-match filters on a run's pairs without an anchor."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    reason: UnanchoredReason | None = None
    source: UrlKey | None = None
    target: UrlKey | None = None


class ExcludedFilter(BaseModel):
    """Exact-match filter on the pages prepare-corpus kept out."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    reason: ExclusionReason | None = None


class OutputReader:
    """Tenant- and run-scoped reads of the output collections; never writes."""

    def __init__(self, db: AsyncDatabase[Document]) -> None:
        self._db = db

    async def latest_run(self, tenant_id: str) -> RunInfo | None:
        """The tenant's latest complete run; a run still writing is never served."""
        _require_tenant(tenant_id)
        with store_errors("read the latest run"):
            document = await self._db[RUNS].find_one(
                {"tenantId": tenant_id, "status": "complete"},
                sort=[("completed_at", DESCENDING)],
            )
        return None if document is None else from_document(RunInfo, document)

    async def recommendations(
        self,
        tenant_id: str,
        run_id: str,
        filters: RecommendationFilter,
        after: str | None = None,
        limit: int = 50,
    ) -> Listing[Recommendation]:
        query: Document = {}
        if filters.source is not None:
            query["source_url"] = filters.source
        if filters.target is not None:
            query["target_url"] = filters.target
        if filters.action_types:
            query["action_type"] = {"$in": [str(action) for action in filters.action_types]}
        if filters.tier is not None:
            query["tier"] = filters.tier
        items, cursor, total = await self._page(
            RECOMMENDATIONS, Recommendation, tenant_id, run_id, query, after, limit
        )
        return Listing[Recommendation](items=items, next_cursor=cursor, total=total)

    async def recommendation(
        self, tenant_id: str, run_id: str, recommendation_id: str
    ) -> Recommendation | None:
        document = await self._find_one(
            RECOMMENDATIONS, tenant_id, run_id, {"id": recommendation_id}
        )
        return None if document is None else from_document(Recommendation, document)

    async def pages(
        self,
        tenant_id: str,
        run_id: str,
        filters: PageFilter,
        after: str | None = None,
        limit: int = 50,
    ) -> Listing[PageProfile]:
        query: Document = {}
        if filters.hub is not None:
            query["hub_id"] = filters.hub
        if filters.orphan is not None:
            query["is_orphan"] = filters.orphan
        if filters.orphan_label is not None:
            query["orphan_label"] = str(filters.orphan_label)
        if filters.dead_end is not None:
            query["is_dead_end"] = filters.dead_end
        if filters.duplicate is not None:
            query["duplicate_group"] = {"$ne": None} if filters.duplicate else None
        items, cursor, total = await self._page(
            PAGES, PageProfile, tenant_id, run_id, query, after, limit
        )
        return Listing[PageProfile](items=items, next_cursor=cursor, total=total)

    async def page(self, tenant_id: str, run_id: str, url: str) -> PageDetail | None:
        """A page's profile and the records on its links; None when the run has no such page."""
        document = await self._find_one(PAGES, tenant_id, run_id, {"url": url})
        if document is None:
            return None
        scope: Document = {"tenantId": tenant_id, "runId": run_id}
        new_links = [str(action) for action in NEW_LINK_ACTIONS]
        with store_errors("read a page's recommendations"):
            outgoing = (
                await self._db[RECOMMENDATIONS]
                .find({**scope, "source_url": url})
                .sort("ordinal", ASCENDING)
                .to_list()
            )
            incoming = await self._db[RECOMMENDATIONS].count_documents(
                {**scope, "target_url": url, "action_type": {"$in": new_links}}
            )
        return PageDetail(
            profile=from_document(PageProfile, document),
            outgoing=tuple(from_document(Recommendation, item) for item in outgoing),
            incoming_total=incoming,
        )

    async def hubs(
        self, tenant_id: str, run_id: str, after: str | None = None, limit: int = 50
    ) -> Listing[HubSummary]:
        items, cursor, total = await self._page(
            HUBS, HubSummary, tenant_id, run_id, {}, after, limit
        )
        return Listing[HubSummary](items=items, next_cursor=cursor, total=total)

    async def bridges(
        self,
        tenant_id: str,
        run_id: str,
        filters: BridgeFilter,
        after: str | None = None,
        limit: int = 50,
    ) -> Listing[BridgePair]:
        query: Document = {}
        if filters.hub is not None:
            query["$or"] = [{"hub_a": filters.hub}, {"hub_b": filters.hub}]
        items, cursor, total = await self._page(
            BRIDGES, BridgePair, tenant_id, run_id, query, after, limit
        )
        return Listing[BridgePair](items=items, next_cursor=cursor, total=total)

    async def duplicates(
        self, tenant_id: str, run_id: str, after: str | None = None, limit: int = 50
    ) -> Listing[DuplicateGroup]:
        items, cursor, total = await self._page(
            DUPLICATES, DuplicateGroup, tenant_id, run_id, {}, after, limit
        )
        return Listing[DuplicateGroup](items=items, next_cursor=cursor, total=total)

    async def unanchored(
        self,
        tenant_id: str,
        run_id: str,
        filters: UnanchoredFilter,
        after: str | None = None,
        limit: int = 50,
    ) -> Listing[UnanchoredOut]:
        query: Document = {}
        if filters.reason is not None:
            query["reason"] = str(filters.reason)
        if filters.source is not None:
            query["source_url"] = filters.source
        if filters.target is not None:
            query["target_url"] = filters.target
        items, cursor, total = await self._page(
            UNANCHORED, UnanchoredOut, tenant_id, run_id, query, after, limit
        )
        return Listing[UnanchoredOut](items=items, next_cursor=cursor, total=total)

    async def target_fixes(
        self, tenant_id: str, run_id: str, after: str | None = None, limit: int = 50
    ) -> Listing[TargetFix]:
        items, cursor, total = await self._page(
            TARGET_FIXES, TargetFix, tenant_id, run_id, {}, after, limit
        )
        return Listing[TargetFix](items=items, next_cursor=cursor, total=total)

    async def excluded_pages(
        self,
        tenant_id: str,
        filters: ExcludedFilter,
        after: str | None = None,
        limit: int = 50,
    ) -> Listing[ExcludedPage]:
        """The pages the latest preparation excluded, by url; not run-scoped. The cursor is
        the last url."""
        _require_tenant(tenant_id)
        _require_limit(limit)
        query: Document = {"tenantId": tenant_id}
        if filters.reason is not None:
            query["reason"] = str(filters.reason)
        page_query: Document = query if after is None else {**query, "url": {"$gt": after}}
        collection = self._db[EXCLUDED_PAGES]
        with store_errors("read excluded pages"):
            documents = (
                await collection.find(page_query, dict.fromkeys(_EXCLUDED_KEYS, 1))
                .sort("url", ASCENDING)
                .limit(limit + 1)
                .to_list()
            )
            total = await collection.count_documents(query)
        items = tuple(_excluded_page(document) for document in documents[:limit])
        cursor = items[-1].url if len(documents) > limit else None
        return Listing[ExcludedPage](items=items, next_cursor=cursor, total=total)

    async def _find_one(
        self, collection: str, tenant_id: str, run_id: str, query: Document
    ) -> Document | None:
        _require_tenant(tenant_id)
        with store_errors(f"read {collection}"):
            return await self._db[collection].find_one(
                {**query, "tenantId": tenant_id, "runId": run_id}
            )

    async def _page[M: BaseModel](
        self,
        collection: str,
        model: type[M],
        tenant_id: str,
        run_id: str,
        filters: Document,
        after: str | None,
        limit: int,
    ) -> tuple[tuple[M, ...], str | None, int]:
        """Items after the cursor in ordinal order, the next cursor, and the filtered total."""
        _require_tenant(tenant_id)
        _require_limit(limit)
        query: Document = {**filters, "tenantId": tenant_id, "runId": run_id}
        page_query: Document = (
            query if after is None else {**query, "ordinal": {"$gt": _ordinal(after)}}
        )
        with store_errors(f"read {collection}"):
            documents = (
                await self._db[collection]
                .find(page_query)
                .sort("ordinal", ASCENDING)
                .limit(limit + 1)
                .to_list()
            )
            total = await self._db[collection].count_documents(query)
        items = tuple(from_document(model, document) for document in documents[:limit])
        cursor = str(documents[limit - 1]["ordinal"]) if len(documents) > limit else None
        return items, cursor, total


@contextmanager
def store_errors(what: str, *, write: bool = False) -> Iterator[None]:
    """Translate a driver failure into the project's store errors."""
    try:
        yield
    except ConnectionFailure as error:
        raise DatabaseUnavailableError("mongodb", f"{what}: server unavailable") from error
    except PyMongoError as error:
        if isinstance(error, OperationFailure) and error.code in _AUTH_CODES:
            raise DatabaseAuthError("mongodb", f"{what}: {error}") from error
        kind = DatabaseWriteError if write else DatabaseReadError
        raise kind("mongodb", f"{what}: {error}") from error


def _ordinal(cursor: str) -> int:
    if not _ORDINAL.fullmatch(cursor):
        raise InvalidCursorError(f"invalid cursor {cursor!r}")
    return int(cursor)


def _excluded_page(document: Mapping[str, object]) -> ExcludedPage:
    # prepare-corpus stores these camelCase, unlike the run output.
    data = {field: document[key] for key, field in _EXCLUDED_KEYS.items() if key in document}
    try:
        return ExcludedPage.model_validate(data)
    except ValidationError as error:
        raise DatabaseReadError(
            "mongodb", f"excluded page document does not fit ExcludedPage: {error}"
        ) from error


def _require_tenant(tenant_id: str) -> None:
    if not tenant_id.strip():
        raise ValueError("tenant_id must be a non-empty string")


def _require_limit(limit: int) -> None:
    if limit < 1:
        raise ValueError("limit must be at least 1")
