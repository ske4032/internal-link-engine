"""The output API's routes: read-only views of a tenant's latest complete run.

Listings page in their stable order: pass a page's ``next_cursor`` as ``after`` for the next
one, until it is null. Filters are exact matches; urls are normalised first.
"""

from typing import Annotated, Final, Literal

from fastapi import APIRouter, Path, Query, status
from pydantic import BaseModel, ConfigDict

from linking_engine.api.deps import AuthorisedTenant, LatestRun, Reader, not_found
from linking_engine.models import (
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
    SiteSummary,
    TargetFix,
    UnanchoredOut,
    UnanchoredReason,
)
from linking_engine.output.reader import (
    BridgeFilter,
    ExcludedFilter,
    PageFilter,
    RecommendationFilter,
    UnanchoredFilter,
)
from linking_engine.urls import UrlKey

# Stored integers are int64; a filter never needs more than this.
_MAX_ID: Final = 2**31 - 1


class ErrorDetail(BaseModel):
    """The body of every error but a request validation error."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    detail: str


class Health(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    status: Literal["ok"] = "ok"


_ERRORS: Final[dict[int | str, dict[str, object]]] = {
    status.HTTP_401_UNAUTHORIZED: {"model": ErrorDetail, "description": "Invalid or missing key."},
    status.HTTP_404_NOT_FOUND: {
        "model": ErrorDetail,
        "description": "No such resource, or no complete run, for the key's tenant.",
    },
    status.HTTP_503_SERVICE_UNAVAILABLE: {
        "model": ErrorDetail,
        "description": "The output store is unavailable.",
    },
}

After = Annotated[
    str | None,
    Query(
        pattern=r"^(0|[1-9][0-9]{0,17})$",
        description="The previous page's next_cursor; omit for the first page.",
    ),
]
Limit = Annotated[int, Query(ge=1, le=500, description="Items per page.")]
UrlFilter = Annotated[UrlKey | None, Query(description="A page url; normalised before matching.")]
HubFilter = Annotated[int | None, Query(ge=0, le=_MAX_ID)]

service = APIRouter(tags=["service"])
router = APIRouter(prefix="/v1/tenants/{tenant}", tags=["output"], responses=_ERRORS)


@service.get("/health", summary="Liveness; needs no key")
async def health() -> Health:
    return Health()


@router.get("/recommendations", summary="The run's recommendations, by source page")
async def recommendations(
    run: LatestRun,
    reader: Reader,
    source: UrlFilter = None,
    target: UrlFilter = None,
    action_type: Annotated[
        list[ActionType] | None, Query(description="Repeat to match any of several.")
    ] = None,
    tier: Annotated[int | None, Query(ge=1, le=_MAX_ID)] = None,
    after: After = None,
    limit: Limit = 50,
) -> Listing[Recommendation]:
    filters = RecommendationFilter(
        source=source, target=target, action_types=tuple(action_type or ()), tier=tier
    )
    return await reader.recommendations(run.tenant_id, run.run_id, filters, after, limit)


@router.get("/recommendations/{recommendation_id}", summary="One recommendation by its id")
async def recommendation(
    run: LatestRun,
    reader: Reader,
    recommendation_id: Annotated[str, Path(pattern=r"^[0-9a-f]{16}$")],
) -> Recommendation:
    found = await reader.recommendation(run.tenant_id, run.run_id, recommendation_id)
    if found is None:
        raise not_found()
    return found


# A complete run always has its summary.
@router.get("/summary", response_model=SiteSummary, summary="Counts over the run's output")
async def summary(run: LatestRun) -> SiteSummary | None:
    return run.summary


@router.get("/runs/latest", summary="The run being served and what it read")
async def runs_latest(run: LatestRun) -> RunInfo:
    return run


@router.get("/pages", summary="The run's page profiles, by url")
async def pages(
    run: LatestRun,
    reader: Reader,
    hub: HubFilter = None,
    orphan: bool | None = None,
    orphan_label: OrphanLabel | None = None,
    dead_end: bool | None = None,
    duplicate: Annotated[bool | None, Query(description="In a duplicate group.")] = None,
    after: After = None,
    limit: Limit = 50,
) -> Listing[PageProfile]:
    filters = PageFilter(
        hub=hub, orphan=orphan, orphan_label=orphan_label, dead_end=dead_end, duplicate=duplicate
    )
    return await reader.pages(run.tenant_id, run.run_id, filters, after, limit)


@router.get("/page", summary="One page's profile and the recommendations on its links")
async def page(
    run: LatestRun,
    reader: Reader,
    url: Annotated[UrlKey, Query(description="The page url; normalised before matching.")],
) -> PageDetail:
    detail = await reader.page(run.tenant_id, run.run_id, url)
    if detail is None:
        raise not_found()
    return detail


@router.get("/orphans", summary="Pages no body link reaches, by url")
async def orphans(
    run: LatestRun,
    reader: Reader,
    orphan_label: OrphanLabel | None = None,
    after: After = None,
    limit: Limit = 50,
) -> Listing[PageProfile]:
    filters = PageFilter(orphan=True, orphan_label=orphan_label)
    return await reader.pages(run.tenant_id, run.run_id, filters, after, limit)


@router.get("/hubs", summary="The run's topic hubs, by id")
async def hubs(
    run: LatestRun, reader: Reader, after: After = None, limit: Limit = 50
) -> Listing[HubSummary]:
    return await reader.hubs(run.tenant_id, run.run_id, after, limit)


@router.get("/bridges", summary="Hub pairs and the links proposed to bridge them")
async def bridges(
    run: LatestRun,
    reader: Reader,
    hub: Annotated[int | None, Query(ge=0, le=_MAX_ID, description="A hub on either side.")] = None,
    after: After = None,
    limit: Limit = 50,
) -> Listing[BridgePair]:
    return await reader.bridges(run.tenant_id, run.run_id, BridgeFilter(hub=hub), after, limit)


@router.get("/duplicates", summary="Groups of pages serving the same body")
async def duplicates(
    run: LatestRun, reader: Reader, after: After = None, limit: Limit = 50
) -> Listing[DuplicateGroup]:
    return await reader.duplicates(run.tenant_id, run.run_id, after, limit)


@router.get("/unanchored", summary="Ranked pairs without a usable anchor, and why")
async def unanchored(
    run: LatestRun,
    reader: Reader,
    reason: UnanchoredReason | None = None,
    source: UrlFilter = None,
    target: UrlFilter = None,
    after: After = None,
    limit: Limit = 50,
) -> Listing[UnanchoredOut]:
    filters = UnanchoredFilter(reason=reason, source=source, target=target)
    return await reader.unanchored(run.tenant_id, run.run_id, filters, after, limit)


@router.get("/target-fixes", summary="Target pages that need a keyword before links can point in")
async def target_fixes(
    run: LatestRun, reader: Reader, after: After = None, limit: Limit = 50
) -> Listing[TargetFix]:
    return await reader.target_fixes(run.tenant_id, run.run_id, after, limit)


@router.get(
    "/excluded-pages",
    summary="Pages kept out of the pipeline, by url; served without a complete run",
)
async def excluded_pages(
    tenant: AuthorisedTenant,
    reader: Reader,
    reason: ExclusionReason | None = None,
    after: Annotated[
        str | None,
        Query(min_length=1, max_length=4096, description="The previous page's next_cursor."),
    ] = None,
    limit: Limit = 50,
) -> Listing[ExcludedPage]:
    return await reader.excluded_pages(tenant, ExcludedFilter(reason=reason), after, limit)
