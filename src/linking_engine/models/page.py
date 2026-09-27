"""Graph entities: `Page`, `Keyword`, and the `LINKS_TO` edge between two pages.

Field names are the snake_case form of the Neo4j property names in the Data
Model. Every model is frozen and forbids extra fields, so a renamed property
fails at the boundary instead of silently dropping data.

Urls are normalised keys (see :func:`linking_engine.urls.normalise_url`): no
scheme, www, query or trailing slash, so one page has one identity across tenants.
"""

from datetime import datetime
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from linking_engine.models.enums import ActionType, AnchorType, IssueFlag, LifecycleStage, PageType
from linking_engine.urls import UrlKey

# Vectors are ``tuple[float, ...]`` rather than ``list[float]``: these models are
# frozen and therefore hashable, and a list field would raise on hash. The tuple
# is also the typed hand-off the embedding stage uses to pass vectors to the
# graph stage, since Pydantic models are the only thing crossing a boundary.


class Page(BaseModel):
    """A crawled page, plus everything the pipeline computes about it.

    Fields the crawler does not provide stay None. A placeholder is a link
    target that was never crawled and carries only its url.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    # ── crawl ───────────────────────────────────────────────────────────────
    url: UrlKey
    is_placeholder: bool = False
    status_code: int | None = Field(default=None, ge=100, le=599)
    content_hash: str | None = None
    # sha256 of the embedded body text; content_hash is the crawler's raw-page hash.
    body_hash: str | None = None
    word_count: int | None = Field(default=None, ge=0)
    page_type: PageType | None = None
    is_indexable: bool | None = None
    crawl_depth: int | None = Field(default=None, ge=0)
    language: str | None = None
    freshness: float | None = Field(default=None, ge=0, le=1)
    published_at: datetime | None = None
    lifecycle_stage: LifecycleStage | None = None
    # Distinct crawled pages linking here from menu or footer template lines; never edges.
    menu_inlinks: int | None = Field(default=None, ge=0)
    footer_inlinks: int | None = Field(default=None, ge=0)

    # ── graph analytics ─────────────────────────────────────────────────────
    # pageRank is over body links only; nav and footer links are never captured.
    page_rank: float | None = Field(default=None, ge=0)
    # Share of the tenant's crawled pages scoring strictly lower; placeholders have no scores.
    page_rank_percentile: float | None = Field(default=None, ge=0, lt=1)
    betweenness: float | None = Field(default=None, ge=0)
    betweenness_percentile: float | None = Field(default=None, ge=0, lt=1)
    link_community_id: int | None = None
    keyword_community_id: int | None = None
    # HDBSCAN cluster label over content_embedding. -1 is the noise label, a
    # real assignment meaning "in no dense region", not a missing value.
    hub_id: int | None = None

    # ── embedding ───────────────────────────────────────────────────────────
    # Reserved: chunked embedding is not in the MVP and this is always False.
    is_chunked: bool = False
    embedding_model: str | None = None
    embedding_dimensions: int | None = Field(default=None, ge=1)
    # Resume marker: body_hash of the stored vector, written in the same transaction.
    embedded_body_hash: str | None = None
    embedded_at: datetime | None = None
    content_embedding: tuple[float, ...] | None = None
    gnn_embedding: tuple[float, ...] | None = None


class Keyword(BaseModel):
    """A search term a page targets or ranks for.

    `is_strategic` marks a client-supplied term; observed and inferred terms
    carry ``False``. The distinction is on the `TARGETS_KEYWORD` edge as well,
    via :class:`~linking_engine.models.enums.KeywordSource`.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    text: str = Field(min_length=1)
    language: str
    search_volume: int | None = Field(default=None, ge=0)
    difficulty: float | None = Field(default=None, ge=0, le=100)
    is_strategic: bool = False


class Link(BaseModel):
    """An existing body link, the `LINKS_TO` edge.

    Only editorial links reach this model. Nav, header, footer and sidebar links
    are discarded at extraction, so no filtering is needed at graph-build time.

    `surrounding_embedding` is precomputed in the embedding stage because Neo4j
    cannot embed text at query time. It lives on the relationship; Community
    Edition has no relationship vector index, which is acceptable because the
    audit scans every edge anyway rather than searching them.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_url: UrlKey
    target_url: UrlKey
    # Ordinal in the source body; (source, position) identifies the edge.
    position: int = Field(ge=0)
    anchor_text: str
    anchor_type: AnchorType | None = None
    # normalise_anchor(anchor_text); None when the anchor normalises to "".
    anchor_key: str | None = None
    # A closed vocabulary of one, for the same reason ActionType has no
    # REPOSITION: the crawler discards nav, header, footer and sidebar links at
    # extraction, so "body" is the only value that can ever reach this model.
    # A Literal makes a nav link arriving here a loud validation error rather
    # than silent bad data, which a bare `str` would wave through.
    link_position: Literal["body"] = "body"
    weight: float | None = Field(default=None, ge=0)
    is_follow: bool = True
    surrounding_text: str
    surrounding_embedding: tuple[float, ...] | None = None
    target_status_code: int | None = Field(default=None, ge=100, le=599)
    issue_flags: frozenset[IssueFlag] = frozenset()
    # FIX whenever the target answers 3xx, 4xx or 5xx.
    verdict: ActionType | None = None


class TenantGraphCounts(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    pages: int = Field(ge=0)
    placeholders: int = Field(ge=0)
    links: int = Field(ge=0)
    redirected_pages: int = Field(default=0, ge=0)
    broken_pages: int = Field(default=0, ge=0)
    fix_links: int = Field(default=0, ge=0)


class PageCentrality(BaseModel):
    """PageRank and betweenness of one crawled page, with its percentile among crawled pages."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    # The stored key as read in the snapshot; written back verbatim, never re-normalised.
    url: str = Field(min_length=1)
    page_rank: float = Field(ge=0, le=1)
    page_rank_percentile: float = Field(ge=0, lt=1)
    betweenness: float = Field(ge=0)
    betweenness_percentile: float = Field(ge=0, lt=1)


class CentralityReport(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    tenant_id: str = Field(min_length=1)
    pages: int = Field(ge=0)
    placeholders: int = Field(ge=0)
    pagerank_s: float = Field(ge=0)
    betweenness_s: float = Field(ge=0)
    write_s: float = Field(ge=0)


class LinkGraphSnapshot(BaseModel):
    """A tenant's pages and body links, read in one transaction; index in ``pages`` is the node id."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    tenant_id: str = Field(min_length=1)
    pages: tuple[str, ...]
    placeholders: tuple[bool, ...]
    # (source url, target url), one per LINKS_TO edge.
    links: tuple[tuple[str, str], ...]

    @model_validator(mode="after")
    def _aligned(self) -> Self:
        if len(self.placeholders) != len(self.pages):
            raise ValueError("placeholders must have one flag per page")
        return self
