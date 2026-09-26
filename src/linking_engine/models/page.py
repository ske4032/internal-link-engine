"""Graph entities: `Page`, `Keyword`, and the `LINKS_TO` edge between two pages.

Field names are the snake_case form of the Neo4j property names in the Data
Model. Every model is frozen and forbids extra fields, so a renamed property
fails at the boundary instead of silently dropping data.

Urls are :class:`~pydantic.HttpUrl`, which normalises on construction - ``/foo``
and ``/foo/`` remain distinct urls and normalisation is not reversible. Canonical
form is decided by the crawler before a `Page` is built; nothing here rewrites a
url after the fact.
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl

from linking_engine.models.enums import ActionType, AnchorType, IssueFlag, LifecycleStage, PageType

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
    url: HttpUrl
    is_placeholder: bool = False
    status_code: int | None = Field(default=None, ge=100, le=599)
    content_hash: str | None = None
    word_count: int | None = Field(default=None, ge=0)
    page_type: PageType | None = None
    is_indexable: bool | None = None
    crawl_depth: int | None = Field(default=None, ge=0)
    language: str | None = None
    freshness: float | None = Field(default=None, ge=0, le=1)
    published_at: datetime | None = None
    lifecycle_stage: LifecycleStage | None = None

    # ── graph analytics ─────────────────────────────────────────────────────
    # pageRank is over body links only; nav and footer links are never captured.
    page_rank: float | None = Field(default=None, ge=0)
    betweenness: float | None = Field(default=None, ge=0)
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
    # Resume marker. Written in the same transaction as the vector it describes,
    # so an interrupted run never re-embeds content it already paid for.
    embedded_content_hash: str | None = None
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

    source_url: HttpUrl
    target_url: HttpUrl
    # Ordinal in the source body; (source, position) identifies the edge.
    position: int = Field(ge=0)
    anchor_text: str
    anchor_type: AnchorType | None = None
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
