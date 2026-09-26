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

from linking_engine.models.enums import AnchorType, LifecycleStage, PageType

# Vectors are ``tuple[float, ...]`` rather than ``list[float]``: these models are
# frozen and therefore hashable, and a list field would raise on hash. The tuple
# is also the typed hand-off the embedding stage uses to pass vectors to the
# graph stage, since Pydantic models are the only thing crossing a boundary.


class Page(BaseModel):
    """A crawled page, plus everything the pipeline computes about it.

    A freshly ingested page carries only the crawl fields. Graph analytics fills
    in `page_rank`, `betweenness` and the community ids; content clustering fills
    in `hub_id`; the embedding stage fills in the vectors and the resume marker.
    Everything computed downstream therefore defaults to ``None``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    # ── crawl ───────────────────────────────────────────────────────────────
    url: HttpUrl
    page_type: PageType
    is_indexable: bool
    http_status: int = Field(ge=100, le=599)
    word_count: int = Field(ge=0)
    crawl_depth: int = Field(ge=0)
    language: str
    freshness: float | None = Field(default=None, ge=0, le=1)
    published_at: datetime | None = None
    lifecycle_stage: LifecycleStage

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
    anchor_text: str
    anchor_type: AnchorType
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
    target_http_status: int | None = Field(default=None, ge=100, le=599)
