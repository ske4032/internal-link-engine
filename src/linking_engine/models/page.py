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

from linking_engine.models.enums import (
    ActionType,
    AnchorType,
    IssueFlag,
    LifecycleStage,
    OrphanLabel,
    PageType,
)
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
    # Leiden over the kNN graph of content embeddings; also covers pages without links.
    content_community_id: int | None = None
    # The member nearest its community's content centroid, PageRank breaking ties.
    is_link_pillar: bool | None = None
    is_keyword_pillar: bool | None = None
    is_content_pillar: bool | None = None
    is_hub_pillar: bool | None = None
    # No body link from another page, and none to another page.
    is_orphan: bool | None = None
    is_dead_end: bool | None = None
    orphan_label: OrphanLabel | None = None
    # HDBSCAN cluster label over content_embedding. -1 is the noise label, a
    # real assignment meaning "in no dense region", not a missing value.
    hub_id: int | None = None

    # ── exact duplicates ────────────────────────────────────────────────────
    # The group of crawled pages serving this page's body in its language, and whether this is
    # the group's canonical copy; both None outside any group.
    duplicate_group: int | None = Field(default=None, ge=0)
    is_canonical: bool | None = None

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
    # Written only by the link audit (FIX, REANCHOR or REMOVE); None until audited or when healthy.
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


class PageCommunities(BaseModel):
    """Communities, pillar flags and link state of one crawled page; None means no community."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    # The stored key as read in the snapshot; written back verbatim, never re-normalised.
    url: str = Field(min_length=1)
    link_community_id: int | None = Field(default=None, ge=0)
    keyword_community_id: int | None = Field(default=None, ge=0)
    content_community_id: int | None = Field(default=None, ge=0)
    is_link_pillar: bool = False
    is_keyword_pillar: bool = False
    is_content_pillar: bool = False
    is_orphan: bool
    is_dead_end: bool
    orphan_label: OrphanLabel | None = None

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if (self.orphan_label is not None) != self.is_orphan:
            raise ValueError("orphan_label is set exactly when the page is an orphan")
        for pillar, community in (
            (self.is_link_pillar, self.link_community_id),
            (self.is_keyword_pillar, self.keyword_community_id),
            (self.is_content_pillar, self.content_community_id),
        ):
            if pillar and community is None:
                raise ValueError("a pillar must belong to a community")
        return self


class CommunityContext(BaseModel):
    """What the community stage reads about a crawled page before it writes."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    url: str = Field(min_length=1)
    menu_inlinks: int = Field(default=0, ge=0)
    footer_inlinks: int = Field(default=0, ge=0)
    # The previous run's labels, to measure drift.
    link_community_id: int | None = None
    keyword_community_id: int | None = None
    content_community_id: int | None = None
    hub_id: int | None = None


class PageHub(BaseModel):
    """HDBSCAN hub of one crawled page: -1 is noise, None means the page has no content vector."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    # The stored key as read in the snapshot; written back verbatim, never re-normalised.
    url: str = Field(min_length=1)
    hub_id: int | None = Field(default=None, ge=-1)
    is_hub_pillar: bool = False

    @model_validator(mode="after")
    def _pillar_in_a_hub(self) -> Self:
        if self.is_hub_pillar and (self.hub_id is None or self.hub_id < 0):
            raise ValueError("a hub pillar must belong to a hub")
        return self


class PageFacts(BaseModel):
    """What the served output reads of one crawled page: its place in the site and its link
    state, as the analytics stages stored them."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    url: str = Field(min_length=1)
    language: str | None = None
    page_type: PageType | None = None
    word_count: int = Field(default=0, ge=0)
    # Body links from and to other crawled pages of the tenant, distinct pages each.
    inbound: int = Field(ge=0)
    outbound: int = Field(ge=0)
    crawl_depth: int | None = Field(default=None, ge=0)
    page_rank_percentile: float | None = Field(default=None, ge=0, lt=1)
    # -1 is HDBSCAN noise.
    hub_id: int | None = Field(default=None, ge=-1)
    is_hub_pillar: bool = False
    is_orphan: bool = False
    orphan_label: OrphanLabel | None = None
    is_dead_end: bool = False
    duplicate_group: int | None = Field(default=None, ge=0)
    is_canonical: bool | None = None

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if self.orphan_label is not None and not self.is_orphan:
            raise ValueError("only an orphan carries an orphan label")
        if (self.duplicate_group is None) != (self.is_canonical is None):
            raise ValueError("is_canonical is set exactly for a page in a duplicate group")
        return self


class HubNode(BaseModel):
    """A stored Hub node. A retired hub keeps its id, inactive, with size 0 and no pillar."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    hub_id: int = Field(ge=0)
    size: int = Field(ge=0)
    pillar_url: str | None = None
    active: bool


class InboundAnchorText(BaseModel):
    """One anchor text of the body links into a crawled page from crawled pages of one
    language, and how many links carry it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    target_url: str = Field(min_length=1)
    source_language: str | None = None
    anchor_text: str
    links: int = Field(ge=1)


class HubCentroid(BaseModel):
    """One active hub: its stable id, size, content centroid and pillar page."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    hub_id: int = Field(ge=0)
    size: int = Field(ge=1)
    centroid: tuple[float, ...] = Field(min_length=1)
    pillar_url: str | None = None


class HubReport(BaseModel):
    """One HDBSCAN run over the tenant's page vectors."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    tenant_id: str = Field(min_length=1)
    pages: int = Field(ge=0)
    hubs: int = Field(ge=0)
    noise: int = Field(ge=0)
    noise_pct: float = Field(ge=0, le=1)
    largest_hub_pct: float = Field(ge=0, le=1)
    median_hub_size: float = Field(ge=0)
    # hdbscan's fast DBCV estimate; noise lowers it.
    relative_validity: float | None = None
    persistence_mean: float | None = None
    persistence_min: float | None = None
    persistence_weighted: float | None = None
    # Hub ids carried over from the previous run, new ones, and previous hubs with no match.
    matched_hubs: int = Field(ge=0)
    new_hubs: int = Field(ge=0)
    retired_hubs: int = Field(ge=0)
    # ARI against the previous run's hub ids; noise pages count as singletons.
    drift_ari: float | None = None
    agreement_link: float | None = None
    agreement_content: float | None = None
    # Pages and noise pages per first url path segment.
    section_pages: dict[str, int]
    section_noise: dict[str, int]
    runtime_s: float = Field(ge=0)
    write_s: float = Field(ge=0)


class PassReport(BaseModel):
    """One Leiden pass. Pages without an edge in the pass graph get no community."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    pages: int = Field(ge=0)
    edges: int = Field(ge=0)
    communities: int = Field(ge=0)
    singletons: int = Field(ge=0)
    largest_community_pct: float = Field(ge=0, le=1)
    median_community_size: float = Field(ge=0)
    modularity: float
    disconnected_communities: int = Field(ge=0)
    # Pairwise ARI across the seeded run and the extra seeds; None below two pages.
    seed_stability_ari_mean: float | None = None
    seed_stability_ari_min: float | None = None
    # ARI against the previous run's labels on the pages both runs placed.
    drift_ari: float | None = None
    pillars: int = Field(ge=0)
    runtime_s: float = Field(ge=0)
    stability_s: float = Field(ge=0)


class CommunityReport(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    tenant_id: str = Field(min_length=1)
    crawled_pages: int = Field(ge=0)
    seen_not_crawled: int = Field(ge=0)
    link: PassReport
    keyword: PassReport
    content: PassReport
    keywords: int = Field(ge=0)
    keywords_dropped: int = Field(ge=0)
    pages_with_keywords: int = Field(ge=0)
    pages_with_embeddings: int = Field(ge=0)
    agreement_link_content: float | None = None
    agreement_link_keyword: float | None = None
    agreement_keyword_content: float | None = None
    orphans: int = Field(ge=0)
    dead_ends: int = Field(ge=0)
    orphan_labels: dict[OrphanLabel, int]
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
