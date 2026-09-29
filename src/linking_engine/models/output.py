"""What the recommendations stage writes per tenant and run, and the output API serves: page
profiles, hubs, bridges, pairs without an anchor, target pages without a keyword, the site
summary and the run that produced them. Recommendations and duplicate groups have their own
models."""

from datetime import datetime
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from linking_engine.models.enums import (
    ActionType,
    BridgeReason,
    ExclusionReason,
    IssueFlag,
    KeywordRung,
    OrphanLabel,
    PageType,
    ScorerName,
    UnanchoredReason,
)
from linking_engine.models.recommendation import Recommendation
from linking_engine.urls import UrlKey


class AnchorMix(BaseModel):
    """How the body links into a page are anchored, by anchor type."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    exact: int = Field(default=0, ge=0)
    partial: int = Field(default=0, ge=0)
    natural: int = Field(default=0, ge=0)
    branded: int = Field(default=0, ge=0)


class PageProfile(BaseModel):
    """One crawled page of the pipeline: its place in the site and what the run proposes."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    url: UrlKey
    title: str | None = None
    language: str | None = None
    page_type: PageType | None = None
    word_count: int = Field(ge=0)
    # Distinct crawled pages linking in, and linked to, from body copy.
    inbound: int = Field(ge=0)
    outbound: int = Field(ge=0)
    # Clicks from the site's roots through body links; None when none reaches the page.
    crawl_depth: int | None = Field(default=None, ge=0)
    page_rank_percentile: float | None = Field(default=None, ge=0, lt=1)
    hub_id: int | None = Field(default=None, ge=0)
    is_hub_pillar: bool = False
    is_orphan: bool = False
    # What still links to an orphan: menus, footer, both, or nothing.
    orphan_label: OrphanLabel | None = None
    is_dead_end: bool = False
    duplicate_group: int | None = Field(default=None, ge=0)
    # Within its duplicate group; None outside one.
    is_canonical: bool | None = None
    target_keyword: str | None = Field(default=None, min_length=1)
    keyword_rung: KeywordRung | None = None
    anchor_mix: AnchorMix = AnchorMix()
    # New-link recommendations from and into the page, and audit verdicts on its links.
    recommendations_out: int = Field(default=0, ge=0)
    recommendations_in: int = Field(default=0, ge=0)
    audit_verdicts_out: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if self.orphan_label is not None and not self.is_orphan:
            raise ValueError("only an orphan carries an orphan label")
        if (self.duplicate_group is None) != (self.is_canonical is None):
            raise ValueError("is_canonical is set exactly for a page in a duplicate group")
        if (self.target_keyword is None) != (self.keyword_rung is None):
            raise ValueError("target_keyword and keyword_rung are set together")
        return self


class PageDetail(BaseModel):
    """A page's profile, the new links and verdicts proposed on it, and how many new links
    are proposed into it (listed by the recommendations endpoint's target filter)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    profile: PageProfile
    outgoing: tuple[Recommendation, ...]
    incoming_total: int = Field(ge=0)


class HubSummary(BaseModel):
    """One topic hub: its main page, its size and the hubs it is bridged to."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    hub_id: int = Field(ge=0)
    language: str | None = None
    size: int = Field(ge=0)
    pillar_url: UrlKey | None = None
    pillar_title: str | None = None
    orphan_pages: int = Field(ge=0)
    dead_end_pages: int = Field(ge=0)
    # New-link recommendations whose target is in the hub.
    recommendations_in: int = Field(ge=0)
    bridge_hubs: tuple[int, ...] = ()


class BridgeLinkOut(BaseModel):
    """A link proposed to bridge two hubs, and the recommendation it became, if any."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    hub_from: int = Field(ge=0)
    hub_to: int = Field(ge=0)
    slot: int = Field(ge=0)
    rank: int = Field(ge=0)
    source_url: UrlKey
    target_url: UrlKey
    similarity: float
    source_page_rank_percentile: float | None = Field(default=None, ge=0, lt=1)
    anchor_keyword: str | None = None
    anchor_rung: KeywordRung | None = None
    reasons: tuple[BridgeReason, ...] = ()
    # Set when the link is also among its source page's new-link recommendations.
    recommendation_id: str | None = Field(default=None, pattern=r"^[0-9a-f]{16}$")


class BridgePair(BaseModel):
    """Two hubs, how they are linked and related today, and the bridge links proposed."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    language: str | None = None
    hub_a: int = Field(ge=0)
    hub_b: int = Field(ge=0)
    size_a: int = Field(ge=0)
    size_b: int = Field(ge=0)
    pages_ab: int = Field(ge=0)
    pages_ba: int = Field(ge=0)
    link_density: float = Field(ge=0)
    centroid_cosine: float
    query_jaccard: float | None = None
    shared_queries: tuple[str, ...] = ()
    bridge_gap: float
    reasons: tuple[BridgeReason, ...] = ()
    links: tuple[BridgeLinkOut, ...] = ()


class UnanchoredOut(BaseModel):
    """A ranked pair without a usable anchor, why, and what to do about it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_url: UrlKey
    target_url: UrlKey
    reason: UnanchoredReason
    advice: str = Field(min_length=1)
    best_score: float | None = Field(default=None, ge=0)
    rank_in_source: int = Field(ge=1)
    # Whether the pair became a CONTENT_GAP recommendation.
    recommended: bool = False


class TargetFix(BaseModel):
    """A target page that source pages would link to, if it had a target keyword."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    target_url: UrlKey
    title: str | None = None
    fix: str = Field(min_length=1)
    # Distinct source pages with a ranked pair into the page that failed for want of a keyword.
    waiting_sources: int = Field(ge=1)
    # Up to five of them, best ranked first.
    best_sources: tuple[UrlKey, ...] = Field(min_length=1, max_length=5)


class SiteSummary(BaseModel):
    """Counts over one run's output: the landing view of a tenant."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    pages: int = Field(ge=0)
    excluded_pages: dict[ExclusionReason, int] = Field(default_factory=dict)
    orphan_pages: dict[OrphanLabel, int] = Field(default_factory=dict)
    dead_end_pages: int = Field(ge=0)
    duplicate_groups: int = Field(ge=0)
    duplicate_copies: int = Field(ge=0)
    hubs: int = Field(ge=0)
    bridge_pairs: int = Field(ge=0)
    # Proposed bridge links, one per slot; their alternatives are not counted.
    bridge_links: int = Field(ge=0)
    recommendations: dict[ActionType, int] = Field(default_factory=dict)
    # New-link recommendations by tier.
    tiers: dict[int, int] = Field(default_factory=dict)
    sources_with_recommendations: int = Field(ge=0)
    # Source pages with fewer ADD_LINK recommendations than the limit.
    sources_below_limit: int = Field(ge=0)
    links_audited: int = Field(ge=0)
    unverified_links: int = Field(ge=0)
    audit_flags: dict[IssueFlag, int] = Field(default_factory=dict)
    unanchored: dict[UnanchoredReason, int] = Field(default_factory=dict)
    target_fixes: int = Field(ge=0)


class QualitySnapshot(BaseModel):
    """The tenant's latest quality evaluation (#74), as MLflow holds it; no urls."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    mlflow_run_id: str = Field(min_length=1)
    finished_at: datetime | None = None
    metrics: dict[str, float] = Field(default_factory=dict)


class RunInfo(BaseModel):
    """What produced a tenant's output: the run, the versions and the stage outputs it read.
    ``summary`` is set once the run is complete."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    tenant_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    status: Literal["writing", "complete"]
    started_at: datetime
    completed_at: datetime | None = None
    scorer: ScorerName
    model_version: str | None = None
    weights_version: str = Field(min_length=1)
    feature_code: str = Field(min_length=1)
    package_version: str = Field(min_length=1)
    link_audit_run_id: str | None = None
    # Stage output (file name, or "link_audit") -> when it was written.
    inputs: dict[str, datetime] = Field(default_factory=dict)
    limit_per_source: int = Field(ge=1)
    # Content gaps listed per source page, beside its new links; 0 lists none.
    content_gap_limit: int = Field(ge=0)
    quality: QualitySnapshot | None = None
    summary: SiteSummary | None = None

    @model_validator(mode="after")
    def _complete(self) -> Self:
        complete = self.status == "complete"
        if complete != (self.completed_at is not None) or complete != (self.summary is not None):
            raise ValueError("completed_at and summary are set exactly for a complete run")
        return self


class RecommendationReport(BaseModel):
    """Outcome of one recommendations run, as logged; no urls."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    tenant_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    scorer: ScorerName
    model_version: str | None = None
    limit_per_source: int = Field(ge=1)
    content_gap_limit: int = Field(ge=0)
    summary: SiteSummary
    # Ranked pairs walked past because they had no anchor choice and no unanchored reason.
    pairs_not_assessed: int = Field(default=0, ge=0)
    seconds: float = Field(ge=0)
    finished_at: datetime


class ApiKeyInfo(BaseModel):
    """An issued API key as it may be shown: never the key or its hash."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    key_id: str = Field(pattern=r"^[0-9a-f]{8}$")
    tenant_id: str = Field(min_length=1)
    label: str | None = None
    created_at: datetime
    revoked_at: datetime | None = None


class Listing[T: BaseModel](BaseModel):
    """One page of a listing in its stable order; pass ``next_cursor`` as ``after`` for the
    next, until it is None."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    items: tuple[T, ...]
    next_cursor: str | None = None
    # Items matching the filters, over all pages.
    total: int = Field(ge=0)
