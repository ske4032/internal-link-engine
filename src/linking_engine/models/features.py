"""The context feature vector for one candidate (source, target) pair.

Assembled after retrieval and consumed by the scorer and the ranker. Grouped in
the order the Core Build Plan lists the feature families: GSC, lifecycle,
strategic, target structural, source structural, cluster, semantic.

None of these are eligibility filters. Eligibility is ``isIndexable AND
source != target AND NOT already linked``, and everything here is a signal the
ranker weighs afterwards.
"""

from pydantic import BaseModel, ConfigDict, Field, HttpUrl

from linking_engine.models.enums import LifecycleStage


class PairFeatures(BaseModel):
    """~30 context features describing one eligible source→target pair.

    GSC fields are optional because a page published last week has no search
    data at all, and zero impressions is a different statement from no data.
    `has_gsc_data` makes that distinction explicit rather than leaving the
    ranker to infer it from a ``None``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    # ── identity ────────────────────────────────────────────────────────────
    source_url: HttpUrl
    target_url: HttpUrl

    # ── GSC ─────────────────────────────────────────────────────────────────
    # Impressions are log-scaled: the raw distribution spans several orders of
    # magnitude and a linear feature lets the head dominate every split.
    source_impressions_log: float | None = None
    target_impressions_log: float | None = None
    # Banded rather than continuous, because the value of moving from 11 to 9 is
    # not the value of moving from 51 to 49. None is the explicit NULL bucket,
    # and it is the only thing that means "no data": a band of 0 is a measured
    # value like any other, which is what has_gsc_data disambiguates.
    target_position_band: int | None = None
    # Actual CTR minus the tenant's own CTR curve at the target's position.
    target_ctr_gap: float | None = None
    target_query_count: int | None = Field(default=None, ge=0)
    has_gsc_data: bool

    # ── lifecycle ───────────────────────────────────────────────────────────
    source_lifecycle_stage: LifecycleStage
    target_lifecycle_stage: LifecycleStage
    # None where the page has no publishedAt to measure age from.
    source_page_age_days: int | None = Field(default=None, ge=0)
    target_page_age_days: int | None = Field(default=None, ge=0)

    # ── strategic ───────────────────────────────────────────────────────────
    target_kw_count: int = Field(ge=0)
    target_max_priority: int | None = Field(default=None, ge=1, le=5)
    # Strategic keywords the target aims at minus the ones it already ranks for.
    target_keyword_gap: int = Field(ge=0)
    # Jaccard: both keyword sets are ~5 terms, so set sizes are comparable.
    pair_kw_overlap: float = Field(ge=0, le=1)

    # ── target structural ───────────────────────────────────────────────────
    target_inbound_count: int = Field(ge=0)
    target_is_orphan: bool
    target_crawl_depth: int = Field(ge=0)
    # Actual inbound links divided by expected inbound for the target's PageRank
    # tier. Above 1 means the page is already better linked than its authority
    # predicts; unbounded above, so no upper limit.
    target_saturation_ratio: float = Field(ge=0)

    # ── source structural ───────────────────────────────────────────────────
    # Equity divides across a page's outbound links, so a 151st link on an
    # already saturated source is close to worthless whatever the target is.
    source_outbound_count: int = Field(ge=0)
    source_outbound_density: float = Field(ge=0)
    source_link_equity_share: float = Field(ge=0, le=1)

    # ── cluster ─────────────────────────────────────────────────────────────
    # Two Leiden passes (link graph, keyword graph) plus the HDBSCAN content
    # hub. None where the page was absent from that projection; hub_id -1 is a
    # real HDBSCAN noise label, not a missing value.
    source_link_community_id: int | None = None
    target_link_community_id: int | None = None
    source_keyword_community_id: int | None = None
    target_keyword_community_id: int | None = None
    source_hub_id: int | None = None
    target_hub_id: int | None = None
    # How far the three clusterings agree that this pair belongs together.
    cluster_agreement: float = Field(ge=0, le=1)

    # ── semantic ────────────────────────────────────────────────────────────
    # Cosine, not Jaccard: a 3-token anchor against a 2,000-token page maxes out
    # near 0.0015 on Jaccard and every score collapses into noise.
    content_cosine: float = Field(ge=-1, le=1)
    context_relevance: float = Field(ge=0, le=1)
    anchor_target_fit: float = Field(ge=0, le=1)
