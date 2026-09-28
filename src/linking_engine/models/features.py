"""The context feature vector for one candidate (source, target) pair.

Assembled after retrieval and consumed by the scorer and the ranker. None of these are
eligibility filters: everything here is a signal the ranker weighs afterwards.

GSC and strategic keywords are optional enrichment. A tenant with neither still gets a
full vector from content and link structure, with ``has_gsc_data`` telling "no data" from
zero. Urls are kept verbatim, as stored.
"""

from datetime import datetime
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from linking_engine.models.enums import ClusterAgreement


class PageStructure(BaseModel):
    """What the graph knows about one crawled page, read once per run for every pair."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    url: str = Field(min_length=1)
    language: str | None = None
    word_count: int = Field(default=0, ge=0)
    # Body links from and to other crawled pages of the tenant, distinct pages each.
    inbound: int = Field(ge=0)
    outbound: int = Field(ge=0)
    is_orphan: bool | None = None
    page_rank_percentile: float | None = Field(default=None, ge=0, lt=1)
    crawl_depth: int | None = Field(default=None, ge=0)
    link_community_id: int | None = Field(default=None, ge=0)
    keyword_community_id: int | None = Field(default=None, ge=0)
    content_community_id: int | None = Field(default=None, ge=0)
    # -1 is HDBSCAN noise.
    hub_id: int | None = Field(default=None, ge=-1)
    is_hub_pillar: bool = False


class PairFeatures(BaseModel):
    """The features of one eligible source -> target pair."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    # ── identity ────────────────────────────────────────────────────────────
    source_url: str = Field(min_length=1)
    target_url: str = Field(min_length=1)

    # ── GSC (optional enrichment) ───────────────────────────────────────────
    # log1p of 28-day impressions; None without GSC data for the page.
    source_impressions_log: float | None = Field(default=None, ge=0)
    target_impressions_log: float | None = Field(default=None, ge=0)
    # 0: 1-3, 1: 4-10, 2: 11-20, 3: 21-50, 4: over 50. None is the explicit NULL bucket.
    target_position_band: int | None = Field(default=None, ge=0, le=4)
    # Actual CTR minus the tenant's own CTR curve at the target's position.
    target_ctr_gap: float | None = Field(default=None, ge=-1, le=1)
    target_query_count: int | None = Field(default=None, ge=0)
    has_gsc_data: bool

    # ── overlap (#13) ───────────────────────────────────────────────────────
    pair_query_overlap: float = Field(ge=0, le=1)
    pair_kw_overlap: float = Field(ge=0, le=1)
    target_kw_count: int = Field(ge=0)
    target_max_priority: int | None = Field(default=None, ge=1, le=5)
    # None when the tenant has no GSC data at all.
    target_keyword_gap: int | None = Field(default=None, ge=0)

    # ── target structure ────────────────────────────────────────────────────
    target_inbound_count: int = Field(ge=0)
    target_is_orphan: bool
    target_crawl_depth: int | None = Field(default=None, ge=0)
    target_page_rank_percentile: float | None = Field(default=None, ge=0, lt=1)
    # Inbound links over the median inbound of the target's PageRank decile. Unbounded above.
    target_saturation_ratio: float = Field(ge=0)

    # ── source structure ────────────────────────────────────────────────────
    # Equity divides across a page's outbound links, so a 151st link is close to worthless.
    source_outbound_count: int = Field(ge=0)
    # Outbound body links per 1,000 words.
    source_outbound_density: float = Field(ge=0)
    # 1 / (outbound + 1).
    source_link_equity_share: float = Field(gt=0, le=1)
    # Authority the source passes on; None when the source has no PageRank yet.
    source_page_rank_percentile: float | None = Field(default=None, ge=0, lt=1)

    # ── hub structure: links mostly stay inside a hub ───────────────────────
    same_hub: bool | None
    source_is_hub_pillar: bool
    target_is_hub_pillar: bool
    # Members of the page's hub; None outside any hub (no vector or noise).
    source_hub_size: int | None = Field(default=None, ge=1)
    target_hub_size: int | None = Field(default=None, ge=1)
    # Share of the target's other hub members that already link to it; None outside a hub.
    target_hub_coverage: float | None = Field(default=None, ge=0, le=1)

    # ── clusters (#14): per-run labels, never compared across runs ──────────
    source_link_community_id: int | None = None
    target_link_community_id: int | None = None
    source_keyword_community_id: int | None = None
    target_keyword_community_id: int | None = None
    source_content_community_id: int | None = None
    target_content_community_id: int | None = None
    source_hub_id: int | None = None
    target_hub_id: int | None = None
    same_link_community: bool | None
    same_keyword_community: bool | None
    same_content_community: bool | None
    cluster_agreement: ClusterAgreement
    content_agreement: ClusterAgreement

    # ── semantic ────────────────────────────────────────────────────────────
    content_cosine: float = Field(ge=-1, le=1)
    # From #16; None until it exists.
    context_relevance: float | None = Field(default=None, ge=0, le=1)
    anchor_target_fit: float | None = Field(default=None, ge=0, le=1)

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if self.source_url == self.target_url:
            raise ValueError("a page cannot be its own source")
        if (self.target_hub_coverage is None) != (self.target_hub_size is None):
            raise ValueError("hub coverage is set exactly when the target is in a hub")
        return self


class FeatureReport(BaseModel):
    """One feature assembly run over a tenant's candidate pairs."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    tenant_id: str = Field(min_length=1)
    pairs: int = Field(ge=0)
    # Matrix columns, in the persisted order.
    columns: tuple[str, ...] = Field(min_length=1)
    chunks: int = Field(ge=0)
    # Data gaps of this tenant, reported rather than failed.
    all_null_columns: tuple[str, ...]
    constant_columns: tuple[str, ...]
    null_share: dict[str, float]
    has_gsc_data_share: float | None = Field(default=None, ge=0, le=1)
    cache_key: str = Field(min_length=1)
    cache_hit: bool
    seconds: float = Field(ge=0)
    finished_at: datetime

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if len(set(self.columns)) != len(self.columns):
            raise ValueError("duplicate columns")
        unknown = (
            set(self.all_null_columns) | set(self.constant_columns) | set(self.null_share)
        ) - set(self.columns)
        if unknown:
            raise ValueError(f"unknown columns {sorted(unknown)}")
        if any(not 0 <= share <= 1 for share in self.null_share.values()):
            raise ValueError("null shares are in [0, 1]")
        return self
