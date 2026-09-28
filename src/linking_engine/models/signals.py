"""Pair signals: GSC query and keyword overlap, and cluster membership.

Signals are ranker inputs, never gates: a pair with no overlap still goes on, carrying a
weak signal. Cluster ids are labels of one run, compared within that run only.
"""

from itertools import pairwise
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from linking_engine.models.enums import ClusterAgreement


class GscQuery(BaseModel):
    """One stored GSC query row of a page, as the signals read it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    url: str = Field(min_length=1)
    query: str


class GscQueryStats(BaseModel):
    """One stored GSC query row with its metrics, for the CTR curve and opportunity value."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    url: str = Field(min_length=1)
    query: str
    impressions: int = Field(ge=0)
    clicks: int = Field(default=0, ge=0)
    position: float = Field(ge=1)


class GscMetrics(BaseModel):
    """A page's GSC totals over the last 28 days."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    url: str = Field(min_length=1)
    impressions_28d: int = Field(ge=0)
    clicks_28d: int = Field(ge=0)
    avg_position: float | None = Field(default=None, ge=1)
    query_count: int = Field(ge=0)


class CtrCurve(BaseModel):
    """A tenant's expected CTR per integer position 1..len(ctr), from its own GSC data.

    Non-increasing by construction: a lower position never expects a higher CTR.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    ctr: tuple[float, ...] = Field(min_length=1)
    rows: int = Field(ge=0)
    impressions: int = Field(ge=0)

    @model_validator(mode="after")
    def _non_increasing(self) -> Self:
        if any(not 0 <= value <= 1 for value in self.ctr):
            raise ValueError("ctr values are shares in [0, 1]")
        if any(b > a for a, b in pairwise(self.ctr)):
            raise ValueError("ctr must not increase with position")
        return self


class StrategicKeyword(BaseModel):
    """A keyword the client assigns to one of its pages."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    url: str = Field(min_length=1)
    keyword: str = Field(min_length=1)
    language: str = Field(min_length=1)
    priority: int | None = Field(default=None, ge=1, le=5)
    is_primary: bool = False


class PageSignals(BaseModel):
    """What one crawled page brings to every pair it is part of, computed once per run."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    url: str = Field(min_length=1)
    # Normalised GSC queries the page has impressions for.
    queries: frozenset[str]
    # Normalised strategic keywords the client assigns to the page.
    keywords: frozenset[str]
    # Strategic keywords without a GSC query for them. None when the
    # tenant has no GSC data at all: no data is not the same as not ranking.
    keyword_gap: int | None = Field(default=None, ge=0)
    link_community_id: int | None = Field(default=None, ge=0)
    keyword_community_id: int | None = Field(default=None, ge=0)
    content_community_id: int | None = Field(default=None, ge=0)
    # -1 is HDBSCAN noise: a real label, not a missing one.
    hub_id: int | None = Field(default=None, ge=-1)

    @model_validator(mode="after")
    def _gap_within_keywords(self) -> Self:
        if self.keyword_gap is not None and self.keyword_gap > len(self.keywords):
            raise ValueError("keyword_gap cannot exceed the targeted keywords")
        return self


class PairSignals(BaseModel):
    """The overlap and cluster signals of one source -> target pair."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    # Jaccard of the two pages' sets; 0.0 when both are empty.
    query_overlap: float = Field(ge=0, le=1)
    keyword_overlap: float = Field(ge=0, le=1)
    # None when either page has no id in that clustering. Two noise pages never share a hub.
    same_link_community: bool | None
    same_keyword_community: bool | None
    same_content_community: bool | None
    same_hub: bool | None
    # The keyword community, then the content community, as the topic against the link
    # community.
    cluster_agreement: ClusterAgreement
    content_agreement: ClusterAgreement


class SignalReport(BaseModel):
    """Pair signals over one run's candidate pairs."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    tenant_id: str = Field(min_length=1)
    pages: int = Field(ge=0)
    pages_with_queries: int = Field(ge=0)
    pages_with_keywords: int = Field(ge=0)
    pages_with_gap: int = Field(ge=0)
    # Distinct GSC urls that are not crawled pages of the tenant.
    unmatched_query_urls: int = Field(ge=0)
    pairs: int = Field(ge=0)
    query_overlap_pairs: int = Field(ge=0)
    query_overlap_mean: float | None = Field(default=None, ge=0, le=1)
    keyword_overlap_pairs: int = Field(ge=0)
    keyword_overlap_mean: float | None = Field(default=None, ge=0, le=1)
    same_hub_pairs: int = Field(ge=0)
    # Pairs where either page is HDBSCAN noise.
    noise_pairs: int = Field(ge=0)
    cluster_agreement: dict[ClusterAgreement, int]
    content_agreement: dict[ClusterAgreement, int]
    seconds: float = Field(ge=0)

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if max(self.pages_with_queries, self.pages_with_keywords, self.pages_with_gap) > self.pages:
            raise ValueError("page counts cannot exceed the pages")
        if (
            max(self.query_overlap_pairs, self.keyword_overlap_pairs, self.same_hub_pairs)
            > self.pairs
        ):
            raise ValueError("pair counts cannot exceed the pairs")
        if self.noise_pairs > self.pairs:
            raise ValueError("pair counts cannot exceed the pairs")
        if (self.query_overlap_mean is None) != (self.pairs == 0) or (
            self.keyword_overlap_mean is None
        ) != (self.pairs == 0):
            raise ValueError("overlap means are set exactly when there are pairs")
        for name in ("cluster_agreement", "content_agreement"):
            counts: dict[ClusterAgreement, int] = getattr(self, name)
            if any(count < 0 for count in counts.values()) or sum(counts.values()) != self.pairs:
                raise ValueError(f"{name} counts must add up to the pairs")
        return self
