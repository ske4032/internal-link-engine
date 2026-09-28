"""Existing links scored against their target: the sentence around the link and the anchor.

Both are Neo4j's normalised cosine in [0, 1] against the target's content vector. They
describe links that already exist; candidate pairs get them once the anchor ladder has
chosen a sentence and an anchor.
"""

from datetime import datetime
from typing import Final, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

HISTOGRAM_BINS: Final = 20


class LinkRelevance(BaseModel):
    """One existing body link between crawled pages, scored against its target."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_url: str = Field(min_length=1)
    position: int = Field(ge=0)
    target_url: str = Field(min_length=1)
    # The surrounding sentence against the target page.
    context_relevance: float | None = Field(default=None, ge=0, le=1)
    # The anchor against the target page; None for generic anchors, which name no topic.
    anchor_target_fit: float | None = Field(default=None, ge=0, le=1)
    anchor_generic: bool = False

    @model_validator(mode="after")
    def _no_fit_for_generic(self) -> Self:
        if self.anchor_generic and self.anchor_target_fit is not None:
            raise ValueError("a generic anchor has no anchor_target_fit")
        return self


class ScoreDistribution(BaseModel):
    """The spread of one relevance score over a tenant's links, and where it splits."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    count: int = Field(ge=1)
    mean: float = Field(ge=0, le=1)
    p10: float = Field(ge=0, le=1)
    p25: float = Field(ge=0, le=1)
    p50: float = Field(ge=0, le=1)
    p75: float = Field(ge=0, le=1)
    p90: float = Field(ge=0, le=1)
    # The boundary between the low and high mode, derived from this tenant's scores; None
    # when the scores do not separate into two modes.
    split: float | None = Field(default=None, ge=0, le=1)
    # Share of links below the split.
    low_share: float | None = Field(default=None, ge=0, le=1)
    # Counts in HISTOGRAM_BINS equal bins over [0, 1]; the edges are fixed, so not stored.
    histogram: tuple[int, ...] = Field(min_length=HISTOGRAM_BINS, max_length=HISTOGRAM_BINS)

    @model_validator(mode="after")
    def _ordered(self) -> Self:
        if not self.p10 <= self.p25 <= self.p50 <= self.p75 <= self.p90:
            raise ValueError("percentiles must not decrease")
        if (self.split is None) != (self.low_share is None):
            raise ValueError("split and low_share are set together")
        if any(n < 0 for n in self.histogram) or sum(self.histogram) != self.count:
            raise ValueError("histogram counts must add up to count")
        return self


class LinkRelevanceReport(BaseModel):
    """One link-relevance run over a tenant's existing links."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    tenant_id: str = Field(min_length=1)
    # Body links between the tenant's crawled pages, and those whose target has a vector.
    links: int = Field(ge=0)
    scored: int = Field(ge=0)
    generic_anchors: int = Field(ge=0)
    # Scored links whose anchor has no stored vector.
    without_anchor_vector: int = Field(ge=0)
    context: ScoreDistribution | None = None
    anchor: ScoreDistribution | None = None
    seconds: float = Field(ge=0)
    finished_at: datetime

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if self.scored > self.links:
            raise ValueError("more scored links than links")
        if self.generic_anchors + self.without_anchor_vector > self.scored:
            raise ValueError("more unscored anchors than scored links")
        if (self.context is None) != (self.scored == 0):
            raise ValueError("the context distribution exists exactly when links were scored")
        return self
