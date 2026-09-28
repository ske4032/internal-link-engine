"""The hand-weighted baseline scorer: its weights and the report of one scoring run.

Weights are configuration: a default ships with the package and a tenant can override it
without a deploy. A feature a pair lacks drops out of that pair's score and the remaining
weights rescale, so optional data (GSC, strategic keywords) is never a penalty.
"""

from datetime import datetime
from typing import Final, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

Direction = Literal["higher", "lower"]
# identity: the value is already a share in [0, 1]; percentile: rank within the run.
Normalisation = Literal["identity", "percentile"]

SCORE_HISTOGRAM_BINS: Final = 20


class FeatureWeight(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    column: str = Field(min_length=1)
    weight: float = Field(gt=0)
    # "lower" scores 1 - x: fewer outbound links, lower saturation, and so on.
    direction: Direction = "higher"
    normalisation: Normalisation = "identity"


class ScorerWeights(BaseModel):
    """The scorer's priors, versioned so a score can be traced to the weights behind it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    version: str = Field(min_length=1)
    features: tuple[FeatureWeight, ...] = Field(min_length=1)
    # Share of the run's pairs in tier 1, then tier 2; the rest are tier 3.
    tier_shares: tuple[float, float] = (0.10, 0.30)

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        columns = [f.column for f in self.features]
        if len(set(columns)) != len(columns):
            raise ValueError("a column is weighted twice")
        if any(not 0 < share < 1 for share in self.tier_shares) or sum(self.tier_shares) >= 1:
            raise ValueError("tier shares are in (0, 1) and leave room for tier 3")
        return self


class ScoreReport(BaseModel):
    """One scoring run over a tenant's feature matrix."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    tenant_id: str = Field(min_length=1)
    pairs: int = Field(ge=0)
    weights: ScorerWeights
    weights_hash: str = Field(min_length=1)
    # Pairs per tier, 1 highest.
    tiers: dict[int, int]
    score_p10: float | None = Field(default=None, ge=0, le=100)
    score_p50: float | None = Field(default=None, ge=0, le=100)
    score_p90: float | None = Field(default=None, ge=0, le=100)
    # Counts of the 0-100 score in SCORE_HISTOGRAM_BINS equal bins.
    score_histogram: tuple[int, ...] = Field(
        min_length=SCORE_HISTOGRAM_BINS, max_length=SCORE_HISTOGRAM_BINS
    )
    # How often each feature was a pair's largest contributor.
    top_contributors: dict[str, int]
    # Share of pairs where each weighted feature was missing and dropped out.
    missing_share: dict[str, float]
    feature_cache_key: str = Field(min_length=1)
    seconds: float = Field(ge=0)
    finished_at: datetime

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if set(self.tiers) - {1, 2, 3} or any(count < 0 for count in self.tiers.values()):
            raise ValueError("tiers are 1, 2 and 3 with non-negative counts")
        if sum(self.tiers.values()) != self.pairs:
            raise ValueError("tier counts must add up to the pairs")
        if any(n < 0 for n in self.score_histogram) or sum(self.score_histogram) != self.pairs:
            raise ValueError("score histogram counts must add up to the pairs")
        stats = (self.score_p10, self.score_p50, self.score_p90)
        if any((stat is None) != (self.pairs == 0) for stat in stats):
            raise ValueError("score percentiles are set exactly when there are pairs")
        if sum(self.top_contributors.values()) > self.pairs:
            raise ValueError("more top contributors than pairs")
        if any(not 0 <= share <= 1 for share in self.missing_share.values()):
            raise ValueError("missing shares are in [0, 1]")
        return self
