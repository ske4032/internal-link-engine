"""Hub bridges: links mostly stay inside a hub, and every hub is connected to the others.

A hub pair counts as connected in one direction when at least ``floor_share`` of the source hub's
pages link into the other hub. Bridges are proposed until each covered pair reaches that floor.
"""

from datetime import datetime
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from linking_engine.models.enums import BridgeReason, KeywordRung


class HubPair(BaseModel):
    """Two hubs of one language, scored for a bridge. ``hub_a`` < ``hub_b``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    language: str | None = None
    hub_a: int = Field(ge=0)
    hub_b: int = Field(ge=0)
    size_a: int = Field(ge=1)
    size_b: int = Field(ge=1)
    # Distinct pages of one hub with at least one body link into the other.
    pages_ab: int = Field(ge=0)
    pages_ba: int = Field(ge=0)
    # Body links between the two hubs, both directions, over size_a x size_b.
    link_density: float = Field(ge=0)
    centroid_cosine: float = Field(ge=-1, le=1)
    # Jaccard of the hubs' GSC query sets; None when the tenant has no GSC data.
    query_jaccard: float | None = Field(default=None, ge=0, le=1)
    # The queries both hubs have impressions for, the ones most pages share first; empty
    # without GSC data.
    shared_queries: tuple[str, ...] = ()
    bridge_gap: float
    reasons: tuple[BridgeReason, ...] = ()

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if self.hub_a >= self.hub_b:
            raise ValueError("hub_a must be lower than hub_b")
        if self.pages_ab > self.size_a or self.pages_ba > self.size_b:
            raise ValueError("more linking pages than hub pages")
        if len(set(self.reasons)) != len(self.reasons):
            raise ValueError("duplicate reasons")
        if len(set(self.shared_queries)) != len(self.shared_queries):
            raise ValueError("duplicate shared queries")
        if self.shared_queries and self.query_jaccard is None:
            raise ValueError("shared queries need GSC data")
        return self


class BridgeLink(BaseModel):
    """One proposed body link from a page of one hub into another hub."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    language: str | None = None
    hub_from: int = Field(ge=0)
    hub_to: int = Field(ge=0)
    # The slot this link fills towards the floor, 1 first; rank 1 is the proposal and 2-3 its
    # alternative targets for the same source page.
    slot: int = Field(ge=1)
    rank: int = Field(ge=1, le=3)
    source_url: str = Field(min_length=1)
    target_url: str = Field(min_length=1)
    # Cosine of the target's content vector to the source page's.
    similarity: float = Field(ge=-1, le=1)
    source_page_rank_percentile: float | None = Field(default=None, ge=0, lt=1)
    # The target's resolved keyword (#19), the anchor to extract; None when it has none.
    anchor_keyword: str | None = None
    anchor_rung: KeywordRung | None = None
    reasons: tuple[BridgeReason, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if self.hub_from == self.hub_to:
            raise ValueError("a bridge joins two different hubs")
        if self.source_url == self.target_url:
            raise ValueError("a page cannot link to itself")
        if (self.anchor_keyword is None) != (self.anchor_rung is None):
            raise ValueError("anchor_keyword and anchor_rung are set together")
        return self


class BridgeReport(BaseModel):
    """One hub-bridge run over a tenant."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    tenant_id: str = Field(min_length=1)
    floor_share: float = Field(gt=0, lt=1)
    hubs: int = Field(ge=0)
    noise_pages: int = Field(ge=0)
    hub_pairs: int = Field(ge=0)
    # Connected components of the hub graph (pairs meeting the floor both ways), summed over
    # languages, before and after the proposed bridges.
    components_before: int = Field(ge=0)
    components_after: int = Field(ge=0)
    # Directions of covered pairs below the floor, the links needed to lift them, and the
    # proposals made (rank 1) and their alternatives.
    directions_below_floor: int = Field(ge=0)
    links_needed: int = Field(ge=0)
    bridge_links: int = Field(ge=0)
    alternatives: int = Field(ge=0)
    # Directions that could not reach the floor: not enough eligible source pages or targets.
    directions_short: int = Field(ge=0)
    by_reason: dict[BridgeReason, int]
    gsc_used: bool
    seconds: float = Field(ge=0)
    finished_at: datetime

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if self.components_after > self.components_before:
            raise ValueError("bridges cannot split the hub graph")
        if self.bridge_links > self.links_needed:
            raise ValueError("more bridge links than needed")
        if self.directions_short > self.directions_below_floor:
            raise ValueError("more short directions than directions below the floor")
        if any(count < 0 for count in self.by_reason.values()):
            raise ValueError("reason counts cannot be negative")
        return self
