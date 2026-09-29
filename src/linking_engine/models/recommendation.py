"""Pipeline output: a proposed action on a pair of pages, and its anchor options.

One `Recommendation` is one document of the tenant's latest run in the `recommendations`
collection, and one item the output API serves.
"""

from collections.abc import Mapping
from datetime import datetime
from typing import Annotated, Final, Literal, Self

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, model_validator

from linking_engine.models.enums import (
    ActionType,
    AnchorType,
    BridgeReason,
    ContentGapFinding,
    IssueFlag,
    RecommendationStatus,
)
from linking_engine.urls import UrlKey

# A new link to add, or copy to write before one can be added: ranked, capped per source page.
NEW_LINK_ACTIONS: Final = frozenset({ActionType.ADD_LINK, ActionType.CONTENT_GAP})
# A verdict on a link that already exists: never capped.
AUDIT_ACTIONS: Final = frozenset({ActionType.FIX, ActionType.REANCHOR, ActionType.REMOVE})


def _as_ordered_pairs(value: object) -> object:
    """Normalise a signal mapping to the pairs the field stores.

    Mongo holds `signals` as a JSON object and that is the natural shape to
    hand in, but a dict field would make the frozen model unhashable. Mapping
    order is insertion order, so the ranked breakdown survives the conversion.
    """
    if isinstance(value, Mapping):
        return tuple(value.items())
    return value


# The ordered (feature name, contribution) breakdown behind a score. A tuple of
# pairs is immutable, hashable and free of `Any`, none of which a mapping value
# typed loosely enough to hold anything would be.
SignalBreakdown = Annotated[
    tuple[tuple[str, float], ...],
    BeforeValidator(_as_ordered_pairs),
]


class AnchorPlacement(BaseModel):
    """Where an extracted anchor phrase sits in the source page's clean body text."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    sentence: str = Field(min_length=1)
    sentence_index: int = Field(ge=0)
    # Character offsets of the phrase in the body text, end exclusive.
    start: int = Field(ge=0)
    end: int = Field(ge=1)

    @model_validator(mode="after")
    def _ordered(self) -> Self:
        if self.end <= self.start:
            raise ValueError("end must be after start")
        return self


class AnchorCandidate(BaseModel):
    """One anchor phrase offered for a link, with its score and provenance.

    `source` distinguishes text already present in the source body from text
    that would require a copy edit to insert. Post-v5 it should always be
    ``EXTRACTED``, because anchor text is extracted and never generated; the
    field stays so acceptance can be tracked separately if that ever changes.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    text: str = Field(min_length=1, max_length=120)
    anchor_type: AnchorType
    source: Literal["EXTRACTED", "GENERATED"]
    # None when the proposing stage gives no score: the audit's REANCHOR phrase.
    score: float | None = Field(default=None, ge=0, le=1)
    # The target keyword the phrase matches.
    keyword: str | None = Field(default=None, min_length=1)
    placement: AnchorPlacement | None = None


class BridgeMark(BaseModel):
    """A new link that is also a proposed bridge between two hubs."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    hub_from: int = Field(ge=0)
    hub_to: int = Field(ge=0)
    reasons: tuple[BridgeReason, ...] = Field(min_length=1)


class Recommendation(BaseModel):
    """A reviewable action on one source→target pair.

    New-link actions (``ADD_LINK``, ``CONTENT_GAP``) are ranked within their source page, so
    they carry `score`, `tier` and `rank_in_source`, and no `position`. Links come first: a
    page's ``ADD_LINK`` records rank 1..limit among themselves, and its ``CONTENT_GAP`` records,
    a separate and shorter list, rank 1..gap limit among themselves. Audit verdicts
    (``FIX``, ``REANCHOR``, ``REMOVE``) are on an existing link: they carry its `position`
    and `current_anchor`, their issue flags, and no score, tier or rank.

    - `finding` and `advice` are set for ``CONTENT_GAP`` only.
    - `proposed_anchors` is never empty; required for ``ADD_LINK``, set for ``REANCHOR`` when
      the copy holds a better phrase, None otherwise. The first is the chosen one.
    - `fix_target` is ``FIX`` only: the canonical page to link to instead of its copy.
    - `bridge` is ``ADD_LINK`` only.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    # Stable across runs for the same tenant, action and link, so feedback survives a rerun.
    id: str = Field(pattern=r"^[0-9a-f]{16}$")
    run_id: str = Field(min_length=1)
    source_url: UrlKey
    target_url: UrlKey
    action_type: ActionType
    # The action in plain words, as the reviewer sees it.
    label: str = Field(min_length=1)
    finding: ContentGapFinding | None = None
    advice: str | None = Field(default=None, min_length=1)

    # 0-100: how the pair ranks among all the tenant's candidate pairs.
    score: float | None = Field(default=None, ge=0, le=100)
    # Priority band, 1 highest, from the tenant's tier shares.
    tier: int | None = Field(default=None, ge=1)
    rank_in_source: int | None = Field(default=None, ge=1)
    # Ordinal of the existing link in the source body; (source_url, position) identifies it.
    position: int | None = Field(default=None, ge=0)
    status: RecommendationStatus

    current_anchor: str | None = None
    proposed_anchors: tuple[AnchorCandidate, ...] | None = None
    issue_flags: tuple[IssueFlag, ...] = ()
    fix_target: UrlKey | None = None
    bridge: BridgeMark | None = None

    rationale: str
    signals: SignalBreakdown
    created_at: datetime

    @model_validator(mode="after")
    def _fits_action(self) -> Self:
        action = self.action_type
        ranked = (self.score, self.tier, self.rank_in_source)
        if action in NEW_LINK_ACTIONS:
            if any(value is None for value in ranked) or self.position is not None:
                raise ValueError(f"{action} needs score, tier and rank_in_source, no position")
            if self.current_anchor is not None or self.issue_flags:
                raise ValueError(f"{action} is a new link: no current anchor or issue flags")
        elif any(value is not None for value in ranked) or self.position is None:
            raise ValueError(f"{action} needs the link's position, no score, tier or rank")
        gap = action is ActionType.CONTENT_GAP
        if (self.finding is not None) != gap or (self.advice is not None) != gap:
            raise ValueError("finding and advice are set for CONTENT_GAP only")
        if action is ActionType.ADD_LINK and not self.proposed_anchors:
            raise ValueError("ADD_LINK needs at least one proposed anchor")
        if self.proposed_anchors is not None and (
            not self.proposed_anchors or action not in {ActionType.ADD_LINK, ActionType.REANCHOR}
        ):
            raise ValueError("proposed_anchors is ADD_LINK and REANCHOR only, and never empty")
        if self.fix_target is not None and action is not ActionType.FIX:
            raise ValueError("fix_target is set for FIX only")
        if self.bridge is not None and action is not ActionType.ADD_LINK:
            raise ValueError("bridge is set for ADD_LINK only")
        if list(self.issue_flags) != sorted(set(self.issue_flags)):
            raise ValueError("issue_flags must be unique and sorted")
        return self
