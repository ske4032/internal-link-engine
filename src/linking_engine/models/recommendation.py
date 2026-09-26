"""Pipeline output: a proposed action on a pair of pages, and its anchor options.

One `Recommendation` becomes one `SUGGESTED_ACTION` edge plus the full payload in
the `recommendations` collection.
"""

from collections.abc import Mapping
from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, HttpUrl

from linking_engine.models.enums import (
    ActionType,
    AnchorType,
    ContentGapFinding,
    RecommendationStatus,
)


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
    score: float = Field(ge=0, le=1)


class Recommendation(BaseModel):
    """A scored, reviewable action on one source→target pair.

    Three fields are conditional on `action_type`, matching the
    `SUGGESTED_ACTION` properties:

    - `finding` is set for ``CONTENT_GAP`` only, and ``None`` otherwise.
    - `current_anchor` is ``None`` for ``ADD_LINK``: there is no existing anchor.
    - `proposed_anchors` is ``None`` for ``REMOVE`` and ``FIX``: neither needs a
      new phrase.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_url: HttpUrl
    target_url: HttpUrl
    action_type: ActionType
    finding: ContentGapFinding | None

    score: float = Field(ge=0, le=100)
    # Priority band, 1 highest. The Data Model names `tier` next to `score` and
    # `status` but fixes no vocabulary for it, so the band boundaries belong to
    # the stage that materialises actions rather than to this contract.
    tier: int = Field(ge=1)
    status: RecommendationStatus

    current_anchor: str | None
    proposed_anchors: tuple[AnchorCandidate, ...] | None

    rationale: str
    signals: SignalBreakdown
    created_at: datetime
