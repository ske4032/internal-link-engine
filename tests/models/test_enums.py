"""The enum tables, member for member.

These are design decisions, not incidental data. The absence of `REPOSITION` is the
clearest case: the crawler extracts body links only, so there is no footer to move a
link out of, and a future edit that "helpfully" adds it would change what the audit is
allowed to recommend. Member sets are asserted against `__members__` rather than
iteration, because an alias (`REPOSITION = ADD_LINK`) does not show up in iteration.
"""

from __future__ import annotations

from enum import StrEnum

import pytest

from linking_engine.models.enums import (
    ActionType,
    AnchorType,
    ContentGapFinding,
    IssueFlag,
    KeywordSource,
    LifecycleStage,
    PageType,
    RecommendationStatus,
)

ENUM_CASES = [
    (ActionType, ("ADD_LINK", "REANCHOR", "REMOVE", "FIX", "CONTENT_GAP")),
    (AnchorType, ("EXACT", "PARTIAL", "NATURAL", "BRANDED")),
    (
        IssueFlag,
        (
            "GENERIC",
            "MISALIGNED",
            "OFF_TOPIC",
            "OVER_OPTIMISED",
            "WASTED_EQUITY",
            "BROKEN",
            "REDIRECTED",
            "NOFOLLOW",
            "NOINDEX_TARGET",
        ),
    ),
    (LifecycleStage, ("NEW", "EMERGING", "ESTABLISHED", "MATURE")),
    (PageType, ("PILLAR", "CATEGORY", "PRODUCT", "ARTICLE")),
    (KeywordSource, ("CLIENT_STRATEGIC", "GSC_OBSERVED", "INFERRED")),
    (ContentGapFinding, ("NO_TOPICAL_MENTION", "AWKWARD_PHRASING")),
    (RecommendationStatus, ("PENDING", "ACCEPTED", "MODIFIED", "DISMISSED")),
]

ENUM_IDS = [enum_type.__name__ for enum_type, _ in ENUM_CASES]


@pytest.mark.parametrize(("enum_type", "expected"), ENUM_CASES, ids=ENUM_IDS)
def test_members_are_exactly_the_documented_set(enum_type, expected) -> None:
    declared = set(enum_type.__members__)
    assert declared == set(expected), (
        f"{enum_type.__name__} drifted from the spec — "
        f"unexpected: {sorted(declared - set(expected))}, "
        f"missing: {sorted(set(expected) - declared)}"
    )
    assert len(list(enum_type)) == len(expected), (
        f"{enum_type.__name__} declares an alias member; aliases hide from iteration "
        "and from every membership check written against it"
    )


@pytest.mark.parametrize(("enum_type", "expected"), ENUM_CASES, ids=ENUM_IDS)
def test_is_a_strenum_whose_values_equal_their_names(enum_type, expected) -> None:
    assert issubclass(enum_type, StrEnum), (
        f"{enum_type.__name__} must be a StrEnum: the graph and Mongo store the bare "
        "string, and a plain Enum serialises as a member repr"
    )
    assert len(expected) > 0
    for member in enum_type:
        assert isinstance(member, str)
        assert member.value == member.name, f"{enum_type.__name__}.{member.name} value drift"
        assert member == member.name


def test_action_type_has_no_reposition() -> None:
    """Body links only — there is no footer to move a link out of."""
    assert "REPOSITION" not in ActionType.__members__, (
        "REPOSITION is deliberately absent: the crawler discards nav, header, footer "
        "and sidebar links at extraction, so no link can be repositioned"
    )
    assert "REPOSITION" not in {member.value for member in ActionType}


def test_content_gap_is_an_action_type_and_has_its_own_finding_enum() -> None:
    """CONTENT_GAP is the ladder's rung 4: the two findings only apply to it."""
    assert ActionType.CONTENT_GAP in set(ActionType)
    assert set(ContentGapFinding.__members__) == {"NO_TOPICAL_MENTION", "AWKWARD_PHRASING"}
