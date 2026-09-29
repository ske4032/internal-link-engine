"""The audit models refuse results and reports that contradict themselves: a proposal without
REANCHOR, a fix target without FIX, a flag or verdict without a reason, an unverified link with a
score, or counts that do not add up."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from linking_engine.models import (
    ActionType,
    AuditCutoff,
    AuditEdge,
    AuditReason,
    IssueFlag,
    LinkAuditReport,
    LinkAuditResult,
)

AT = datetime(2026, 9, 29, 12, 0, tzinfo=UTC)
FIX, REANCHOR, REMOVE = ActionType.FIX, ActionType.REANCHOR, ActionType.REMOVE


def result(**fields: object) -> LinkAuditResult:
    return LinkAuditResult.model_validate(
        {
            "source_url": "example.com/a",
            "position": 0,
            "target_url": "example.com/b",
            "run_id": "run-1",
            "issue_flags": frozenset(),
            "verdict": None,
            "audited_at": AT,
            **fields,
        }
    )


def report(**fields: object) -> LinkAuditReport:
    return LinkAuditReport.model_validate(
        {
            "tenant_id": "test-tenant",
            "run_id": "run-1",
            "links": 4,
            "unverified": 1,
            "source_pages": 2,
            "index_like_pages": 0,
            "by_flag": {IssueFlag.GENERIC: 1},
            "by_verdict": {REANCHOR: 1, FIX: 1},
            "by_reason": {AuditReason.GENERIC_ANCHOR: 1},
            "healthy": 1,
            "ladder_pairs": 1,
            "proposals": 1,
            "cutoffs": (AuditCutoff(name="outbound_q3", value=2.0, reason="Q3 of 2 sources"),),
            "embeddings": True,
            "keyword_cosines": 3,
            "seconds": 0.5,
            "finished_at": AT,
            **fields,
        }
    )


def test_a_consistent_result_and_report_validate() -> None:
    reanchored = result(
        issue_flags={IssueFlag.GENERIC},
        verdict=REANCHOR,
        reasons=("generic", "better phrase"),
        proposed_anchor="map case",
    )
    fixed = result(verdict=FIX, reasons=("a copy",), fix_target="example.com/canon")
    ghost = result(unverified=True, reasons=("not crawled",))

    assert (reanchored.proposed_anchor, fixed.fix_target, ghost.unverified) == (
        "map case",
        "example.com/canon",
        True,
    )
    assert report().links == 4


@pytest.mark.parametrize(
    ("fields", "problem"),
    [
        ({"verdict": ActionType.ADD_LINK, "reasons": ("x",)}, "FIX, REANCHOR or REMOVE"),
        ({"verdict": FIX, "reasons": ("x",), "proposed_anchor": "map case"}, "only a REANCHOR"),
        ({"verdict": REMOVE, "reasons": ("x",), "fix_target": "example.com/c"}, "only a FIX"),
        ({"issue_flags": {IssueFlag.GENERIC, IssueFlag.NOFOLLOW}, "reasons": ("x",)}, "a reason"),
        ({"verdict": REANCHOR}, "a reason"),
        ({"reasons": (" ",)}, "blank"),
        ({"unverified": True, "keyword_alignment": 0.5}, "unverified"),
        ({"unverified": True, "issue_flags": {IssueFlag.BROKEN}, "reasons": ("x",)}, "unverified"),
    ],
    ids=[
        "add-link-verdict",
        "proposal-without-reanchor",
        "fix-target-without-fix",
        "flag-without-reason",
        "verdict-without-reason",
        "blank-reason",
        "unverified-with-score",
        "unverified-with-flag",
    ],
)
def test_a_self_contradicting_result_is_refused(fields: dict[str, object], problem: str) -> None:
    with pytest.raises(ValidationError, match=problem):
        result(**fields)


@pytest.mark.parametrize(
    ("fields", "problem"),
    [
        ({"by_verdict": {ActionType.CONTENT_GAP: 1, FIX: 1}}, "FIX, REANCHOR or REMOVE"),
        ({"by_flag": {IssueFlag.GENERIC: -1}}, "negative"),
        ({"healthy": 2}, "unverified, healthy or has one verdict"),
        ({"proposals": 2}, "more proposals"),
        ({"index_like_pages": 3}, "index-like"),
        ({"embeddings": False}, "A1 alone"),
        ({"embeddings_skipped_reason": "no scores"}, "A1 alone"),
        ({"keyword_cosines": 4}, "keyword cosines"),
    ],
    ids=[
        "content-gap-verdict",
        "negative-count",
        "links-do-not-add-up",
        "proposals-over-reanchor",
        "index-like-over-sources",
        "a1-without-reason",
        "a2-with-reason",
        "cosines-over-verified",
    ],
)
def test_a_report_whose_counts_disagree_is_refused(fields: dict[str, object], problem: str) -> None:
    with pytest.raises(ValidationError, match=problem):
        report(**fields)


def test_a_non_canonical_copy_cannot_be_its_own_canonical_page() -> None:
    with pytest.raises(ValidationError, match="its own canonical page"):
        AuditEdge(
            source_url="example.com/a",
            position=0,
            target_url="example.com/copy",
            anchor_text="dry sack",
            target_canonical_url="example.com/copy",
        )


# A report with no audited links, and one whose every link points at an uncrawled page.
NO_LINKS: dict[str, object] = {
    "links": 0,
    "unverified": 0,
    "healthy": 0,
    "by_verdict": {},
    "proposals": 0,
    "keyword_cosines": 0,
}
ALL_UNVERIFIED = {**NO_LINKS, "links": 2, "unverified": 2}


@pytest.mark.parametrize(
    ("fields", "rates"),
    [
        ({}, (2 / 4, 2 / 3)),
        ({"healthy": 3, "by_verdict": {}, "proposals": 0}, (0.0, 0.0)),
        (ALL_UNVERIFIED, (0.0, None)),
        (NO_LINKS, (None, None)),
    ],
    ids=["verdicts", "no-verdicts", "all-unverified", "no-links"],
)
def test_the_fixable_rates_are_verdicts_over_all_links_and_over_verified_links(
    fields: dict[str, object], rates: tuple[float | None, float | None]
) -> None:
    audited = report(**fields)

    assert (audited.fixable_rate, audited.verified_fixable_rate) == rates
