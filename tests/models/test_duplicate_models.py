from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from linking_engine.models import DuplicateGroup, DuplicateInput, DuplicateReport, Page


def group(group_id: int, canonical: str, *copies: str) -> DuplicateGroup:
    return DuplicateGroup(group_id=group_id, canonical=canonical, copies=copies)


GROUPS = (
    group(0, "example.com/a", "example.com/b", "example.com/c"),
    group(1, "example.com/d", "example.com/e"),
)


def report(**fields: object) -> DuplicateReport:
    values: dict[str, object] = {
        "tenant_id": "acme",
        "groups": GROUPS,
        "pages_in_groups": 5,
        "non_canonical": 3,
        "largest_group": 3,
        "seconds": 0.25,
        "finished_at": datetime(2026, 9, 28, tzinfo=UTC),
    }
    return DuplicateReport.model_validate({**values, **fields})


def test_a_consistent_report_is_accepted() -> None:
    assert report().largest_group == 3
    empty = report(groups=(), pages_in_groups=0, non_canonical=0, largest_group=0)
    assert empty.groups == ()


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        pytest.param({"pages_in_groups": 4}, "pages_in_groups", id="pages"),
        pytest.param({"non_canonical": 2}, "non_canonical", id="copies"),
        pytest.param({"largest_group": 2}, "largest_group", id="largest"),
        pytest.param(
            {"groups": (GROUPS[0], group(2, "example.com/d", "example.com/e"))},
            "count up from 0",
            id="gap-in-ids",
        ),
        pytest.param({"groups": (GROUPS[1], GROUPS[0])}, "count up from 0", id="ids-out-of-order"),
        pytest.param(
            {
                "groups": (
                    group(0, "example.com/d", "example.com/e"),
                    group(1, "example.com/a", "example.com/b", "example.com/c"),
                )
            },
            "ordered by canonical",
            id="canonicals-out-of-order",
        ),
        pytest.param(
            {"groups": (GROUPS[0], group(1, "example.com/d", "example.com/a"))},
            "one group only",
            id="page-in-two-groups",
        ),
        pytest.param({"tenant_id": ""}, "tenant_id", id="blank-tenant"),
    ],
)
def test_an_inconsistent_report_is_refused(fields: dict[str, object], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        report(**fields)


@pytest.mark.parametrize(
    ("canonical", "copies", "message"),
    [
        pytest.param("example.com/a", (), "at least 1", id="no-copies"),
        pytest.param(
            "example.com/a", ("example.com/c", "example.com/b"), "ascending", id="unsorted"
        ),
        pytest.param(
            "example.com/a", ("example.com/b", "example.com/b"), "unique", id="repeated-copy"
        ),
        pytest.param("example.com/a", ("example.com/a",), "canonical", id="canonical-copy"),
        pytest.param("", ("example.com/a",), "at least 1", id="blank-canonical"),
        pytest.param("example.com/a", ("",), "at least 1", id="blank-copy"),
    ],
)
def test_an_inconsistent_group_is_refused(
    canonical: str, copies: tuple[str, ...], message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        DuplicateGroup(group_id=0, canonical=canonical, copies=copies)


def test_group_ids_and_inbound_counts_are_never_negative() -> None:
    with pytest.raises(ValidationError):
        DuplicateGroup(group_id=-1, canonical="example.com/a", copies=("example.com/b",))
    with pytest.raises(ValidationError):
        DuplicateInput(url="example.com/a", body_hash="a" * 64, indexable=True, inbound=-1)
    with pytest.raises(ValidationError):
        Page(url="example.com/a", duplicate_group=-1)


def test_models_are_frozen_and_forbid_extra_fields() -> None:
    with pytest.raises(ValidationError):
        report(groups_found=2)
    with pytest.raises(ValidationError):
        GROUPS[0].canonical = "example.com/z"  # type: ignore[misc]
