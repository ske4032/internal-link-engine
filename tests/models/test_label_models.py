"""Validators of the #29/#82 hand-label models: each rejection beside the valid case."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest
from pydantic import ValidationError

from linking_engine.models import (
    ExportedPair,
    LabelEvent,
    LabelExport,
    LabelImportReport,
    RecommendationStatus,
    ScorerName,
)

AT = datetime(2026, 9, 30, 12, 0, tzinfo=UTC)
ACCEPTED, MODIFIED, DISMISSED = (
    RecommendationStatus.ACCEPTED,
    RecommendationStatus.MODIFIED,
    RecommendationStatus.DISMISSED,
)


def pair_fields(**changes: Any) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "pair_id": "p1",
        "source_url": "https://example.com/a",
        "target_url": "https://example.com/t",
        "anchor": "dome tent",
        "anchor_type": "EXACT",
        "rung": "EXACT",
        "keyword": "dome tent",
        "sentence": "Our dome tent guide covers wet nights.",
        "anchor_start": 4,
        "score": 0.8,
        "rank_in_source": 1,
        "band": 1,
    }
    return {**fields, **changes}


def export_fields(**changes: Any) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "export_id": "e1",
        "created_at": AT,
        "seed": 7,
        "pages": 2,
        "pairs_per_page": 3,
        "pairs": 6,
        "scorer": ScorerName.BASELINE,
        "model_version": None,
        "weights_version": "w1",
        "eligible_pages": 2,
        "eligible_pairs": 6,
        "small_pages": 0,
        "file_name": "labels.csv",
    }
    return {**fields, **changes}


def event_fields(status: RecommendationStatus, **changes: Any) -> dict[str, Any]:
    grade = {ACCEPTED: 3, MODIFIED: 2, DISMISSED: 1}.get(status, 3)
    used = {ACCEPTED: "dome tent", MODIFIED: "tent guide"}.get(status)
    fields: dict[str, Any] = {
        **pair_fields(),
        "import_id": "i1",
        "export_id": "e1",
        "scorer": ScorerName.BASELINE,
        "weights_version": "w1",
        "status": status,
        "accepted": status is not DISMISSED,
        "grade": grade,
        "anchor_used": used,
        "reviewer": "qa",
        "created_at": AT,
    }
    return {**fields, **changes}


def test_a_pair_has_distinct_pages_and_its_anchor_at_its_offset() -> None:
    assert ExportedPair(**pair_fields()).anchor_start == 4

    with pytest.raises(ValidationError, match="cannot link to itself"):
        ExportedPair(**pair_fields(target_url="https://example.com/a"))
    with pytest.raises(ValidationError, match="not at its offset"):
        ExportedPair(**pair_fields(anchor_start=5))


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"pairs": 5, "eligible_pairs": 6}, "pairs_per_page pairs of each page"),
        ({"eligible_pages": 1}, "eligible pages and pairs only"),
        ({"eligible_pairs": 5}, "eligible pages and pairs only"),
        ({"scorer": ScorerName.LEARNED}, "model_version is set when the learned ranker"),
        ({"model_version": "4"}, "model_version is set when the learned ranker"),
    ],
)
def test_an_export_rejects_inconsistent_counts_and_snapshots(
    changes: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        LabelExport(**export_fields(**changes))


def test_an_export_of_the_learned_ranker_names_its_model() -> None:
    export = LabelExport(**export_fields(scorer=ScorerName.LEARNED, model_version="4"))

    assert (export.pairs, export.model_version) == (6, "4")
    assert LabelExport(**export_fields()).model_version is None


@pytest.mark.parametrize("status", [ACCEPTED, MODIFIED, DISMISSED])
def test_each_label_status_has_its_grade_and_anchor(status: RecommendationStatus) -> None:
    event = LabelEvent(**event_fields(status))

    assert event.grade == {ACCEPTED: 3, MODIFIED: 2, DISMISSED: 1}[status]


@pytest.mark.parametrize(
    ("status", "changes", "message"),
    [
        (RecommendationStatus.PENDING, {}, "accepted, modified or dismissed"),
        (ACCEPTED, {"grade": 2}, "ACCEPTED label has grade 3"),
        (DISMISSED, {"grade": 2}, "DISMISSED label has grade 1"),
        (MODIFIED, {"accepted": False}, "dismissed ones are not"),
        (DISMISSED, {"accepted": True}, "dismissed ones are not"),
        (ACCEPTED, {"anchor_used": None}, "anchor_used is the proposed anchor"),
        (ACCEPTED, {"anchor_used": "tent guide"}, "anchor_used is the proposed anchor"),
        (MODIFIED, {"anchor_used": None}, "anchor_used is the proposed anchor"),
        (MODIFIED, {"anchor_used": "dome tent"}, "anchor_used is the proposed anchor"),
        (DISMISSED, {"anchor_used": "dome tent"}, "anchor_used is the proposed anchor"),
    ],
)
def test_a_label_event_rejects_an_inconsistent_label(
    status: RecommendationStatus, changes: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        LabelEvent(**event_fields(status, **changes))


def test_an_import_report_splits_its_labelled_rows_by_status() -> None:
    fields: dict[str, Any] = {"tenant_id": "test-a", "export_id": "e1", "rows": 3}
    report = LabelImportReport(**fields, labelled=2, by_status={ACCEPTED: 1, DISMISSED: 1})

    assert report.import_id is None
    with pytest.raises(ValidationError, match="split by status"):
        LabelImportReport(**fields, labelled=2, by_status={ACCEPTED: 1})
    with pytest.raises(ValidationError, match="never exceed the rows"):
        LabelImportReport(**fields, labelled=4, by_status={ACCEPTED: 4})
