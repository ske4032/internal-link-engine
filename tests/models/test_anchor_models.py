"""Validators of the #20/#21 anchor extraction models, and of the semantic rung's fields (#22):
each rejection beside the boundary that passes."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from linking_engine.anchor.extraction import DEFAULT_THRESHOLD
from linking_engine.models import (
    AnchorMatch,
    AnchorReport,
    AnchorRung,
    ExtractionSettings,
    KeywordSource,
)
from linking_engine.models.anchors import NO_LANGUAGE, SENTENCE_INDEX_BINS, STEM_JACCARD_BINS

SENTENCE = "Our trail shoes grip wet rock."
EXACT, STEMMED, STEM_SET, SEMANTIC = tuple(AnchorRung)


def only_error(exc_info: pytest.ExceptionInfo[ValidationError]) -> dict[str, object]:
    errors = exc_info.value.errors()
    assert len(errors) == 1, f"expected exactly one validation error, got {errors}"
    return dict(errors[0])


def test_no_language_is_the_undetermined_code_and_the_bins_are_fixed() -> None:
    assert NO_LANGUAGE == "und"
    assert (STEM_JACCARD_BINS, SENTENCE_INDEX_BINS) == (20, (0, 1, 2, 3, 6, 11, 21))


# ── ExtractionSettings ──────────────────────────────────────────────────────


def test_the_stem_set_threshold_defaults_to_the_ladders_and_may_reach_one() -> None:
    assert ExtractionSettings().stem_set_threshold == DEFAULT_THRESHOLD == 0.6
    assert ExtractionSettings(stem_set_threshold=1.0).stem_set_threshold == 1.0


@pytest.mark.parametrize("threshold", [0.0, -0.1, 1.01])
def test_the_stem_set_threshold_is_in_the_half_open_unit_interval(threshold: float) -> None:
    with pytest.raises(ValidationError) as exc_info:
        ExtractionSettings(stem_set_threshold=threshold)
    assert only_error(exc_info)["loc"] == ("stem_set_threshold",)


def test_the_semantic_threshold_is_derived_unless_a_cosine_is_set() -> None:
    assert ExtractionSettings().semantic_threshold is None
    assert ExtractionSettings(semantic_threshold=-1.0).semantic_threshold == -1.0
    assert ExtractionSettings(semantic_threshold=1.0).semantic_threshold == 1.0


@pytest.mark.parametrize("threshold", [-1.01, 1.01])
def test_the_semantic_threshold_is_a_cosine(threshold: float) -> None:
    with pytest.raises(ValidationError) as exc_info:
        ExtractionSettings(semantic_threshold=threshold)
    assert only_error(exc_info)["loc"] == ("semantic_threshold",)


def test_settings_forbid_unknown_fields() -> None:
    with pytest.raises(ValidationError) as exc_info:
        ExtractionSettings.model_validate({"stem_set_threshold": 0.5, "stemSetThreshold": 0.4})
    assert only_error(exc_info)["type"] == "extra_forbidden"


# ── AnchorMatch ─────────────────────────────────────────────────────────────


def match(**fields: object) -> AnchorMatch:
    values: dict[str, object] = {
        "source_url": "example.com/guide",
        "target_url": "example.com/shoes",
        "keyword": "trail shoes",
        "keyword_rank": 1,
        "keyword_source": KeywordSource.CLIENT_STRATEGIC,
        "rung": EXACT,
        "phrase": "trail shoes",
        # The sentence starts at 100 in the body; the phrase at 4 in the sentence.
        "start": 104,
        "end": 115,
        "sentence": SENTENCE,
        "sentence_index": 3,
        "sentence_start": 100,
        **fields,
    }
    return AnchorMatch.model_validate(values)


def test_a_valid_match_builds_on_every_rung() -> None:
    assert match().phrase == SENTENCE[4:15]
    assert match(rung=STEMMED, keyword="trail shoe").rung is STEMMED
    assert match(rung=STEM_SET, keyword="shoes trail", stem_jaccard=1.0).stem_jaccard == 1.0
    for cosine in (-1.0, 1.0):
        semantic = match(rung=SEMANTIC, keyword="hiking footwear", semantic_similarity=cosine)
        assert semantic.semantic_similarity == cosine


def test_a_phrase_at_the_very_start_of_its_sentence_is_valid() -> None:
    found = match(phrase="Our", start=100, end=103, sentence_start=100)
    assert (found.start, found.end) == (100, 103)


def test_a_page_never_links_to_itself() -> None:
    with pytest.raises(ValidationError, match="cannot link to itself"):
        match(target_url="example.com/guide")


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        pytest.param({"start": 99, "end": 110}, "inside its sentence", id="before-the-sentence"),
        pytest.param({"end": 104}, "inside its sentence", id="empty-span"),
        pytest.param({"start": 105, "end": 116}, "not at its offsets", id="shifted"),
        pytest.param({"end": 116}, "do not span the phrase", id="end-past-the-phrase"),
        pytest.param(
            {"phrase": "trail shoes grip"}, "do not span the phrase", id="phrase-too-long"
        ),
        pytest.param({"phrase": "trail shoos"}, "not at its offsets", id="other-text"),
    ],
)
def test_offsets_must_locate_the_phrase_in_its_sentence(
    fields: dict[str, object], message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        match(**fields)


@pytest.mark.parametrize(
    "fields",
    [
        pytest.param({"stem_jaccard": 0.8}, id="exact-with-jaccard"),
        pytest.param({"rung": STEMMED, "stem_jaccard": 0.8}, id="stemmed-with-jaccard"),
        pytest.param({"rung": STEM_SET}, id="stem-set-without-jaccard"),
    ],
)
def test_stem_jaccard_belongs_to_the_stem_set_rung_only(fields: dict[str, object]) -> None:
    with pytest.raises(ValidationError, match="stem set rung only"):
        match(**fields)


@pytest.mark.parametrize(
    "fields",
    [
        pytest.param({"semantic_similarity": 0.8}, id="exact-with-similarity"),
        pytest.param(
            {"rung": STEM_SET, "stem_jaccard": 1.0, "semantic_similarity": 0.8},
            id="stem-set-with-similarity",
        ),
        pytest.param({"rung": SEMANTIC}, id="semantic-without-similarity"),
    ],
)
def test_semantic_similarity_belongs_to_the_semantic_rung_only(fields: dict[str, object]) -> None:
    with pytest.raises(ValidationError, match="semantic rung only"):
        match(**fields)


@pytest.mark.parametrize("cosine", [-1.01, 1.01])
def test_semantic_similarity_is_a_cosine(cosine: float) -> None:
    with pytest.raises(ValidationError) as exc_info:
        match(rung=SEMANTIC, semantic_similarity=cosine)
    assert ("semantic_similarity",) in [error["loc"] for error in exc_info.value.errors()]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("keyword_rank", 0),
        ("start", -1),
        ("sentence_index", -1),
        ("sentence_start", -1),
        ("keyword", ""),
        ("phrase", ""),
        ("sentence", ""),
        ("source_url", ""),
    ],
)
def test_anchor_match_bounds(field: str, value: object) -> None:
    with pytest.raises(ValidationError) as exc_info:
        match(**{field: value})
    assert (field,) in [error["loc"] for error in exc_info.value.errors()]


@pytest.mark.parametrize("jaccard", [0.0, 1.01])
def test_stem_jaccard_is_a_positive_share(jaccard: float) -> None:
    with pytest.raises(ValidationError) as exc_info:
        match(rung=STEM_SET, stem_jaccard=jaccard)
    assert ("stem_jaccard",) in [error["loc"] for error in exc_info.value.errors()]


# ── AnchorReport ────────────────────────────────────────────────────────────


def report(**fields: object) -> AnchorReport:
    values: dict[str, object] = {
        "tenant_id": "acme",
        "stem_set_threshold": 0.5,
        "pairs": 10,
        "bridge_pairs": 2,
        "pairs_with_keywords": 8,
        "pairs_matched": 5,
        "primary_matched": 3,
        "matches": 7,
        "by_rung": {EXACT: 4, STEMMED: 2, STEM_SET: 1},
        "best_rung": {EXACT: 3, STEMMED: 1, STEM_SET: 1},
        "by_keyword_rank": {1: 3, 2: 3, 3: 1},
        "by_rung_and_rank": {EXACT: {1: 3, 2: 1}, STEMMED: {2: 2}, STEM_SET: {3: 1}},
        "stem_jaccard_histogram": (*([0] * 12), 1, *([0] * 7)),
        "sentence_index_histogram": (2, 1, 1, 1, 1, 1, 0),
        "overlapping_existing_anchors": 2,
        "existing_anchors_located": 3,
        "existing_anchors_unlocated": 1,
        "single_token_keywords": 1,
        "source_pages": 6,
        "sources_without_body": 1,
        "stemmed_languages": {"en": 4, "de": 1},
        "unstemmed_languages": {NO_LANGUAGE: 1},
        "seconds": 0.4,
        "finished_at": datetime(2026, 9, 28, tzinfo=UTC),
        **fields,
    }
    return AnchorReport.model_validate(values)


def test_the_report_boundaries_pass() -> None:
    assert report().matches == 7
    equal = report(
        pairs=5,
        pairs_with_keywords=5,
        primary_matched=5,
        bridge_pairs=5,
        matches=5,
        by_rung={EXACT: 5},
        best_rung={EXACT: 5},
        by_keyword_rank={1: 5},
        by_rung_and_rank={EXACT: {1: 5, 2: 0}},
        stem_jaccard_histogram=(0,) * 20,
        sentence_index_histogram=(5, 0, 0, 0, 0, 0, 0),
        sources_without_body=6,
        overlapping_existing_anchors=3,
        existing_anchors_unlocated=0,
    )
    assert (equal.pairs, equal.primary_matched, equal.bridge_pairs) == (5, 5, 5)
    assert equal.overlapping_existing_anchors == equal.existing_anchors_located == 3


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        pytest.param({"primary_matched": 6}, "primary <= matched", id="primary-over-matched"),
        pytest.param(
            {
                "pairs_matched": 9,
                "matches": 9,
                "by_rung": {EXACT: 9},
                "by_keyword_rank": {1: 9},
                "best_rung": {EXACT: 9},
            },
            "primary <= matched",
            id="matched-over-keywords",
        ),
        pytest.param({"pairs_with_keywords": 11}, "primary <= matched", id="keywords-over-pairs"),
        pytest.param({"bridge_pairs": 11}, "more bridge pairs", id="bridges-over-pairs"),
        pytest.param({"by_rung": {EXACT: 4, STEMMED: 2}}, "matches by rung", id="by-rung-sum"),
        pytest.param(
            {"best_rung": {EXACT: 5, STEMMED: 1}}, "pairs by best rung", id="best-rung-sum"
        ),
        pytest.param({"by_keyword_rank": {1: 7, 2: 1}}, "by keyword rank", id="rank-sum"),
        pytest.param(
            {
                "matches": 4,
                "by_rung": {EXACT: 4},
                "by_keyword_rank": {1: 4},
                "by_rung_and_rank": {EXACT: {1: 4}},
                "stem_jaccard_histogram": (0,) * 20,
                "sentence_index_histogram": (4, 0, 0, 0, 0, 0, 0),
            },
            "every matched pair",
            id="fewer-matches-than-matched-pairs",
        ),
        pytest.param({"sources_without_body": 7}, "sources without a body", id="bodies"),
        pytest.param(
            {"overlapping_existing_anchors": 4},
            "more blocking anchors than located",
            id="blocking-over-located",
        ),
        pytest.param(
            {"sentence_index_histogram": (8, -1, 0, 0, 0, 0, 0)},
            "cannot be negative",
            id="negative-count",
        ),
        pytest.param(
            {"stemmed_languages": {"en": -1}}, "cannot be negative", id="negative-language"
        ),
        pytest.param(
            {
                "by_keyword_rank": {0: 3, 2: 4},
                "by_rung_and_rank": {EXACT: {0: 3, 2: 1}, STEMMED: {2: 2}, STEM_SET: {2: 1}},
            },
            "ranks start at 1",
            id="rank-zero",
        ),
        pytest.param(
            {"by_rung_and_rank": {EXACT: {1: 3}, STEMMED: {2: 2}, STEM_SET: {3: 1}}},
            "add up to the matches by rung",
            id="rung-and-rank-by-rung",
        ),
        pytest.param(
            {"by_rung_and_rank": {EXACT: {1: 2, 2: 2}, STEMMED: {2: 2}, STEM_SET: {3: 1}}},
            "add up to the matches by rank",
            id="rung-and-rank-by-rank",
        ),
        pytest.param(
            {"stem_jaccard_histogram": (1, *([0] * 18))}, "Jaccard histogram", id="jaccard-19-bins"
        ),
        pytest.param({"stem_jaccard_histogram": (0,) * 20}, "Jaccard histogram", id="jaccard-sum"),
        pytest.param(
            {"sentence_index_histogram": (2, 1, 1, 1, 1, 1)},
            "sentence histogram",
            id="sentence-6-bins",
        ),
        pytest.param(
            {"sentence_index_histogram": (2, 1, 1, 1, 1, 1, 1)},
            "sentence histogram",
            id="sentence-sum",
        ),
    ],
)
def test_report_counts_must_be_consistent(fields: dict[str, object], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        report(**fields)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("stem_set_threshold", 0.0),
        ("pairs", -1),
        ("tenant_id", ""),
        ("seconds", -1.0),
        ("existing_anchors_located", -1),
        ("existing_anchors_unlocated", -1),
        ("identifier_mismatches", -1),
    ],
)
def test_report_field_bounds(field: str, value: object) -> None:
    with pytest.raises(ValidationError) as exc_info:
        report(**{field: value})
    assert (field,) in [error["loc"] for error in exc_info.value.errors()]
