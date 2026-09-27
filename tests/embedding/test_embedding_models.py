from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from linking_engine.models import (
    AnchorKeyUpdate,
    EdgeRef,
    EmbeddingBatch,
    EmbeddingModelCount,
    EmbeddingTarget,
    EmbedRunReport,
    LinkEmbedReport,
    LinkText,
    PageEmbedding,
    PageText,
    SentenceTarget,
    TenantEmbedReport,
)


def embedding() -> PageEmbedding:
    return PageEmbedding(
        url="https://example.com/a",
        vector=(0.6, 0.8),
        tokens=50,
        original_tokens=80,
        truncated=True,
    )


def test_page_text_rejects_empty_url() -> None:
    with pytest.raises(ValidationError, match="url"):
        PageText(url="", text="body")


def test_page_text_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError, match="title"):
        PageText(url="https://example.com/a", text="body", title="A")  # type: ignore[call-arg]


def test_page_text_is_frozen_and_hashable() -> None:
    text = PageText(url="https://example.com/a", text="body")
    with pytest.raises(ValidationError):
        text.text = "changed"  # type: ignore[misc]
    assert hash(text) == hash(PageText(url="https://example.com/a", text="body"))


def test_page_embedding_is_frozen_and_hashable() -> None:
    result = embedding()
    with pytest.raises(ValidationError):
        result.truncated = False  # type: ignore[misc]
    assert isinstance(result.vector, tuple)
    assert hash(result) == hash(embedding())


def test_page_embedding_keeps_both_token_counts() -> None:
    result = embedding()
    assert (result.tokens, result.original_tokens, result.truncated) == (50, 80, True)


@pytest.mark.parametrize(
    ("field", "value"),
    [("url", ""), ("vector", ()), ("tokens", -1), ("original_tokens", -1)],
)
def test_page_embedding_rejects_out_of_range_fields(field: str, value: object) -> None:
    values = {**embedding().model_dump(), field: value}
    with pytest.raises(ValidationError, match=field):
        PageEmbedding(**values)


def test_page_embedding_accepts_zero_token_counts() -> None:
    result = PageEmbedding(
        url="https://example.com/empty",
        vector=(1.0,),
        tokens=0,
        original_tokens=0,
        truncated=False,
    )
    assert (result.tokens, result.original_tokens, result.vector) == (0, 0, (1.0,))


# --- EmbeddingBatch ---------------------------------------------------------------------


def test_embedding_batch_keeps_embeddings_in_order_and_api_tokens() -> None:
    second = embedding().model_copy(update={"url": "https://example.com/b"})
    batch = EmbeddingBatch(embeddings=(embedding(), second), api_tokens=0)
    assert [item.url for item in batch.embeddings] == ["https://example.com/a", second.url]
    assert batch.api_tokens == 0


def test_embedding_batch_rejects_empty_embeddings() -> None:
    with pytest.raises(ValidationError, match="embeddings") as exc_info:
        EmbeddingBatch(embeddings=(), api_tokens=10)
    assert [error["type"] for error in exc_info.value.errors()] == ["too_short"]


def test_embedding_batch_rejects_negative_api_tokens() -> None:
    with pytest.raises(ValidationError, match="api_tokens"):
        EmbeddingBatch(embeddings=(embedding(),), api_tokens=-1)


# --- EmbeddingTarget and EmbeddingModelCount ---------------------------------------------


def test_embedding_target_allows_a_missing_body_hash_but_not_an_empty_url() -> None:
    assert EmbeddingTarget(url="https://example.com/a", body_hash=None).body_hash is None
    with pytest.raises(ValidationError, match="url"):
        EmbeddingTarget(url="", body_hash=None)


def test_embedding_model_count_allows_no_model_but_needs_a_vector() -> None:
    assert EmbeddingModelCount(embedding_model=None, vectors=1).embedding_model is None
    with pytest.raises(ValidationError, match="vectors"):
        EmbeddingModelCount(embedding_model="voyage-4-large", vectors=0)


# --- EmbedRunReport ---------------------------------------------------------------------

# Distinct values, so a sum that drops or double-counts one field cannot reach 15.
HANDLED = {
    "embedded": 1,
    "skipped_not_usable": 2,
    "skipped_empty_body": 3,
    "skipped_missing": 4,
    "skipped_hash_mismatch": 5,
}


def report(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "tenant_id": "t",
        "embedding_model": "voyage-4-large",
        "dimensions": 2048,
        "selected": sum(HANDLED.values()),
        **HANDLED,
        "up_to_date": 6,
        "placeholders": 7,
        "non_2xx": 8,
        "flushes": 1,
        "api_tokens": 100,
        "tokens": 90,
        "truncated": 0,
        "elapsed_s": 0.5,
        "finished_at": datetime(2026, 9, 27, tzinfo=UTC),
    }
    values.update(overrides)
    return values


def test_run_report_accepts_selected_equal_to_embedded_plus_skipped() -> None:
    result = EmbedRunReport.model_validate(report())
    assert result.selected == sum(getattr(result, name) for name in HANDLED) == 15


@pytest.mark.parametrize("selected", [14, 16])
def test_run_report_rejects_unaccounted_selected(selected: int) -> None:
    with pytest.raises(ValidationError, match="selected must equal embedded plus every skipped"):
        EmbedRunReport.model_validate(report(selected=selected))


@pytest.mark.parametrize("name", HANDLED)
def test_every_handled_count_is_part_of_the_sum(name: str) -> None:
    bumped = HANDLED[name] + 1
    with pytest.raises(ValidationError, match="selected must equal"):
        EmbedRunReport.model_validate(report(**{name: bumped}))
    assert EmbedRunReport.model_validate(report(**{name: bumped, "selected": 16})).selected == 16


@pytest.mark.parametrize("name", ["up_to_date", "placeholders", "non_2xx"])
def test_counts_outside_the_selection_are_not_part_of_the_sum(name: str) -> None:
    assert getattr(EmbedRunReport.model_validate(report(**{name: 1000})), name) == 1000


def test_run_report_rejects_a_naive_finish_time() -> None:
    with pytest.raises(ValidationError, match="finished_at"):
        EmbedRunReport.model_validate(report(finished_at=datetime(2026, 9, 27)))


@pytest.mark.parametrize(
    ("field", "value", "error_type"),
    [
        ("tenant_id", "", "string_too_short"),
        ("embedding_model", "", "string_too_short"),
        ("dimensions", 0, "greater_than_equal"),
        ("dimensions", -1, "greater_than_equal"),
    ],
)
def test_run_report_rejects_an_empty_name_or_a_non_positive_dimension(
    field: str, value: object, error_type: str
) -> None:
    with pytest.raises(ValidationError, match=field) as exc_info:
        EmbedRunReport.model_validate(report(**{field: value}))
    assert [error["type"] for error in exc_info.value.errors()] == [error_type]


def test_run_report_accepts_a_one_char_name_and_one_dimension() -> None:
    result = EmbedRunReport.model_validate(report(tenant_id="t", embedding_model="m", dimensions=1))
    assert (result.tenant_id, result.embedding_model, result.dimensions) == ("t", "m", 1)


def test_run_report_rejects_negative_counts() -> None:
    with pytest.raises(ValidationError, match="skipped_missing"):
        EmbedRunReport.model_validate(report(selected=7, skipped_missing=-4))


# --- link-embedding models ----------------------------------------------------------------


def test_sentence_target_needs_a_hash_and_at_least_one_edge() -> None:
    edge = EdgeRef(source_url="example.com/a", position=0)
    assert SentenceTarget(sentence_hash="ab", edges=(edge,)).edges == (edge,)
    with pytest.raises(ValidationError, match="edges"):
        SentenceTarget(sentence_hash="ab", edges=())
    with pytest.raises(ValidationError, match="sentence_hash"):
        SentenceTarget(sentence_hash="", edges=(edge,))


@pytest.mark.parametrize("model", [EdgeRef, LinkText])
def test_an_edge_needs_a_source_and_a_non_negative_position(model: type[EdgeRef]) -> None:
    extra = {"anchor_text": "a", "surrounding_text": "s"} if model is LinkText else {}
    with pytest.raises(ValidationError, match="source_url"):
        model(source_url="", position=0, **extra)
    with pytest.raises(ValidationError, match="position"):
        model(source_url="example.com/a", position=-1, **extra)


def test_anchor_key_update_requires_an_explicit_key_even_when_none() -> None:
    update = AnchorKeyUpdate(
        source_url="example.com/a", position=0, anchor_key=None, anchor_generic=False
    )
    assert update.anchor_key is None
    with pytest.raises(ValidationError, match="anchor_key"):
        AnchorKeyUpdate(source_url="example.com/a", position=0, anchor_generic=False)  # type: ignore[call-arg]


# --- LinkEmbedReport ----------------------------------------------------------------------


# Distinct values, so a sum that drops or double-counts one field cannot match.
ANCHOR_COUNTS = {"generic_anchors": 2, "anchors_cached": 3, "anchors_embedded": 4}
SENTENCE_COUNTS = {"sentences_cached": 5, "sentences_embedded": 6}


def link_report(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "tenant_id": "t",
        "embedding_model": "voyage-4-large",
        "dimensions": 2048,
        "edges": 30,
        "keys_written": 30,
        "empty_anchors": 3,
        "unique_anchors": 9,
        "anchor_dedupe_ratio": 3.0,
        **ANCHOR_COUNTS,
        "generic_edges": 7,
        "empty_sentences": 8,
        "surrounding_cleared": 1,
        "unique_sentences": 11,
        "sentence_dedupe_ratio": 2.0,
        **SENTENCE_COUNTS,
        "surrounding_edges_written": 12,
        "anchor_flushes": 1,
        "sentence_flushes": 2,
        "api_tokens": 100,
        "tokens": 90,
        "truncated": 0,
        "elapsed_s": 0.5,
        "finished_at": datetime(2026, 9, 27, tzinfo=UTC),
    }
    values.update(overrides)
    return values


def test_link_report_accepts_accounted_anchors_and_sentences() -> None:
    result = LinkEmbedReport.model_validate(link_report())
    assert result.unique_anchors == 2 + 3 + 4
    assert result.unique_sentences == 5 + 6


@pytest.mark.parametrize("name", ANCHOR_COUNTS)
def test_every_anchor_count_is_part_of_unique_anchors(name: str) -> None:
    bumped = {name: ANCHOR_COUNTS[name] + 1}
    with pytest.raises(ValidationError, match="unique_anchors must equal"):
        LinkEmbedReport.model_validate(link_report(**bumped))
    result = LinkEmbedReport.model_validate(link_report(**bumped, unique_anchors=10))
    assert result.unique_anchors == 10


@pytest.mark.parametrize("name", SENTENCE_COUNTS)
def test_every_sentence_count_is_part_of_unique_sentences(name: str) -> None:
    bumped = {name: SENTENCE_COUNTS[name] + 1}
    with pytest.raises(ValidationError, match="unique_sentences must equal"):
        LinkEmbedReport.model_validate(link_report(**bumped))
    result = LinkEmbedReport.model_validate(link_report(**bumped, unique_sentences=12))
    assert result.unique_sentences == 12


def test_surrounding_cleared_is_required() -> None:
    values = link_report()
    del values["surrounding_cleared"]
    with pytest.raises(ValidationError, match="surrounding_cleared"):
        LinkEmbedReport.model_validate(values)


def test_link_report_allows_no_ratio_when_nothing_was_counted() -> None:
    empty = link_report(
        edges=0,
        keys_written=0,
        empty_anchors=0,
        unique_anchors=0,
        anchor_dedupe_ratio=None,
        generic_anchors=0,
        generic_edges=0,
        anchors_cached=0,
        anchors_embedded=0,
        empty_sentences=0,
        surrounding_cleared=0,
        unique_sentences=0,
        sentence_dedupe_ratio=None,
        sentences_cached=0,
        sentences_embedded=0,
        surrounding_edges_written=0,
    )
    result = LinkEmbedReport.model_validate(empty)
    assert (result.anchor_dedupe_ratio, result.sentence_dedupe_ratio) == (None, None)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("tenant_id", ""),
        ("embedding_model", ""),
        ("dimensions", 0),
        ("edges", -1),
        ("surrounding_cleared", -1),
        ("anchor_dedupe_ratio", -0.5),
        ("finished_at", datetime(2026, 9, 27)),
    ],
)
def test_link_report_rejects_bad_fields(field: str, value: object) -> None:
    with pytest.raises(ValidationError, match=field):
        LinkEmbedReport.model_validate(link_report(**{field: value}))


def test_tenant_report_carries_both_stages() -> None:
    pages = EmbedRunReport.model_validate(report())
    links = LinkEmbedReport.model_validate(link_report())
    result = TenantEmbedReport(pages=pages, links=links)
    assert (result.pages, result.links) == (pages, links)
    rebuilt = TenantEmbedReport.model_validate(result.model_dump(mode="json"))
    assert rebuilt == result
