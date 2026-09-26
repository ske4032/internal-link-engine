from __future__ import annotations

import pytest
from pydantic import ValidationError

from linking_engine.models import PageEmbedding, PageText


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
