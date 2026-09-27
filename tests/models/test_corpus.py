"""body_hash on the Mongo page documents: required, lowercase sha256 hex."""

from __future__ import annotations

import pytest
from pydantic import BaseModel, ValidationError

from linking_engine.models import Heading, PageRecord, PageSummary

HASH = "0123456789abcdef" * 4

RECORD: dict[str, object] = {
    "url": "https://example.com/a",
    "status_code": 200,
    "usable": True,
    "meta_title": None,
    "meta_description": None,
    "h1": None,
    "headings": (Heading(level=1, text="H"),),
    "body_text": "text",
    "word_count": 1,
    "link_count": 0,
    "content_hash": None,
    "body_hash": HASH,
    "scraped_at": None,
    "source": "test",
    "crawl_url": "https://example.com/a",
}
SUMMARY: dict[str, object] = {
    "url": "https://example.com/a",
    "status_code": 200,
    "word_count": 1,
    "content_hash": None,
    "body_hash": HASH,
}
MODELS = [
    pytest.param(PageRecord, RECORD, id="PageRecord"),
    pytest.param(PageSummary, SUMMARY, id="PageSummary"),
]


@pytest.mark.parametrize(("model", "data"), MODELS)
def test_a_64_char_lowercase_hex_body_hash_is_kept(
    model: type[BaseModel], data: dict[str, object]
) -> None:
    assert model.model_validate(data).model_dump()["body_hash"] == HASH


@pytest.mark.parametrize(("model", "data"), MODELS)
def test_missing_body_hash_is_rejected(model: type[BaseModel], data: dict[str, object]) -> None:
    without = {key: value for key, value in data.items() if key != "body_hash"}
    with pytest.raises(ValidationError, match="body_hash") as exc_info:
        model.model_validate(without)
    assert [error["type"] for error in exc_info.value.errors()] == ["missing"]


@pytest.mark.parametrize(("model", "data"), MODELS)
@pytest.mark.parametrize(
    "value",
    [
        pytest.param(HASH.upper(), id="uppercase"),
        pytest.param(HASH[:63], id="short"),
        pytest.param(HASH + "0", id="long"),
        pytest.param(HASH[:63] + "g", id="non-hex"),
        pytest.param("", id="empty"),
    ],
)
def test_malformed_body_hash_is_rejected(
    model: type[BaseModel], data: dict[str, object], value: str
) -> None:
    with pytest.raises(ValidationError, match="body_hash") as exc_info:
        model.model_validate({**data, "body_hash": value})
    assert [error["type"] for error in exc_info.value.errors()] == ["string_pattern_mismatch"]


@pytest.mark.parametrize(("model", "data"), MODELS)
def test_null_body_hash_is_rejected(model: type[BaseModel], data: dict[str, object]) -> None:
    with pytest.raises(ValidationError, match="body_hash"):
        model.model_validate({**data, "body_hash": None})


@pytest.mark.parametrize(("model", "data"), MODELS)
def test_template_inlinks_default_to_zero_and_are_never_negative(
    model: type[BaseModel], data: dict[str, object]
) -> None:
    dumped = model.model_validate(data).model_dump()
    assert (dumped["menu_inlinks"], dumped["footer_inlinks"]) == (0, 0)
    for field in ("menu_inlinks", "footer_inlinks"):
        with pytest.raises(ValidationError, match=field):
            model.model_validate({**data, field: -1})
