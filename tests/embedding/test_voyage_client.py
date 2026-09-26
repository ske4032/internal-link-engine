from __future__ import annotations

import json
import math
import time

import pytest
from structlog.testing import capture_logs
from voyage_fakes import (
    API_TOKEN_DRIFT,
    DIMENSION,
    MODEL,
    RAW_COMPONENT,
    FakeVoyage,
    client,
    page,
    settings,
    word_tokenizer,
    words,
)
from voyageai.error import (
    APIConnectionError,
    APIError,
    AuthenticationError,
    InvalidRequestError,
    MalformedRequestError,
    RateLimitError,
    ServerError,
    ServiceUnavailableError,
    Timeout,
    VoyageError,
)

from linking_engine.embedding import voyage_client
from linking_engine.embedding.voyage_client import VoyageSettings
from linking_engine.errors import (
    EmbeddingAuthError,
    EmbeddingError,
    EmbeddingRequestError,
    EmbeddingResponseError,
    EmbeddingUnavailableError,
)
from linking_engine.models import PageEmbedding, PageText


def norm(vector: tuple[float, ...]) -> float:
    return math.sqrt(sum(x * x for x in vector))


def events(logs: list[dict[str, object]], name: str) -> list[dict[str, object]]:
    return [entry for entry in logs if entry.get("event") == name]


REQUEST_ID = "req_test_0001"
HANG_S = 10.0


def sdk_error(
    cls: type[VoyageError], status: int, detail: str, *, json_detail: bool
) -> VoyageError:
    """Built as the SDK requestor builds it, request-id header included."""
    headers = {"request-id": REQUEST_ID}
    if not json_detail:
        # 5xx and non-JSON bodies: fixed message, body never parsed.
        return cls(detail, "<html>error page</html>", status, headers=headers)
    json_body = {"detail": detail}
    body = json.dumps(json_body)
    # Unmapped 4xx: the SDK folds body, status and headers into APIError's message.
    message = f"{detail} {body} {status} {json_body} {headers}" if cls is APIError else detail
    return cls(message, body, status, json_body, headers)


# (SDK class, status, provider detail, detail comes from a JSON body)
REJECTED = [
    (InvalidRequestError, 400, "input too long", True),
    (AuthenticationError, 401, "Provided API key is invalid.", True),
    (APIError, 403, "Forbidden for this organisation.", True),
    (APIError, 404, "Model not found.", True),
    (MalformedRequestError, 422, "texts must be a list", True),
]
RETRYABLE = [
    (RateLimitError, 429, "Rate limit exceeded.", True),
    (ServerError, 500, "The server failed to process the request.", False),
    (ServiceUnavailableError, 502, "The server is overloaded or not ready yet.", False),
    (ServiceUnavailableError, 503, "The server is overloaded or not ready yet.", False),
    (ServiceUnavailableError, 504, "The server is overloaded or not ready yet.", False),
    # 5xx the SDK has no class for, with a non-JSON body: retried on status alone.
    (APIError, 501, "HTTP code 501 from API (<html>error page</html>)", False),
]


def case_id(case: tuple[type[VoyageError], int, str, bool]) -> str:
    return f"{case[1]}-{case[0].__name__}"


def transient() -> list[VoyageError]:
    return [Timeout("Request timed out"), APIConnectionError("Error communicating with Voyage")]


def record_waits(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    waits: list[float] = []

    async def no_sleep(seconds: float) -> None:
        waits.append(seconds)

    monkeypatch.setattr(voyage_client, "backoff_sleep", no_sleep)
    return waits


# --- tokens ---------------------------------------------------------------------------


def test_count_tokens_encodes_the_whole_list_in_one_call(monkeypatch: pytest.MonkeyPatch) -> None:
    tokenizer = word_tokenizer()
    seen: list[list[str]] = []
    real = tokenizer.encode_batch

    def spy(texts, *args, **kwargs):
        seen.append(list(texts))
        return real(texts, *args, **kwargs)

    monkeypatch.setattr(tokenizer, "encode_batch", spy)
    counts = client(FakeVoyage(), tokenizer=tokenizer).count_tokens(["a b c", "d", ""])
    assert counts == [3, 1, 0]
    assert seen == [["a b c", "d", ""]], f"expected one encode_batch call, got {seen}"


async def test_embed_tokenizes_once_per_call(monkeypatch: pytest.MonkeyPatch) -> None:
    tokenizer = word_tokenizer()
    calls: list[int] = []
    real = tokenizer.encode_batch

    def spy(texts, *args, **kwargs):
        calls.append(len(texts))
        return real(texts, *args, **kwargs)

    monkeypatch.setattr(tokenizer, "encode_batch", spy)
    pages = [page(0, 80), page(1, 10), page(2, 30), page(3, 5), page(4, 20)]
    await client(FakeVoyage(), tokenizer=tokenizer).embed(pages)
    assert calls == [5], f"expected one encode_batch over all 5 pages, got {calls}"


# --- batching ---------------------------------------------------------------------------


async def test_no_call_exceeds_token_budget_or_item_cap() -> None:
    fake = FakeVoyage()
    lengths = [37, 5, 50, 12, 49, 1, 33, 20, 20, 20, 20, 44, 7, 50, 50]
    await client(fake).embed([page(i, n) for i, n in enumerate(lengths)])
    tokenizer = word_tokenizer()
    assert fake.calls, "the SDK was never called"
    for call in fake.calls:
        encodings = tokenizer.encode_batch(list(call.texts))
        sent = sum(len(e.ids) for e in encodings)
        assert sent <= 120, f"call sent {sent} tokens, budget 120"
        assert len(call.texts) <= 4, f"call sent {len(call.texts)} items, cap 4"
    assert sum(len(c.texts) for c in fake.calls) == len(lengths)


async def test_5000_and_200_token_pages_embed_in_one_call() -> None:
    fake = FakeVoyage(dimension=2048)
    config = settings(
        context_tokens=32_000,
        max_request_tokens=120_000,
        request_token_budget=110_000,
        max_batch_items=1_000,
    )
    result = await client(fake, config).embed([page(0, 5_000), page(1, 200)])
    assert len(fake.calls) == 1, f"expected one call, got {len(fake.calls)}"
    assert [words(t) for t in fake.calls[0].texts] == [5_000, 200]
    assert [(r.tokens, r.original_tokens, r.truncated) for r in result] == [
        (5_000, 5_000, False),
        (200, 200, False),
    ]


async def test_realistic_corpus_never_exceeds_110k_per_call() -> None:
    fake = FakeVoyage(dimension=2048)
    config = settings(
        context_tokens=32_000,
        max_request_tokens=120_000,
        request_token_budget=110_000,
        max_batch_items=1_000,
    )
    result = await client(fake, config).embed([page(i, 3_300) for i in range(40)])
    assert len(result) == 40
    assert all(sum(words(t) for t in c.texts) <= 110_000 for c in fake.calls)
    assert len(fake.calls) == 2, f"40 pages of 3,300 tokens need 2 calls, got {len(fake.calls)}"


async def test_order_is_preserved_across_batches() -> None:
    fake = FakeVoyage()
    pages = [page(i, 30) for i in range(10)]
    with capture_logs() as logs:
        result = await client(fake).embed(pages)
    assert len(fake.calls) == 3, "10 pages at cap 4 should take 3 calls"
    assert [r.url for r in result] == [p.url for p in pages]
    tags = [max(range(DIMENSION), key=r.vector.__getitem__) for r in result]
    assert tags == list(range(10)), f"vectors attached to the wrong pages: {tags}"
    batches = events(logs, "embedding.batch")
    assert [(e["items"], e["tokens"], e["retries"]) for e in batches] == [
        (4, 120, 0),
        (4, 120, 0),
        (2, 60, 0),
    ]
    for e in batches:
        assert e["api_tokens"] == e["tokens"] + API_TOKEN_DRIFT, (
            "api_tokens must be the SDK's count"
        )
        assert isinstance(e["latency_ms"], int | float) and e["latency_ms"] >= 0
        assert e["truncated"] == 0


async def test_empty_input_makes_no_call() -> None:
    fake = FakeVoyage()
    with capture_logs() as logs:
        assert await client(fake).embed([]) == []
    assert fake.calls == []
    assert events(logs, "embedding.batch") == []


# --- truncation --------------------------------------------------------------------------


async def test_over_context_page_is_truncated_and_logged() -> None:
    fake = FakeVoyage()
    pages = [page(0, 80), page(1, 70), page(2, 20)]
    with capture_logs() as logs:
        result = await client(fake).embed(pages)

    # 50 + 50 + 20 = 120 fits the budget only because counts are taken after truncation.
    assert len(fake.calls) == 1, f"truncated pages should share one call, got {len(fake.calls)}"
    sent = fake.calls[0].texts
    assert [words(t) for t in sent] == [50, 50, 20]
    for original, text in zip(pages, sent, strict=True):
        assert original.text.startswith(text), "truncation must keep the start of the page"
    assert [(r.tokens, r.original_tokens, r.truncated) for r in result] == [
        (50, 80, True),
        (50, 70, True),
        (20, 20, False),
    ]

    truncated = events(logs, "embedding.truncated")
    assert [(e["url"], e["original_tokens"], e["limit"]) for e in truncated] == [
        (pages[0].url, 80, 50),
        (pages[1].url, 70, 50),
    ]
    (batch,) = events(logs, "embedding.batch")
    assert batch["tokens"] == 120
    assert batch["truncated"] == 2


async def test_page_at_exact_context_limit_is_not_truncated() -> None:
    fake = FakeVoyage()
    with capture_logs() as logs:
        (result,) = await client(fake).embed([page(0, 50)])
    assert (result.tokens, result.original_tokens, result.truncated) == (50, 50, False)
    assert fake.calls[0].texts == (page(0, 50).text,)
    assert events(logs, "embedding.truncated") == []


async def test_truncation_uses_token_offsets_not_characters() -> None:
    # Irregular whitespace and long words: a character cut would not land on 50 tokens.
    text = "  ".join(f"word{i:04d}\n" for i in range(90))
    fake = FakeVoyage(respond=lambda texts: [[RAW_COMPONENT] * DIMENSION for _ in texts])
    (result,) = await client(fake).embed([PageText(url="https://example.com/long", text=text)])
    (sent,) = fake.calls[0].texts
    assert words(sent) == 50
    assert sent.split() == text.split()[:50]
    assert (result.tokens, result.original_tokens, result.truncated) == (50, 90, True)


# --- request shape -----------------------------------------------------------------------


async def test_every_call_is_a_document_request_for_the_configured_model() -> None:
    fake = FakeVoyage()
    await client(fake).embed([page(i, 40) for i in range(7)])
    assert len(fake.calls) > 1
    for call in fake.calls:
        assert call.input_type == "document"
        assert call.model == MODEL
        assert call.output_dimension == DIMENSION
        assert call.truncation is True


# --- vectors -----------------------------------------------------------------------------


async def test_vectors_are_unit_length_at_2048_dimensions() -> None:
    fake = FakeVoyage(dimension=2048, respond=lambda texts: [[RAW_COMPONENT] * 2048 for _ in texts])
    result = await client(fake).embed([page(0, 10), page(1, 20)])
    assert all(isinstance(r, PageEmbedding) for r in result)
    for r in result:
        assert len(r.vector) == 2048
        assert abs(norm(r.vector) - 1.0) <= 1e-6, f"norm {norm(r.vector)}"
        # Every raw component was 3.0, so each normalised one is 1/sqrt(2048).
        assert r.vector == pytest.approx((1 / math.sqrt(2048),) * 2048)


@pytest.mark.parametrize(
    "respond",
    [
        pytest.param(lambda texts: [[1.0] * (DIMENSION - 1) for _ in texts], id="short-dimension"),
        pytest.param(lambda texts: [[1.0] * (DIMENSION + 1) for _ in texts], id="long-dimension"),
        pytest.param(lambda texts: [[1.0] * DIMENSION for _ in texts[1:]], id="one-missing"),
        pytest.param(lambda texts: [[1.0] * DIMENSION for _ in [*texts, "x"]], id="one-extra"),
        pytest.param(
            lambda texts: [[0.0] * DIMENSION] + [[1.0] * DIMENSION for _ in texts[1:]],
            id="zero-vector",
        ),
        pytest.param(
            lambda texts: [[math.nan] * DIMENSION for _ in texts],
            id="nan-vector",
        ),
    ],
)
async def test_malformed_response_raises(respond) -> None:
    fake = FakeVoyage(respond=respond)
    with pytest.raises(EmbeddingResponseError):
        await client(fake).embed([page(0, 10), page(1, 10)])


# --- retries -----------------------------------------------------------------------------


async def test_429_twice_then_success() -> None:
    fake = FakeVoyage(
        failures=[
            RateLimitError("rate limited", http_status=429),
            RateLimitError("rate limited", http_status=429),
        ]
    )
    with capture_logs() as logs:
        result = await client(fake).embed([page(0, 10), page(1, 10)])
    assert len(fake.calls) == 3, f"expected 2 retries then success, got {len(fake.calls)} calls"
    assert [r.url for r in result] == [page(0, 10).url, page(1, 10).url]
    (batch,) = events(logs, "embedding.batch")
    assert batch["retries"] == 2


@pytest.mark.parametrize(
    "error",
    [sdk_error(*case[:3], json_detail=case[3]) for case in RETRYABLE] + transient(),
    ids=lambda e: f"{e.http_status}-{type(e).__name__}",
)
async def test_one_retryable_failure_then_success(error: VoyageError) -> None:
    fake = FakeVoyage(failures=[error])
    with capture_logs() as logs:
        result = await client(fake).embed([page(0, 10)])
    assert [r.url for r in result] == [page(0, 10).url]
    assert len(fake.calls) == 2, f"{type(error).__name__} {error.http_status} retried once"
    (batch,) = events(logs, "embedding.batch")
    assert batch["retries"] == 1
    (retry,) = events(logs, "embedding.retry")
    assert (retry["attempt"], retry["error"], retry["http_status"]) == (
        1,
        type(error).__name__,
        error.http_status,
    )


@pytest.mark.parametrize("case", REJECTED, ids=case_id)
async def test_rejected_status_fails_once_with_status_and_type(
    case: tuple[type[VoyageError], int, str, bool],
) -> None:
    cls, status, detail, json_detail = case
    error = sdk_error(cls, status, detail, json_detail=json_detail)
    fake = FakeVoyage(failures=[error])
    with pytest.raises(EmbeddingRequestError) as exc_info:
        await client(fake).embed([page(0, 10)])
    exc = exc_info.value
    message = str(exc)
    assert len(fake.calls) == 1, f"HTTP {status} must not be retried, got {len(fake.calls)} calls"
    expected = EmbeddingAuthError if status in (401, 403) else EmbeddingRequestError
    assert type(exc) is expected, f"HTTP {status} should raise {expected.__name__}, got {exc!r}"
    assert (exc.status_code, exc.error_type) == (status, cls.__name__)
    assert f"HTTP {status}" in message and cls.__name__ in message, message
    assert message.endswith(f": HTTP {status} {cls.__name__}: {detail}"), message
    assert "retries exhausted" not in message, message
    assert REQUEST_ID not in message, f"request-id prefix leaked: {message}"
    assert exc.__cause__ is error


@pytest.mark.parametrize("case", RETRYABLE, ids=case_id)
async def test_retryable_status_exhausts_with_status_and_type(
    case: tuple[type[VoyageError], int, str, bool], monkeypatch: pytest.MonkeyPatch
) -> None:
    cls, status, detail, json_detail = case
    error = sdk_error(cls, status, detail, json_detail=json_detail)
    waits = record_waits(monkeypatch)
    fake = FakeVoyage(failures=[error] * 10)
    with pytest.raises(EmbeddingUnavailableError) as exc_info:
        await client(fake, settings(max_attempts=4)).embed([page(0, 10)])
    exc = exc_info.value
    message = str(exc)
    assert len(fake.calls) == 4, f"expected max_attempts=4 calls, got {len(fake.calls)}"
    assert (exc.status_code, exc.error_type) == (status, cls.__name__)
    assert "retries exhausted after 4 attempts" in message, message
    assert f"HTTP {status}" in message, message
    assert message.endswith(
        f"retries exhausted after 4 attempts: HTTP {status} {cls.__name__}: {detail}"
    ), message
    assert REQUEST_ID not in message, f"request-id prefix leaked: {message}"
    assert exc.__cause__ is error
    assert waits == [0.0, 0.0, 0.0], f"zero backoff settings requested waits {waits}"


async def test_unclassified_sdk_error_is_not_retried() -> None:
    # No status and not a transport class: nothing says a retry would help.
    error = APIError("Invalid response object from API")
    fake = FakeVoyage(failures=[error])
    with pytest.raises(EmbeddingUnavailableError) as exc_info:
        await client(fake).embed([page(0, 10)])
    exc = exc_info.value
    assert len(fake.calls) == 1
    assert (exc.status_code, exc.error_type) == (None, "APIError")
    assert str(exc).endswith(": APIError: Invalid response object from API"), str(exc)
    assert "HTTP" not in str(exc) and "retries exhausted" not in str(exc), str(exc)
    assert exc.__cause__ is error


async def test_default_max_attempts_is_used() -> None:
    fake = FakeVoyage(failures=[ServerError("down", http_status=500) for _ in range(10)])
    config = VoyageSettings(
        api_key="test",
        context_tokens=50,
        max_request_tokens=130,
        request_token_budget=120,
        max_batch_items=4,
        backoff_initial_s=0.0,
        backoff_max_s=0.0,
    )
    with pytest.raises(EmbeddingUnavailableError) as exc_info:
        await client(fake, config).embed([page(0, 10)])
    assert len(fake.calls) == config.max_attempts == 8
    assert "retries exhausted after 8 attempts" in str(exc_info.value), str(exc_info.value)


async def test_backoff_waits_before_every_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    # Real settings, no real sleeping: the waits requested are what is asserted.
    waits = record_waits(monkeypatch)
    fake = FakeVoyage(failures=[RateLimitError("rate limited", http_status=429) for _ in range(5)])
    config = settings(max_attempts=6, backoff_initial_s=100.0, backoff_max_s=100.0)
    await client(fake, config).embed([page(0, 10)])
    assert len(fake.calls) == 6
    assert len(waits) == 5, f"expected a wait before each of 5 retries, got {waits}"
    assert all(0 <= w <= 100.0 for w in waits), f"a wait exceeded backoff_max_s: {waits}"
    assert max(waits) > 1.0, f"backoff settings ignored, waits {waits}"


async def test_backoff_wait_is_capped_by_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    waits = record_waits(monkeypatch)
    fake = FakeVoyage(failures=[RateLimitError("rate limited", http_status=429) for _ in range(5)])
    config = settings(max_attempts=6, backoff_initial_s=1.0, backoff_max_s=0.75)
    await client(fake, config).embed([page(0, 10)])
    assert len(waits) == 5
    assert all(0 <= w <= 0.75 for w in waits), f"a wait exceeded backoff_max_s=0.75: {waits}"


async def test_zero_backoff_settings_do_not_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    waits = record_waits(monkeypatch)
    fake = FakeVoyage(failures=[RateLimitError("rate limited", http_status=429) for _ in range(3)])
    await client(fake, settings(max_attempts=4)).embed([page(0, 10)])
    assert len(fake.calls) == 4
    assert waits == [0.0, 0.0, 0.0], f"zero backoff settings requested waits {waits}"


@pytest.mark.parametrize("error", transient(), ids=lambda e: type(e).__name__)
async def test_transport_failure_exhausts_without_status(error: VoyageError) -> None:
    fake = FakeVoyage(failures=[error] * 10)
    with pytest.raises(EmbeddingUnavailableError) as exc_info:
        await client(fake, settings(max_attempts=3)).embed([page(0, 10)])
    exc = exc_info.value
    message = str(exc)
    assert len(fake.calls) == 3, f"{type(error).__name__} should be retried to max_attempts=3"
    assert (exc.status_code, exc.error_type) == (None, type(error).__name__)
    assert message.endswith(
        f"retries exhausted after 3 attempts: {type(error).__name__}: {error.user_message}"
    ), message
    assert "HTTP" not in message, message
    assert exc.__cause__ is error


async def test_hung_call_is_retried_then_unavailable() -> None:
    # A hang far above the deadline: finishing at all within HANG_S proves the deadline cut it.
    fake = FakeVoyage(hang_s=HANG_S)
    started = time.perf_counter()
    with pytest.raises(EmbeddingUnavailableError) as exc_info:
        await client(fake, settings(max_attempts=3, timeout_s=0.05)).embed([page(0, 10)])
    elapsed = time.perf_counter() - started
    exc = exc_info.value
    message = str(exc)
    assert isinstance(exc.__cause__, TimeoutError)
    assert len(fake.calls) == 3, (
        f"a hung call should be retried to max_attempts=3, got {len(fake.calls)}"
    )
    assert (exc.status_code, exc.error_type) == (None, "TimeoutError")
    assert "retries exhausted after 3 attempts" in message, message
    assert message.endswith("TimeoutError: no response within 0.05s"), message
    assert "HTTP" not in message, message
    assert fake.cancelled == 3, f"expected every hung call cancelled, got {fake.cancelled}"
    assert elapsed < HANG_S, f"timeout_s=0.05 was not enforced: 3 attempts took {elapsed:.2f}s"


async def test_one_hung_call_then_success() -> None:
    fake = FakeVoyage(hang_s=HANG_S, hang_calls=1)
    with capture_logs() as logs:
        result = await client(fake, settings(timeout_s=0.05)).embed([page(0, 10)])
    assert len(result) == 1
    assert len(fake.calls) == 2
    assert fake.cancelled == 1
    (batch,) = events(logs, "embedding.batch")
    assert batch["retries"] == 1


def test_embedding_errors_share_a_base() -> None:
    for cls in (
        EmbeddingAuthError,
        EmbeddingRequestError,
        EmbeddingResponseError,
        EmbeddingUnavailableError,
    ):
        assert issubclass(cls, EmbeddingError)
    assert not issubclass(EmbeddingRequestError, EmbeddingUnavailableError)
    # Auth is a rejected request: callers catching EmbeddingRequestError still see it.
    assert issubclass(EmbeddingAuthError, EmbeddingRequestError)
    assert not issubclass(EmbeddingAuthError, EmbeddingUnavailableError)


def test_embedding_error_defaults_type_to_its_class() -> None:
    exc = EmbeddingUnavailableError("down")
    assert (exc.status_code, exc.error_type, str(exc)) == (
        None,
        "EmbeddingUnavailableError",
        "down",
    )


def test_default_sdk_deadline_sits_above_the_client_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    built: list[dict[str, object]] = []

    class RecordingClient:
        def __init__(self, **kwargs: object) -> None:
            built.append(kwargs)

    monkeypatch.setattr(voyage_client, "AsyncClient", RecordingClient)
    voyage_client.VoyageClient(
        settings(api_key="pa-key", timeout_s=12.5),
        model=MODEL,
        dimension=DIMENSION,
        tokenizer=word_tokenizer(),
    )
    assert built == [{"api_key": "pa-key", "max_retries": 0, "timeout": 17.5}]


@pytest.mark.parametrize(("model", "dimension"), [(" ", DIMENSION), (MODEL, 0)])
def test_client_rejects_blank_model_or_non_positive_dimension(model: str, dimension: int) -> None:
    with pytest.raises(ValueError):
        voyage_client.VoyageClient(
            settings(),
            model=model,
            dimension=dimension,
            sdk=FakeVoyage(),
            tokenizer=word_tokenizer(),
        )
