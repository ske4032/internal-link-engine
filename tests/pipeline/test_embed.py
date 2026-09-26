"""Embedding pipeline acceptance: skips, committed flushes, resume, model guard, progress, memory."""

from __future__ import annotations

import gc
import math
import re
import tracemalloc
from contextlib import aclosing
from datetime import timedelta
from typing import TYPE_CHECKING, NoReturn

import numpy as np
import pytest
from pymongo import AsyncMongoClient
from structlog.testing import capture_logs
from voyage_fakes import (
    API_TOKEN_DRIFT,
    MODEL,
    FakeVoyage,
    client,
    page_index,
    tagged_vector,
    words,
)
from voyageai.error import InvalidRequestError, RateLimitError

from linking_engine.errors import (
    DatabaseReadError,
    DatabaseWriteError,
    EmbeddingError,
    EmbeddingModelMismatchError,
    EmbeddingRequestError,
    EmbeddingResponseError,
    EmbeddingUnavailableError,
    SchemaError,
)
from linking_engine.ingest.graph_load import load_tenant_graph
from linking_engine.ingest.markdown_clean import body_hash
from linking_engine.models import (
    EmbeddingBatch,
    EmbeddingModelCount,
    EmbedRunReport,
    LinkRecord,
    Page,
    PageRecord,
)
from linking_engine.pipeline.embed import FLUSH_SIZE, check_models, embed_tenant

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Awaitable, Callable, Iterable, Mapping, Sequence

    from linking_engine.embedding.voyage_client import VoyageClient
    from linking_engine.graph.repo import GraphRepo
    from linking_engine.ingest.mongo_repo import MongoRepo
    from linking_engine.models import PageEmbedding, PageText

BASE = "example.com"
GHOST = f"{BASE}/ghost"
DIM = 2048
OTHER_MODEL = "voyage-3-large"
WORDS = 30


def url(i: int) -> str:
    return f"{BASE}/p{i:03d}"


def body(i: int, length: int = WORDS, *, tag: int | None = None) -> str:
    """`length` words; the first names the fake's vector spike, so argmax identifies the text."""
    return " ".join([f"p{i if tag is None else tag}"] + ["w"] * (length - 1))


def record(
    i: int,
    text: str | None = None,
    *,
    status: int = 200,
    usable: bool | None = True,
    links: int = 0,
) -> PageRecord:
    text = body(i) if text is None else text
    return PageRecord(
        url=url(i),
        crawl_url=url(i),
        status_code=status,
        usable=usable,
        meta_title=None,
        meta_description=None,
        h1=None,
        headings=(),
        body_text=text,
        word_count=len(text.split()),
        link_count=links,
        content_hash=None,
        body_hash=body_hash(text),
        scraped_at=None,
        source="test",
    )


def sent_texts(fake: FakeVoyage) -> list[str]:
    return [text for call in fake.calls for text in call.texts]


def sdk_tokens(fake: FakeVoyage) -> int:
    """What the fake reported as total_tokens over every call."""
    return sum(words(text) for text in sent_texts(fake)) + API_TOKEN_DRIFT * len(fake.calls)


def events(logs: Sequence[Mapping[str, object]], name: str) -> list[Mapping[str, object]]:
    return [entry for entry in logs if entry.get("event") == name]


REASONS = ("not_usable", "empty_body", "missing", "hash_mismatch")
TALLY = ("embedded", *REASONS, "api_tokens", "tokens")


def tally_of(entry: Mapping[str, object]) -> dict[str, object]:
    return {key: entry[key] for key in TALLY}


def tally(
    embedded: int = 0, *, api_tokens: int = 0, tokens: int = 0, **reasons: int
) -> dict[str, object]:
    skips = {reason: reasons.get(reason, 0) for reason in REASONS}
    return {"embedded": embedded, **skips, "api_tokens": api_tokens, "tokens": tokens}


def assert_failed_at_startup(logs: Sequence[Mapping[str, object]], error: BaseException) -> None:
    (failed,) = events(logs, "embedding.run.failed")
    assert (failed["flush"], failed["pages_done"]) == (0, 0)
    assert tally_of(failed) == tally()
    assert (failed["error_type"], failed["error"]) == (type(error).__name__, str(error))
    assert failed["log_level"] == "error"
    assert events(logs, "embedding.run.start") == []
    assert events(logs, "embedding.flush") == []


async def seed(
    mongo: MongoRepo,
    graph: GraphRepo,
    tenant: str,
    pages: Sequence[PageRecord],
    links: Sequence[LinkRecord] = (),
) -> None:
    await mongo.write_pages(tenant, pages, links)
    await load_tenant_graph(mongo, graph, tenant)


async def seed_plain(mongo: MongoRepo, graph: GraphRepo, tenant: str, count: int) -> None:
    await seed(mongo, graph, tenant, [record(i) for i in range(count)])


async def stored(graph: GraphRepo, tenant: str, urls: Iterable[str]) -> dict[str, Page]:
    pages = await graph.get_pages(tenant, list(urls), include_vectors=True)
    return {str(page.url): page for page in pages}


def assert_embedded(page: Page, text: str) -> None:
    """Vector computed from `text` (the full body), with marker, model, dimensions and time."""
    assert page.content_embedding is not None, f"{page.url} has no vector"
    vector = np.asarray(page.content_embedding)
    assert vector.shape == (DIM,)
    assert int(np.argmax(vector)) == page_index(text) % DIM, f"{page.url} holds another vector"
    assert abs(float(np.linalg.norm(vector)) - 1.0) < 1e-5
    assert page.embedded_body_hash == page.body_hash == body_hash(text)
    assert (page.embedding_model, page.embedding_dimensions) == (MODEL, DIM)
    assert page.embedded_at is not None
    assert page.embedded_at.utcoffset() == timedelta(0)


def assert_untouched(page: Page) -> None:
    written = (
        page.content_embedding,
        page.embedded_body_hash,
        page.embedding_model,
        page.embedding_dimensions,
        page.embedded_at,
    )
    assert written == (None, None, None, None, None), f"{page.url} was written: {written[1:]}"


# A mixed tenant. p000..p006 are embedded: p001 has usable=None, p002 is cut to the
# 50-token context. p007 is unusable, p008 empty, p009 whitespace; p010..p012 answer
# 3xx/4xx/5xx and p000 links to one uncrawled page. At flush size 3 the flushes are
# [0-2] [3-5] [6-8] [9], and the last one is all skipped.
ELIGIBLE = tuple(range(7))
LONG = 2
NOT_USABLE, EMPTY, BLANK = 7, 8, 9
NON_2XX = {10: 301, 11: 404, 12: 503}
MIXED_FLUSH = 3
MIXED_URLS = [url(i) for i in range(13)] + [GHOST]


def mixed_body(i: int) -> str:
    return body(i, 60) if i == LONG else body(i)


async def seed_mixed(mongo: MongoRepo, graph: GraphRepo, tenant: str) -> None:
    pages = [
        record(0, links=1),
        record(1, usable=None),
        record(LONG, mixed_body(LONG)),
        *(record(i) for i in range(3, 7)),
        record(NOT_USABLE, usable=False),
        record(EMPTY, ""),
        record(BLANK, " \n\t "),
        *(record(i, status=code) for i, code in NON_2XX.items()),
    ]
    ghost_link = LinkRecord(
        source_url=str(url(0)),
        position=0,
        target_url=str(GHOST),
        anchor_text="ghost",
        surrounding_text="a link to a page never crawled",
        is_internal=True,
    )
    await seed(mongo, graph, tenant, pages, [ghost_link])


# ── 1-3: full run, rerun, one edited body ───────────────────────────────────


@pytest.mark.integration
async def test_full_run_embeds_exactly_the_eligible_pages(
    mongo: MongoRepo, graph: GraphRepo, tenant: str
) -> None:
    await seed_mixed(mongo, graph, tenant)
    fake = FakeVoyage(dimension=DIM)
    report = await embed_tenant(mongo, graph, client(fake), tenant, flush_size=MIXED_FLUSH)

    sent = sent_texts(fake)
    assert sorted(page_index(text) for text in sent) == list(ELIGIBLE)
    for text in sent:
        index = page_index(text)
        if index == LONG:
            assert mixed_body(LONG).startswith(text)
            assert words(text) == 50
        else:
            assert text == mixed_body(index), "the text sent must be the stored body"

    pages = await stored(graph, tenant, MIXED_URLS)
    assert len(pages) == len(MIXED_URLS)
    for i in ELIGIBLE:
        assert_embedded(pages[url(i)], mixed_body(i))
    for other in [url(i) for i in (NOT_USABLE, EMPTY, BLANK, *NON_2XX)] + [GHOST]:
        assert_untouched(pages[other])
    assert pages[GHOST].is_placeholder

    assert report.model_dump(exclude={"elapsed_s", "finished_at"}) == {
        "tenant_id": tenant,
        "embedding_model": MODEL,
        "dimensions": DIM,
        "selected": 10,
        "embedded": 7,
        "skipped_not_usable": 1,
        "skipped_empty_body": 2,
        "skipped_missing": 0,
        "skipped_hash_mismatch": 0,
        "up_to_date": 0,
        "placeholders": 1,
        "non_2xx": 3,
        "flushes": 4,
        "api_tokens": sdk_tokens(fake),
        "tokens": sum(words(text) for text in sent),
        "truncated": 1,
    }
    assert report.api_tokens == report.tokens + API_TOKEN_DRIFT * len(fake.calls)
    assert report.elapsed_s >= 0
    assert report.finished_at.utcoffset() == timedelta(0)


@pytest.mark.integration
async def test_immediate_rerun_embeds_nothing_and_makes_no_voyage_call(
    mongo: MongoRepo, graph: GraphRepo, tenant: str
) -> None:
    await seed_mixed(mongo, graph, tenant)
    await embed_tenant(
        mongo, graph, client(FakeVoyage(dimension=DIM)), tenant, flush_size=MIXED_FLUSH
    )
    before = await stored(graph, tenant, MIXED_URLS)

    fake = FakeVoyage(dimension=DIM)
    report = await embed_tenant(mongo, graph, client(fake), tenant, flush_size=MIXED_FLUSH)

    assert fake.call_count == 0
    assert (report.embedded, report.up_to_date, report.api_tokens, report.tokens) == (0, 7, 0, 0)
    # Skipped pages stay selected: each run reads them from Mongo and skips them again.
    assert (report.selected, report.skipped_not_usable, report.skipped_empty_body) == (3, 1, 2)
    assert await stored(graph, tenant, MIXED_URLS) == before


@pytest.mark.integration
async def test_editing_one_body_re_embeds_exactly_that_page(
    mongo: MongoRepo, graph: GraphRepo, tenant: str
) -> None:
    urls = [url(i) for i in range(6)]
    await seed_plain(mongo, graph, tenant, 6)
    await embed_tenant(mongo, graph, client(FakeVoyage(dimension=DIM)), tenant, flush_size=4)
    before = await stored(graph, tenant, urls)

    # A new first word moves the fake's spike, so the new vector differs from the old one.
    edited = body(4, tag=1004)
    await mongo.write_pages(tenant, [record(4, edited)], [])
    await load_tenant_graph(mongo, graph, tenant)
    fake = FakeVoyage(dimension=DIM)
    report = await embed_tenant(mongo, graph, client(fake), tenant, flush_size=4)

    assert [call.texts for call in fake.calls] == [(edited,)]
    assert (report.selected, report.embedded, report.up_to_date) == (1, 1, 5)
    after = await stored(graph, tenant, urls)
    unchanged = [u for u in urls if u != url(4)]
    assert [after[u] for u in unchanged] == [before[u] for u in unchanged]
    assert_embedded(after[url(4)], edited)
    old, new = before[url(4)], after[url(4)]
    assert new.content_embedding != old.content_embedding
    assert old.embedded_at is not None
    assert new.embedded_at is not None
    assert new.embedded_at > old.embedded_at


# ── 4: mixed-model guard ────────────────────────────────────────────────────


def unit_vectors(indices: Sequence[int]) -> np.ndarray:
    matrix = np.asarray([tagged_vector(body(i), DIM) for i in indices], dtype=np.float64)
    return (matrix / np.linalg.norm(matrix, axis=1, keepdims=True)).astype(np.float32)


async def store_vectors(graph: GraphRepo, tenant: str, indices: Sequence[int], model: str) -> None:
    await graph.write_embeddings(
        tenant,
        [url(i) for i in indices],
        [body_hash(body(i)) for i in indices],
        unit_vectors(indices),
        model=model,
        dimensions=DIM,
    )


async def store_vector_without_model(graph: GraphRepo, tenant: str, index: int) -> None:
    await graph._auto(
        "MATCH (p:Page {tenantId: $t, url: $u}) SET p.content_embedding = $vec",
        t=tenant,
        u=url(index),
        vec=unit_vectors([index])[0].tolist(),
    )


async def two_models(graph: GraphRepo, tenant: str) -> None:
    await store_vectors(graph, tenant, [0], MODEL)
    await store_vectors(graph, tenant, [1], OTHER_MODEL)


async def a_null_model(graph: GraphRepo, tenant: str) -> None:
    await store_vectors(graph, tenant, [0], MODEL)
    await store_vector_without_model(graph, tenant, 1)


async def another_model(graph: GraphRepo, tenant: str) -> None:
    await store_vectors(graph, tenant, [0, 1], OTHER_MODEL)


@pytest.mark.integration
@pytest.mark.parametrize(
    ("setup", "problem"),
    [
        pytest.param(
            two_models,
            "stored vectors mix 2 embedding models; found voyage-3-large (1), voyage-4-large (1)",
            id="two-models",
        ),
        pytest.param(
            a_null_model,
            "some stored vectors have no embeddingModel; found voyage-4-large (1), <no model> (1)",
            id="null-model",
        ),
        pytest.param(
            another_model,
            "stored vectors use a different model than configured; found voyage-3-large (2)",
            id="different-model",
        ),
    ],
)
async def test_mixed_models_fail_the_run_before_any_voyage_call(
    mongo: MongoRepo,
    graph: GraphRepo,
    tenant: str,
    setup: Callable[[GraphRepo, str], Awaitable[None]],
    problem: str,
) -> None:
    await seed_plain(mongo, graph, tenant, 3)
    await setup(graph, tenant)
    before = await stored(graph, tenant, [url(i) for i in range(3)])
    fake = FakeVoyage(dimension=DIM)
    expected = (
        f"tenant {tenant}: {problem}; configured {MODEL}. "
        "Vectors from different models are not comparable"
    )

    with (
        capture_logs() as logs,
        pytest.raises(EmbeddingModelMismatchError, match=f"^{re.escape(expected)}$") as exc_info,
    ):
        await embed_tenant(mongo, graph, client(fake), tenant)

    assert fake.call_count == 0
    assert await stored(graph, tenant, [url(i) for i in range(3)]) == before
    assert_failed_at_startup(logs, exc_info.value)


@pytest.mark.integration
async def test_another_tenants_models_do_not_block_this_tenant(
    mongo: MongoRepo, graph: GraphRepo, tenant: str
) -> None:
    other = f"{tenant}-other"
    await seed_plain(mongo, graph, other, 2)
    await store_vectors(graph, other, [0], OTHER_MODEL)
    await store_vector_without_model(graph, other, 1)
    await seed_plain(mongo, graph, tenant, 3)

    report = await embed_tenant(mongo, graph, client(FakeVoyage(dimension=DIM)), tenant)
    assert (report.selected, report.embedded) == (3, 3)
    with pytest.raises(EmbeddingModelMismatchError, match=f"^tenant {re.escape(other)}: some"):
        await embed_tenant(mongo, graph, client(FakeVoyage(dimension=DIM)), other)


async def corrupt_model_value(graph: GraphRepo, tenant: str) -> None:
    await graph._auto(
        "MATCH (p:Page {tenantId: $t, url: $u}) "
        "SET p.content_embedding = [1.0, 0.0], p.embeddingModel = 7",
        t=tenant,
        u=url(0),
    )


async def corrupt_target(graph: GraphRepo, tenant: str) -> None:
    await graph._auto(
        "MATCH (p:Page {tenantId: $t, url: $u}) SET p.bodyHash = 42", t=tenant, u=url(0)
    )


@pytest.mark.integration
@pytest.mark.parametrize(
    ("corrupt", "message"),
    [
        pytest.param(
            corrupt_model_value, "embedding model counts do not fit the model", id="models-read"
        ),
        pytest.param(corrupt_target, "embedding targets do not fit the model", id="selection-read"),
    ],
)
async def test_a_startup_read_error_logs_flush_zero_and_is_re_raised(
    mongo: MongoRepo,
    graph: GraphRepo,
    tenant: str,
    corrupt: Callable[[GraphRepo, str], Awaitable[None]],
    message: str,
) -> None:
    await seed_plain(mongo, graph, tenant, 3)
    await corrupt(graph, tenant)
    fake = FakeVoyage(dimension=DIM)

    with capture_logs() as logs, pytest.raises(DatabaseReadError, match=message) as exc_info:
        await embed_tenant(mongo, graph, client(fake), tenant)

    assert fake.call_count == 0
    assert_failed_at_startup(logs, exc_info.value)


@pytest.mark.integration
async def test_a_selection_error_is_re_raised_unchanged(
    mongo: MongoRepo, graph: GraphRepo, tenant: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    error = DatabaseReadError("neo4j", "selection failed")

    async def failing_selection(tenant_id: str) -> NoReturn:
        raise error

    monkeypatch.setattr(graph, "embedding_selection", failing_selection)
    with capture_logs() as logs, pytest.raises(DatabaseReadError, match="selection failed") as info:
        await embed_tenant(mongo, graph, client(FakeVoyage(dimension=DIM)), tenant)

    assert info.value is error
    assert_failed_at_startup(logs, error)


# ── 5-6: failures leave a committed prefix ──────────────────────────────────


@pytest.mark.integration
@pytest.mark.parametrize(
    ("failures", "error", "message"),
    [
        pytest.param(
            {2: InvalidRequestError("input too long", http_status=400)},
            EmbeddingRequestError,
            "HTTP 400 InvalidRequestError: input too long",
            id="rejected",
        ),
        pytest.param(
            {n: RateLimitError("rate limited", http_status=429) for n in (2, 3, 4)},
            EmbeddingUnavailableError,
            "retries exhausted after 3 attempts: HTTP 429 RateLimitError",
            id="retries-exhausted",
        ),
    ],
)
async def test_voyage_failure_keeps_earlier_flushes_and_a_rerun_embeds_the_rest(
    mongo: MongoRepo,
    graph: GraphRepo,
    tenant: str,
    failures: dict[int, BaseException],
    error: type[EmbeddingError],
    message: str,
) -> None:
    urls = [url(i) for i in range(10)]
    await seed_plain(mongo, graph, tenant, 10)
    fake = FakeVoyage(dimension=DIM, fail_on=failures)

    with capture_logs() as logs, pytest.raises(error, match=message) as exc_info:
        await embed_tenant(mongo, graph, client(fake), tenant, flush_size=4)

    pages = await stored(graph, tenant, urls)
    for i in range(4):
        assert_embedded(pages[url(i)], body(i))
    for i in range(4, 10):
        assert_untouched(pages[url(i)])
    (failed,) = events(logs, "embedding.run.failed")
    assert {key: failed[key] for key in ("flush", "pages_done", "error_type", "error")} == {
        "flush": 2,
        "pages_done": 4,
        "error_type": error.__name__,
        "error": str(exc_info.value),
    }
    # Failed attempts spend nothing, so only flush 1's request is counted.
    assert tally_of(failed) == tally(4, api_tokens=4 * WORDS + API_TOKEN_DRIFT, tokens=4 * WORDS)
    assert [entry["flush"] for entry in events(logs, "embedding.flush")] == [1]
    assert events(logs, "embedding.run.done") == []

    healthy = FakeVoyage(dimension=DIM)
    report = await embed_tenant(mongo, graph, client(healthy), tenant, flush_size=4)
    assert sorted(page_index(text) for text in sent_texts(healthy)) == list(range(4, 10))
    assert (report.selected, report.embedded, report.up_to_date) == (6, 6, 4)
    pages = await stored(graph, tenant, urls)
    for i in range(10):
        assert_embedded(pages[url(i)], body(i))


def short_when(index: int) -> Callable[[Sequence[str]], list[list[float]]]:
    """Drops the last vector of the request that carries page `index`."""

    def respond(texts: Sequence[str]) -> list[list[float]]:
        vectors = [tagged_vector(text, DIM) for text in texts]
        return vectors[:-1] if any(page_index(text) == index for text in texts) else vectors

    return respond


@pytest.mark.integration
async def test_a_short_sdk_response_fails_its_flush_before_writing(
    mongo: MongoRepo, graph: GraphRepo, tenant: str
) -> None:
    await seed_plain(mongo, graph, tenant, 10)
    fake = FakeVoyage(dimension=DIM, respond=short_when(4))

    with pytest.raises(EmbeddingResponseError, match="expected 4 vectors, got 3"):
        await embed_tenant(mongo, graph, client(fake), tenant, flush_size=4)

    assert fake.call_count == 2
    pages = await stored(graph, tenant, [url(i) for i in range(10)])
    for i in range(4):
        assert_embedded(pages[url(i)], body(i))
    for i in range(4, 10):
        assert_untouched(pages[url(i)])


@pytest.mark.integration
async def test_a_failure_inside_a_multi_request_flush_writes_none_of_it(
    mongo: MongoRepo, graph: GraphRepo, tenant: str
) -> None:
    await seed_plain(mongo, graph, tenant, 8)
    rejected = InvalidRequestError("input too long", http_status=400)
    fake = FakeVoyage(dimension=DIM, fail_on={2: rejected})

    with pytest.raises(EmbeddingRequestError, match="HTTP 400 InvalidRequestError"):
        await embed_tenant(mongo, graph, client(fake), tenant, flush_size=8)

    # The flush's first request succeeded, but its vectors are only written with the rest.
    assert fake.call_count == 2
    pages = await stored(graph, tenant, [url(i) for i in range(8)])
    for i in range(8):
        assert_untouched(pages[url(i)])


@pytest.mark.integration
async def test_run_failed_counts_what_the_failing_flush_already_spent_and_skipped(
    mongo: MongoRepo, graph: GraphRepo, tenant: str
) -> None:
    # Flush 1 is p000-p007 in two requests. Flush 2 is p008-p015 with p010 unusable, so its
    # requests carry 4 and 3 pages, and the second of them (call 4) is rejected.
    await seed(mongo, graph, tenant, [record(i, usable=i != 10) for i in range(16)])
    fake = FakeVoyage(dimension=DIM, fail_on={4: InvalidRequestError("bad", http_status=400)})

    with capture_logs() as logs, pytest.raises(EmbeddingRequestError, match="HTTP 400"):
        await embed_tenant(mongo, graph, client(fake), tenant, flush_size=8)

    assert [len(call.texts) for call in fake.calls] == [4, 4, 4, 3]
    per_call = [len(call.texts) * WORDS + API_TOKEN_DRIFT for call in fake.calls]
    (flush,) = events(logs, "embedding.flush")
    assert flush["api_tokens"] == sum(per_call[:2])
    (failed,) = events(logs, "embedding.run.failed")
    assert (failed["flush"], failed["pages_done"]) == (2, 8)
    assert tally_of(failed) == tally(
        8, not_usable=1, api_tokens=sum(per_call[:3]), tokens=12 * WORDS
    )
    assert isinstance(failed["elapsed_s"], float)
    pages = await stored(graph, tenant, [url(i) for i in range(16)])
    for i in range(8):
        assert_embedded(pages[url(i)], body(i))
    for i in range(8, 16):
        assert_untouched(pages[url(i)])


@pytest.mark.integration
async def test_a_corrupt_mongo_page_fails_its_flush_and_keeps_earlier_ones(
    mongo: MongoRepo, graph: GraphRepo, tenant: str, mongo_uri: str
) -> None:
    await seed_plain(mongo, graph, tenant, 6)
    raw: AsyncMongoClient[dict[str, object]] = AsyncMongoClient(mongo_uri)
    await raw["linking_engine_test"]["pages"].update_one(
        {"tenantId": tenant, "url": url(5)}, {"$set": {"statusCode": 42}}
    )
    await raw.close()
    fake = FakeVoyage(dimension=DIM)

    with capture_logs() as logs, pytest.raises(DatabaseReadError, match="does not fit PageRecord"):
        await embed_tenant(mongo, graph, client(fake), tenant, flush_size=4)

    assert fake.call_count == 1
    (failed,) = events(logs, "embedding.run.failed")
    assert failed["flush"] == 2
    assert tally_of(failed) == tally(4, api_tokens=sdk_tokens(fake), tokens=4 * WORDS)
    pages = await stored(graph, tenant, [url(i) for i in range(6)])
    for i in range(4):
        assert_embedded(pages[url(i)], body(i))
    for i in (4, 5):
        assert_untouched(pages[url(i)])


@pytest.mark.integration
async def test_a_graph_change_during_the_run_fails_that_flush_and_keeps_earlier_ones(
    mongo: MongoRepo, graph: GraphRepo, tenant: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    await seed_plain(mongo, graph, tenant, 8)
    read = mongo.get_pages
    reads: list[int] = []

    async def racing_read(*args: object, **kwargs: object) -> object:
        reads.append(len(reads) + 1)
        if len(reads) == 2:
            # A concurrent graph reload changes one page of flush 2 after selection.
            await graph._auto(
                "MATCH (p:Page {tenantId: $t, url: $u}) SET p.bodyHash = $h",
                t=tenant,
                u=url(5),
                h=body_hash("a newer body"),
            )
        return await read(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(mongo, "get_pages", racing_read)
    fake = FakeVoyage(dimension=DIM)
    with (
        capture_logs() as logs,
        pytest.raises(DatabaseWriteError, match="wrote 3 of 4 rows, rolled back"),
    ):
        await embed_tenant(mongo, graph, client(fake), tenant, flush_size=4)

    pages = await stored(graph, tenant, [url(i) for i in range(8)])
    for i in range(4):
        assert_embedded(pages[url(i)], body(i))
    for i in range(4, 8):
        assert_untouched(pages[url(i)])
    (failed,) = events(logs, "embedding.run.failed")
    assert (failed["flush"], failed["pages_done"], failed["error_type"]) == (
        2,
        4,
        "DatabaseWriteError",
    )
    # Flush 2 was embedded before its write was refused, so its tokens were spent.
    assert fake.call_count == 2
    assert tally_of(failed) == tally(4, api_tokens=sdk_tokens(fake), tokens=8 * WORDS)


class TamperingVoyage:
    """A real client whose batch holding `url` is rewritten before the pipeline sees it."""

    def __init__(
        self,
        inner: VoyageClient,
        target_url: str,
        tamper: Callable[[EmbeddingBatch], EmbeddingBatch],
    ) -> None:
        self._inner = inner
        self._url = target_url
        self._tamper = tamper

    @property
    def model(self) -> str:
        return self._inner.model

    @property
    def dimension(self) -> int:
        return self._inner.dimension

    async def iter_embed(self, pages: Sequence[PageText]) -> AsyncGenerator[EmbeddingBatch]:
        async with aclosing(self._inner.iter_embed(pages)) as batches:
            async for batch in batches:
                if any(item.url == self._url for item in batch.embeddings):
                    yield self._tamper(batch)
                else:
                    yield batch


def rebuilt(batch: EmbeddingBatch, embeddings: Sequence[PageEmbedding]) -> EmbeddingBatch:
    return EmbeddingBatch(embeddings=tuple(embeddings), api_tokens=batch.api_tokens)


@pytest.mark.integration
@pytest.mark.parametrize(
    ("tamper", "message"),
    [
        pytest.param(
            lambda b: rebuilt(b, b.embeddings[:-1]),
            re.escape("flush 2: 3 vectors for 4 pages, 4 urls and 4 hashes"),
            id="one-short",
        ),
        pytest.param(
            lambda b: rebuilt(b, [*b.embeddings, b.embeddings[-1]]),
            re.escape(f"vector 4 is for {url(7)}, expected no more pages"),
            id="one-extra",
        ),
        pytest.param(
            lambda b: rebuilt(b, [b.embeddings[1], b.embeddings[0], *b.embeddings[2:]]),
            re.escape(f"vector 0 is for {url(5)}, expected {url(4)}"),
            id="out-of-order",
        ),
        pytest.param(
            lambda b: rebuilt(
                b,
                [
                    b.embeddings[0],
                    b.embeddings[1].model_copy(update={"url": url(99)}),
                    *b.embeddings[2:],
                ],
            ),
            re.escape(f"vector 1 is for {url(99)}, expected {url(5)}"),
            id="wrong-url",
        ),
    ],
)
async def test_a_batch_that_does_not_match_its_pages_fails_before_writing(
    mongo: MongoRepo,
    graph: GraphRepo,
    tenant: str,
    tamper: Callable[[EmbeddingBatch], EmbeddingBatch],
    message: str,
) -> None:
    await seed_plain(mongo, graph, tenant, 10)
    tampering = TamperingVoyage(client(FakeVoyage(dimension=DIM)), url(4), tamper)

    with capture_logs() as logs, pytest.raises(EmbeddingResponseError, match=message):
        await embed_tenant(mongo, graph, tampering, tenant, flush_size=4)  # type: ignore[arg-type]

    pages = await stored(graph, tenant, [url(i) for i in range(10)])
    for i in range(4):
        assert_embedded(pages[url(i)], body(i))
    for i in range(4, 10):
        assert_untouched(pages[url(i)])
    (failed,) = events(logs, "embedding.run.failed")
    assert (failed["flush"], failed["error_type"]) == (2, "EmbeddingResponseError")


# ── 7: memory ───────────────────────────────────────────────────────────────

MEMORY_FLUSH = 20
MEMORY_FLUSHES = 10
# The up-front target list is the only per-page state a run keeps; measured at 0.8-0.9 KB.
PER_TARGET_BYTES = 2_000


@pytest.mark.integration
async def test_memory_stays_flat_across_flushes(
    mongo: MongoRepo, graph: GraphRepo, tenant: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    one, many, warm = f"{tenant}-one", f"{tenant}-many", f"{tenant}-warm"
    sizes = ((warm, MEMORY_FLUSH), (one, MEMORY_FLUSH), (many, MEMORY_FLUSH * MEMORY_FLUSHES))
    for name, count in sizes:
        await seed_plain(mongo, graph, name, count)

    write = graph.write_embeddings

    async def collected_first(*args: object, **kwargs: object) -> int:
        # The driver keeps the last write's parameters in a cycle; measure only what is reachable.
        gc.collect()
        return await write(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(graph, "write_embeddings", collected_first)

    async def peak(name: str) -> tuple[EmbedRunReport, int]:
        fake = FakeVoyage(dimension=DIM, record_calls=False)
        gc.collect()
        start = tracemalloc.get_traced_memory()[0]
        tracemalloc.reset_peak()
        report = await embed_tenant(mongo, graph, client(fake), name, flush_size=MEMORY_FLUSH)
        return report, tracemalloc.get_traced_memory()[1] - start

    tracemalloc.start()
    try:
        await peak(warm)
        one_report, one_peak = await peak(one)
        many_report, many_peak = await peak(many)
    finally:
        tracemalloc.stop()

    assert (one_report.flushes, one_report.embedded) == (1, MEMORY_FLUSH)
    assert (many_report.flushes, many_report.embedded) == (
        MEMORY_FLUSHES,
        MEMORY_FLUSH * MEMORY_FLUSHES,
    )
    extra_targets = MEMORY_FLUSH * (MEMORY_FLUSHES - 1)
    assert many_peak - one_peak < PER_TARGET_BYTES * extra_targets, (
        f"1 flush peaked at {one_peak:,} B and {MEMORY_FLUSHES} flushes at {many_peak:,} B; "
        f"allowed growth is {PER_TARGET_BYTES:,} B per extra target"
    )


# ── 8: progress events ──────────────────────────────────────────────────────


@pytest.mark.integration
async def test_one_progress_event_per_flush_with_cumulative_counts(
    mongo: MongoRepo, graph: GraphRepo, tenant: str
) -> None:
    await seed_mixed(mongo, graph, tenant)
    fake = FakeVoyage(dimension=DIM)
    with capture_logs() as logs:
        report = await embed_tenant(mongo, graph, client(fake), tenant, flush_size=MIXED_FLUSH)

    # One request per flush that has pages to send; the all-skipped last flush sends none.
    per_call = [sum(words(text) for text in call.texts) + API_TOKEN_DRIFT for call in fake.calls]
    assert len(per_call) == 3
    keys = ("flush", "flushes", "written", "skipped", *REASONS, "pages_done", "remaining")
    flushes = events(logs, "embedding.flush")
    assert len(flushes) == math.ceil(report.selected / MIXED_FLUSH) == 4
    # Flush 3 skips p007 (unusable) and p008 (empty); flush 4 skips p009 (whitespace).
    assert [{key: entry[key] for key in keys} for entry in flushes] == [
        dict(zip(keys, (1, 4, 3, 0, 0, 0, 0, 0, 3, 7), strict=True)),
        dict(zip(keys, (2, 4, 3, 0, 0, 0, 0, 0, 6, 4), strict=True)),
        dict(zip(keys, (3, 4, 1, 2, 1, 1, 0, 0, 7, 1), strict=True)),
        dict(zip(keys, (4, 4, 0, 1, 0, 1, 0, 0, 7, 0), strict=True)),
    ]
    assert [entry["api_tokens"] for entry in flushes] == [
        per_call[0],
        sum(per_call[:2]),
        sum(per_call),
        sum(per_call),
    ]
    assert all(entry["skipped"] == sum(entry[r] for r in REASONS) for entry in flushes)
    assert flushes[-1]["api_tokens"] == report.api_tokens
    elapsed = [entry["elapsed_s"] for entry in flushes]
    assert elapsed == sorted(elapsed), elapsed
    assert all(entry["tenant_id"] == tenant for entry in flushes)
    assert all(entry["stage"] == "embedding" for entry in flushes)

    (start,) = events(logs, "embedding.run.start")
    assert (start["selected"], start["flushes"], start["flush_size"]) == (10, 4, MIXED_FLUSH)
    assert (start["up_to_date"], start["placeholders"], start["non_2xx"]) == (0, 1, 3)
    (done,) = events(logs, "embedding.run.done")
    assert (done["embedded"], done["flushes"]) == (7, 4)


@pytest.mark.integration
async def test_selection_is_read_once_and_mongo_once_per_flush(
    mongo: MongoRepo, graph: GraphRepo, tenant: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    await seed_mixed(mongo, graph, tenant)
    calls: list[str] = []

    def spy(owner: object, name: str) -> None:
        method = getattr(owner, name)

        async def recorded(*args: object, **kwargs: object) -> object:
            calls.append(name)
            return await method(*args, **kwargs)

        monkeypatch.setattr(owner, name, recorded)

    for name in ("embedding_models", "embedding_selection", "write_embeddings"):
        spy(graph, name)
    spy(mongo, "get_pages")

    await embed_tenant(
        mongo, graph, client(FakeVoyage(dimension=DIM)), tenant, flush_size=MIXED_FLUSH
    )

    flush = ["get_pages", "write_embeddings"]
    assert calls == ["embedding_models", "embedding_selection", *flush * 3, "get_pages"]


@pytest.mark.integration
async def test_a_tenant_with_nothing_to_embed_makes_no_call_and_no_flush(
    mongo: MongoRepo, graph: GraphRepo, tenant: str
) -> None:
    fake = FakeVoyage(dimension=DIM)
    with capture_logs() as logs:
        report = await embed_tenant(mongo, graph, client(fake), tenant)
    assert (report.selected, report.embedded, report.flushes) == (0, 0, 0)
    assert fake.call_count == 0
    assert events(logs, "embedding.flush") == []
    assert len(events(logs, "embedding.run.done")) == 1


# ── skips that do not roll back their flush ─────────────────────────────────


@pytest.mark.integration
async def test_one_flush_with_every_skip_reason_reports_each_on_its_event(
    mongo: MongoRepo, graph: GraphRepo, tenant: str
) -> None:
    # Flush 1: p000 and p004 embedded; p001 unusable, p002 empty, p003 stale, p005 not in
    # Mongo. Flush 2: p006 and p007 embedded, so its per-flush skip counts are all zero.
    pages = [record(0), record(1, usable=False), record(2, ""), record(3), record(4)]
    await seed(mongo, graph, tenant, [*pages, record(6), record(7)])
    await graph.upsert_pages(
        tenant, [Page(url=str(url(5)), status_code=200, body_hash=body_hash(body(5)))]
    )
    await mongo.write_pages(tenant, [record(3, body(3, tag=1003))], [])
    fake = FakeVoyage(dimension=DIM)

    with capture_logs() as logs:
        report = await embed_tenant(mongo, graph, client(fake), tenant, flush_size=6)

    keys = ("flush", "written", "skipped", *REASONS, "pages_done", "remaining")
    assert [{key: entry[key] for key in keys} for entry in events(logs, "embedding.flush")] == [
        dict(zip(keys, (1, 2, 4, 1, 1, 1, 1, 2, 2), strict=True)),
        dict(zip(keys, (2, 2, 0, 0, 0, 0, 0, 4, 0), strict=True)),
    ]
    assert (
        report.skipped_not_usable,
        report.skipped_empty_body,
        report.skipped_missing,
        report.skipped_hash_mismatch,
        report.embedded,
    ) == (1, 1, 1, 1, 4)
    assert [e["sample_urls"] for e in events(logs, "embedding.flush.missing")] == [[url(5)]]
    assert [e["sample_urls"] for e in events(logs, "embedding.flush.hash_mismatch")] == [[url(3)]]
    assert sorted(page_index(text) for text in sent_texts(fake)) == [0, 4, 6, 7]


@pytest.mark.integration
async def test_a_stale_graph_body_hash_is_skipped_with_a_warning(
    mongo: MongoRepo, graph: GraphRepo, tenant: str
) -> None:
    await seed_plain(mongo, graph, tenant, 4)
    # Mongo has the new body; the graph was not reloaded, so its bodyHash is stale.
    await mongo.write_pages(tenant, [record(1, body(1, tag=1001))], [])
    fake = FakeVoyage(dimension=DIM)

    with capture_logs() as logs:
        report = await embed_tenant(mongo, graph, client(fake), tenant, flush_size=4)

    assert (report.selected, report.embedded, report.skipped_hash_mismatch) == (4, 3, 1)
    assert report.flushes == 1
    (warning,) = events(logs, "embedding.flush.hash_mismatch")
    assert (warning["flush"], warning["count"], warning["sample_urls"]) == (1, 1, [url(1)])
    assert warning["log_level"] == "warning"
    assert "load_graph" in str(warning["hint"])
    assert sorted(page_index(text) for text in sent_texts(fake)) == [0, 2, 3]
    pages = await stored(graph, tenant, [url(i) for i in range(4)])
    assert_untouched(pages[url(1)])
    for i in (0, 2, 3):
        assert_embedded(pages[url(i)], body(i))


@pytest.mark.integration
async def test_a_null_graph_body_hash_fails_before_any_read_or_embed(
    mongo: MongoRepo, graph: GraphRepo, tenant: str
) -> None:
    await seed_plain(mongo, graph, tenant, 3)
    # A page loaded before bodyHash existed: the graph needs a reload, so nothing is embedded.
    await graph._auto(
        "MATCH (p:Page {tenantId: $t, url: $u}) REMOVE p.bodyHash", t=tenant, u=url(1)
    )
    fake = FakeVoyage(dimension=DIM)
    with (
        capture_logs() as logs,
        pytest.raises(DatabaseReadError, match="1 selected pages have no bodyHash"),
    ):
        await embed_tenant(mongo, graph, client(fake), tenant)

    assert fake.call_count == 0
    (failed,) = events(logs, "embedding.run.failed")
    assert (failed["flush"], failed["error_type"]) == (0, "DatabaseReadError")
    pages = await stored(graph, tenant, [url(i) for i in range(3)])
    for i in range(3):
        assert_untouched(pages[url(i)])


@pytest.mark.integration
async def test_graph_pages_missing_from_mongo_are_skipped_with_a_capped_sample(
    mongo: MongoRepo, graph: GraphRepo, tenant: str
) -> None:
    await seed_plain(mongo, graph, tenant, 2)
    graph_only = list(range(10, 17))
    await graph.upsert_pages(
        tenant,
        [Page(url=str(url(i)), status_code=200, body_hash=body_hash(body(i))) for i in graph_only],
    )
    with capture_logs() as logs:
        report = await embed_tenant(mongo, graph, client(FakeVoyage(dimension=DIM)), tenant)

    assert (report.selected, report.embedded, report.skipped_missing) == (9, 2, 7)
    (warning,) = events(logs, "embedding.flush.missing")
    assert (warning["count"], warning["sample_urls"]) == (7, [url(i) for i in graph_only[:5]])
    pages = await stored(graph, tenant, [url(i) for i in (0, 1, *graph_only)])
    assert_embedded(pages[url(0)], body(0))
    assert_embedded(pages[url(1)], body(1))
    for i in graph_only:
        assert_untouched(pages[url(i)])


# ── guards that run before any I/O ──────────────────────────────────────────


class Untouchable:
    """Stands in for a store the code under test must not reach."""

    def __getattr__(self, name: str) -> NoReturn:
        raise AssertionError(f"{name} was used before the argument checks")


@pytest.mark.parametrize(
    ("tenant_id", "flush_size", "message"),
    [
        (" ", FLUSH_SIZE, "tenant_id must be a non-empty string"),
        ("acme", 0, "flush_size must be at least 1"),
        ("acme", -1, "flush_size must be at least 1"),
    ],
)
async def test_bad_arguments_raise_before_any_io(
    tenant_id: str, flush_size: int, message: str
) -> None:
    fake = FakeVoyage(dimension=DIM)
    with pytest.raises(ValueError, match=message):
        await embed_tenant(
            Untouchable(),  # type: ignore[arg-type]
            Untouchable(),  # type: ignore[arg-type]
            client(fake),
            tenant_id,
            flush_size=flush_size,
        )
    assert fake.call_count == 0


async def test_a_client_dimension_other_than_the_index_fails_before_any_io() -> None:
    fake = FakeVoyage(dimension=16)
    with pytest.raises(SchemaError, match="embedding dimension 16 does not match the 2048d"):
        await embed_tenant(Untouchable(), Untouchable(), client(fake), "acme")  # type: ignore[arg-type]
    assert fake.call_count == 0


# ── check_models ────────────────────────────────────────────────────────────


def rows(*pairs: tuple[str | None, int]) -> tuple[EmbeddingModelCount, ...]:
    return tuple(EmbeddingModelCount(embedding_model=model, vectors=n) for model, n in pairs)


@pytest.mark.parametrize(
    "found", [(), rows((MODEL, 3))], ids=["no-stored-vectors", "only-the-configured-model"]
)
def test_check_models_passes(found: tuple[EmbeddingModelCount, ...]) -> None:
    check_models(found, MODEL, tenant_id="acme")


@pytest.mark.parametrize(
    ("found", "problem", "listing"),
    [
        pytest.param(
            rows((MODEL, 2), (None, 1)),
            "some stored vectors have no embeddingModel",
            "voyage-4-large (2), <no model> (1)",
            id="configured-and-none",
        ),
        pytest.param(
            rows((None, 4)),
            "some stored vectors have no embeddingModel",
            "<no model> (4)",
            id="only-none",
        ),
        pytest.param(
            rows((OTHER_MODEL, 1), (MODEL, 5), (None, 2)),
            "some stored vectors have no embeddingModel",
            "voyage-3-large (1), voyage-4-large (5), <no model> (2)",
            id="none-is-reported-before-mixing",
        ),
        pytest.param(
            rows((OTHER_MODEL, 1), (MODEL, 5)),
            "stored vectors mix 2 embedding models",
            "voyage-3-large (1), voyage-4-large (5)",
            id="two-models",
        ),
        pytest.param(
            rows(("a", 1), ("b", 2), ("c", 3)),
            "stored vectors mix 3 embedding models",
            "a (1), b (2), c (3)",
            id="three-models",
        ),
        pytest.param(
            rows((OTHER_MODEL, 7)),
            "stored vectors use a different model than configured",
            "voyage-3-large (7)",
            id="one-other-model",
        ),
    ],
)
def test_check_models_names_tenant_problem_and_every_model(
    found: tuple[EmbeddingModelCount, ...], problem: str, listing: str
) -> None:
    expected = (
        f"tenant acme: {problem}; found {listing}; configured {MODEL}. "
        "Vectors from different models are not comparable"
    )
    with pytest.raises(EmbeddingModelMismatchError, match=f"^{re.escape(expected)}$"):
        check_models(found, MODEL, tenant_id="acme")


def test_check_models_rejects_a_blank_configured_model() -> None:
    with pytest.raises(ValueError, match="configured model must be a non-empty string"):
        check_models(rows((MODEL, 1)), " ", tenant_id="acme")


def test_a_model_mismatch_is_an_embedding_error() -> None:
    assert issubclass(EmbeddingModelMismatchError, EmbeddingError)
