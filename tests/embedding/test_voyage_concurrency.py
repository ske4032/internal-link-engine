"""Concurrent Voyage requests paced to a token budget: in flight up to the cap, batches in input
order, tokens over any 60 seconds within the budget on a fake clock, an oversized request
waiting rather than failing, and a failure cancelling the rest."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pytest
from pydantic import ValidationError
from structlog.testing import capture_logs
from voyage_fakes import (
    FakeResult,
    VoyageSettings,
    page,
    page_index,
    settings,
    tagged_vector,
    word_tokenizer,
    words,
)
from voyageai.error import ServiceUnavailableError

from linking_engine.embedding import voyage_client
from linking_engine.embedding.voyage_client import TokenPacer, VoyageClient
from linking_engine.errors import EmbeddingUnavailableError

if TYPE_CHECKING:
    from collections.abc import Sequence

DIMENSION = 16


@dataclass
class FakeClock:
    """The pacer's clock. A sleep first lets every task that is ready run at the current time,
    as they would on a real clock, then moves the clock forward at once."""

    now: float = 0.0
    slept: list[float] = field(default_factory=list)

    def time(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        await asyncio.sleep(0)
        self.now += seconds


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> FakeClock:
    found = FakeClock()
    monkeypatch.setattr(voyage_client, "pace_clock", found.time)
    monkeypatch.setattr(voyage_client, "pace_sleep", found.sleep)
    return found


@dataclass
class ScriptedVoyage:
    """An SDK whose requests take ``delays[page]`` seconds of real time, fail on ``failing``
    pages, and record how many are in flight and when each started on the fake clock."""

    delays: dict[int, float] = field(default_factory=dict)
    failing: set[int] = field(default_factory=set)
    clock: FakeClock | None = None
    in_flight: int = 0
    most_in_flight: int = 0
    started: list[tuple[int, float]] = field(default_factory=list)
    cancelled: int = 0

    async def embed(
        self,
        texts: Sequence[str],
        *,
        model: str,
        input_type: str,
        output_dimension: int,
        truncation: bool,
    ) -> FakeResult:
        first = page_index(texts[0])
        self.started.append((first, self.clock.now if self.clock else 0.0))
        self.in_flight += 1
        self.most_in_flight = max(self.most_in_flight, self.in_flight)
        try:
            await asyncio.sleep(self.delays.get(first, 0.0))
        except asyncio.CancelledError:
            self.cancelled += 1
            raise
        finally:
            self.in_flight -= 1
        if first in self.failing:
            raise ServiceUnavailableError("down", http_status=503)
        return FakeResult(
            [tagged_vector(text, output_dimension) for text in texts],
            sum(words(text) for text in texts),
        )


def voyage(sdk: ScriptedVoyage, **overrides: object) -> VoyageClient:
    values: dict[str, object] = {"max_batch_items": 1, "max_attempts": 1, **overrides}
    return VoyageClient(
        settings(**values),
        model="voyage-4-large",
        dimension=DIMENSION,
        sdk=sdk,
        tokenizer=word_tokenizer(),
    )


# ── settings ────────────────────────────────────────────────────────────────


def test_the_defaults_are_sixteen_requests_and_three_million_tokens_a_minute() -> None:
    found = VoyageSettings(api_key="f" * 64)  # type: ignore[arg-type]

    assert (found.max_concurrent_requests, found.tokens_per_minute) == (16, 3_000_000)


def test_both_limits_come_from_the_environment_and_must_be_positive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VOYAGE_API_KEY", "f" * 64)
    monkeypatch.setenv("VOYAGE_MAX_CONCURRENT_REQUESTS", "3")
    monkeypatch.setenv("VOYAGE_TOKENS_PER_MINUTE", "1000")

    found = VoyageSettings()  # type: ignore[call-arg]

    assert (found.max_concurrent_requests, found.tokens_per_minute) == (3, 1000)
    for name in ("VOYAGE_MAX_CONCURRENT_REQUESTS", "VOYAGE_TOKENS_PER_MINUTE"):
        monkeypatch.setenv(name, "0")
        with pytest.raises(ValidationError):
            VoyageSettings()  # type: ignore[call-arg]
        monkeypatch.setenv(name, "3")


# ── concurrency ─────────────────────────────────────────────────────────────


async def test_small_requests_run_together_up_to_the_cap_and_never_beyond() -> None:
    sdk = ScriptedVoyage(delays=dict.fromkeys(range(10), 0.02))

    embedded = await voyage(sdk, max_concurrent_requests=3).embed([page(i, 5) for i in range(10)])

    assert len(embedded) == 10
    assert sdk.most_in_flight == 3


async def test_batches_come_back_in_input_order_whatever_order_requests_finish() -> None:
    # The earlier the request, the longer it takes: they finish in reverse.
    sdk = ScriptedVoyage(delays={i: 0.05 - 0.005 * i for i in range(8)})

    batches = [
        batch
        async for batch in voyage(sdk, max_concurrent_requests=8).iter_embed(
            [page(i, 5) for i in range(8)]
        )
    ]

    assert [batch.embeddings[0].url for batch in batches] == [page(i, 5).url for i in range(8)]
    assert sdk.most_in_flight == 8


async def test_a_request_failing_after_its_retries_raises_and_cancels_the_rest() -> None:
    # Request 0 is slow, request 1 fails at once, the others would take ten seconds.
    delays = {0: 0.05, 1: 0.0, **dict.fromkeys(range(2, 6), 10.0)}
    sdk = ScriptedVoyage(delays=delays, failing={1})

    with pytest.raises(EmbeddingUnavailableError):
        await asyncio.wait_for(
            voyage(sdk, max_concurrent_requests=6).embed([page(i, 5) for i in range(6)]),
            timeout=5,
        )

    assert sdk.cancelled >= 4, "the slow requests were left running"
    assert sdk.in_flight == 0


# ── pacing ──────────────────────────────────────────────────────────────────


async def test_the_pacer_keeps_every_minute_within_the_budget(clock: FakeClock) -> None:
    pacer = TokenPacer(100)

    waits = [await pacer.acquire(60), await pacer.acquire(60), await pacer.acquire(40)]

    # 60 + 60 exceeds the budget: the second waits for the first to age out; 60 + 40 fits.
    assert waits == [0.0, 60.0, 0.0]
    assert clock.now == 60.0


async def test_a_request_larger_than_the_budget_waits_for_an_empty_minute(clock: FakeClock) -> None:
    pacer = TokenPacer(100)

    assert await pacer.acquire(500) == 0.0
    assert await pacer.acquire(10) == 60.0
    assert await pacer.acquire(90) == 0.0
    assert await pacer.acquire(1) == 60.0


async def test_the_pacer_admits_requests_in_the_order_they_ask(clock: FakeClock) -> None:
    pacer = TokenPacer(100)
    admitted: list[int] = []

    async def ask(number: int) -> None:
        await pacer.acquire(50)
        admitted.append(number)

    await asyncio.gather(*(ask(number) for number in range(6)))

    assert admitted == list(range(6))
    assert clock.now == 120.0


def test_a_pacer_needs_a_positive_budget() -> None:
    with pytest.raises(ValueError, match="positive"):
        TokenPacer(0)


async def test_big_requests_stay_within_the_token_budget_of_every_minute(
    clock: FakeClock,
) -> None:
    sdk = ScriptedVoyage(clock=clock)
    client = voyage(sdk, max_concurrent_requests=8, tokens_per_minute=100)

    with capture_logs() as logs:
        embedded = await client.embed([page(i, 40) for i in range(8)])

    assert [e.url for e in embedded] == [page(i, 40).url for i in range(8)]
    # 40 tokens each against 100 a minute: two requests start per minute, in input order.
    assert sdk.started == [(i, 60.0 * (i // 2)) for i in range(8)]
    for _, start in sdk.started:
        window = [s for _, s in sdk.started if start <= s < start + 60]
        assert 40 * len(window) <= 100
    throttled = [line["throttle_ms"] for line in logs if line["event"] == "embedding.batch"]
    # Each request asked at 0: its wait, queueing behind the earlier ones included.
    assert throttled == [0, 0, 60_000, 60_000, 120_000, 120_000, 180_000, 180_000]


async def test_one_request_over_the_budget_is_sent_rather_than_refused(clock: FakeClock) -> None:
    sdk = ScriptedVoyage(clock=clock)
    client = voyage(sdk, tokens_per_minute=30, context_tokens=50)

    embedded = await client.embed([page(0, 45), page(1, 5)])

    assert len(embedded) == 2
    assert sdk.started == [(0, 0.0), (1, 60.0)]


async def test_a_slow_first_request_is_awaited_without_polling_the_finished_ones(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Later requests that are already done must not wake the wait for the first again and
    again; each wake-up is a request finishing."""
    wait = asyncio.wait
    wakeups = 0

    async def counted(*args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        nonlocal wakeups
        wakeups += 1
        return await wait(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(voyage_client.asyncio, "wait", counted)
    sdk = ScriptedVoyage(delays={0: 0.2, 1: 0.0, 2: 0.0})

    embedded = await voyage(sdk, max_concurrent_requests=3).embed([page(i, 5) for i in range(3)])

    assert [e.url for e in embedded] == [page(i, 5).url for i in range(3)]
    assert 1 <= wakeups <= 3, f"{wakeups} wake-ups for three requests"
