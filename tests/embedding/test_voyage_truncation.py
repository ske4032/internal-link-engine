"""Truncation against a byte-level BPE, where one character can span several tokens."""

from __future__ import annotations

from structlog.testing import capture_logs
from voyage_fakes import DIMENSION, RAW_COMPONENT, FakeVoyage, bpe_tokenizer, client, settings

from linking_engine.models import PageEmbedding, PageText

TOKENIZER = bpe_tokenizer()
SPECIALS = len(TOKENIZER.encode("")) - len(TOKENIZER.encode("", add_special_tokens=False))
# Longest character in the pages, in tokens: an emoji is 4 byte tokens.
MAX_TOKENS_PER_CHAR = 4
ROOMY = 100_000

PAGES = [
    PageText(
        url="https://example.com/mixed",
        text="Café naïve résumé 日本語の文章 😀 the word 👩‍💻 東京 🎉 word " * 2,
    ),
    PageText(url="https://example.com/emoji", text="😀😀😀 日本 é 🇬🇧 ok the words"),
    PageText(url="https://example.com/ascii", text="the word the word plain ascii text"),
    PageText(url="https://example.com/cjk", text="日本語日本語東京東京"),
]


def count(text: str) -> int:
    return len(TOKENIZER.encode(text))


def content_ids(text: str) -> list[int]:
    return TOKENIZER.encode(text, add_special_tokens=False).ids


LONGEST = max(count(p.text) for p in PAGES)
# Every limit from "specials only" to "nothing truncated", so a cut lands on every token.
LIMITS = range(SPECIALS, LONGEST + 2)


async def run(limit: int) -> tuple[list[str], list[PageEmbedding]]:
    fake = FakeVoyage(respond=lambda texts: [[RAW_COMPONENT] * DIMENSION for _ in texts])
    config = settings(
        context_tokens=limit,
        max_request_tokens=ROOMY,
        request_token_budget=ROOMY,
        max_batch_items=len(PAGES),
    )
    result = await client(fake, config, tokenizer=TOKENIZER).embed(PAGES)
    sent = [text for call in fake.calls for text in call.texts]
    assert len(sent) == len(PAGES)
    return sent, result


def test_fixture_splits_characters_across_tokens() -> None:
    # Guards the fixture itself: without these splits the tests below prove nothing.
    encoding = TOKENIZER.encode("é😀日")
    assert SPECIALS == 2
    assert encoding.offsets[0] == encoding.offsets[-1] == (0, 0), "BOS/EOS must be zero-width"
    assert encoding.offsets[1:-1] == [(0, 1)] + [(1, 2)] * 4 + [(2, 3)] * 2
    assert all(count(p.text) > len(p.text) for p in PAGES[:2]), "pages must have split chars"


async def test_truncated_text_fits_the_context_limit() -> None:
    over = []
    for limit in LIMITS:
        sent, _ = await run(limit)
        over += [
            (limit, p.url, count(s)) for p, s in zip(PAGES, sent, strict=True) if count(s) > limit
        ]
    assert over == [], f"(context_tokens, url, re-encoded tokens) over the limit: {over[:5]}"


async def test_truncated_text_is_a_whole_character_token_prefix() -> None:
    bad = []
    for limit in LIMITS:
        sent, _ = await run(limit)
        for p, s in zip(PAGES, sent, strict=True):
            ids = content_ids(s)
            if s != p.text[: len(s)] or ids != content_ids(p.text)[: len(ids)]:
                bad.append((limit, p.url, s[-3:]))
    assert bad == [], f"sent text is not a clean prefix of the page: {bad[:5]}"


async def test_truncation_drops_at_most_one_partial_character() -> None:
    wasteful = []
    for limit in LIMITS:
        sent, _ = await run(limit)
        for p, s in zip(PAGES, sent, strict=True):
            floor = min(count(p.text), limit - (MAX_TOKENS_PER_CHAR - 1))
            if count(s) < floor:
                wasteful.append((limit, p.url, count(s)))
    assert wasteful == [], (
        f"(context_tokens, url, kept tokens) cut more than needed: {wasteful[:5]}"
    )


async def test_reported_tokens_are_the_sent_texts_token_count() -> None:
    wrong = []
    for limit in LIMITS:
        sent, result = await run(limit)
        for p, s, r in zip(PAGES, sent, result, strict=True):
            expected = (count(s), count(p.text), s != p.text)
            if (r.tokens, r.original_tokens, r.truncated) != expected:
                wrong.append((limit, p.url, (r.tokens, r.original_tokens, r.truncated), expected))
    assert wrong == [], (
        "(context_tokens, url, reported (tokens, original, truncated), re-encoded) "
        f"differ in {len(wrong)} cases: {wrong[:5]}"
    )


async def test_context_below_the_special_tokens_sends_empty_text() -> None:
    fake = FakeVoyage(respond=lambda texts: [[RAW_COMPONENT] * DIMENSION for _ in texts])
    config = settings(context_tokens=1, max_request_tokens=ROOMY, request_token_budget=ROOMY)
    with capture_logs() as logs:
        (result,) = await client(fake, config, tokenizer=TOKENIZER).embed(PAGES[:1])
    assert fake.calls[0].texts == ("",)
    # Only BOS/EOS remain; that is what the model sees, so that is the count reported.
    assert (result.tokens, result.original_tokens, result.truncated) == (
        count(""),
        count(PAGES[0].text),
        True,
    )
    (truncated,) = [e for e in logs if e["event"] == "embedding.truncated"]
    assert truncated["limit"] == 1
