"""Keyword resolution: the keyword a page's inbound anchors should be about.

The chain, first match wins: the client's strategic keyword, the GSC query with the most
click upside that the page's own copy can anchor, the H1, the title. Displayed text is kept
as written, whitespace collapsed; matching uses ``normalise_term``.

H1 and title formats differ per tenant, so the fallbacks assume none: brand affixes are
detected from the tenant's own titles, and a fallback that is generic, only the brand, or
shared by several pages is rejected with its reason.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import TYPE_CHECKING, Final

from linking_engine.anchor.generic import is_generic
from linking_engine.gsc import normalise_term, opportunity_value
from linking_engine.models import KeywordRung, ResolvedKeyword, StrategicKeyword

if TYPE_CHECKING:
    from collections.abc import Collection, Iterable, Sequence

    from linking_engine.models import CtrCurve, GscQueryStats, PageRecord

MIN_QUERY_IMPRESSIONS: Final = 50
BRAND_SUFFIX_SHARE: Final = 0.3
# An affix is shared, so one title alone never makes one, however few titles there are.
MIN_BRAND_TITLES: Final = 2
MIN_TOKEN_LENGTH: Final = 2
# A fallback on this many pages is a template text that cannot tell them apart.
REPEATED_FALLBACK_PAGES: Final = 3
# Longer fallbacks are kept, but counted: they read as sentences rather than keywords.
MAX_KEYWORD_TOKENS: Final = 12
# Further GSC queries ranked after a page's resolved and strategic keywords.
MAX_SECONDARY_QUERIES: Final = 4
# Headings that stand in for a page's topic without naming it, per language. A keyword-only
# list: as anchors these words are judged by the anchor audit's own generic vocabulary.
PLACEHOLDER_HEADINGS: Final = frozenset(
    {
        # en
        "home",
        "homepage",
        "home page",
        "welcome",
        "untitled",
        "index",
        "page not found",
        "404",
        "not found",
        "error",
        # de
        "startseite",
        "willkommen",
        "herzlich willkommen",
        "seite nicht gefunden",
        # fr
        "accueil",
        "bienvenue",
        "page introuvable",
        # es
        "inicio",
        "bienvenido",
        "bienvenida",
        "página no encontrada",
        # it
        "benvenuto",
        "benvenuti",
        "pagina non trovata",
    }
)
# Pipe, hyphen, en dash, em dash or double colon, with a space either side.
_SEPARATOR: Final = re.compile(r" (?:\||-|\u2013|\u2014|::) ")
_TOKEN: Final = re.compile(r"\w+")


def display_text(text: str) -> str:
    return " ".join(text.split())


def tokens(text: str) -> frozenset[str]:
    """Normalised word tokens of ``text``."""
    return frozenset(_TOKEN.findall(normalise_term(text)))


def brand_affixes(titles: Iterable[str | None]) -> tuple[str | None, str | None]:
    """(prefix, suffix): the first and the last title segment, each when shared by at least
    ``BRAND_SUFFIX_SHARE`` of the non-blank titles and by ``MIN_BRAND_TITLES`` of them."""
    counted = 0
    firsts: Counter[str] = Counter()
    lasts: Counter[str] = Counter()
    written: dict[str, Counter[str]] = {}
    for title in titles:
        if not title or not title.strip():
            continue
        counted += 1
        segments = [segment.strip() for segment in _SEPARATOR.split(display_text(title))]
        if len(segments) < 2:
            continue
        for counter, segment in ((firsts, segments[0]), (lasts, segments[-1])):
            if segment:
                key = normalise_term(segment)
                counter[key] += 1
                written.setdefault(key, Counter())[segment] += 1

    def common(counter: Counter[str]) -> str | None:
        if not counter:
            return None
        key, count = min(counter.items(), key=lambda item: (-item[1], item[0]))
        if count < max(MIN_BRAND_TITLES, BRAND_SUFFIX_SHARE * counted):
            return None
        return min(written[key].items(), key=lambda item: (-item[1], item[0]))[0]

    return common(firsts), common(lasts)


def brand_suffix(titles: Iterable[str | None]) -> str | None:
    """The last title segment shared by at least ``BRAND_SUFFIX_SHARE`` of the titles."""
    return brand_affixes(titles)[1]


def brand_prefix(titles: Iterable[str | None]) -> str | None:
    """The first title segment shared by at least ``BRAND_SUFFIX_SHARE`` of the titles."""
    return brand_affixes(titles)[0]


def clean_title(title: str | None, suffix: str | None, prefix: str | None = None) -> str | None:
    """``title`` with a brand affix and its separator stripped from either end; None when
    nothing is left."""
    if title is None:
        return None
    text = display_text(title)
    brands = {normalise_term(affix) for affix in (prefix, suffix) if affix}
    separators = list(_SEPARATOR.finditer(text)) if brands else []
    if separators and normalise_term(text[separators[-1].end() :]) in brands:
        text = text[: separators[-1].start()].strip()
        separators.pop()
    if separators and normalise_term(text[: separators[0].start()]) in brands:
        text = text[separators[0].end() :].strip()
    return text or None


def h1_candidates(page: PageRecord) -> tuple[str, ...]:
    """The page's H1, then its other level-1 headings, in order and without repeats."""
    found: list[str] = []
    for text in (page.h1, *(heading.text for heading in page.headings if heading.level == 1)):
        if text is not None and (shown := display_text(text)) and shown not in found:
            found.append(shown)
    return tuple(found)


def fallback_texts(page: PageRecord) -> tuple[str, ...]:
    """Every H1 and title candidate of the page, before cleaning."""
    title = display_text(page.meta_title) if page.meta_title else ""
    return (*h1_candidates(page), title) if title else h1_candidates(page)


def repeated_fallbacks(
    pages: Iterable[tuple[Sequence[str], str]], suffix: str | None, prefix: str | None = None
) -> frozenset[str]:
    """Normalised cleaned fallbacks found over at least ``REPEATED_FALLBACK_PAGES`` different
    bodies; each entry of ``pages`` is one page's ``fallback_texts`` and body hash. A template
    heading sits over different content; the same article served at several urls has one body
    and keeps its heading."""
    bodies: defaultdict[str, set[str]] = defaultdict(set)
    for texts, body in pages:
        cleaned = (clean_title(text, suffix, prefix) for text in texts)
        for text in {normalise_term(text) for text in cleaned if text}:
            bodies[text].add(body)
    return frozenset(
        text for text, found in bodies.items() if len(found) >= REPEATED_FALLBACK_PAGES
    )


def fallback_reason(
    text: str | None,
    brand_tokens: frozenset[str],
    repeated: Collection[str],
    *,
    generic_add: frozenset[str] = frozenset(),
    generic_remove: frozenset[str] = frozenset(),
) -> str | None:
    """Why a cleaned H1 or title cannot be the page's keyword; None when it can."""
    if text is None:
        return "missing"
    # is_generic has no key for punctuation-only text ("---"), so letters are checked here too.
    if (
        not any(char.isalpha() for char in text)
        or is_generic(text, add=generic_add, remove=generic_remove)
        or ((term := normalise_term(text)) in PLACEHOLDER_HEADINGS and term not in generic_remove)
    ):
        return "generic"
    if brand_tokens and tokens(text) <= brand_tokens:
        return "brand"
    if normalise_term(text) in repeated:
        return "repeated"
    return None


def usable_query(query: str, page_text: str, impressions: int, brand: str | None) -> bool:
    """A GSC query is usable when it has enough impressions, is more than the brand, and
    every token of it occurs in ``page_text``, so an anchor can be extracted."""
    return _usable(query, tokens(page_text), impressions, tokens(brand) if brand else frozenset())


def _usable(
    query: str, page_tokens: frozenset[str], impressions: int, brand_tokens: frozenset[str]
) -> bool:
    words = {token for token in tokens(query) if len(token) >= MIN_TOKEN_LENGTH}
    return (
        impressions >= MIN_QUERY_IMPRESSIONS
        and bool(words)
        and not words <= brand_tokens
        and words <= page_tokens
    )


def usable_strategic(row: StrategicKeyword) -> bool:
    """A strategic row can become a keyword: non-blank text and a language code."""
    return bool(row.keyword.strip()) and len(row.language) >= MIN_TOKEN_LENGTH


def strategic_rank(row: StrategicKeyword) -> tuple[bool, int, str, str]:
    # Primary first, then priority 5 down to 1 (unset last), then text, then text as written.
    return (
        not row.is_primary,
        -(row.priority or 0),
        normalise_term(row.keyword),
        display_text(row.keyword),
    )


def merge_strategic(rows: Iterable[StrategicKeyword]) -> list[StrategicKeyword]:
    """The usable rows, one per page, normalised keyword and language, ordered by url then
    ``strategic_rank``. A client's rows can repeat a keyword, so duplicates merge: highest
    priority, primary if any row is, and the best-ranked row's text."""
    groups: dict[tuple[str, str, str], list[StrategicKeyword]] = {}
    for row in rows:
        if usable_strategic(row):
            key = (row.url, normalise_term(row.keyword), row.language)
            groups.setdefault(key, []).append(row)
    merged = [
        StrategicKeyword(
            url=url,
            keyword=display_text(min(group, key=strategic_rank).keyword),
            language=language,
            priority=max((row.priority for row in group if row.priority), default=None),
            is_primary=any(row.is_primary for row in group),
        )
        for (url, _, language), group in groups.items()
    ]
    return sorted(merged, key=lambda row: (row.url, strategic_rank(row)))


def choose_strategic(strategic: Iterable[StrategicKeyword]) -> StrategicKeyword | None:
    """The primary strategic keyword, else the highest-priority one, ties by text, after
    duplicates merge."""
    return min(merge_strategic(strategic), key=strategic_rank, default=None)


def _brand_tokens(prefix: str | None, suffix: str | None) -> frozenset[str]:
    return frozenset().union(*(tokens(affix) for affix in (prefix, suffix) if affix))


def _ranked_queries(
    page: PageRecord,
    queries: Sequence[GscQueryStats],
    curve: CtrCurve | None,
    brand_tokens: frozenset[str],
) -> tuple[list[tuple[float, str]], int]:
    """The page's usable GSC queries as (opportunity value, text), best first, and how many
    the quality bar rejected; nothing without a curve."""
    own = [row for row in queries if row.url == page.url]
    if curve is None or not own:
        return [], 0
    page_tokens = tokens(" ".join(filter(None, (page.h1, page.meta_title, page.body_text))))
    usable: list[tuple[float, str]] = []
    for row in own:
        if _usable(row.query, page_tokens, row.impressions, brand_tokens):
            usable.append(
                (opportunity_value(curve, row.impressions, row.position), display_text(row.query))
            )
    usable.sort(key=lambda item: (-item[0], normalise_term(item[1]), item[1]))
    return usable, len(own) - len(usable)


def secondary_queries(
    page: PageRecord,
    queries: Sequence[GscQueryStats],
    curve: CtrCurve | None,
    taken: Iterable[str],
    *,
    suffix: str | None = None,
    prefix: str | None = None,
    limit: int = MAX_SECONDARY_QUERIES,
) -> tuple[str, ...]:
    """Up to ``limit`` usable GSC queries of the page by opportunity value, skipping any whose
    normalised text is in ``taken`` (the page's keywords so far) or already chosen."""
    ranked, _ = _ranked_queries(page, queries, curve, _brand_tokens(prefix, suffix))
    seen = {normalise_term(text) for text in taken}
    chosen: list[str] = []
    for _, text in ranked:
        if len(chosen) >= limit:
            break
        if (key := normalise_term(text)) not in seen:
            seen.add(key)
            chosen.append(text)
    return tuple(chosen)


def resolve_page(
    page: PageRecord,
    language: str,
    strategic: Sequence[StrategicKeyword],
    queries: Sequence[GscQueryStats],
    curve: CtrCurve | None,
    suffix: str | None,
    *,
    prefix: str | None = None,
    repeated: Collection[str] = frozenset(),
    generic_add: frozenset[str] = frozenset(),
    generic_remove: frozenset[str] = frozenset(),
) -> tuple[ResolvedKeyword | None, int, tuple[str, ...]]:
    """The page's keyword, how many of its GSC queries the quality bar rejected (counted only
    when the chain reaches the GSC rung), and why each H1 or title fallback tried was rejected
    ("h1_repeated", "title_generic", ...)."""
    chosen = choose_strategic(row for row in strategic if row.url == page.url)
    if chosen is not None:
        return (
            ResolvedKeyword(
                url=page.url,
                text=display_text(chosen.keyword),
                language=chosen.language,
                rung=KeywordRung.STRATEGIC,
            ),
            0,
            (),
        )

    brand_tokens = _brand_tokens(prefix, suffix)
    ranked, rejected = _ranked_queries(page, queries, curve, brand_tokens)
    if ranked:
        value, text = ranked[0]
        return (
            ResolvedKeyword(
                url=page.url,
                text=text,
                language=language,
                rung=KeywordRung.GSC,
                opportunity_value=value,
            ),
            rejected,
            (),
        )

    reasons: list[str] = []
    title = display_text(page.meta_title) if page.meta_title else ""
    for rung, name, candidates in (
        (KeywordRung.H1, "h1", h1_candidates(page)),
        (KeywordRung.TITLE, "title", (title,) if title else ()),
    ):
        if not candidates:
            reasons.append(f"{name}_missing")
        for candidate in candidates:
            cleaned = clean_title(candidate, suffix, prefix)
            reason = fallback_reason(
                cleaned,
                brand_tokens,
                repeated,
                generic_add=generic_add,
                generic_remove=generic_remove,
            )
            if cleaned is not None and reason is None:
                return (
                    ResolvedKeyword(url=page.url, text=cleaned, language=language, rung=rung),
                    rejected,
                    tuple(reasons),
                )
            reasons.append(f"{name}_{reason}")
    return None, rejected, tuple(reasons)


def resolve_keyword(
    page: PageRecord,
    language: str,
    strategic: Sequence[StrategicKeyword],
    queries: Sequence[GscQueryStats],
    curve: CtrCurve | None,
    suffix: str | None,
    *,
    prefix: str | None = None,
    repeated: Collection[str] = frozenset(),
    generic_add: frozenset[str] = frozenset(),
    generic_remove: frozenset[str] = frozenset(),
) -> ResolvedKeyword | None:
    """The first rung of the chain that yields a keyword for the page; None when none does."""
    return resolve_page(
        page,
        language,
        strategic,
        queries,
        curve,
        suffix,
        prefix=prefix,
        repeated=repeated,
        generic_add=generic_add,
        generic_remove=generic_remove,
    )[0]


def is_long(keyword: ResolvedKeyword) -> bool:
    """An H1 or title keyword longer than ``MAX_KEYWORD_TOKENS`` words."""
    return (
        keyword.rung in (KeywordRung.H1, KeywordRung.TITLE)
        and len(keyword.text.split()) > MAX_KEYWORD_TOKENS
    )
