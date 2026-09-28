"""The keyword chain: strategic, then GSC by opportunity behind a quality bar, then the H1,
then the title without its brand suffix. Every rung has its own fixture: a corpus where every
page has a strategic keyword never reaches rungs 2 and 3."""

from __future__ import annotations

import pytest

from linking_engine.anchor.keywords import (
    MAX_KEYWORD_TOKENS,
    MAX_SECONDARY_QUERIES,
    REPEATED_FALLBACK_PAGES,
    brand_affixes,
    brand_prefix,
    brand_suffix,
    clean_title,
    fallback_reason,
    h1_candidates,
    is_long,
    merge_strategic,
    repeated_fallbacks,
    resolve_keyword,
    resolve_page,
    secondary_queries,
    usable_query,
)
from linking_engine.ingest.markdown_clean import body_hash
from linking_engine.models import (
    CtrCurve,
    GscQueryStats,
    Heading,
    KeywordRung,
    PageRecord,
    ResolvedKeyword,
    StrategicKeyword,
)

URL = "example.com/shoes"
CURVE = CtrCurve(ctr=(0.3, 0.2, 0.1, 0.05), rows=100, impressions=10_000)


def page(
    *,
    title: str | None = "Trail Shoes | Acme",
    h1: str | None = None,
    body: str = "",
    headings: tuple[Heading, ...] = (),
) -> PageRecord:
    return PageRecord.model_validate(
        {
            "url": URL,
            "status_code": 200,
            "usable": True,
            "meta_title": title,
            "meta_description": None,
            "h1": h1,
            "headings": headings,
            "body_text": body,
            "word_count": len(body.split()),
            "link_count": 0,
            "content_hash": None,
            "body_hash": body_hash(body),
            "scraped_at": None,
            "source": "test",
            "crawl_url": f"https://{URL}",
            "language": "en",
        }
    )


def strategic(
    keyword: str, *, priority: int | None = None, primary: bool = False
) -> StrategicKeyword:
    return StrategicKeyword(
        url=URL, keyword=keyword, language="en", priority=priority, is_primary=primary
    )


def query(text: str, impressions: int, position: float) -> GscQueryStats:
    return GscQueryStats(
        url=URL, query=text, impressions=impressions, clicks=impressions // 20, position=position
    )


SHOE_PAGE = page(
    title="Trail Shoes | Acme",
    h1="Waterproof Trail Running Shoes",
    body="Our waterproof trail shoes grip wet rock on every run.",
)
SHOE_QUERIES = [
    # Already first: no upside.
    query("trail running shoes", 2000, 1.2),
    # 400 x (0.3 - 0.1) = 80.
    query("waterproof trail shoes", 400, 3),
    # 5000 x (0.3 - 0.05) = 1250, but the page never mentions boots.
    query("hiking boots", 5000, 4),
]


# ── brand_suffix and clean_title ────────────────────────────────────────────


def titles(branded: int, total: int, brand: str = "Acme", sep: str = " | ") -> list[str]:
    return [f"Page {i}{sep}{brand}" if i < branded else f"Page {i}" for i in range(total)]


@pytest.mark.parametrize(
    ("branded", "suffix"),
    [(4, "Acme"), (3, "Acme"), (2, None), (10, "Acme"), (0, None)],
    ids=["40pct", "30pct-boundary", "20pct", "all", "none"],
)
def test_the_brand_suffix_ends_at_least_30_percent_of_titles(
    branded: int, suffix: str | None
) -> None:
    assert brand_suffix(titles(branded, 10)) == suffix


def test_every_separator_counts_towards_the_same_suffix() -> None:
    mixed = [
        "Trail Shoes | Acme",
        "Rain Jackets - Acme",
        "Tents \u2013 Acme",
        "Stoves \u2014 Acme",
        "Lanterns :: Acme",
        *(f"Page {i}" for i in range(10)),
    ]
    # Five of fifteen, but only because the five separators are pooled.
    assert brand_suffix(mixed) == "Acme"


def test_the_last_segment_is_the_suffix_candidate() -> None:
    nested = [f"Shoes | Trail | Acme {i}" for i in range(2)] + ["Guides | Trail | Acme"] * 4
    assert brand_suffix([*nested, "Page a", "Page b", "Page c", "Page d"]) == "Acme"


def test_no_titles_have_no_suffix() -> None:
    assert brand_suffix([]) is None


@pytest.mark.parametrize(
    ("title", "suffix", "cleaned"),
    [
        pytest.param("Trail Shoes | Acme", "Acme", "Trail Shoes", id="pipe"),
        pytest.param("Trail Shoes \u2013 Acme", "Acme", "Trail Shoes", id="en-dash"),
        pytest.param("Trail Shoes :: Acme", "Acme", "Trail Shoes", id="double-colon"),
        pytest.param("Trail Shoes", "Acme", "Trail Shoes", id="no-suffix"),
        pytest.param("Acme Tents | Shop", "Acme", "Acme Tents | Shop", id="brand-not-last"),
        pytest.param("  Trail Shoes | Acme ", "Acme", "Trail Shoes", id="whitespace"),
        pytest.param("Trail Shoes | Acme", None, "Trail Shoes | Acme", id="tenant-without-suffix"),
    ],
)
def test_clean_title_strips_only_a_trailing_brand_suffix(
    title: str, suffix: str | None, cleaned: str
) -> None:
    assert clean_title(title, suffix) == cleaned


# ── usable_query ────────────────────────────────────────────────────────────

TEXT = "Waterproof Trail Running Shoes. Trail Shoes | Acme. Our shoes grip wet rock."


@pytest.mark.parametrize(
    ("text", "impressions", "brand", "usable"),
    [
        pytest.param("trail running shoes", 50, "Acme", True, id="at-the-impression-bar"),
        pytest.param("trail running shoes", 49, "Acme", False, id="below-the-impression-bar"),
        pytest.param("TRAIL  Running\u00a0Shoes", 500, "Acme", True, id="normalised"),
        pytest.param("trail running boots", 500, "Acme", False, id="term-not-on-page"),
        pytest.param("w trail shoes", 500, "Acme", True, id="one-letter-tokens-ignored"),
        pytest.param("acme", 5000, "Acme", False, id="only-the-brand"),
        pytest.param("ACME", 5000, "Acme", False, id="only-the-brand-any-case"),
        pytest.param("acme trail shoes", 500, "Acme", True, id="brand-plus-terms"),
        pytest.param("acme", 5000, None, True, id="no-brand-known"),
        pytest.param("", 5000, None, False, id="empty"),
    ],
)
def test_a_gsc_query_must_be_in_demand_and_extractable_from_the_page(
    text: str, impressions: int, brand: str | None, usable: bool
) -> None:
    assert usable_query(text, TEXT, impressions, brand) is usable


# ── resolve_keyword ─────────────────────────────────────────────────────────


def test_a_strategic_primary_always_wins() -> None:
    found = resolve_keyword(
        SHOE_PAGE,
        "en",
        [strategic("Rain Jackets", priority=5), strategic("Trail Running Shoes", primary=True)],
        SHOE_QUERIES,
        CURVE,
        "Acme",
    )

    assert found is not None
    assert (found.rung, found.text, found.language, found.url) == (
        KeywordRung.STRATEGIC,
        "Trail Running Shoes",
        "en",
        URL,
    )
    assert found.opportunity_value is None


@pytest.mark.parametrize(
    ("keywords", "chosen"),
    [
        pytest.param([("tents", 2), ("stoves", 5), ("boots", None)], "stoves", id="highest"),
        pytest.param([("tents", 3), ("boots", 3)], "boots", id="tie-by-text"),
        pytest.param([("tents", None), ("boots", 1)], "boots", id="any-priority-beats-none"),
        pytest.param([("tents", None)], "tents", id="unprioritised"),
    ],
)
def test_without_a_primary_the_highest_priority_strategic_keyword_wins(
    keywords: list[tuple[str, int | None]], chosen: str
) -> None:
    rows = [strategic(text, priority=priority) for text, priority in keywords]

    found = resolve_keyword(SHOE_PAGE, "en", rows, SHOE_QUERIES, CURVE, "Acme")

    assert found is not None
    assert (found.rung, found.text) == (KeywordRung.STRATEGIC, chosen)


def test_the_gsc_rung_picks_the_usable_query_with_the_most_click_upside() -> None:
    found = resolve_keyword(SHOE_PAGE, "en", [], SHOE_QUERIES, CURVE, "Acme")

    assert found is not None
    assert (found.rung, found.text) == (KeywordRung.GSC, "waterproof trail shoes")
    assert found.opportunity_value == pytest.approx(80.0)


def test_a_query_in_the_title_only_still_passes_the_bar() -> None:
    bare = page(title="Rain Jackets | Acme", h1=None, body="")

    found = resolve_keyword(bare, "en", [], [query("rain jackets", 300, 4)], CURVE, "Acme")

    assert found is not None
    assert (found.rung, found.text) == (KeywordRung.GSC, "rain jackets")


def test_without_a_curve_gsc_is_skipped_for_the_h1() -> None:
    found = resolve_keyword(SHOE_PAGE, "en", [], SHOE_QUERIES, None, "Acme")

    assert found is not None
    assert (found.rung, found.text) == (KeywordRung.H1, "Waterproof Trail Running Shoes")


@pytest.mark.parametrize(
    "queries",
    [
        pytest.param([query("hiking boots", 5000, 4)], id="not-on-the-page"),
        pytest.param([query("waterproof trail shoes", 49, 4)], id="too-few-impressions"),
        pytest.param([query("acme", 9000, 4)], id="brand-only"),
        pytest.param([], id="no-gsc-data"),
    ],
)
def test_weak_gsc_data_falls_through_to_the_h1(queries: list[GscQueryStats]) -> None:
    found = resolve_keyword(SHOE_PAGE, "en", [], queries, CURVE, "Acme")

    assert found is not None
    assert (found.rung, found.opportunity_value) == (KeywordRung.H1, None)


def test_the_h1_loses_a_trailing_brand_suffix() -> None:
    branded = page(h1="Trail Running Shoes | Acme", title="Shop | Acme")

    found = resolve_keyword(branded, "de", [], [], None, "Acme")

    assert found is not None
    assert (found.rung, found.text, found.language) == (KeywordRung.H1, "Trail Running Shoes", "de")


@pytest.mark.parametrize("h1", [None, "", "   "])
def test_without_an_h1_the_title_is_used_without_its_suffix(h1: str | None) -> None:
    found = resolve_keyword(page(h1=h1, title="Trail Shoes | Acme"), "en", [], [], None, "Acme")

    assert found is not None
    assert (found.rung, found.text) == (KeywordRung.TITLE, "Trail Shoes")


@pytest.mark.parametrize("title", [None, "", "  "])
def test_a_page_with_nothing_to_go_on_stays_unresolved(title: str | None) -> None:
    assert resolve_keyword(page(h1=None, title=title), "en", [], [], CURVE, "Acme") is None


def test_rows_of_other_pages_are_ignored() -> None:
    other = "example.com/other"
    rows = [StrategicKeyword(url=other, keyword="Rain Jackets", language="en", is_primary=True)]
    queries = [
        GscQueryStats(url=other, query="waterproof trail shoes", impressions=900, position=4)
    ]

    found = resolve_keyword(SHOE_PAGE, "en", rows, queries, CURVE, "Acme")

    assert found is not None
    assert (found.rung, found.text) == (KeywordRung.H1, "Waterproof Trail Running Shoes")


def test_a_blank_strategic_keyword_falls_through() -> None:
    found = resolve_keyword(SHOE_PAGE, "en", [strategic("   ", primary=True)], [], None, "Acme")

    assert found is not None
    assert found.rung is KeywordRung.H1


def test_a_strategic_keyword_keeps_its_own_language() -> None:
    """The resolved edge must point at the same Keyword node as the row's strategic edge."""
    row = StrategicKeyword(url=URL, keyword="Laufschuhe", language="de", is_primary=True)

    found = resolve_keyword(SHOE_PAGE, "en", [row], [], None, "Acme")

    assert found is not None
    assert (found.rung, found.text, found.language) == (KeywordRung.STRATEGIC, "Laufschuhe", "de")


# ── Amendment 1: fallbacks that survive any tenant's format ─────────────────


def test_the_fallback_constants_are_the_contracts() -> None:
    assert (REPEATED_FALLBACK_PAGES, MAX_KEYWORD_TOKENS) == (3, 12)


def test_a_brand_prefix_is_detected_like_the_suffix() -> None:
    leading = [f"Acme | Page {i}" if i < 4 else f"Page {i}" for i in range(10)]

    assert brand_prefix(leading) == "Acme"
    assert brand_suffix(leading) is None
    assert brand_affixes(leading) == ("Acme", None)
    assert brand_prefix([f"Acme | Page {i}" if i < 2 else f"Page {i}" for i in range(10)]) is None


@pytest.mark.parametrize(
    ("title", "suffix", "prefix", "cleaned"),
    [
        pytest.param("Acme | Trail Shoes", None, "Acme", "Trail Shoes", id="prefix"),
        pytest.param("Acme | Trail Shoes | Acme", "Acme", "Acme", "Trail Shoes", id="both-ends"),
        pytest.param("ACME - Trail Shoes", None, "Acme", "Trail Shoes", id="any-case"),
        pytest.param("Trail Shoes | Acme", None, "Acme", "Trail Shoes", id="prefix-found-at-end"),
        pytest.param("Shop | Trail Shoes", None, "Acme", "Shop | Trail Shoes", id="other-first"),
    ],
)
def test_clean_title_strips_a_brand_at_either_end(
    title: str, suffix: str | None, prefix: str | None, cleaned: str
) -> None:
    assert clean_title(title, suffix, prefix) == cleaned


def test_every_level_one_heading_is_an_h1_candidate_in_order() -> None:
    shown = page(
        h1="Welcome",
        headings=(
            Heading(level=1, text="Trail Shoes"),
            Heading(level=2, text="Sizing"),
            Heading(level=1, text="Trail  Shoes"),
            Heading(level=1, text="Rain Jackets"),
        ),
    )
    assert h1_candidates(shown) == ("Welcome", "Trail Shoes", "Rain Jackets")
    assert h1_candidates(page(h1=None, headings=(Heading(level=1, text="Tents"),))) == ("Tents",)


def resolved(
    record: PageRecord, **options: object
) -> tuple[ResolvedKeyword | None, tuple[str, ...]]:
    found, _, reasons = resolve_page(record, "en", [], [], None, "Acme", **options)  # type: ignore[arg-type]
    return found, reasons


def test_a_generic_first_h1_gives_way_to_the_next_level_one_heading() -> None:
    found, reasons = resolved(
        page(h1="Read more", headings=(Heading(level=1, text="Trail Running Shoes"),))
    )

    assert found is not None
    assert (found.rung, found.text) == (KeywordRung.H1, "Trail Running Shoes")
    assert reasons == ("h1_generic",)


@pytest.mark.parametrize("h1", ["2024", "123 456"])
def test_a_letter_free_h1_is_rejected_as_generic(h1: str) -> None:
    found, reasons = resolved(page(h1=h1, title="Trail Shoes | Acme"))

    assert found is not None
    assert (found.rung, found.text, reasons) == (KeywordRung.TITLE, "Trail Shoes", ("h1_generic",))


@pytest.mark.parametrize("h1", ["---", "\u00bb", "***", "..."])
def test_a_symbols_only_h1_is_rejected_as_letter_free(h1: str) -> None:
    found, reasons = resolved(page(h1=h1, title="Trail Shoes | Acme"))

    assert found is not None
    assert (found.rung, found.text) == (KeywordRung.TITLE, "Trail Shoes")
    assert reasons == ("h1_generic",)


def test_an_h1_that_is_only_the_brand_is_rejected() -> None:
    found, reasons = resolved(page(h1="ACME", title="Trail Shoes | Acme"))

    assert found is not None
    assert (found.rung, reasons) == (KeywordRung.TITLE, ("h1_brand",))


def test_fallbacks_over_at_least_three_different_bodies_are_repeated_templates() -> None:
    pages = [
        (("Our Products", "Tents | Acme"), "body-1"),
        (("Our Products", "Stoves | Acme"), "body-2"),
        (("our  products", "Boots | Acme"), "body-3"),
        (("Unique", "Tents | Acme"), "body-4"),
        (("Sale | Acme", "Sale"), "body-5"),
    ]

    assert repeated_fallbacks(pages, "Acme") == {"our products"}


def test_one_article_served_at_several_urls_keeps_its_heading() -> None:
    # The same body at four urls is one page published four times, not a template.
    same = [(("What is Patching?", "What is Patching? | Acme"), "body-1") for _ in range(4)]
    listing = [(("Blog", "Blog | Acme"), f"page-{i}") for i in range(3)]

    assert repeated_fallbacks([*same, *listing], "Acme") == {"blog"}


def test_a_repeated_h1_falls_through_and_a_repeated_title_leaves_the_page_unresolved() -> None:
    found, reasons = resolved(
        page(h1="Our Products", title="Tents | Acme"), repeated={"our products"}
    )
    assert found is not None
    assert (found.rung, found.text, reasons) == (KeywordRung.TITLE, "Tents", ("h1_repeated",))

    found, reasons = resolved(
        page(h1="Our Products", title="Tents | Acme"), repeated={"our products", "tents"}
    )
    assert (found, reasons) == (None, ("h1_repeated", "title_repeated"))


def test_the_tenants_generic_overrides_apply_to_fallbacks() -> None:
    blocked, reasons = resolved(
        page(h1="Trail Shoes", title="Rain Jackets | Acme"), generic_add=frozenset({"trail shoes"})
    )
    assert blocked is not None
    assert (blocked.text, reasons) == ("Rain Jackets", ("h1_generic",))

    allowed, _ = resolved(page(h1="Read more"), generic_remove=frozenset({"read more"}))
    assert allowed is not None
    assert (allowed.rung, allowed.text) == (KeywordRung.H1, "Read more")


def test_missing_fallbacks_are_recorded() -> None:
    assert resolved(page(h1=None, title=None)) == (None, ("h1_missing", "title_missing"))
    found, reasons = resolved(page(h1=None, title="Tents | Acme"))
    assert found is not None
    assert (found.text, reasons) == ("Tents", ("h1_missing",))


@pytest.mark.parametrize(
    ("words", "rung", "long"),
    [(13, KeywordRung.H1, True), (12, KeywordRung.H1, False), (13, KeywordRung.STRATEGIC, False)],
)
def test_only_long_h1_or_title_fallbacks_count_as_long(
    words: int, rung: KeywordRung, long: bool
) -> None:
    keyword = ResolvedKeyword(url=URL, text=" ".join(["word"] * words), language="en", rung=rung)
    assert is_long(keyword) is long


def test_a_missing_title_or_cleaned_candidate_is_reported_missing() -> None:
    assert clean_title(None, "Acme") is None
    assert fallback_reason(None, frozenset(), frozenset()) == "missing"


@pytest.mark.parametrize(
    "h1", ["Home", "Welcome", "Untitled", "Startseite", "Willkommen", "Accueil", "Bienvenue"]
)
def test_placeholder_headings_in_any_language_are_generic(h1: str) -> None:
    found, reasons = resolved(page(h1=h1, title="Trail Shoes | Acme"))

    assert found is not None
    assert (found.rung, found.text, reasons) == (KeywordRung.TITLE, "Trail Shoes", ("h1_generic",))


def test_the_tenant_can_keep_a_placeholder_heading() -> None:
    found, reasons = resolved(page(h1="Home"), generic_remove=frozenset({"home"}))

    assert found is not None
    assert (found.rung, found.text, reasons) == (KeywordRung.H1, "Home", ())


# ── Amendment 2: secondary GSC queries of the ranked keyword set ────────────

WIDE_PAGE = page(
    title="Trail Shoes | Acme",
    h1="Trail Running Shoes",
    body="Waterproof trail shoes, trail gaiters, trail poles, trail maps, trail lights and tents.",
)


def ranked_queries(*rows: tuple[str, int]) -> list[GscQueryStats]:
    """All at position 3, so opportunity follows impressions."""
    return [query(text, impressions, 3) for text, impressions in rows]


def test_at_most_four_secondary_queries_by_opportunity() -> None:
    rows = ranked_queries(
        ("trail lights", 300),
        ("trail gaiters", 600),
        ("waterproof trail shoes", 700),
        ("trail maps", 400),
        ("trail poles", 500),
    )

    found = secondary_queries(WIDE_PAGE, rows, CURVE, ["Trail Running Shoes"])

    assert MAX_SECONDARY_QUERIES == 4
    assert found == ("waterproof trail shoes", "trail gaiters", "trail poles", "trail maps")
    assert secondary_queries(WIDE_PAGE, rows, CURVE, [], limit=2) == (
        "waterproof trail shoes",
        "trail gaiters",
    )


def test_secondary_queries_skip_the_pages_keywords_and_their_own_duplicates() -> None:
    rows = ranked_queries(
        ("TRAIL RUNNING  shoes", 900),
        ("Tents", 800),
        ("trail gaiters", 600),
        ("Trail  Gaiters", 550),
    )

    found = secondary_queries(WIDE_PAGE, rows, CURVE, ["Trail Running Shoes", "tents"])

    assert found == ("trail gaiters",)


def test_secondary_queries_pass_the_same_quality_bar() -> None:
    rows = ranked_queries(("hiking boots", 5000), ("trail maps", 49), ("acme", 900), ("tents", 60))
    assert secondary_queries(WIDE_PAGE, rows, CURVE, [], suffix="Acme") == ("tents",)


def test_other_pages_queries_are_never_secondaries() -> None:
    other = GscQueryStats(url="example.com/other", query="tents", impressions=900, position=3)
    assert secondary_queries(WIDE_PAGE, [other], CURVE, []) == ()


def test_without_a_curve_there_are_no_secondary_queries() -> None:
    assert secondary_queries(WIDE_PAGE, ranked_queries(("tents", 900)), None, []) == ()


def test_duplicate_strategic_rows_merge_per_page_keyword_and_language() -> None:
    rows = [
        StrategicKeyword(url=URL, keyword="Trail Shoes", language="en", priority=2),
        StrategicKeyword(url=URL, keyword="trail  shoes", language="en", priority=5),
        StrategicKeyword(url=URL, keyword="TRAIL   SHOES", language="en", is_primary=True),
        StrategicKeyword(url=URL, keyword="Trail Shoes", language="de", priority=1),
        StrategicKeyword(url=URL, keyword="tents", language="en", priority=4),
        StrategicKeyword(url=URL, keyword="   ", language="en", priority=5),
        StrategicKeyword(url=URL, keyword="stoves", language="e", priority=5),
        StrategicKeyword(url="example.com/a", keyword="boots", language="en"),
    ]

    merged = merge_strategic(rows)

    # Highest priority, primary if any row is, the best-ranked row's text with whitespace
    # collapsed; ordered by url, then primary, priority and text.
    assert [(r.url, r.keyword, r.language, r.priority, r.is_primary) for r in merged] == [
        ("example.com/a", "boots", "en", None, False),
        (URL, "TRAIL SHOES", "en", 5, True),
        (URL, "tents", "en", 4, False),
        (URL, "Trail Shoes", "de", 1, False),
    ]
