"""The extraction ladder: exact, stemmed with modifier drop, then stem sets, on real Snowball
stemmers, with sentence and character offsets that point into the stored body."""

from __future__ import annotations

from importlib import resources

import pytest

from linking_engine.anchor.extraction import (
    MAX_INNER_STOP_WORDS,
    MAX_SPAN,
    MIN_SHARED_STEMS,
    MIN_SPAN,
    SNOWBALL,
    SourceIndex,
    Stems,
    algorithm_for,
    anchor_report,
    existing_spans,
    extract,
    keyword_tokens,
    sentences,
    stop_words_for,
    summarise_anchors,
    tokens,
)
from linking_engine.models import AnchorMatch, AnchorRung, KeywordSource

EXACT, STEMMED, STEM_SET = AnchorRung.EXACT, AnchorRung.STEMMED, AnchorRung.STEM_SET
SOURCE, TARGET = "example.com/guide", "example.com/target"


def index(body: str, *, language: str | None = "en", headings: tuple[str, ...] = ()) -> SourceIndex:
    return SourceIndex(SOURCE, body, headings, Stems(language))


def find(
    body: str,
    keyword: str,
    *,
    language: str | None = "en",
    headings: tuple[str, ...] = (),
    existing: tuple[tuple[int, int], ...] = (),
    threshold: float = 0.5,
) -> AnchorMatch | None:
    matches, _, _ = extract(
        index(body, language=language, headings=headings),
        TARGET,
        [(1, keyword, KeywordSource.CLIENT_STRATEGIC)],
        existing=existing,
        threshold=threshold,
    )
    assert len(matches) <= 1
    return matches[0] if matches else None


def assert_at_offsets(found: AnchorMatch, body: str) -> None:
    assert body[found.start : found.end] == found.phrase
    assert body[found.sentence_start : found.sentence_start + len(found.sentence)] == found.sentence


def test_the_span_constants_are_the_contracts() -> None:
    assert (MIN_SPAN, MAX_SPAN, MIN_SHARED_STEMS, MAX_INNER_STOP_WORDS) == (2, 5, 2, 1)


# ── issue acceptance ────────────────────────────────────────────────────────


def test_the_verbatim_keyword_resolves_at_rung_one_whatever_its_case() -> None:
    body = "Our range grows every year. The Hydraulic Press is our flagship."

    found = find(body, "hydraulic press")

    assert found is not None
    assert (found.rung, found.phrase, found.sentence_index) == (EXACT, "Hydraulic Press", 1)
    assert found.sentence == "The Hydraulic Press is our flagship."
    assert_at_offsets(found, body)


@pytest.mark.parametrize(
    ("keyword", "body", "phrase"),
    [
        pytest.param("trail shoe", "Our trail shoes grip wet rock.", "trail shoes", id="plural"),
        pytest.param("trail running", "We trail run every weekend.", "trail run", id="-ing"),
        pytest.param("painted wall", "Paint walls before winter.", "Paint walls", id="-ed"),
    ],
)
def test_plural_ing_and_ed_forms_resolve_at_rung_two(keyword: str, body: str, phrase: str) -> None:
    found = find(body, keyword)

    assert found is not None
    assert (found.rung, found.phrase) == (STEMMED, phrase)
    assert_at_offsets(found, body)


@pytest.mark.parametrize(
    ("language", "keyword", "body", "phrase"),
    [
        pytest.param("de", "haus", "Unsere Häuser stehen am See.", "Häuser", id="german"),
        pytest.param(
            "fr",
            "chaussure de randonnée",
            "Nos chaussures de randonnées tiennent.",
            "chaussures de randonnées",
            id="french",
        ),
        pytest.param(
            "es", "correr", "Seguimos corriendo por la montaña.", "corriendo", id="spanish"
        ),
    ],
)
def test_each_page_uses_its_languages_stemmer(
    language: str, keyword: str, body: str, phrase: str
) -> None:
    found = find(body, keyword, language=language)

    assert found is not None
    assert (found.rung, found.phrase) == (STEMMED, phrase)


@pytest.mark.parametrize(
    ("keyword", "body"),
    [
        pytest.param("haus", "Unsere Häuser stehen am See.", id="german-text"),
        pytest.param("correr", "Seguimos corriendo por la montaña.", id="spanish-text"),
    ],
)
def test_the_wrong_languages_stemmer_would_miss_the_variant(keyword: str, body: str) -> None:
    assert find(body, keyword, language="en") is None


@pytest.mark.parametrize("language", ["ja", None, "zz-ZZ"])
def test_a_language_without_a_stemmer_compares_casefolded_tokens(language: str | None) -> None:
    body = "Our Trail Shoes grip. Other trail shoe here."

    exact = find(body, "trail shoes", language=language)
    assert exact is not None
    assert (exact.rung, exact.phrase) == (EXACT, "Trail Shoes")
    # No stemmer: the singular keyword never meets the plural.
    plural = find("Our trail shoes grip.", "trail shoe", language=language)
    assert plural is None


@pytest.mark.parametrize(
    ("language", "algorithm"),
    [
        ("en", "english"),
        ("de-AT", "german"),
        ("pt_BR", "portuguese"),
        ("NB", "norwegian"),
        ("nn", "norwegian"),
        ("ja", None),
        ("", None),
        (None, None),
    ],
)
def test_the_stemmer_follows_the_primary_language_subtag(
    language: str | None, algorithm: str | None
) -> None:
    assert algorithm_for(language) == algorithm
    assert Stems(language).stemmed is (algorithm is not None)


@pytest.mark.parametrize(
    "body",
    ["The pressure rises fast.", "Impressive results.", "A pressman works here."],
)
def test_no_match_inside_a_larger_word(body: str) -> None:
    assert find(body, "press") is None


@pytest.mark.parametrize(
    ("keyword", "found"),
    [
        pytest.param("press-fit bearing", (EXACT, "press-fit bearing"), id="compound-as-written"),
        # Rungs 1-2 need the keyword's own gap; the whole compound shares every stem.
        pytest.param("press fit", (STEM_SET, "press-fit"), id="whole-compound"),
        pytest.param("press", None, id="ends-inside"),
        pytest.param("fit bearing", None, id="starts-inside"),
    ],
)
def test_a_phrase_never_cuts_a_hyphenated_compound(
    keyword: str, found: tuple[AnchorRung, str] | None
) -> None:
    match = find("A press-fit bearing.", keyword, threshold=0.01)
    assert ((match.rung, match.phrase) if match else None) == found


def test_no_rung_cuts_a_compound_even_where_the_stems_would_match() -> None:
    assert find("A hydraulic-press brake.", "press brake", threshold=0.01) is None
    found = find("A hydraulic-press brake.", "hydraulic press brake")
    assert found is not None
    assert (found.rung, found.phrase) == (STEM_SET, "hydraulic-press brake")
    # "CVE-2026" would share two stems, but only inside the identifiers.
    assert (
        find(
            "Issues CVE-2026-20002 and CVE-2026-30003 are critical.",
            "CVE-2026-10001 \u2013 Widget",
            threshold=0.01,
        )
        is None
    )


# ── the ladder in detail ────────────────────────────────────────────────────


def test_within_a_rung_the_earliest_sentence_then_offset_wins() -> None:
    body = "Tents first. Trail shoes and trail shoes here. Trail shoes again."

    found = find(body, "trail shoes")

    assert found is not None
    assert (found.sentence_index, found.start) == (1, body.index("Trail shoes and"))


def test_a_lower_rung_beats_an_earlier_sentence() -> None:
    body = "We sell trail shoe models. Our trail shoes grip."

    found = find(body, "trail shoes")

    assert found is not None
    assert (found.rung, found.sentence_index, found.phrase) == (EXACT, 1, "trail shoes")


def test_heading_lines_never_hold_an_anchor_but_keep_their_place() -> None:
    body = "Trail Shoes\nOur trail shoes grip wet rock."

    found = find(body, "trail shoes", headings=("  trail   SHOES ",))

    assert found is not None
    assert (found.sentence_index, found.sentence) == (1, "Our trail shoes grip wet rock.")
    assert find("Trail Shoes", "trail shoes", headings=("Trail Shoes",)) is None


@pytest.mark.parametrize(
    ("body", "found"),
    [
        pytest.param("Get trail  shoes now.", (EXACT, "trail  shoes"), id="whitespace"),
        pytest.param("Get trail-shoes now.", (STEM_SET, "trail-shoes"), id="hyphen"),
        pytest.param("Get trail/shoes now.", (STEM_SET, "trail/shoes"), id="slash"),
        pytest.param("Get trail - shoes now.", None, id="spaced-hyphen"),
        pytest.param("Get trail\nshoes now.", None, id="line-break"),
        pytest.param("Get trail, shoes now.", None, id="comma"),
        pytest.param("Get trail -- shoes now.", None, id="double-hyphen"),
    ],
)
def test_rungs_one_and_two_join_a_phrase_by_whitespace_only(
    body: str, found: tuple[AnchorRung, str] | None
) -> None:
    match = find(body, "trail shoes", threshold=0.01)
    assert ((match.rung, match.phrase) if match else None) == found


@pytest.mark.parametrize(
    ("keyword", "body", "phrase"),
    [
        pytest.param(
            "industrial hydraulic press", "Our hydraulic press line.", "hydraulic press", id="head"
        ),
        pytest.param(
            "hydraulic press machine", "Our hydraulic press line.", "hydraulic press", id="tail"
        ),
    ],
)
def test_a_modifier_drops_at_either_end_of_a_long_keyword(
    keyword: str, body: str, phrase: str
) -> None:
    found = find(body, keyword)
    assert found is not None
    assert (found.rung, found.phrase) == (STEMMED, phrase)


def test_two_token_keywords_drop_nothing() -> None:
    assert find("Our press line.", "hydraulic press") is None


@pytest.mark.parametrize(
    ("body", "keyword"),
    [
        pytest.param("What is included in the price.", "what is seo", id="what-is"),
        pytest.param("Learn how to cook rice today.", "how to install", id="how-to"),
    ],
)
def test_a_dropped_modifier_never_leaves_only_stop_words(body: str, keyword: str) -> None:
    assert find(body, keyword, threshold=0.01) is None


def test_a_dropped_modifier_never_leaves_a_stop_word_at_an_edge() -> None:
    found = find("The hydraulic press for the plant.", "hydraulic press for sale", threshold=0.6)

    # Not "hydraulic press for" at rung two; the stem set rung finds the content words.
    assert found is not None
    assert (found.rung, found.phrase) == (STEM_SET, "hydraulic press")


def test_a_dropped_modifier_may_keep_a_stop_word_inside() -> None:
    found = find("Our tents for winter ship now.", "best tents for winter")

    assert found is not None
    assert (found.rung, found.phrase) == (STEMMED, "tents for winter")


def test_a_compound_is_one_word_so_e_commerce_never_stands_for_the_keyword() -> None:
    # Not a dropped modifier, and on the stem set rung one shared word of two.
    assert find("We build e-commerce sites.", "e-commerce platform", threshold=0.01) is None
    found = find("We build e-commerce platforms.", "e-commerce platform")
    assert found is not None
    assert (found.rung, found.phrase) == (STEMMED, "e-commerce platforms")


def test_a_compound_holding_every_content_stem_passes_the_stem_set_rung() -> None:
    found = find("Our range of press-fit parts.", "press fit")

    assert found is not None
    assert (found.rung, found.phrase, found.stem_jaccard) == (STEM_SET, "press-fit", 1.0)


def test_the_stem_set_rung_catches_what_the_stemmed_rung_misses() -> None:
    body = "Tune the press for hydraulic loads."

    found = find(body, "industrial hydraulic press")

    # The stop word "for" counts in neither set: press, hydraul against industri, hydraul, press.
    assert found is not None
    assert (found.rung, found.phrase) == (STEM_SET, "press for hydraulic")
    assert found.stem_jaccard == pytest.approx(2 / 3)
    assert_at_offsets(found, body)


@pytest.mark.parametrize(("threshold", "found"), [(0.5, True), (0.51, False)])
def test_the_stem_set_threshold_is_inclusive(threshold: float, found: bool) -> None:
    # press, heavi, hydraul against industri, hydraul, press: 2 shared of 4.
    match = find(
        "Set the press to heavy hydraulic mode.", "industrial hydraulic press", threshold=threshold
    )
    got = (match.phrase, match.stem_jaccard) if match else None
    assert got == (("press to heavy hydraulic", 0.5) if found else None)


def test_a_stem_set_span_holds_at_most_one_stop_word() -> None:
    # "press for the hydraulic" would score 2 of 3 but holds two stop words.
    assert find("Tune the press for the hydraulic loads.", "industrial hydraulic press") is None
    # "install updates when the System" would score 1.
    shorter = find(
        "Always install updates when the System is idle.",
        "how to install a system update",
        threshold=0.6,
    )
    assert shorter is not None
    assert shorter.phrase == "install updates"
    assert shorter.stem_jaccard == pytest.approx(2 / 3)
    kept = find("We cover log rotation for Linux hosts.", "log rotation in linux", threshold=0.6)
    assert kept is not None
    assert (kept.rung, kept.phrase, kept.stem_jaccard) == (
        STEM_SET,
        "log rotation for Linux",
        1.0,
    )


def test_a_stem_set_span_never_starts_or_ends_on_a_stop_word() -> None:
    body = "If System Update cannot install a required update, retry."

    found = find(body, "how to install a system update", threshold=0.6)

    # Not "System Update cannot install a": the keyword's stop words are no stems to end on.
    assert found is not None
    assert (found.rung, found.phrase, found.stem_jaccard) == (
        STEM_SET,
        "System Update cannot install",
        1.0,
    )
    assert_at_offsets(found, body)


def test_exact_and_stemmed_match_a_keywords_stop_words_as_written() -> None:
    body = "Buy tents for the winter now."
    exact = find(body, "tents for the winter")
    stemmed = find(body, "tent for the winters")
    assert exact is not None
    assert stemmed is not None
    assert (exact.rung, stemmed.rung) == (EXACT, STEMMED)
    assert exact.phrase == stemmed.phrase == "tents for the winter"


def test_a_keyword_with_one_content_stem_skips_the_stem_set_rung() -> None:
    assert find("Read about hydraulic press.", "about the press", threshold=0.01) is None


@pytest.mark.parametrize("language", [None, "ja"])
def test_a_language_without_a_stop_word_list_counts_every_word(language: str | None) -> None:
    found = find(
        "Tune the press for hydraulic loads.", "industrial hydraulic press", language=language
    )

    # "for" counts here: press, for, hydraulic against industrial, hydraulic, press.
    assert found is not None
    assert (found.phrase, found.stem_jaccard) == ("press for hydraulic", 0.5)


def test_stop_words_never_make_a_stem_set_span() -> None:
    # Only "tents" is a content word with a keyword stem; "for" and "the" never count.
    assert find("Read the guide for the tents.", "tents for the winter") is None


def test_a_stem_set_span_needs_two_of_the_keywords_stems() -> None:
    # The span "press press" has Jaccard 0.5 with {hydraul, press} but shares one stem.
    assert find("The press press works.", "hydraulic press") is None


def test_a_keyword_with_one_distinct_stem_skips_the_stem_set_rung() -> None:
    # "press presses" has one distinct stem, so no span can share two with it.
    assert find("The press machine.", "press presses", threshold=0.01) is None


def test_the_best_stem_set_span_has_the_highest_jaccard_then_the_earliest_sentence() -> None:
    body = "Industrial steel hydraulic parts. Industrial and hydraulic kits. Press industrial sets."

    found = find(body, "industrial hydraulic press")

    # The "steel" span shares 2 of 4 stems; the "and" span 2 of 3, as "and" is a stop word.
    assert found is not None
    assert (found.sentence_index, found.phrase) == (1, "Industrial and hydraulic")
    assert found.stem_jaccard == pytest.approx(2 / 3)
    tie = find(
        "Press industrial kits. Industrial and hydraulic sets.", "industrial hydraulic press"
    )
    assert tie is not None
    assert (tie.sentence_index, tie.phrase) == (0, "Press industrial")


def test_a_stem_set_span_starts_and_ends_on_keyword_stems() -> None:
    found = find("See the hydraulic and industrial systems.", "industrial hydraulic press")

    assert found is not None
    assert found.phrase == "hydraulic and industrial"


def test_an_existing_anchor_pushes_the_match_to_the_next_occurrence() -> None:
    body = "Trail shoes rock. Buy trail shoes today."
    first = (0, len("Trail shoes"))

    matches, skipped, _ = extract(
        index(body), TARGET, [(1, "trail shoes", KeywordSource.CLIENT_STRATEGIC)], existing=(first,)
    )

    [found] = matches
    assert (found.sentence_index, found.phrase, skipped) == (1, "trail shoes", frozenset({first}))
    covered = (first, (body.index("trail shoes today"), body.index(" today")))
    none, skipped_all, _ = extract(
        index(body), TARGET, [(1, "trail shoes", KeywordSource.CLIENT_STRATEGIC)], existing=covered
    )
    assert (none, skipped_all) == ([], frozenset(covered))


def test_an_existing_anchor_blocking_several_keywords_counts_once() -> None:
    body = "Trail shoes rock. Buy trail shoes today."
    first = (0, len("Trail shoes"))
    unrelated = (body.index("today"), len(body) - 1)
    keywords = [
        (1, "trail shoes", KeywordSource.CLIENT_STRATEGIC),
        (2, "trail shoe", KeywordSource.GSC_OBSERVED),
    ]

    matches, blocking, _ = extract(index(body), TARGET, keywords, existing=(first, unrelated))

    assert [(m.keyword, m.sentence_index) for m in matches] == [
        ("trail shoes", 1),
        ("trail shoe", 1),
    ]
    assert blocking == {first}


def test_each_found_keyword_is_one_match_carrying_its_rank_and_source() -> None:
    body = "Our trail shoes grip. Tents stand firm."
    keywords = [
        (1, "trail shoes", KeywordSource.CLIENT_STRATEGIC),
        (2, "tent", KeywordSource.GSC_OBSERVED),
        (3, "stoves", KeywordSource.INFERRED),
    ]

    matches, skipped, _ = extract(index(body), TARGET, keywords)

    assert [(m.keyword, m.keyword_rank, m.keyword_source, m.rung) for m in matches] == [
        ("trail shoes", 1, KeywordSource.CLIENT_STRATEGIC, EXACT),
        ("tent", 2, KeywordSource.GSC_OBSERVED, STEMMED),
    ]
    assert skipped == frozenset()
    for found in matches:
        assert (found.source_url, found.target_url) == (SOURCE, TARGET)
        assert_at_offsets(found, body)


@pytest.mark.parametrize("threshold", [0.0, 1.01])
def test_a_threshold_outside_the_unit_interval_is_refused(threshold: float) -> None:
    with pytest.raises(ValueError, match="threshold"):
        extract(index("text"), TARGET, [], threshold=threshold)


# ── casefolding, headings and gaps ──────────────────────────────────────────


def test_a_dotted_capital_i_matches_a_plain_i() -> None:
    body = "\u0130stanbul otelleri burada."

    found = find(body, "istanbul otelleri", language="tr")

    assert found is not None
    assert (found.rung, found.phrase) == (EXACT, "\u0130stanbul otelleri")
    assert_at_offsets(found, body)


@pytest.mark.parametrize(
    ("heading", "keyword"),
    [
        pytest.param("Step 1. Install the press", "install the press", id="numbered-step"),
        pytest.param("What is torque? A torque guide", "torque guide", id="question"),
    ],
)
def test_a_heading_with_sentence_punctuation_yields_nothing(heading: str, keyword: str) -> None:
    assert find(f"{heading}\nSee the bench.", keyword, headings=(heading,)) is None


def test_sentences_after_a_heading_keep_their_place() -> None:
    heading = "Step 1. Install the press"
    body = f"{heading}\nInstall the press on the bench."

    found = find(body, "install the press", headings=(heading,))

    assert found is not None
    assert (found.sentence, found.start) == ("Install the press on the bench.", len(heading) + 1)
    # Both sentences of the heading keep their place.
    assert found.sentence_index == 2


@pytest.mark.parametrize(
    ("body", "keyword", "phrase"),
    [
        pytest.param(
            "Fast Node.js hosting for teams.", "node.js hosting", "Node.js hosting", id="dot"
        ),
        pytest.param("Every 3.5mm jack is here.", "3.5mm jack", "3.5mm jack", id="decimal"),
        pytest.param("Tips & Tricks for gardening.", "tips & tricks", "Tips & Tricks", id="amp"),
        pytest.param(
            "Tips & Tricks for gardening.", "tips  &  tricks", "Tips & Tricks", id="spaces"
        ),
        pytest.param("A C# tutorial for beginners.", "c# tutorial", "C# tutorial", id="hash"),
    ],
)
def test_a_keyword_with_inner_punctuation_matches_verbatim(
    body: str, keyword: str, phrase: str
) -> None:
    found = find(body, keyword)

    assert found is not None
    assert (found.rung, found.phrase) == (EXACT, phrase)


@pytest.mark.parametrize(
    ("body", "keyword"),
    [
        pytest.param("Fast Node.js hosting for teams.", "js hosting", id="after-a-dot"),
        pytest.param("Every 3.5mm jack is here.", "5mm jack", id="after-a-decimal-point"),
        pytest.param("A C# tutorial for beginners.", "c tutorial", id="gap-unlike-the-keyword"),
    ],
)
def test_no_phrase_starts_inside_a_compound_or_skips_a_gap(body: str, keyword: str) -> None:
    assert find(body, keyword, threshold=0.01) is None


def test_a_spaced_hyphen_ends_the_phrase_like_a_dash() -> None:
    assert (
        find("Buy the press - hydraulic models ship fast.", "press hydraulic", threshold=0.01)
        is None
    )


# ── sentences, tokens and existing spans ────────────────────────────────────


def test_sentences_split_at_line_breaks_and_closing_punctuation_with_their_offsets() -> None:
    body = "\n  First one. Second?  Third!\nFourth\u2026 Fifth 3.5 kg.\n\nSixth"

    found = sentences(body)

    assert [(s.index, s.text) for s in found] == [
        (0, "First one."),
        (1, "Second?"),
        (2, "Third!"),
        (3, "Fourth\u2026"),
        (4, "Fifth 3.5 kg."),
        (5, "Sixth"),
    ]
    assert all(body[s.start : s.start + len(s.text)] == s.text for s in found)


@pytest.mark.parametrize(
    ("language", "listed", "unlisted"),
    [
        pytest.param("en", {"for", "the", "and", "don't"}, {"die", "press"}, id="english"),
        pytest.param("de-AT", {"die", "der", "und"}, {"for", "zelt"}, id="german-subtag"),
    ],
)
def test_stop_words_follow_the_languages_stemmer(
    language: str, listed: set[str], unlisted: set[str]
) -> None:
    words = stop_words_for(language)
    assert listed <= words
    assert not unlisted & words


@pytest.mark.parametrize("language", ["ja", None, "", "tr"])
def test_a_language_without_a_stop_word_list_has_none(language: str | None) -> None:
    # Turkish has a stemmer but no packaged list.
    assert stop_words_for(language) == frozenset()


LISTED = ("da", "de", "en", "es", "fi", "fr", "ga", "hu", "id", "it", "nb", "nl", "pt", "ru", "sv")


def test_the_stop_word_lists_load_for_all_fifteen_languages_under_their_license() -> None:
    assert len({SNOWBALL[language] for language in LISTED}) == 15
    loaded = {language: stop_words_for(language) for language in LISTED}
    assert [language for language, words in loaded.items() if not words] == []
    # Keys compare like tokens: casefolded, with straight apostrophes.
    keys = set().union(*loaded.values())
    assert keys == {key.casefold() for key in keys}
    assert not [key for key in keys if set(key) & set("\u2019\u2018\u02bc\u2032")]
    listed = resources.files("linking_engine.anchor").joinpath("stopwords")
    assert listed.joinpath("LICENSE").is_file()


def test_tokens_are_casefolded_words_with_offsets_split_at_hyphens() -> None:
    found = tokens("Don't Stop-Me l\u2019Eau", offset=10)

    # A typographic apostrophe compares as "'".
    assert [t.folded for t in found] == ["don't", "stop", "me", "l'eau"]
    assert [(t.start, t.end) for t in found] == [(10, 15), (16, 20), (21, 23), (24, 29)]
    assert keyword_tokens("  Trail   SHOES ") == ["trail", "shoes"]


@pytest.mark.parametrize("apostrophe", ["\u2019", "\u2018", "\u02bc", "\u2032"])
def test_every_typographic_apostrophe_matches_a_straight_one_at_rung_one(apostrophe: str) -> None:
    found = find(f"Yes, it{apostrophe}s ready.", "it's")

    assert found is not None
    assert (found.rung, found.phrase) == (EXACT, f"it{apostrophe}s")


def test_a_typographic_possessive_stems_like_a_plural() -> None:
    body = "Read the organization\u2019s policy today."

    found = find(body, "organization policies")

    assert found is not None
    assert (found.rung, found.phrase) == (STEMMED, "organization\u2019s policy")
    assert_at_offsets(found, body)


def test_a_straight_apostrophe_keyword_matches_a_typographic_one_as_written() -> None:
    body = "Don\u2019t stop at the summit."

    found = find(body, "don't stop")

    assert found is not None
    assert (found.rung, found.phrase) == (EXACT, "Don\u2019t stop")


def test_existing_spans_locate_each_anchor_inside_its_surrounding_text() -> None:
    body = "Intro. Read our trail shoes guide today. Tents too."
    links = [
        ("trail shoes", "Read our trail shoes guide today."),
        ("trail shoes", "Read our trail shoes guide today."),
        ("stoves", "Read our trail shoes guide today."),
        ("tents", "Not in the body at all."),
        ("", "Intro."),
    ]

    spans = existing_spans(body, links)

    start = body.index("trail shoes")
    assert spans == [(start, start + len("trail shoes"))]


# ── the report ──────────────────────────────────────────────────────────────


def match(
    source: str,
    rung: AnchorRung,
    rank: int,
    *,
    sentence_index: int = 0,
    jaccard: float | None = None,
) -> AnchorMatch:
    sentence = "Our trail shoes grip."
    return AnchorMatch(
        source_url=f"example.com/{source}",
        target_url=TARGET,
        keyword="trail shoes",
        keyword_rank=rank,
        keyword_source=KeywordSource.CLIENT_STRATEGIC,
        rung=rung,
        phrase="trail shoes",
        start=4,
        end=15,
        sentence=sentence,
        sentence_index=sentence_index,
        sentence_start=0,
        stem_jaccard=jaccard,
    )


def test_the_report_counts_rungs_ranks_best_rungs_histograms_and_languages() -> None:
    matches = [
        match("a", STEMMED, 1, sentence_index=0),
        match("a", EXACT, 2, sentence_index=2),
        match("b", STEM_SET, 2, sentence_index=3, jaccard=0.5),
        match("b", STEM_SET, 3, sentence_index=5, jaccard=1.0),
        match("c", STEMMED, 2, sentence_index=6),
        match("c", EXACT, 4, sentence_index=11),
        match("d", EXACT, 1, sentence_index=21),
        match("d", EXACT, 2, sentence_index=500),
    ]

    report = anchor_report(
        "acme",
        matches,
        threshold=0.5,
        pairs=6,
        bridge_pairs=2,
        pairs_with_keywords=5,
        overlapping=3,
        located_anchors=4,
        unlocated_anchors=1,
        keywords=[
            ("trail shoes", "en"),
            ("trail shoes", None),
            ("tents", "en"),
            ("tents", "en"),
            ("Stoves", "ja"),
            ("the stoves", "en"),
            # Two content stems in English, one in German, where "die" is a stop word.
            ("die Zelte", "en"),
            ("die Zelte", "de"),
        ],
        source_languages={
            "example.com/a": "en",
            "example.com/b": "de-AT",
            "example.com/c": "ja",
            "example.com/d": None,
            "example.com/e": "en",
        },
        sources_without_body=2,
        missing_sources=1,
        started=0.0,
    )

    assert (report.pairs_matched, report.primary_matched, report.matches) == (4, 2, 8)
    assert report.by_rung == {EXACT: 4, STEMMED: 2, STEM_SET: 2}
    # a: EXACT beats STEMMED; b: STEM_SET only; c and d: EXACT.
    assert report.best_rung == {EXACT: 3, STEMMED: 0, STEM_SET: 1}
    assert report.by_keyword_rank == {1: 2, 2: 4, 3: 1, 4: 1}
    assert report.by_rung_and_rank == {
        EXACT: {1: 1, 2: 2, 4: 1},
        STEMMED: {1: 1, 2: 1},
        STEM_SET: {2: 1, 3: 1},
    }
    assert report.stem_jaccard_histogram == (*([0] * 10), 1, *([0] * 8), 1)
    # Bins 0, 1, 2, 3-5, 6-10, 11-20, 21+.
    assert report.sentence_index_histogram == (1, 0, 1, 2, 1, 1, 2)
    # tents, Stoves, the stoves and die Zelte, each counted once.
    assert report.single_token_keywords == 4
    # One more source page is not stored at all: it counts as a source without a body only.
    assert (report.source_pages, report.sources_without_body) == (6, 2)
    assert report.stemmed_languages == {"de-AT": 1, "en": 2}
    assert report.unstemmed_languages == {"ja": 1, "und": 1}
    assert (report.overlapping_existing_anchors, report.bridge_pairs) == (3, 2)
    assert (report.existing_anchors_located, report.existing_anchors_unlocated) == (4, 1)


def test_the_summary_states_the_counts_without_urls_or_phrases() -> None:
    report = anchor_report(
        "acme",
        [match("a", EXACT, 1)],
        threshold=0.5,
        pairs=2,
        bridge_pairs=1,
        pairs_with_keywords=2,
        overlapping=0,
        located_anchors=0,
        unlocated_anchors=0,
        keywords=[("trail shoes", "en")],
        source_languages={"example.com/a": "en"},
        sources_without_body=0,
        identifier_mismatches=4,
        started=0.0,
    )

    summary = summarise_anchors(report)

    assert report.identifier_mismatches == 4
    facts = ("acme", "2 pairs", "1 from hub bridges", "1 pairs matched", "Stemmed: en 1")
    for fact in (*facts, "4 stemmed or stem set places refused"):
        assert fact in summary, f"{fact!r} missing from:\n{summary}"
    assert not [
        text
        for text in ("example.com/a", "trail shoes", "Our trail shoes grip.")
        if text in summary
    ]
