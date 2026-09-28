"""Anchor scoring and choice (#23): the four parts and their weights, the rank weight, the
profile bonus, the types, and the choice per pair with its alternatives or content gap."""

from __future__ import annotations

import pytest

from linking_engine.anchor.extraction import Stems
from linking_engine.anchor.keywords import tokens as words_of
from linking_engine.anchor.scoring import (
    ALTERNATIVES,
    AWKWARD_FLOOR,
    COSINE_SHARE,
    DIVERSITY_WEIGHT,
    KEYWORD_WEIGHT,
    LENGTH_WEIGHT,
    PROFILE_BONUS,
    SECONDARY_WEIGHT,
    SEMANTIC_WEIGHT,
    STAGE,
    STEM_SHARE,
    Brand,
    Candidate,
    ExistingAnchor,
    PairCandidates,
    anchor_type,
    brand_tokens,
    branded,
    choose,
    existing_type,
    score_candidate,
    selection_report,
    stem_jaccard,
    summarise_selection,
    unit,
    word_count,
)
from linking_engine.models import (
    AnchorMatch,
    AnchorRung,
    AnchorType,
    AnchorTypeProfile,
    ContentGapFinding,
    KeywordSource,
    SemanticThreshold,
    UnanchoredReason,
)
from linking_engine.models.anchors import UNANCHORED_ADVICE, content_gap_finding

EXACT, STEMMED, STEM_SET, SEMANTIC = tuple(AnchorRung)
NO_MENTION = UnanchoredReason.SOURCE_DOES_NOT_MENTION_TOPIC
NO_GOOD_PHRASE = UnanchoredReason.TOPIC_MENTIONED_BUT_NO_GOOD_PHRASE
NO_KEYWORD = UnanchoredReason.TARGET_PAGE_HAS_NO_KEYWORD
NO_TEXT = UnanchoredReason.SOURCE_PAGE_TEXT_UNAVAILABLE
NOT_SEARCHED = UnanchoredReason.MEANING_SEARCH_NOT_RUN
SOURCE, TARGET, OTHER = "example.com/guide", "example.com/shoes", "example.com/tents"
PROFILE = AnchorTypeProfile()
NO_BRAND: Brand = ()
SUMMIT: Brand = (("summit",),)
EN = Stems("en")


def match(
    phrase: str,
    *,
    rung: AnchorRung = EXACT,
    rank: int = 1,
    sentence_index: int = 0,
    lead: str = "Read about ",
    source: str = SOURCE,
    target: str = TARGET,
) -> AnchorMatch:
    sentence = f"{lead}{phrase} today."
    start = 100 + len(lead)
    return AnchorMatch(
        source_url=source,
        target_url=target,
        keyword="trail shoes",
        keyword_rank=rank,
        keyword_source=KeywordSource.CLIENT_STRATEGIC,
        rung=rung,
        phrase=phrase,
        start=start,
        end=start + len(phrase),
        sentence=sentence,
        sentence_index=sentence_index,
        sentence_start=100,
        stem_jaccard=0.5 if rung is STEM_SET else None,
        semantic_similarity=0.7 if rung is SEMANTIC else None,
    )


def candidate(
    phrase: str,
    *,
    stem: float = 1.0,
    target_cosine: float | None = 0.6,
    keyword_cosine: float | None = 0.2,
    rung: AnchorRung = EXACT,
    rank: int = 1,
    sentence_index: int = 0,
    lead: str = "Read about ",
    source: str = SOURCE,
    target: str = TARGET,
) -> Candidate:
    found = match(
        phrase,
        rung=rung,
        rank=rank,
        sentence_index=sentence_index,
        lead=lead,
        source=source,
        target=target,
    )
    return Candidate(
        match=found,
        stem_jaccard=stem,
        words=words_of(phrase),
        word_count=word_count(phrase),
        target_cosine=target_cosine,
        keyword_cosine=keyword_cosine,
    )


def score(
    found: Candidate,
    kind: AnchorType = AnchorType.EXACT,
    *,
    against: tuple[frozenset[str], ...] = (),
    counts: dict[AnchorType, int] | None = None,
    profile: AnchorTypeProfile = PROFILE,
) -> float:
    return score_candidate(found, kind, against=against, counts=counts or {}, profile=profile).total


# ── the parts and their weights ─────────────────────────────────────────────


def test_the_weights_are_the_contracts() -> None:
    weights = (SEMANTIC_WEIGHT, KEYWORD_WEIGHT, DIVERSITY_WEIGHT, LENGTH_WEIGHT)
    assert weights == (0.4, 0.4, 0.1, 0.1)
    assert sum(weights) == pytest.approx(1.0)
    assert (STEM_SHARE, COSINE_SHARE) == (0.7, 0.3)
    assert (SECONDARY_WEIGHT, PROFILE_BONUS, AWKWARD_FLOOR, ALTERNATIVES) == (0.7, 0.1, 0.4, 2)
    assert STAGE == "anchor-selection"


@pytest.mark.parametrize(
    ("cosine", "mapped"), [(-1.0, 0.0), (0.0, 0.5), (0.6, 0.8), (1.0, 1.0), (1.2, 1.0), (-1.2, 0.0)]
)
def test_a_cosine_maps_to_the_unit_interval_as_link_scores_do(cosine: float, mapped: float) -> None:
    assert unit(cosine) == pytest.approx(mapped)


@pytest.mark.parametrize(
    ("phrase", "count"),
    [
        ("trail", 1),
        ("trail shoes", 2),
        ("e-commerce platform", 2),
        ("node.js hosting", 2),
        ("Tips & Tricks", 2),
        ("light trail running shoes for wet rock", 7),
    ],
)
def test_a_compound_without_whitespace_is_one_word(phrase: str, count: int) -> None:
    assert word_count(phrase) == count


@pytest.mark.parametrize(
    ("phrase", "keyword", "jaccard"),
    [
        pytest.param("Trail Shoes", "trail shoe", 1.0, id="stems-equal"),
        pytest.param("hydraulic press for sale", "hydraulic press", 2 / 3, id="stop-words-out"),
        pytest.param("for the", "for the win", 2 / 3, id="only-stop-words"),
        pytest.param("tents", "trail shoes", 0.0, id="disjoint"),
    ],
)
def test_the_stem_jaccard_compares_content_stems(phrase: str, keyword: str, jaccard: float) -> None:
    assert stem_jaccard(phrase, keyword, EN) == pytest.approx(jaccard)


def test_every_part_weighs_into_the_total() -> None:
    found = candidate("trail shoes", target_cosine=0.6, keyword_cosine=0.2)

    parts = score_candidate(found, AnchorType.EXACT, against=(), counts={}, profile=PROFILE)

    # semantic (1 + 0.6) / 2; keyword 0.7 x 1 + 0.3 x (1 + 0.2) / 2.
    assert (parts.semantic, parts.keyword, parts.diversity, parts.length) == pytest.approx(
        (0.8, 0.88, 1.0, 1.0)
    )
    assert (parts.rank_weight, parts.profile_bonus) == pytest.approx((1.0, 0.1 * PROFILE.exact))
    weighted = SEMANTIC_WEIGHT * 0.8 + KEYWORD_WEIGHT * 0.88 + DIVERSITY_WEIGHT + LENGTH_WEIGHT
    assert parts.total == pytest.approx(weighted + 0.1 * PROFILE.exact)


def test_without_the_target_cosine_the_other_parts_rescale() -> None:
    found = candidate("trail shoes", target_cosine=None, keyword_cosine=0.2)

    parts = score_candidate(found, AnchorType.EXACT, against=(), counts={}, profile=PROFILE)

    assert parts.semantic is None
    rescaled = (KEYWORD_WEIGHT * 0.88 + DIVERSITY_WEIGHT + LENGTH_WEIGHT) / (1 - SEMANTIC_WEIGHT)
    assert parts.total == pytest.approx(rescaled + 0.1 * PROFILE.exact)


def test_without_the_keyword_cosine_the_keyword_part_is_the_stem_jaccard() -> None:
    found = candidate("trail shoes", stem=0.5, keyword_cosine=None)

    parts = score_candidate(found, AnchorType.PARTIAL, against=(), counts={}, profile=PROFILE)

    assert parts.keyword == 0.5


def test_a_perfect_candidate_without_vectors_is_not_penalised() -> None:
    bare = candidate("trail shoes", target_cosine=None, keyword_cosine=None)
    full = candidate("trail shoes", target_cosine=1.0, keyword_cosine=1.0)
    no_bonus = AnchorTypeProfile(exact=0.0, partial=0.0, natural=0.0, branded=0.0)

    assert score(bare, profile=no_bonus) == pytest.approx(1.0)
    assert score(full, profile=no_bonus) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("phrase", "length"),
    [
        ("trail", 0.6),
        ("trail shoes", 1.0),
        ("light trail running shoes", 1.0),
        ("light trail running shoes today", 0.6),
        ("light trail running shoes for rock", 0.3),
    ],
)
def test_two_to_four_words_read_best(phrase: str, length: float) -> None:
    parts = score_candidate(
        candidate(phrase), AnchorType.PARTIAL, against=(), counts={}, profile=PROFILE
    )
    assert parts.length == length


def test_diversity_is_one_minus_the_closest_word_jaccard() -> None:
    found = candidate("trail shoes guide")
    against = (frozenset({"trail", "shoes"}), frozenset({"tents"}), frozenset[str]())

    parts = score_candidate(found, AnchorType.EXACT, against=against, counts={}, profile=PROFILE)

    assert parts.diversity == pytest.approx(1 - 2 / 3)
    assert (
        score_candidate(found, AnchorType.EXACT, against=(), counts={}, profile=PROFILE).diversity
        == 1.0
    )


def test_a_secondary_keyword_weighs_less_but_keeps_its_profile_bonus() -> None:
    primary = score_candidate(
        candidate("trail shoes", rank=1), AnchorType.PARTIAL, against=(), counts={}, profile=PROFILE
    )
    secondary = score_candidate(
        candidate("trail shoes", rank=2), AnchorType.PARTIAL, against=(), counts={}, profile=PROFILE
    )

    assert (primary.rank_weight, secondary.rank_weight) == (1.0, 0.7)
    bonus = 0.1 * PROFILE.partial
    assert secondary.total - bonus == pytest.approx(0.7 * (primary.total - bonus))


@pytest.mark.parametrize(
    ("kind", "deficit"),
    [
        # 3 exact and 1 natural so far: exact is over its 0.15, natural 0.25 under its 0.5.
        (AnchorType.EXACT, 0.0),
        (AnchorType.NATURAL, 0.25),
        (AnchorType.BRANDED, 0.15),
        (AnchorType.PARTIAL, 0.2),
    ],
)
def test_the_profile_bonus_is_the_types_deficit(kind: AnchorType, deficit: float) -> None:
    counts = {AnchorType.EXACT: 3, AnchorType.NATURAL: 1}

    parts = score_candidate(
        candidate("trail shoes"), kind, against=(), counts=counts, profile=PROFILE
    )

    assert parts.profile_bonus == pytest.approx(PROFILE_BONUS * deficit)


# ── types ───────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("rung", "rank", "kind"),
    [
        (EXACT, 1, AnchorType.EXACT),
        (EXACT, 2, AnchorType.PARTIAL),
        (STEMMED, 1, AnchorType.PARTIAL),
        (STEM_SET, 1, AnchorType.PARTIAL),
        (SEMANTIC, 1, AnchorType.NATURAL),
        (SEMANTIC, 3, AnchorType.NATURAL),
    ],
)
def test_a_match_is_typed_by_its_rung_and_keyword_rank(
    rung: AnchorRung, rank: int, kind: AnchorType
) -> None:
    assert anchor_type(match("trail shoes", rung=rung, rank=rank), NO_BRAND) is kind


@pytest.mark.parametrize("rung", list(AnchorRung))
def test_a_brand_word_makes_any_match_branded(rung: AnchorRung) -> None:
    assert anchor_type(match("Summit trail shoes", rung=rung), SUMMIT) is (AnchorType.BRANDED)


def test_the_brand_comes_from_the_tenants_title_affixes() -> None:
    titles = ["Trail shoes | Summit", "Tents | Summit", "Stoves | Summit", "Maps", None]

    assert brand_tokens(titles) == SUMMIT
    assert brand_tokens(["Trail shoes", "Tents"]) == ()
    both = ["Summit Gear | Trail shoes | Summit Gear", "Summit Gear | Tents | Summit Gear"]
    assert brand_tokens(both) == (("summit", "gear"),)


@pytest.mark.parametrize(
    ("text", "held"),
    [
        pytest.param("Summit Gear boots", True, id="whole-brand"),
        pytest.param("new summit  GEAR", True, id="casefolded"),
        pytest.param("hiking gear", False, id="one-brand-word"),
        pytest.param("gear from summit", False, id="not-contiguous"),
        pytest.param("summitgear", False, id="not-its-tokens"),
    ],
)
def test_a_multi_word_brand_counts_only_as_a_whole(text: str, held: bool) -> None:
    brand: Brand = (("summit", "gear"),)
    assert branded(text, brand) is held
    kind = anchor_type(match(text), brand)
    assert kind is (AnchorType.BRANDED if held else AnchorType.EXACT)


@pytest.mark.parametrize(
    ("text", "kind"),
    [
        pytest.param("Summit shoes", AnchorType.BRANDED, id="brand"),
        pytest.param("Trail  SHOES", AnchorType.EXACT, id="primary-words"),
        pytest.param("running shoe", AnchorType.PARTIAL, id="secondary-verbatim"),
        pytest.param("our trail guide", AnchorType.PARTIAL, id="shares-a-stem"),
        pytest.param("read more about grip", AnchorType.NATURAL, id="shares-nothing"),
    ],
)
def test_an_existing_anchor_is_typed_against_the_ranked_keywords(
    text: str, kind: AnchorType
) -> None:
    keywords = ["trail shoes", "running shoe"]
    assert existing_type(text, keywords, EN, SUMMIT) is kind


@pytest.mark.parametrize(
    ("text", "kind"),
    [
        pytest.param("Summit Gear packs", AnchorType.BRANDED, id="whole-brand"),
        pytest.param("hiking gear", AnchorType.NATURAL, id="one-brand-word"),
        pytest.param("summit trail shoes", AnchorType.PARTIAL, id="other-brand-word"),
    ],
)
def test_an_existing_anchor_is_branded_only_by_the_whole_brand(text: str, kind: AnchorType) -> None:
    assert existing_type(text, ["trail shoes"], EN, (("summit", "gear"),)) is kind


def test_a_section_prefix_is_a_brand_only_as_a_whole() -> None:
    titled = [f"Trail Blog | {topic}" for topic in ("Boots", "Tents", "Stoves")]
    brand = brand_tokens(titled)

    assert brand == (("trail", "blog"),)
    assert anchor_type(match("trail shoes"), brand) is AnchorType.EXACT
    assert existing_type("trail shoes", ["trail shoes"], EN, brand) is AnchorType.EXACT
    assert anchor_type(match("our trail blog notes"), brand) is AnchorType.BRANDED


def test_a_one_word_section_prefix_still_brands_its_word() -> None:
    brand = brand_tokens([f"Blog | {topic}" for topic in ("Boots", "Tents", "Stoves")])

    assert brand == (("blog",),)
    assert anchor_type(match("trail blog tips"), brand) is AnchorType.BRANDED
    assert existing_type("gear blog", ["trail shoes"], EN, brand) is AnchorType.BRANDED


def test_an_existing_anchor_into_a_page_without_keywords_is_natural() -> None:
    assert existing_type("trail shoes", [], EN, NO_BRAND) is AnchorType.NATURAL


# ── the choice ──────────────────────────────────────────────────────────────


def pair(
    *found: Candidate,
    target: str = TARGET,
    source: str = SOURCE,
    has_keywords: bool = True,
    has_text: bool = True,
    meaning_searched: bool = True,
) -> PairCandidates:
    return PairCandidates(
        source_url=source,
        target_url=target,
        candidates=found,
        has_keywords=has_keywords,
        has_text=has_text,
        meaning_searched=meaning_searched,
    )


def test_the_best_total_is_chosen_with_two_distinct_alternatives() -> None:
    found = (
        candidate("trail shoes", target_cosine=0.9),
        candidate("Trail  Shoes", target_cosine=0.8, sentence_index=1),
        candidate("running shoes", target_cosine=0.7, rung=STEMMED),
        candidate("shoes for trails", target_cosine=0.6, rung=STEMMED),
        candidate("wet rock shoes", target_cosine=0.5, rung=STEMMED),
    )

    choices, unanchored = choose([pair(*found)], existing={}, profile=PROFILE, brand=NO_BRAND)

    assert unanchored == []
    assert [(c.rank, c.match.phrase) for c in choices] == [
        (1, "trail shoes"),
        (2, "running shoes"),
        (3, "shoes for trails"),
    ]
    assert [c.anchor_type for c in choices] == [
        AnchorType.EXACT,
        AnchorType.PARTIAL,
        AnchorType.PARTIAL,
    ]
    assert all(c.context_relevance is None and c.anchor_target_fit is None for c in choices)
    assert choices[0].score.total > choices[1].score.total > choices[2].score.total


def test_ties_go_to_the_earlier_sentence_then_the_earlier_offset() -> None:
    later = candidate("trail shoes", sentence_index=2)
    earlier = candidate("trail shoes", sentence_index=1, lead="See ")
    first_offset = candidate("trail shoes", sentence_index=1, lead="Get ")

    for order in ((later, earlier), (earlier, later)):
        choices, _ = choose([pair(*order)], existing={}, profile=PROFILE, brand=NO_BRAND)
        assert [c.match.sentence_index for c in choices] == [1]
    choices, _ = choose(
        [
            pair(
                later, candidate("trail shoes", sentence_index=1, lead="Please read "), first_offset
            )
        ],
        existing={},
        profile=PROFILE,
        brand=NO_BRAND,
    )
    assert choices[0].match.start == first_offset.match.start


def test_a_source_without_a_candidate_does_not_mention_the_topic() -> None:
    choices, unanchored = choose([pair()], existing={}, profile=PROFILE, brand=NO_BRAND)

    assert choices == []
    [gap] = unanchored
    assert (gap.source_url, gap.target_url) == (SOURCE, TARGET)
    assert (gap.reason, gap.best_score) == (NO_MENTION, None)
    assert content_gap_finding(gap.reason) is ContentGapFinding.NO_TOPICAL_MENTION


@pytest.mark.parametrize(
    ("has_keywords", "has_text", "meaning_searched", "reason"),
    [
        pytest.param(False, False, False, NO_KEYWORD, id="target-without-keywords-first"),
        pytest.param(True, False, False, NO_TEXT, id="source-text-unavailable"),
        pytest.param(True, True, False, NOT_SEARCHED, id="meaning-search-not-run"),
        pytest.param(True, True, True, NO_MENTION, id="searched-on-every-rung"),
    ],
)
def test_an_unsearched_pair_says_why_and_only_a_full_search_is_a_content_gap(
    has_keywords: bool, has_text: bool, meaning_searched: bool, reason: UnanchoredReason
) -> None:
    found = pair(has_keywords=has_keywords, has_text=has_text, meaning_searched=meaning_searched)

    choices, unanchored = choose([found], existing={}, profile=PROFILE, brand=NO_BRAND)

    assert choices == []
    [gap] = unanchored
    assert (gap.reason, gap.best_score) == (reason, None)
    assert (content_gap_finding(gap.reason) is None) is (reason is not NO_MENTION)


def weak(total: float) -> Candidate:
    """A one-word secondary phrase, no vectors, whose total is ``total`` under a zero profile:
    (keyword weight x stem + diversity weight + length weight x 0.6) / (1 - semantic weight)
    x the secondary weight."""
    rest = DIVERSITY_WEIGHT + LENGTH_WEIGHT * 0.6
    stem = (total / SECONDARY_WEIGHT * (1 - SEMANTIC_WEIGHT) - rest) / KEYWORD_WEIGHT
    assert 0 <= stem <= 1, stem
    return candidate("shoes", stem=stem, target_cosine=None, keyword_cosine=None, rank=2)


def test_phrases_that_all_score_below_the_floor_are_awkward_phrasing() -> None:
    zero = AnchorTypeProfile(exact=0.0, partial=0.0, natural=0.0, branded=0.0)

    choices, unanchored = choose(
        [pair(weak(0.35), weak(0.3))], existing={}, profile=zero, brand=NO_BRAND
    )

    assert choices == []
    [gap] = unanchored
    assert gap.reason is NO_GOOD_PHRASE
    assert content_gap_finding(gap.reason) is ContentGapFinding.AWKWARD_PHRASING
    assert gap.best_score == pytest.approx(0.35)
    above, _ = choose([pair(weak(0.4 + 1e-9))], existing={}, profile=zero, brand=NO_BRAND)
    assert [c.score.total for c in above] == [pytest.approx(0.4)]
    _, below = choose([pair(weak(0.4 - 1e-9))], existing={}, profile=zero, brand=NO_BRAND)
    assert [gap.reason for gap in below] == [NO_GOOD_PHRASE]


PARTIAL_ONLY = AnchorTypeProfile(exact=0.0, partial=1.0, natural=0.0, branded=0.0)


def test_the_profile_bonus_never_lifts_a_phrase_over_the_floor() -> None:
    # 0.381 before a 0.1 bonus for its wanted type: 0.481 in all, still refused.
    choices, unanchored = choose(
        [pair(weak(0.381))], existing={}, profile=PARTIAL_ONLY, brand=NO_BRAND
    )

    assert choices == []
    [gap] = unanchored
    assert gap.reason is NO_GOOD_PHRASE
    assert gap.best_score == pytest.approx(0.381)
    lifted, _ = choose([pair(weak(0.4 + 1e-9))], existing={}, profile=PARTIAL_ONLY, brand=NO_BRAND)
    [chosen] = lifted
    assert (chosen.score.total, chosen.score.profile_bonus) == pytest.approx((0.5, 0.1))


def test_a_phrase_below_the_floor_is_never_an_alternative_whatever_its_bonus() -> None:
    # No vectors, a fifth of the stems: above the floor, and no bonus for an exact phrase.
    good = candidate("trail shoes", stem=0.2, target_cosine=None, keyword_cosine=None)
    good_total = (KEYWORD_WEIGHT * 0.2 + DIVERSITY_WEIGHT + LENGTH_WEIGHT) / (1 - SEMANTIC_WEIGHT)
    assert AWKWARD_FLOOR < good_total < 0.481
    poor = weak(0.381)

    choices, unanchored = choose(
        [pair(poor, good)], existing={}, profile=PARTIAL_ONLY, brand=NO_BRAND
    )

    assert unanchored == []
    assert poor.match.phrase != good.match.phrase
    # The poor phrase totals 0.481 with its bonus, above the good one.
    assert [(c.rank, c.match.phrase) for c in choices] == [(1, "trail shoes")]
    assert choices[0].score.total == pytest.approx(good_total)


def test_diversity_accumulates_across_a_targets_pairs_and_stops_at_the_target() -> None:
    def both(source: str, target: str = TARGET) -> PairCandidates:
        found = (
            candidate("trail shoes", target_cosine=0.9, source=source, target=target),
            candidate("shoes for rock", target_cosine=0.8, source=source, target=target),
        )
        return pair(*found, source=source, target=target)

    first, second, elsewhere = (
        both("example.com/a"),
        both("example.com/b"),
        both("example.com/c", OTHER),
    )

    choices, _ = choose([first, elsewhere, second], existing={}, profile=PROFILE, brand=NO_BRAND)

    chosen = {(c.match.source_url, c.match.target_url): c for c in choices if c.rank == 1}
    assert chosen[("example.com/a", TARGET)].match.phrase == "trail shoes"
    assert chosen[("example.com/c", OTHER)].match.phrase == "trail shoes"
    # "trail shoes" is already an anchor of the target: its diversity is now 0.
    repeat = chosen[("example.com/b", TARGET)]
    assert repeat.match.phrase == "shoes for rock"
    assert repeat.score.diversity == pytest.approx(1 - 1 / 4)


def test_existing_anchors_count_for_diversity_and_the_profile() -> None:
    existing = {TARGET: [ExistingAnchor(frozenset({"trail", "shoes"}), AnchorType.EXACT)] * 3}
    found = candidate("trail shoes", target_cosine=0.6)

    choices, _ = choose([pair(found)], existing=existing, profile=PROFILE, brand=NO_BRAND)

    [chosen] = choices
    assert chosen.score.diversity == 0.0
    # Exact is already 3 of 3, above its 0.15 share.
    assert chosen.score.profile_bonus == 0.0


def test_the_profile_only_nudges_and_never_drops_a_pair() -> None:
    natural_only = AnchorTypeProfile(exact=0.0, partial=0.0, natural=1.0, branded=0.0)
    found = (
        candidate("trail shoes", target_cosine=0.9),
        candidate("footwear for rock", stem=0.0, target_cosine=0.7, rung=SEMANTIC),
    )

    choices, unanchored = choose([pair(*found)], existing={}, profile=natural_only, brand=NO_BRAND)

    assert unanchored == []
    # The semantic phrase gains the whole bonus of 0.1 but trails by more.
    assert [(c.rank, c.anchor_type) for c in choices] == [
        (1, AnchorType.EXACT),
        (2, AnchorType.NATURAL),
    ]
    assert choices[1].score.profile_bonus == pytest.approx(0.1)
    # 0.004 behind before the bonus, 0.096 ahead after it.
    close = (
        candidate("trail shoes", target_cosine=0.62),
        candidate("footwear for rock", target_cosine=0.6, rung=SEMANTIC),
    )
    nudged, _ = choose([pair(*close)], existing={}, profile=natural_only, brand=NO_BRAND)
    assert nudged[0].anchor_type is AnchorType.NATURAL


# ── the report and summary ──────────────────────────────────────────────────


THRESHOLD = SemanticThreshold(
    value=0.62, quantile=0.99, negatives=400, positives=20, positive_recall=0.85
)


def test_the_report_counts_choices_gaps_types_ranks_and_histograms() -> None:
    choices, unanchored = choose(
        [
            pair(candidate("trail shoes", target_cosine=0.9), candidate("running shoes")),
            pair(
                candidate("footwear", rung=SEMANTIC, rank=2, source="example.com/b"),
                source="example.com/b",
            ),
            pair(source="example.com/c"),
            pair(weak(0.3), source="example.com/d"),
            pair(candidate("tent pegs", target=OTHER), target=OTHER),
            pair(target="example.com/bare", has_keywords=False),
            pair(source="example.com/e", has_text=False),
            pair(source="example.com/f", meaning_searched=False),
        ],
        existing={},
        profile=AnchorTypeProfile(exact=0.0, partial=0.0, natural=0.0, branded=0.0),
        brand=NO_BRAND,
    )
    filled = [
        choice.model_copy(update={"context_relevance": 0.7, "anchor_target_fit": 0.8})
        if choice.rank == 1 and choice.match.source_url == SOURCE
        else choice
        for choice in choices
    ]

    report = selection_report(
        "acme",
        filled,
        unanchored,
        pairs=8,
        lexical_pairs=3,
        semantic_invocations=2,
        semantic_similarities=[0.72, -0.2],
        zero_overlap_matches=1,
        threshold=THRESHOLD,
        semantic_skipped_reason=None,
        embedding_skipped_reason=None,
        sentences_embedded=12,
        sentences_cached=3,
        phrases_embedded=40,
        phrases_cached=5,
        profile=PROFILE,
        targets=4,
        started=0.0,
    )

    assert (report.chosen, report.alternatives) == (3, 1)
    assert report.unanchored == {
        NO_MENTION: 1,
        NO_GOOD_PHRASE: 1,
        NO_KEYWORD: 1,
        NO_TEXT: 1,
        NOT_SEARCHED: 1,
    }
    assert report.chosen_types == {
        AnchorType.EXACT: 2,
        AnchorType.PARTIAL: 0,
        AnchorType.NATURAL: 1,
        AnchorType.BRANDED: 0,
    }
    assert report.chosen_ranks == {1: 2, 2: 1}
    assert (report.targets, report.targets_with_anchor, report.features_filled) == (4, 2, 2)
    assert report.semantic_matched == 2
    # -0.2 clips into the first bin, 0.72 falls in bin 14.
    assert report.semantic_histogram == (1, *([0] * 13), 1, *([0] * 5))
    assert sum(report.score_histogram) == 3
    assert report.profile == PROFILE


def test_totals_above_one_fall_in_the_last_bin() -> None:
    choices, _ = choose(
        [pair(candidate("trail shoes", target_cosine=1.0, keyword_cosine=1.0))],
        existing={},
        profile=AnchorTypeProfile(exact=1.0),
        brand=NO_BRAND,
    )
    assert choices[0].score.total == pytest.approx(1.1)

    report = selection_report(
        "acme",
        choices,
        [],
        pairs=1,
        lexical_pairs=1,
        semantic_invocations=0,
        semantic_similarities=[],
        zero_overlap_matches=0,
        threshold=THRESHOLD,
        semantic_skipped_reason="no Voyage key",
        embedding_skipped_reason="no Voyage key",
        sentences_embedded=0,
        sentences_cached=0,
        phrases_embedded=0,
        phrases_cached=0,
        profile=PROFILE,
        targets=1,
        started=0.0,
    )

    assert report.score_histogram == (*([0] * 19), 1)
    assert report.semantic_skipped_reason == report.embedding_skipped_reason == "no Voyage key"


@pytest.mark.parametrize(
    ("threshold", "skipped", "facts"),
    [
        pytest.param(
            THRESHOLD,
            None,
            (
                "threshold 0.620, the 99% quantile of 400",
                "85.0% of 20",
                "matched 1",
                "Every text needed was embedded",
                "refused 2 phrases whose numbers disagreed",
                "and 3 closer to another target",
            ),
            id="derived",
        ),
        pytest.param(
            SemanticThreshold(value=0.6, quantile=0.99, negatives=12, positives=0, fallback=True),
            None,
            ("threshold 0.600, the default", "no descriptive existing anchors"),
            id="fallback",
        ),
        pytest.param(
            SemanticThreshold(
                value=0.7,
                quantile=0.99,
                negatives=0,
                positives=4,
                positive_recall=0.5,
                overridden=True,
            ),
            None,
            ("at configured threshold 0.700;", "50.0% of 4"),
            id="overridden",
        ),
        pytest.param(
            SemanticThreshold(value=0.9, quantile=0.99, negatives=900, positives=3, bounded=True),
            None,
            ("bounded",),
            id="bounded",
        ),
        pytest.param(
            THRESHOLD,
            "no Voyage key",
            ("skipped: no Voyage key", "were not embedded: no Voyage key"),
            id="skipped",
        ),
    ],
)
def test_the_summary_states_counts_and_the_threshold_without_text(
    threshold: SemanticThreshold, skipped: str | None, facts: tuple[str, ...]
) -> None:
    phrases = ("trail shoes", "footwear")
    choices, unanchored = choose(
        [
            pair(candidate(phrases[0])),
            pair(
                candidate(phrases[1], rung=SEMANTIC, source="example.com/b"),
                source="example.com/b",
            ),
            pair(source="example.com/c"),
        ],
        existing={},
        profile=PROFILE,
        brand=NO_BRAND,
    )
    report = selection_report(
        "acme",
        choices,
        unanchored,
        pairs=3,
        lexical_pairs=1,
        semantic_invocations=0 if skipped else 1,
        semantic_similarities=[] if skipped else [0.7],
        semantic_rejected_identifier=2,
        semantic_rejected_other_target=3,
        zero_overlap_matches=0,
        threshold=threshold,
        semantic_skipped_reason=skipped,
        embedding_skipped_reason=skipped,
        sentences_embedded=4,
        sentences_cached=0,
        phrases_embedded=9,
        phrases_cached=0,
        profile=PROFILE,
        targets=1,
        started=0.0,
    )

    summary = summarise_selection(report)

    advice = UNANCHORED_ADVICE[NO_MENTION]
    for fact in ("acme", "3 pairs", "2 anchors chosen", "1 of 1 targets", advice, *facts):
        assert fact in summary, f"{fact!r} missing from:\n{summary}"
    sentences = [c.match.sentence for c in choices]
    leaked = [t for t in (SOURCE, TARGET, "example.com/b", *phrases, *sentences) if t in summary]
    assert leaked == []
