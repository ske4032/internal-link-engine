"""The semantic rung's pure parts (#22): candidate phrases by the ladder's rules, top sentences,
the match with exact offsets, shared stems, the threshold, and the negative sample."""

from __future__ import annotations

import numpy as np
import pytest

from linking_engine.anchor.extraction import SourceIndex, Stems
from linking_engine.anchor.semantic import (
    DEFAULT_SEMANTIC_THRESHOLD,
    MAX_PHRASES_PER_SENTENCE,
    MIN_NEGATIVES,
    THRESHOLD_BOUNDS,
    THRESHOLD_QUANTILE,
    TOP_SENTENCES,
    Phrase,
    SemanticOutcome,
    candidate_phrases,
    cosines,
    derive_threshold,
    negative_phrases,
    relevance,
    sample,
    semantic_match,
    shares_stem,
    top_sentences,
)
from linking_engine.models import AnchorRung, KeywordSource

STEMS = Stems("en")
SOURCE = "example.com/source"
TARGET = "example.com/target"
PLAIN = "Light trail shoes grip wet rock."


def index(body: str, headings: tuple[str, ...] = (), url: str = SOURCE) -> SourceIndex:
    return SourceIndex(url, body, headings, STEMS)


def texts(phrases: list[Phrase]) -> list[str]:
    return [phrase.text for phrase in phrases]


# ── candidate phrases ───────────────────────────────────────────────────────


def test_phrases_are_two_to_five_tokens_most_content_first_then_earliest_capped_at_ten() -> None:
    found = candidate_phrases(index(PLAIN), 0)

    # Six content tokens give 14 spans of 2-5 tokens; the ten with the most content tokens stay.
    assert MAX_PHRASES_PER_SENTENCE == 10
    assert texts(found) == [
        "Light trail shoes grip wet",
        "trail shoes grip wet rock",
        "Light trail shoes grip",
        "trail shoes grip wet",
        "shoes grip wet rock",
        "Light trail shoes",
        "trail shoes grip",
        "shoes grip wet",
        "grip wet rock",
        "Light trail",
    ]
    assert [p.content_tokens for p in found] == [5, 5, 4, 4, 4, 3, 3, 3, 3, 2]
    assert len(candidate_phrases(index(PLAIN), 0, limit=100)) == 14
    assert texts(candidate_phrases(index(PLAIN), 0, limit=2)) == texts(found)[:2]


def test_phrases_never_start_or_end_on_a_stop_word_and_hold_at_most_one_inside() -> None:
    found = candidate_phrases(index("Pack the tent for a cold night."), 0)

    # "tent for a cold" holds two stop words; every other span ends on one.
    assert texts(found) == ["Pack the tent", "cold night"]
    assert [p.content_tokens for p in found] == [2, 2]


def test_phrases_stay_inside_a_block_between_punctuation() -> None:
    found = candidate_phrases(index("Trail shoes, rain jackets and tents."), 0)

    assert texts(found) == [
        "rain jackets and tents",
        "Trail shoes",
        "rain jackets",
        "jackets and tents",
    ]
    assert not [text for text in texts(found) if "," in text]


def test_phrases_never_cut_a_compound() -> None:
    found = candidate_phrases(index("Our e-mail list shares trail tips."), 0, limit=100)

    assert "e-mail" in texts(found), "a compound is kept whole"
    assert not [text for text in texts(found) if text.startswith("mail") or text.endswith("e")]


def test_phrases_overlapping_an_existing_anchor_are_dropped() -> None:
    shoes = PLAIN.index("shoes")

    found = candidate_phrases(index(PLAIN), 0, existing=[(shoes, shoes + len("shoes"))], limit=100)

    assert texts(found) == ["grip wet rock", "Light trail", "grip wet", "wet rock"]


def test_phrases_carry_exact_offsets_into_the_body_and_skip_heading_lines() -> None:
    body = f"Trail Guide\nIntro line here.\n{PLAIN}"
    source = index(body, headings=("Trail Guide",))

    assert [s.text for s in source.sentences] == ["Intro line here.", PLAIN]
    for position in range(len(source.sentences)):
        for phrase in candidate_phrases(source, position, limit=100):
            assert body[phrase.start : phrase.end] == phrase.text
            assert phrase.position == position
            assert "Guide" not in phrase.text


# ── top sentences ───────────────────────────────────────────────────────────


def test_top_sentences_are_the_closest_to_any_keyword_best_first_ties_to_the_earlier() -> None:
    e = np.eye(4)
    sentences = np.stack([e[3], e[0], 0.6 * e[1] + 0.8 * e[3], e[1], e[0]])
    keywords = np.stack([e[0], e[1]])

    assert TOP_SENTENCES == 3
    # Rows 1, 3 and 4 have cosine 1 to a keyword; row 2 has 0.6; row 0 has 0.
    assert top_sentences(sentences, keywords) == [1, 3, 4]
    assert top_sentences(sentences, keywords, count=4) == [1, 3, 4, 2]
    assert top_sentences(sentences, keywords, count=10) == [1, 3, 4, 2, 0]


def test_top_sentences_refuse_a_zero_count_and_a_zero_vector() -> None:
    e = np.eye(3)
    with pytest.raises(ValueError, match="count"):
        top_sentences(e, e, count=0)
    with pytest.raises(ValueError, match="non-zero"):
        top_sentences(np.zeros((2, 3)), e)
    with pytest.raises(ValueError, match="finite"):
        cosines([[np.nan, 1.0, 0.0]], e)


def test_relevance_maps_a_cosine_to_the_unit_interval_like_the_link_scores() -> None:
    a, b = np.array([1.0, 0.0]), np.array([0.6, 0.8])

    assert relevance(a, b) == pytest.approx((1 + 0.6) / 2)
    assert relevance(a, a * 3) == pytest.approx(1.0)
    assert relevance(a, -a) == pytest.approx(0.0)
    assert relevance(None, b) is None
    assert relevance(a, None) is None


def test_cosines_are_clipped_to_the_unit_interval() -> None:
    found = cosines([[1.0, 1e-17], [-2.0, 0.0]], [[3.0, 0.0]])
    assert found.tolist() == [[1.0], [-1.0]]


# ── the semantic match ──────────────────────────────────────────────────────

E = np.eye(6)
KEYWORDS = [
    (1, "hiking boots", KeywordSource.CLIENT_STRATEGIC),
    (2, "footwear", KeywordSource.GSC_OBSERVED),
]
KEYWORD_VECTORS = np.stack([E[0], E[1]])


def vectors_for(phrases: list[Phrase], planted: dict[str, np.ndarray]) -> np.ndarray:
    """Every phrase orthogonal to both keywords, except the planted ones."""
    return np.stack([planted.get(p.text, E[5]) for p in phrases])


def test_the_best_phrase_becomes_a_semantic_match_with_its_keyword_and_exact_offsets() -> None:
    source = index(f"Pack light.\n{PLAIN}")
    phrases = candidate_phrases(source, 1)
    rows = vectors_for(phrases, {"trail shoes grip": 0.8 * E[1] + 0.6 * E[5]})

    match = semantic_match(
        source, TARGET, phrases, rows, KEYWORDS, KEYWORD_VECTORS, threshold=0.8
    ).match

    assert match is not None
    assert (match.rung, match.phrase, match.keyword, match.keyword_rank, match.keyword_source) == (
        AnchorRung.SEMANTIC,
        "trail shoes grip",
        "footwear",
        2,
        KeywordSource.GSC_OBSERVED,
    )
    assert match.semantic_similarity == pytest.approx(0.8)
    assert match.stem_jaccard is None
    assert source.body[match.start : match.end] == "trail shoes grip"
    assert (match.sentence, match.sentence_index, match.sentence_start) == (
        PLAIN,
        1,
        len("Pack light.\n"),
    )
    assert (match.source_url, match.target_url) == (SOURCE, TARGET)


def test_a_match_needs_the_threshold_and_is_accepted_at_it() -> None:
    source = index(PLAIN)
    phrases = candidate_phrases(source, 0)
    rows = vectors_for(phrases, {"grip wet rock": 0.6 * E[0] + 0.8 * E[5]})

    at = semantic_match(
        source, TARGET, phrases, rows, KEYWORDS, KEYWORD_VECTORS, threshold=0.6
    ).match
    above = semantic_match(
        source, TARGET, phrases, rows, KEYWORDS, KEYWORD_VECTORS, threshold=0.6001
    )

    assert at is not None
    assert (at.phrase, at.keyword_rank) == ("grip wet rock", 1)
    assert above == SemanticOutcome(None), "nothing reached the threshold, nothing rejected"


def test_a_tie_goes_to_the_earlier_phrase_then_the_better_ranked_keyword() -> None:
    source = index(PLAIN)
    phrases = candidate_phrases(source, 0)
    both = 0.7 * E[0] + 0.7 * E[1] + np.sqrt(0.02) * E[5]
    rows = vectors_for(phrases, {"grip wet rock": both, "Light trail shoes": both})

    match = semantic_match(
        source, TARGET, phrases, rows, KEYWORDS, KEYWORD_VECTORS, threshold=0.5
    ).match

    # "Light trail shoes" starts first in the sentence, though it comes later in the list; its
    # cosine is the same to both keywords, so the primary one wins.
    assert match is not None
    assert (match.phrase, match.keyword_rank) == ("Light trail shoes", 1)


def test_no_phrase_or_no_keyword_is_no_match_and_misaligned_vectors_are_refused() -> None:
    source = index(PLAIN)
    phrases = candidate_phrases(source, 0)
    rows = vectors_for(phrases, {})

    none = SemanticOutcome(None)
    empty = semantic_match(source, TARGET, [], rows[:0], KEYWORDS, KEYWORD_VECTORS, threshold=0)
    assert empty == none
    assert semantic_match(source, TARGET, phrases, rows, [], KEYWORD_VECTORS, threshold=0) == none
    with pytest.raises(ValueError, match="one vector per phrase"):
        semantic_match(source, TARGET, phrases, rows[:-1], KEYWORDS, KEYWORD_VECTORS, threshold=0)


@pytest.mark.parametrize(
    ("phrase", "keyword", "shared"),
    [
        pytest.param("trail running shoes", "trail shoes", True, id="shared-word"),
        pytest.param("a shoe", "trail shoes", True, id="shared-stem"),
        pytest.param("footwear for rocky paths", "trail shoes", False, id="zero-overlap"),
        pytest.param("with the others", "the shoes", False, id="stop-words-only"),
    ],
)
def test_a_zero_overlap_match_shares_no_content_stem_with_its_keyword(
    phrase: str, keyword: str, shared: bool
) -> None:
    assert shares_stem(phrase, keyword, STEMS) is shared


# ── the threshold ───────────────────────────────────────────────────────────


def test_the_threshold_is_the_high_quantile_of_the_unrelated_cosines() -> None:
    negatives = np.linspace(0.0, 0.8, 1001).tolist()

    found = derive_threshold(negatives, [])

    assert THRESHOLD_QUANTILE == 0.99
    assert found.value == pytest.approx(0.792)
    assert (found.quantile, found.negatives, found.positives) == (0.99, 1001, 0)
    assert (found.bounded, found.fallback, found.positive_recall) == (False, False, None)


@pytest.mark.parametrize(
    ("level", "value"),
    [pytest.param(0.1, 0.35, id="below"), pytest.param(0.95, 0.9, id="above")],
)
def test_a_threshold_outside_the_bounds_is_clipped_and_flagged(level: float, value: float) -> None:
    found = derive_threshold([level] * 500, [])

    assert THRESHOLD_BOUNDS == (0.35, 0.9)
    assert (found.value, found.bounded, found.fallback) == (value, True, False)


def test_too_few_negatives_fall_back_to_the_default() -> None:
    few = derive_threshold([0.5] * (MIN_NEGATIVES - 1), [])
    enough = derive_threshold([0.5] * MIN_NEGATIVES, [])

    assert (few.value, few.fallback, few.bounded, few.negatives) == (
        DEFAULT_SEMANTIC_THRESHOLD,
        True,
        False,
        MIN_NEGATIVES - 1,
    )
    assert (enough.value, enough.fallback) == (0.5, False)


def test_recall_is_the_share_of_existing_anchors_at_or_above_the_value() -> None:
    found = derive_threshold([0.5] * 400, [0.3, 0.5, 0.7, 0.9])

    assert found.value == 0.5
    assert (found.positives, found.positive_recall) == (4, 0.75)


@pytest.mark.parametrize("override", [0.2, 0.5, 0.95])
def test_an_override_is_used_as_is_and_the_negatives_are_ignored(override: float) -> None:
    found = derive_threshold([0.1] * 400, [0.3, 0.6], override=override)

    assert (found.value, found.overridden, found.negatives) == (override, True, 0)
    assert (found.bounded, found.fallback) == (False, False)
    assert found.positive_recall == sum(c >= override for c in (0.3, 0.6)) / 2
    assert not derive_threshold([0.5] * 400, []).overridden


# ── sampling ────────────────────────────────────────────────────────────────


def test_a_sample_keeps_every_item_when_there_are_few_and_their_order_when_cut() -> None:
    items = list(range(100))

    assert sample(items[:10], 10) == items[:10]
    cut = sample(items, 20)
    assert len(cut) == 20
    assert cut == sorted(cut)
    assert set(cut) <= set(items)
    assert sample(items, 20) == cut, "the same seed must draw the same items"
    assert sample(items, 20, seed=7) != cut


def pages() -> list[SourceIndex]:
    return [
        index("Trail shoes grip rock. Light boots dry fast.", url="example.com/trail/a"),
        index("Winter tents stand firm. Warm bags pack small.", url="example.com/tent/b"),
        index("", url="example.com/empty"),
    ]


def topic(url: str) -> str:
    return url.split("/")[1]


def test_negative_phrases_are_distinct_pairs_with_unrelated_targets_only() -> None:
    targets = ["example.com/trail/a", "example.com/tent/b"]
    phrases = {
        page.url: {p.text for i in range(len(page.sentences)) for p in candidate_phrases(page, i)}
        for page in pages()
    }
    # Every phrase of one topic's page, paired with the other topic's target.
    every = {(phrase, t) for t in targets for s in targets if s != t for phrase in phrases[s]}

    found = negative_phrases(pages(), targets, lambda s, t: topic(s) != topic(t), count=5000)

    assert len(found) == len(set(found)), "a pair was drawn twice"
    assert set(found) == every, "the draws stop short of the distinct pairs there are"
    assert len(found) == 24
    capped = negative_phrases(pages(), targets, lambda s, t: topic(s) != topic(t), count=10)
    assert len(capped) == 10
    assert capped == negative_phrases(pages(), targets, lambda s, t: topic(s) != topic(t), count=10)


def test_a_tiny_tenant_has_too_few_negatives_and_falls_back() -> None:
    targets = ["example.com/trail/a", "example.com/tent/b"]

    found = negative_phrases(pages(), targets, lambda s, t: topic(s) != topic(t))
    threshold = derive_threshold([0.1] * len(found), [])

    assert len(found) < MIN_NEGATIVES
    assert (threshold.value, threshold.fallback) == (DEFAULT_SEMANTIC_THRESHOLD, True)


def test_negative_phrases_give_up_when_nothing_is_unrelated() -> None:
    targets = ["example.com/trail/a", "example.com/tent/b"]

    assert negative_phrases(pages(), targets, lambda s, t: False, count=10) == []
    assert negative_phrases(pages()[2:], targets, lambda s, t: True, count=10) == []
    assert negative_phrases(pages(), [], lambda s, t: True, count=10) == []
    # A page is never its own negative, even when the predicate would allow it.
    same = negative_phrases(pages()[:1], ["example.com/trail/a"], lambda s, t: True, count=10)
    assert same == []
