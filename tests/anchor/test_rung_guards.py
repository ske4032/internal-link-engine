"""The rung-3 guards: a phrase stands for a keyword only when both name the same identifiers
(digit tokens, and month names next to one), on the stemmed, stem set and semantic rungs; and a
semantic phrase must be closest to its own target's keywords. Synthetic wording only."""

from __future__ import annotations

from importlib import resources

import numpy as np
import pytest

from linking_engine.anchor.extraction import (
    SNOWBALL,
    SourceIndex,
    Stems,
    extract,
    identifiers,
    month_stems,
    strong_identifier,
)
from linking_engine.anchor.semantic import (
    OTHER_TARGET_CHUNK,
    SemanticOutcome,
    agrees,
    best_other_cosines,
    candidate_phrases,
    eligible_phrases,
    semantic_match,
)
from linking_engine.models import AnchorRung, KeywordSource

EN = Stems("en")
SOURCE = "example.com/source"
TARGET = "example.com/target"
STRATEGIC = KeywordSource.CLIENT_STRATEGIC
# A title keyword as the target's H1 writes it, with an en dash.
CVE_WIDGET = "CVE-2026-10001 \u2013 Widget"


def index(body: str, language: str = "en") -> SourceIndex:
    return SourceIndex(SOURCE, body, (), Stems(language))


def ladder(body: str, keyword: str, language: str = "en") -> tuple[list, frozenset]:  # type: ignore[type-arg]
    matches, _, mismatched = extract(index(body, language), TARGET, [(1, keyword, STRATEGIC)])
    return matches, mismatched


def texts(body: str, spans: frozenset[tuple[int, int]]) -> list[str]:
    return sorted(body[start:end] for start, end in spans)


# ── identifiers ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("text", "found"),
    [
        ("Patch Tuesday July 2024", {"2024", "month:7"}),
        ("May 2024 release notes", {"2024", "month:5"}),
        ("you may update", set()),
        ("march forward", set()),
        ("July", set()),
        # "July" sits next to "2"; the comma starts a new block before 2024.
        ("July 2, 2024", {"2", "2024", "month:7"}),
        ("2 July", {"2", "month:7"}),
        ("CVE-2026-20270", {"2026", "20270"}),
        ("upgrade to widget 10", {"10"}),
        ("trail shoes", set()),
    ],
)
def test_identifiers_are_digit_tokens_and_month_names_next_to_one(
    text: str, found: set[str]
) -> None:
    assert identifiers(text, EN) == found


def test_a_month_name_counts_only_next_to_a_number_in_its_block() -> None:
    assert identifiers("July, 2024", EN) == {"2024"}, "the comma splits the block"
    assert identifiers("July and 2024", EN) == {"2024"}, "one token between them"


@pytest.mark.parametrize(
    ("language", "first", "second"),
    [
        pytest.param("ru", "отчёт за январь 2024", "обновление от 15 января 2024", id="russian"),
        pytest.param("fi", "tammikuu 2024", "tammikuuta 2024", id="finnish"),
    ],
)
def test_inflected_month_names_are_the_same_month(language: str, first: str, second: str) -> None:
    stems = Stems(language)

    assert "month:1" in identifiers(first, stems)
    assert "month:1" in identifiers(second, stems)


@pytest.mark.parametrize("language", [None, "ja"])
def test_a_language_without_a_month_list_gets_the_digit_rule_only(language: str | None) -> None:
    stems = Stems(language)

    assert dict(stems.months) == {}
    assert identifiers("July 2024", stems) == {"2024"}


def test_every_month_list_loads_and_names_all_twelve_months() -> None:
    folder = resources.files("linking_engine.anchor").joinpath("months")
    algorithms = sorted(
        entry.name.removesuffix(".txt") for entry in folder.iterdir() if entry.name.endswith(".txt")
    )
    language_of = {algorithm: code for code, algorithm in SNOWBALL.items()}

    assert len(algorithms) == 15
    for algorithm in algorithms:
        assert algorithm in language_of, f"{algorithm} has no language code"
        found = month_stems(language_of[algorithm])
        assert set(found.values()) == set(range(1, 13)), algorithm
        assert Stems(language_of[algorithm]).months == found


# ── agreement ───────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("phrase", "keyword", "agree"),
    [
        pytest.param("CVE-2026-10002", CVE_WIDGET, False, id="other-cve"),
        pytest.param("the CVE-2026-10001 fix", CVE_WIDGET, True, id="same-cve"),
        pytest.param("June 2024 digest", "Monthly digest July 2024", False, id="other-month"),
        pytest.param("July 2025 update", "Monthly digest July 2024", False, id="other-year"),
        pytest.param("July 2024 patch", "Monthly digest July 2024", True, id="same-month"),
        pytest.param("upgrade to widget", "upgrade to widget 10", False, id="dropped-version"),
        pytest.param("widget 10 upgrades", "upgrade to widget 10", True, id="same-version"),
        pytest.param("footwear for rocky paths", "trail shoes", True, id="none-either-side"),
    ],
)
def test_a_phrase_agrees_with_a_keyword_on_exactly_the_same_identifiers(
    phrase: str, keyword: str, agree: bool
) -> None:
    assert agrees(phrase, keyword, EN) is agree


# ── the lexical rungs ───────────────────────────────────────────────────────


def test_a_dropped_version_number_is_never_a_stemmed_match() -> None:
    body = "Upgrade to widget today. Then upgrade to widget 11 soon."

    matches, mismatched = ladder(body, "upgrade to widget 10")

    assert matches == []
    assert mismatched, "the refused places are reported"
    assert all("10" not in text for text in texts(body, mismatched))


def test_a_stemmed_variant_with_the_same_identifiers_is_accepted() -> None:
    body = "Tips for upgrading widget 10 laptops at home."

    matches, mismatched = ladder(body, "upgrade widget 10 laptop")

    [found] = matches
    assert (found.rung, found.phrase) == (AnchorRung.STEMMED, "upgrading widget 10 laptops")
    assert mismatched == frozenset()


def test_the_exact_rung_is_unchanged() -> None:
    [found], mismatched = ladder("Buy the widget 10 kit now.", "widget 10")

    assert (found.rung, found.phrase, mismatched) == (AnchorRung.EXACT, "widget 10", frozenset())


def test_another_cve_number_is_never_matched() -> None:
    body = "Read the CVE-2026-10002 widget notice before patching."

    matches, mismatched = ladder(body, CVE_WIDGET)

    assert matches == []
    assert texts(body, mismatched), "the CVE-2026-10002 place was refused, not missed"
    assert all("10002" in text for text in texts(body, mismatched))


MONTHS = (
    "You may update any time. Read the May 2024 release notes today. "
    "Our June 2024 release notes list fixes."
)


def test_a_month_keyword_matches_its_own_month_despite_the_verb_elsewhere() -> None:
    [found], mismatched = ladder(MONTHS, "may 2024 release note")

    assert (found.rung, found.phrase) == (AnchorRung.STEMMED, "May 2024 release notes")
    assert mismatched == frozenset()


def test_the_same_year_without_the_month_is_refused_on_the_stem_set_rung() -> None:
    matches, mismatched = ladder(MONTHS, "april 2024 release notes guide")

    assert matches == []
    assert texts(MONTHS, mismatched) == ["2024 release notes", "2024 release notes"]
    assert len(mismatched) == 2


# ── the semantic rung ───────────────────────────────────────────────────────

E = np.eye(6)


def vectors(phrases: list, planted: dict[str, np.ndarray]) -> np.ndarray:  # type: ignore[type-arg]
    """Every phrase far from the keyword, except the planted ones."""
    return np.stack([planted.get(p.text, E[5]) for p in phrases])


def test_the_closest_phrase_naming_another_month_is_refused_for_the_next_that_agrees() -> None:
    source = index("Read the June 2024 digest and the July 2024 patch notes.")
    phrases = candidate_phrases(source, 0, limit=100)
    rows = vectors(
        phrases,
        {"June 2024 digest": E[0], "July 2024 patch": 0.7 * E[0] + np.sqrt(0.51) * E[5]},
    )
    keywords = [(1, "Monthly digest July 2024", STRATEGIC)]

    found = semantic_match(source, TARGET, phrases, rows, keywords, E[:1], threshold=0.5).match

    assert found is not None
    assert (found.phrase, found.semantic_similarity) == ("July 2024 patch", pytest.approx(0.7))


def test_a_pair_whose_every_phrase_names_another_identifier_is_rejected_as_such() -> None:
    source = index("Read the CVE-2026-10002 notice and the widget patch notes.")
    phrases = candidate_phrases(source, 0, limit=100)
    rows = np.stack([E[0] for _ in phrases])
    keywords = [(1, CVE_WIDGET, STRATEGIC)]

    outcome = semantic_match(source, TARGET, phrases, rows, keywords, E[:1], threshold=0.5)

    assert outcome == SemanticOutcome(None, "identifier")
    assert eligible_phrases(phrases, rows, keywords, E[:1], threshold=0.5, stems=EN) == []


PLAIN = "Light trail shoes grip wet rock."
KEYWORDS = [(1, "hiking boots", STRATEGIC), (2, "footwear", KeywordSource.GSC_OBSERVED)]


def rival_outcome(rival: float, planted: dict[str, np.ndarray]) -> SemanticOutcome:
    source = index(PLAIN)
    phrases = candidate_phrases(source, 0)
    shoes = next(i for i, p in enumerate(phrases) if p.text == "trail shoes grip")
    return semantic_match(
        source,
        TARGET,
        phrases,
        vectors(phrases, planted),
        KEYWORDS,
        E[:2],
        threshold=0.5,
        other_best={shoes: rival},
    )


def test_a_generic_phrase_closer_to_another_targets_keyword_is_refused() -> None:
    planted = {"trail shoes grip": 0.8 * E[1] + 0.6 * E[5]}

    assert rival_outcome(0.85, planted) == SemanticOutcome(None, "other_target")
    assert rival_outcome(0.8 + 1e-6, planted) == SemanticOutcome(None, "other_target")
    tie = rival_outcome(0.8, planted).match
    assert tie is not None, "a tie with another target passes"
    assert tie.phrase == "trail shoes grip"


def test_the_next_phrase_without_a_closer_rival_wins() -> None:
    planted = {
        "trail shoes grip": 0.8 * E[1] + 0.6 * E[5],
        "grip wet rock": 0.6 * E[1] + 0.8 * E[5],
    }

    found = rival_outcome(0.95, planted).match

    assert found is not None
    assert (found.phrase, found.semantic_similarity) == ("grip wet rock", pytest.approx(0.6))


def test_nothing_at_the_threshold_is_no_match_and_no_rejection() -> None:
    assert rival_outcome(0.95, {}) == SemanticOutcome(None)


def test_the_best_other_cosine_leaves_out_each_rows_own_keywords() -> None:
    phrases = np.stack([E[0], E[1], E[2]])
    keywords = np.stack([E[0], E[1]])

    assert best_other_cosines(phrases, keywords, [[0], [], [0, 1]]).tolist() == [
        0.0,
        1.0,
        -np.inf,
    ]
    assert OTHER_TARGET_CHUNK == 512
    many = np.stack([E[i % 3] for i in range(7)])
    excluded = [[i % 2] for i in range(7)]
    np.testing.assert_array_equal(
        best_other_cosines(many, keywords, excluded, chunk=2),
        best_other_cosines(many, keywords, excluded),
    )
    assert best_other_cosines(phrases, keywords[:0], [[], [], []]).tolist() == [-np.inf] * 3
    with pytest.raises(ValueError, match="one excluded set per phrase"):
        best_other_cosines(phrases, keywords, [[0]])


# ── brand tokens ────────────────────────────────────────────────────────────

BRAND = frozenset({"acme7"})


@pytest.mark.parametrize(
    ("text", "found"),
    [
        ("Acme7 patch management", set()),
        ("CVE-2026-1 Acme7", {"1", "2026"}),
        # A brand token is no number for a month to stand next to.
        ("July Acme7 release", set()),
        ("July 2024 Acme7", {"2024", "month:7"}),
    ],
)
def test_a_brand_token_with_a_digit_is_never_an_identifier(text: str, found: set[str]) -> None:
    assert identifiers(text, EN, BRAND) == found


def test_the_brand_is_what_lets_a_branded_phrase_agree() -> None:
    assert agrees("Acme7 patch management", "patch management", EN, BRAND)
    assert not agrees("Acme7 patch management", "patch management", EN), "without the brand"
    assert agrees("CVE-2026-1 Acme7", "cve 2026 1 advisory", EN, BRAND)
    assert not agrees("Acme7 advisory", "cve 2026 1 advisory", EN, BRAND), "the CVE still counts"


def test_a_brand_inside_a_stem_set_span_no_longer_refuses_it() -> None:
    body = "Run the patch Acme7 management console daily."
    keyword = [(1, "console patch management tool", STRATEGIC)]

    unbranded, _, refused = extract(SourceIndex(SOURCE, body, (), EN), TARGET, keyword)
    branded, _, none = extract(SourceIndex(SOURCE, body, (), EN, brand=BRAND), TARGET, keyword)

    assert unbranded == []
    assert texts(body, refused) == ["patch Acme7 management console"]
    [found] = branded
    assert (found.rung, found.phrase) == (AnchorRung.STEM_SET, "patch Acme7 management console")
    assert none == frozenset()


def test_the_brand_keeps_a_branded_stemmed_variant() -> None:
    body = "Our Acme7 patch management console keeps servers current."
    index = SourceIndex(SOURCE, body, (), EN, brand=BRAND)

    [found], _, mismatched = extract(index, TARGET, [(1, "acme7 patching console", STRATEGIC)])

    assert (found.rung, found.phrase, mismatched) == (
        AnchorRung.STEMMED,
        "Acme7 patch",
        frozenset(),
    )


# ── strong identifiers on the stem set rung ─────────────────────────────────


@pytest.mark.parametrize(
    ("identifier", "strong"),
    [
        ("kb5034441", True),
        ("22h2", True),
        ("x86", True),
        ("16856", True),
        ("1234", True),
        ("2100", True),
        ("2026", False),
        ("1999", False),
        ("11", False),
        ("10", False),
        ("month:7", False),
    ],
)
def test_a_strong_identifier_is_a_code_or_a_long_number_that_is_not_a_year(
    identifier: str, strong: bool
) -> None:
    assert strong_identifier(identifier) is strong


CODES = "Issues include CVE-2026-16856, CVE-2026-17083."


@pytest.mark.parametrize(
    ("code", "phrase"),
    [("16856", "CVE-2026-16856"), ("17083", "CVE-2026-17083"), ("20270", None)],
)
def test_a_code_written_as_one_compound_matches_its_own_id_only(
    code: str, phrase: str | None
) -> None:
    matches, mismatched = ladder(CODES, f"CVE-2026-{code} \u2013 Widget")

    if phrase is None:
        assert (matches, mismatched) == ([], frozenset())
    else:
        [found] = matches
        # {cve, 2026, code} against {cve, 2026, code, widget}.
        assert (found.rung, found.phrase, found.stem_jaccard) == (
            AnchorRung.STEM_SET,
            phrase,
            0.75,
        )


def test_a_code_matches_beside_other_words_of_the_title() -> None:
    [found], _ = ladder(
        "See the CVE-2026-16856 fix for IBM i systems.", "CVE-2026-16856 \u2013 IBM i"
    )

    assert (found.rung, found.phrase, found.stem_jaccard) == (
        AnchorRung.STEM_SET,
        "CVE-2026-16856",
        0.75,
    )


def test_the_two_word_rule_still_holds_without_a_strong_identifier() -> None:
    [found], _ = ladder("Upgrade to Windows 11 today.", "windows 11 upgrade guide")
    missed, _ = ladder("Our 2023 report is out.", "2023 annual report review")

    assert (found.rung, found.phrase) == (AnchorRung.STEM_SET, "Upgrade to Windows 11")
    assert missed == [], "a year alone grants nothing: 2 of 4 stems is below 0.6"
