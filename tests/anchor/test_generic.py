"""Anchor normalisation and the generic-anchor dictionary (issue #6 gotchas)."""

from __future__ import annotations

import pytest

from linking_engine.anchor.generic import (
    GENERIC_ANCHORS,
    generic_overrides,
    is_generic,
    normalise_anchor,
)

# The 13 entries of GENERIC_ANCHORS in files/linking-engine-docs/dev/scripts/corpus/taxonomy.py.
CORPUS_GENERIC = (
    "click here",
    "here",
    "read more",
    "learn more",
    "this page",
    "find out more",
    "see more",
    "more info",
    "more information",
    "continue reading",
    "view more",
    "check it out",
    "details",
)


# "click here" in fullwidth letters with an ideographic space; NFKC folds it to ASCII.
FULLWIDTH = (
    "".join(chr(ord(char) + 0xFEE0) for char in "click")
    + "\N{IDEOGRAPHIC SPACE}"
    + "".join(chr(ord(char) + 0xFEE0) for char in "here")
)


# ── normalise_anchor ────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "variant",
    [
        "Click Here",
        "click here ",
        "click  here.",
        "  CLICK\tHERE!! ",
        "click\nhere",
        "(click here)",
    ],
)
def test_case_spacing_and_edge_punctuation_variants_share_one_key(variant: str) -> None:
    assert normalise_anchor(variant) == "click here"


@pytest.mark.parametrize(
    ("text", "key"),
    [
        pytest.param(FULLWIDTH, "click here", id="nfkc-fullwidth"),
        pytest.param("click\N{NO-BREAK SPACE}here", "click here", id="nfkc-no-break-space"),
        pytest.param("\N{LATIN SMALL LIGATURE FI}le formats", "file formats", id="nfkc-ligature"),
        pytest.param("Straße", "strasse", id="casefold-not-lower"),
        pytest.param("« click here »", "click here", id="guillemets-and-inner-spaces"),
        pytest.param('"Read more…"', "read more", id="quotes-and-ellipsis"),
        pytest.param("¿Qué es?", "qué es", id="inverted-question-mark"),
    ],
)
def test_nfkc_casefold_and_unicode_punctuation(text: str, key: str) -> None:
    assert normalise_anchor(text) == key


@pytest.mark.parametrize(
    ("text", "key"),
    [
        pytest.param("C#", "c#", id="c-sharp"),
        pytest.param("F#.", "f#", id="f-sharp-trailing-dot"),
        pytest.param("#hashtag", "#hashtag", id="leading-hash"),
        pytest.param("C", "c", id="c"),
    ],
)
def test_hash_is_never_stripped(text: str, key: str) -> None:
    assert normalise_anchor(text) == key


def test_c_sharp_and_c_are_distinct_keys() -> None:
    assert normalise_anchor("C#") != normalise_anchor("C")


@pytest.mark.parametrize(
    ("text", "key"),
    [
        pytest.param("don't miss", "don't miss", id="apostrophe"),
        pytest.param("e-mail  setup", "e-mail setup", id="hyphen"),
        pytest.param("U.S.A.", "u.s.a", id="dots"),
        pytest.param(
            "plus d\N{RIGHT SINGLE QUOTATION MARK}infos",
            "plus d\N{RIGHT SINGLE QUOTATION MARK}infos",
            id="typographic-apostrophe",
        ),
    ],
)
def test_inner_punctuation_is_kept(text: str, key: str) -> None:
    assert normalise_anchor(text) == key


def test_arrows_and_pictographs_leave_the_key_but_meaningful_symbols_stay() -> None:
    assert normalise_anchor("Read more →") == "read more"
    assert normalise_anchor("→ Next") == "next"
    assert normalise_anchor("Learn more 👉") == "learn more"
    assert normalise_anchor("Acme®") == "acme"
    assert normalise_anchor("C++") == "c++"
    assert normalise_anchor("$100") == "$100"
    assert normalise_anchor("→") == "→"


@pytest.mark.parametrize(
    "text", ["2", "10.", "$100", "5 €", "Next page", "Zurück", "Précédent", "older posts"]
)
def test_numbers_and_pagination_are_generic(text: str) -> None:
    assert is_generic(text)


def test_tenant_overrides_add_and_remove_generic_phrases() -> None:
    add, remove = generic_overrides(frozenset({"Download now"}), frozenset({"Details"}))
    assert is_generic("download now!", add=add, remove=remove)
    assert not is_generic("Details", add=add, remove=remove)
    assert is_generic("Details")


@pytest.mark.parametrize("text", ["", "   ", "\t\n", "...!!!", "« »", "“ ”"])
def test_blank_or_punctuation_only_normalises_to_empty(text: str) -> None:
    assert normalise_anchor(text) == ""


@pytest.mark.parametrize("text", ["Click Here", "« (C#) »", FULLWIDTH + ".", "Read more →"])
def test_normalising_is_idempotent(text: str) -> None:
    key = normalise_anchor(text)
    assert normalise_anchor(key) == key


# ── GENERIC_ANCHORS and is_generic ──────────────────────────────────────────


def test_dictionary_holds_every_corpus_entry_already_normalised() -> None:
    assert set(CORPUS_GENERIC) <= GENERIC_ANCHORS
    assert all(normalise_anchor(entry) == entry for entry in GENERIC_ANCHORS), sorted(
        entry for entry in GENERIC_ANCHORS if normalise_anchor(entry) != entry
    )


@pytest.mark.parametrize("entry", CORPUS_GENERIC)
def test_every_corpus_entry_is_generic_in_any_casing(entry: str) -> None:
    assert is_generic(entry)
    assert is_generic(f"  {entry.upper()}. ")


@pytest.mark.parametrize(
    "text",
    [
        "Hier klicken",
        "Mehr erfahren",
        "Weiterlesen",
        "Diese Seite",
        "Cliquez ici",
        "En savoir plus",
        "Lire la suite",
        "plus d'infos",
        "plus d\N{RIGHT SINGLE QUOTATION MARK}infos",
        "Plus d\N{RIGHT SINGLE QUOTATION MARK}informations",
    ],
)
def test_german_and_french_equivalents_are_generic(text: str) -> None:
    assert is_generic(text)


@pytest.mark.parametrize(
    "text",
    [
        "Read more →",
        "→ Read more",
        "▸ Details",
        "Learn more ↗",
        "Read more »",
        "Here \N{SINGLE RIGHT-POINTING ANGLE QUOTATION MARK}",
    ],
)
def test_edge_symbols_and_arrows_do_not_hide_a_generic_anchor(text: str) -> None:
    assert is_generic(text)


@pytest.mark.parametrize(
    "text",
    [
        "trail running shoes",
        "read more about trail shoes",
        "here's why",
        "click",
        "C#",
        "",
        "   ",
        "!!!",
    ],
)
def test_descriptive_empty_and_partial_anchors_are_not_generic(text: str) -> None:
    assert not is_generic(text)


@pytest.mark.parametrize(
    ("text", "key"),
    [
        pytest.param("click here\N{ZERO WIDTH SPACE}", "click here", id="zero-width-space"),
        pytest.param("Docu\N{SOFT HYPHEN}mentation", "documentation", id="soft-hyphen"),
        pytest.param("\N{ZERO WIDTH NO-BREAK SPACE}Trail shoes", "trail shoes", id="bom"),
    ],
)
def test_invisible_format_characters_are_dropped(text: str, key: str) -> None:
    assert normalise_anchor(text) == key


@pytest.mark.parametrize("text", ["→", " ▸ ", "→ »"])
def test_a_symbol_only_anchor_is_generic_and_keeps_its_key(text: str) -> None:
    assert normalise_anchor(text) != ""
    assert is_generic(text)
