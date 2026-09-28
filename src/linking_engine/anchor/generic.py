"""Anchor-text normalisation and the generic-anchor vocabulary."""

import unicodedata
from collections.abc import Mapping
from typing import Final

_TYPOGRAPHIC_APOSTROPHE: Final = "\N{RIGHT SINGLE QUOTATION MARK}"

# Per language, already normalised; matching is on the whole anchor, so the union is safe.
GENERIC_ANCHORS_BY_LANGUAGE: Final[Mapping[str, frozenset[str]]] = {
    "en": frozenset(
        {
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
            "next",
            "previous",
            "prev",
            "back",
            "older",
            "newer",
            "older posts",
            "newer posts",
            "next page",
            "previous page",
        }
    ),
    "de": frozenset(
        {
            "hier klicken",
            "klicken sie hier",
            "hier",
            "mehr erfahren",
            "weiterlesen",
            "mehr lesen",
            "mehr",
            "mehr anzeigen",
            "mehr dazu",
            "mehr informationen",
            "mehr infos",
            "weitere informationen",
            "diese seite",
            "weiter",
            "zurück",
            "nächste",
            "vorherige",
            "nächste seite",
            "vorherige seite",
        }
    ),
    "fr": frozenset(
        {
            "cliquez ici",
            "cliquer ici",
            "ici",
            "en savoir plus",
            "lire la suite",
            "lire plus",
            "voir plus",
            "voir la suite",
            "plus d'infos",
            "plus d'informations",
            "détails",
            "cette page",
            "suivant",
            "suivante",
            "précédent",
            "précédente",
            "page suivante",
            "page précédente",
        }
    ),
}
_ALL: Final = frozenset().union(*GENERIC_ANCHORS_BY_LANGUAGE.values())
# NFKC keeps U+2019, so apostrophe entries get both forms.
GENERIC_ANCHORS: Final[frozenset[str]] = _ALL | frozenset(
    anchor.replace("'", _TYPOGRAPHIC_APOSTROPHE) for anchor in _ALL if "'" in anchor
)
# Arrows are math symbols (Sm) like "+", so they are listed by block.
_ARROW_BLOCKS: Final = ((0x2190, 0x21FF), (0x27F0, 0x27FF), (0x2900, 0x297F), (0x2B00, 0x2BFF))


def normalise_anchor(text: str) -> str:
    """Case-, width- and spacing-insensitive key. Edge punctuation, arrows and pictographs
    (emoji, dingbats, ™, ®) are removed; '#', '+', '$' and other meaningful symbols stay."""
    # Format characters (zero-width space, soft hyphen, BOM) are invisible scraping residue.
    visible = "".join(char for char in text if unicodedata.category(char) != "Cf")
    folded = " ".join(
        unicodedata.normalize("NFKC", unicodedata.normalize("NFKC", visible).casefold()).split()
    )
    # A symbol-only anchor ("→") keeps its symbol, so it still gets a key and reads as generic.
    return _strip_edges(folded, symbols=False) or _strip_edges(
        folded, symbols=False, decorations=False
    )


def generic_overrides(
    add: frozenset[str], remove: frozenset[str]
) -> tuple[frozenset[str], frozenset[str]]:
    """Normalise a tenant's extra generic phrases and never-generic phrases once per run."""
    return (
        frozenset(filter(None, map(normalise_anchor, add))),
        frozenset(filter(None, map(normalise_anchor, remove))),
    )


def is_generic(
    text: str, *, add: frozenset[str] = frozenset(), remove: frozenset[str] = frozenset()
) -> bool:
    """True when the anchor says nothing about its target: a generic phrase, pagination,
    or no letters at all ("2", "→"). Text that normalises to nothing ("»", "...") has no
    key and is not called generic. ``add``/``remove`` are normalised tenant overrides."""
    key = normalise_anchor(text)
    if not key or key in remove:
        return False
    phrase = _strip_edges(key, symbols=True)
    return (
        key in add
        or phrase in add
        or not any(char.isalpha() for char in phrase)
        or phrase in GENERIC_ANCHORS
    )


def _strip_edges(text: str, *, symbols: bool, decorations: bool = True) -> str:
    """Trim edge whitespace and punctuation (and symbols if asked) in one pass, which is the fixpoint."""
    start, end = 0, len(text)
    while start < end and _is_edge(text[start], symbols=symbols, decorations=decorations):
        start += 1
    while end > start and _is_edge(text[end - 1], symbols=symbols, decorations=decorations):
        end -= 1
    return text[start:end]


def _is_edge(char: str, *, symbols: bool, decorations: bool = True) -> bool:
    if char.isspace():
        return True
    # Keeps "c#" and "f#" distinct from "c" and "f".
    if char == "#":
        return False
    category = unicodedata.category(char)
    if category.startswith("P") or (decorations and (category == "So" or _is_arrow(char))):
        return True
    return symbols and category.startswith("S")


def _is_arrow(char: str) -> bool:
    code = ord(char)
    return any(low <= code <= high for low, high in _ARROW_BLOCKS)
