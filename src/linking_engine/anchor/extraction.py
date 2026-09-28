"""The extraction ladder: where a target's keyword is already written in a source page.

Anchor text is extracted, never generated. Rung 1 finds the keyword verbatim, rung 2 as a
stemmed variant or with one modifier dropped, rung 2.5 as a short span sharing most of its
stems. Only the keyword and the page's tokens are stemmed, never the document, so every offset
points into the stored body text. A heading line or an existing link's anchor never yields one.
"""

from __future__ import annotations

import functools
import re
import time
from bisect import bisect_right
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import resources
from itertools import pairwise
from types import MappingProxyType
from typing import TYPE_CHECKING, Final

import numpy as np
import snowballstemmer

from linking_engine.models import AnchorMatch, AnchorReport, AnchorRung, ExtractionSettings
from linking_engine.models.anchors import NO_LANGUAGE, SENTENCE_INDEX_BINS, STEM_JACCARD_BINS

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence

    from linking_engine.models import KeywordSource

STAGE: Final = "anchor-extraction"
DEFAULT_THRESHOLD: Final = ExtractionSettings().stem_set_threshold
# Token counts of the spans the stem set rung compares with the keyword.
MIN_SPAN: Final = 2
MAX_SPAN: Final = 5
# At least this many of the keyword's stems in a stem set span.
MIN_SHARED_STEMS: Final = 2
# Stop words a stem set span may hold between its edges; more reads as a clause, not a phrase.
MAX_INNER_STOP_WORDS: Final = 1
# Language primary subtag (ISO 639-1) -> Snowball stemmer.
SNOWBALL: Final = {
    "ar": "arabic",
    "ca": "catalan",
    "cs": "czech",
    "da": "danish",
    "de": "german",
    "el": "greek",
    "en": "english",
    "eo": "esperanto",
    "es": "spanish",
    "et": "estonian",
    "eu": "basque",
    "fa": "persian",
    "fi": "finnish",
    "fr": "french",
    "ga": "irish",
    "hi": "hindi",
    "hu": "hungarian",
    "hy": "armenian",
    "id": "indonesian",
    "it": "italian",
    "lt": "lithuanian",
    "nb": "norwegian",
    "ne": "nepali",
    "nl": "dutch",
    "nn": "norwegian",
    "no": "norwegian",
    "pl": "polish",
    "pt": "portuguese",
    "ro": "romanian",
    "ru": "russian",
    "sr": "serbian",
    "st": "sesotho",
    "sv": "swedish",
    "ta": "tamil",
    "tr": "turkish",
    "yi": "yiddish",
}
# The lexical rungs, in order; the semantic rung (#22) is not part of the ladder.
_LADDER: Final = (AnchorRung.EXACT, AnchorRung.STEMMED, AnchorRung.STEM_SET)

_APOSTROPHES: Final = (
    "\N{RIGHT SINGLE QUOTATION MARK}"
    "\N{LEFT SINGLE QUOTATION MARK}"
    "\N{MODIFIER LETTER APOSTROPHE}"
    "\N{PRIME}"
)
_TOKEN = re.compile(rf"\w+(?:['{_APOSTROPHES}]\w+)*")
# Typographic apostrophes compare as the straight one, which the stemmers strip possessives on;
# the dot above left by casefolding a dotted capital I is dropped.
_FOLD: Final = str.maketrans({**dict.fromkeys(_APOSTROPHES, "'"), "\N{COMBINING DOT ABOVE}": None})
_LINE = re.compile(r"[^\r\n]+")
# Within a line, a sentence ends after . ! ? or an ellipsis followed by whitespace.
_SENTENCE_END = re.compile(r"(?<=[.!?\u2026])\s+")
_SPACE = re.compile(r"\s+")


@dataclass(frozen=True, slots=True)
class Sentence:
    # The sentence's place among the body's sentences, headings included, 0 first.
    index: int
    start: int
    text: str


@dataclass(frozen=True, slots=True)
class Token:
    start: int
    end: int
    folded: str


def _collapse(text: str) -> str:
    return " ".join(text.split()).casefold()


def sentences(text: str, headings: Iterable[str] = ()) -> list[Sentence]:
    """The body's sentences, stripped, with their offsets. The body is split into lines first;
    a line equal to a heading is left out, its sentences keeping their place in the numbering."""
    skip = {_collapse(heading) for heading in headings}
    found: list[Sentence] = []
    index = 0
    for line in _LINE.finditer(text):
        heading = _collapse(line.group()) in skip
        for piece_start, piece_end in _pieces(line.group()):
            piece = line.group()[piece_start:piece_end]
            stripped = piece.strip()
            if not stripped:
                continue
            if not heading:
                start = line.start() + piece_start + len(piece) - len(piece.lstrip())
                found.append(Sentence(index=index, start=start, text=stripped))
            index += 1
    return found


def _pieces(line: str) -> Iterator[tuple[int, int]]:
    start = 0
    for boundary in _SENTENCE_END.finditer(line):
        yield start, boundary.start()
        start = boundary.end()
    yield start, len(line)


def tokens(text: str, offset: int = 0) -> list[Token]:
    """Unicode word tokens with their offsets (shifted by ``offset``); compared casefolded,
    typographic apostrophes as "'"."""
    return [
        Token(
            start=offset + match.start(),
            end=offset + match.end(),
            folded=match.group().casefold().translate(_FOLD),
        )
        for match in _TOKEN.finditer(text)
    ]


def _gap(gap: str) -> str:
    """A gap between two tokens with its whitespace runs as one space; only whitespace is " "."""
    return _SPACE.sub(" ", gap)


def _compound(gap: str) -> bool:
    """A gap without whitespace joins its tokens into one compound, as in e-mail or node.js."""
    return bool(gap) and not _SPACE.search(gap)


def algorithm_for(language: str | None) -> str | None:
    """The Snowball algorithm of a language's primary subtag; None when there is none."""
    if not language:
        return None
    return SNOWBALL.get(re.split(r"[-_]", language, maxsplit=1)[0].casefold())


def stemmer_for(language: str | None) -> Callable[[str], str] | None:
    """A new Snowball stemmer for the language; None when there is none. A stemmer is not
    thread-safe, so each run keeps its own."""
    name = algorithm_for(language)
    if name is None:
        return None
    stem = snowballstemmer.stemmer(name).stemWord

    def stemmed(token: str) -> str:
        return str(stem(token))

    return stemmed


@functools.cache
def _stop_words(algorithm: str) -> frozenset[str]:
    listed = resources.files("linking_engine.anchor").joinpath("stopwords", f"{algorithm}.txt")
    if not listed.is_file():
        return frozenset()
    words = listed.read_text(encoding="utf-8").split()
    return frozenset(word.casefold().translate(_FOLD) for word in words)


@functools.cache
def _month_stems(algorithm: str) -> Mapping[str, int]:
    listed = resources.files("linking_engine.anchor").joinpath("months", f"{algorithm}.txt")
    if not listed.is_file():
        return MappingProxyType({})
    stem = snowballstemmer.stemmer(algorithm).stemWord
    found: dict[str, int] = {}
    for line in listed.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        number, name = line.split("\t")
        found.setdefault(str(stem(name.casefold().translate(_FOLD))), int(number))
    return MappingProxyType(found)


def month_stems(language: str | None) -> Mapping[str, int]:
    """The packaged month names (full and abbreviated) of the language's stemmer algorithm, as
    stems to month numbers; none for a language without a list."""
    algorithm = algorithm_for(language)
    return _month_stems(algorithm) if algorithm else MappingProxyType({})


def stop_words_for(language: str | None) -> frozenset[str]:
    """The packaged stop words of the language's stemmer algorithm, as token keys; none for a
    language without a list."""
    algorithm = algorithm_for(language)
    return _stop_words(algorithm) if algorithm else frozenset()


class Stems:
    """One language's token stems, cached for the run, and its stop words; casefolded tokens as
    they are when the language has no stemmer."""

    __slots__ = ("_cache", "_stem", "language", "months", "stop_words")

    def __init__(self, language: str | None) -> None:
        self.language = language
        self.stop_words = stop_words_for(language)
        # Month name stems -> month number; empty for a language without a list.
        self.months = month_stems(language)
        self._stem = stemmer_for(language)
        self._cache: dict[str, str] = {}

    @property
    def stemmed(self) -> bool:
        return self._stem is not None

    def __call__(self, token: str) -> str:
        if self._stem is None:
            return token
        found = self._cache.get(token)
        if found is None:
            found = self._cache[token] = self._stem(token)
        return found


def locate_anchors(
    body: str, links: Iterable[tuple[str, str]]
) -> tuple[list[tuple[int, int]], int]:
    """Character spans of existing links' anchors in the body, from (anchor text, surrounding
    text): the surrounding text's first occurrence, then the anchor's inside it. Also how many
    links could not be located, whose anchors therefore protect nothing."""
    spans: set[tuple[int, int]] = set()
    unlocated = 0
    for anchor, surrounding in links:
        around = body.find(surrounding) if anchor and surrounding else -1
        inside = surrounding.find(anchor) if around >= 0 else -1
        if inside < 0:
            unlocated += 1
            continue
        start = around + inside
        spans.add((start, start + len(anchor)))
    return sorted(spans), unlocated


def existing_spans(body: str, links: Iterable[tuple[str, str]]) -> list[tuple[int, int]]:
    """The located spans of `locate_anchors`."""
    return locate_anchors(body, links)[0]


def keyword_tokens(keyword: str) -> list[str]:
    """The keyword's casefolded tokens."""
    return [token.folded for token in tokens(keyword)]


def _block_ids(gaps: Sequence[str]) -> list[int]:
    """The block of each token from the gap before it ("" before the first): a block ends at
    punctuation with whitespace around it (", ", " - "), as in `SourceIndex`."""
    block = 0
    found: list[int] = []
    for i, gap in enumerate(gaps):
        if i and not _compound(gap) and gap != " ":
            block += 1
        found.append(block)
    return found


# A four-digit number of this form is a year, which many texts share.
_YEAR: Final = re.compile(r"(?:19|20)\d\d")


def strong_identifier(identifier: str) -> bool:
    """A code that names one thing: letters with digits (KB5034441, 22H2), or a number of four
    or more digits that is not a year (16856). Years, small numbers and months are not."""
    if identifier.startswith("month:"):
        return False
    if any(char.isalpha() for char in identifier) and any(char.isdigit() for char in identifier):
        return True
    return identifier.isdigit() and len(identifier) >= 4 and not _YEAR.fullmatch(identifier)


def _identifiers(
    folded: Sequence[str], blocks: Sequence[int], stems: Stems, brand: frozenset[str]
) -> frozenset[str]:
    """Tokens holding a digit, and ``month:<n>`` for a month name next to one in its block;
    ``brand`` tokens are neither."""
    digit = [token not in brand and any(char.isdigit() for char in token) for token in folded]
    found = {token for token, has in zip(folded, digit, strict=True) if has}
    if stems.months:
        for i, token in enumerate(folded):
            month = stems.months.get(stems(token)) if token not in brand else None
            if month is not None and any(
                0 <= j < len(folded) and digit[j] and blocks[j] == blocks[i] for j in (i - 1, i + 1)
            ):
                found.add(f"month:{month}")
    return frozenset(found)


def identifiers(text: str, stems: Stems, brand: frozenset[str] = frozenset()) -> frozenset[str]:
    """What a text names that a synonym cannot stand in for: its tokens holding a digit (ids,
    versions, years), folded, and its month names next to one of them ("July 2024", "2 July"),
    as ``month:<n>``. A phrase stands for a keyword only when both hold the same ones. The
    tenant's ``brand`` tokens (folded) never count, even with a digit ("Acme7")."""
    found = tokens(text)
    gaps = ["", *(_gap(text[a.end : b.start]) for a, b in pairwise(found))]
    return _identifiers([token.folded for token in found], _block_ids(gaps), stems, brand)


def content_stems(keyword: Sequence[str], stems: Stems) -> frozenset[str]:
    """The distinct stems of a keyword's tokens that are not stop words, what the stem set rung
    compares."""
    return frozenset(stems(token) for token in keyword if token not in stems.stop_words)


@dataclass(frozen=True, slots=True)
class _Needle:
    """A run of keyword tokens to find, with the gap text accepted after each but the last."""

    values: tuple[str, ...]
    gaps: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _Found:
    rung: AnchorRung
    sentence: int
    first: int
    last: int
    jaccard: float | None = None


@dataclass(frozen=True, slots=True)
class _Keyword:
    exact: _Needle
    # Stemmed: the whole keyword, then each valid modifier drop.
    variants: tuple[_Needle, ...]
    # The distinct stems of its content tokens, what the stem set rung compares.
    stems: frozenset[str]
    identifiers: frozenset[str]


def _keyword(keyword: str, stems: Stems, brand: frozenset[str] = frozenset()) -> _Keyword:
    """The keyword verbatim, its stemmed variants (whole, then with one modifier dropped at
    either end), and its content stems.

    A compound counts as one word; a modifier is dropped only from keywords of three or more
    words, and a dropped variant starts and ends on a content token and keeps two of them.
    """
    found = tokens(keyword)
    folded = [token.folded for token in found]
    gaps = [_gap(keyword[a.end : b.start]) for a, b in pairwise(found)]
    stemmed = [stems(token) for token in folded]
    exact = _Needle(tuple(folded), tuple(gaps))
    variants = [_Needle(tuple(stemmed), tuple(gaps))]
    content = [token not in stems.stop_words for token in folded]
    # First token of each word.
    words = [0, *(i + 1 for i, gap in enumerate(gaps) if not _compound(gap))]
    if len(words) >= 3:
        for first, last in ((words[1], len(folded) - 1), (0, words[-1] - 1)):
            if content[first] and content[last] and sum(content[first : last + 1]) >= 2:
                variants.append(_Needle(tuple(stemmed[first : last + 1]), tuple(gaps[first:last])))
    return _Keyword(
        exact=exact,
        variants=tuple(variants),
        stems=frozenset(stem for stem, kept in zip(stemmed, content, strict=True) if kept),
        identifiers=_identifiers(folded, _block_ids(["", *gaps]), stems, brand),
    )


class SourceIndex:
    """One source page's sentences and tokens, indexed by casefolded text and by stem; built once
    per page and probed for every keyword of every pair the page is the source of."""

    __slots__ = (
        "_blocks",
        "_by_stem",
        "_by_text",
        "_content",
        "_folded",
        "_gaps",
        "_stems",
        "_tokens",
        "_words",
        "body",
        "brand",
        "sentences",
        "stems",
        "url",
    )

    def __init__(
        self,
        url: str,
        body: str,
        headings: Iterable[str],
        stems: Stems,
        *,
        brand: frozenset[str] = frozenset(),
    ) -> None:
        self.url = url
        self.body = body
        self.stems = stems
        # The tenant's brand tokens (folded), which never count as identifiers.
        self.brand = brand
        self.sentences = sentences(body, headings)
        self._tokens: list[list[Token]] = []
        self._folded: list[list[str]] = []
        self._stems: list[list[str]] = []
        # The gap before each token, whitespace runs as one space; "" before the first.
        self._gaps: list[list[str]] = []
        # A block ends at punctuation with whitespace around it (", ", " - ", " & "); a stem set
        # span stays inside one.
        self._blocks: list[list[int]] = []
        # The word each token belongs to: the tokens of a compound are one word.
        self._words: list[list[int]] = []
        # Whether a token is a content word rather than a stop word.
        self._content: list[list[bool]] = []
        self._by_text: dict[str, list[tuple[int, int]]] = {}
        self._by_stem: dict[str, list[tuple[int, int]]] = {}
        for position, sentence in enumerate(self.sentences):
            found = tokens(sentence.text, sentence.start)
            stemmed = [stems(token.folded) for token in found]
            gaps = [""] + [_gap(body[a.end : b.start]) for a, b in pairwise(found)]
            blocks: list[int] = []
            words: list[int] = []
            block = word = 0
            for i, token in enumerate(found):
                if i and not _compound(gaps[i]):
                    word += 1
                    if gaps[i] != " ":
                        block += 1
                blocks.append(block)
                words.append(word)
                self._by_text.setdefault(token.folded, []).append((position, i))
                self._by_stem.setdefault(stemmed[i], []).append((position, i))
            self._tokens.append(found)
            self._folded.append([token.folded for token in found])
            self._stems.append(stemmed)
            self._gaps.append(gaps)
            self._blocks.append(blocks)
            self._words.append(words)
            self._content.append([token.folded not in stems.stop_words for token in found])

    def _whole(self, position: int, first: int, last: int) -> bool:
        """Whether a phrase of these tokens cuts no compound at either end."""
        gaps = self._gaps[position]
        return not _compound(gaps[first]) and (
            last + 1 == len(gaps) or not _compound(gaps[last + 1])
        )

    def _runs(self, needle: _Needle, *, stemmed: bool) -> list[tuple[int, int, int]]:
        """(sentence position, first token, last token) of every contiguous occurrence of the
        needle in body order, probed from its rarest token. Between two of its tokens the body
        has whitespace or the keyword's own gap there."""
        index = self._by_stem if stemmed else self._by_text
        values = needle.values
        if not values or any(value not in index for value in values):
            return []
        pivot = min(range(len(values)), key=lambda k: len(index[values[k]]))
        rows = self._stems if stemmed else self._folded
        wanted = list(values)
        runs: list[tuple[int, int, int]] = []
        for position, i in index[values[pivot]]:
            first, last = i - pivot, i - pivot + len(values) - 1
            row, gaps = rows[position], self._gaps[position]
            if (
                first >= 0
                and last < len(row)
                and row[first : last + 1] == wanted
                and self._whole(position, first, last)
                and all(gaps[first + k + 1] in (" ", gap) for k, gap in enumerate(needle.gaps))
            ):
                runs.append((position, first, last))
        return runs

    def _stem_sets(
        self,
        stems: frozenset[str],
        threshold: float,
        identifiers: frozenset[str] = frozenset(),
    ) -> list[tuple[float, int, int, int]]:
        """(Jaccard, sentence position, first token, last token) of every span inside one block
        whose first and last tokens are content words with a keyword stem, with at most one stop
        word between them and no compound cut, reaching the threshold. The keyword stems it holds
        lie in at least two of its words (a compound is one), unless it holds all of them or the
        keyword has a strong identifier and the span holds exactly the keyword's ``identifiers``
        (a code written as one compound, as CVE-2026-16856). Stop words count in neither set."""
        strong = any(strong_identifier(identifier) for identifier in identifiers)
        spans: list[tuple[float, int, int, int]] = []
        for stem in sorted(stems):
            for position, first in self._by_stem.get(stem, ()):
                row, content = self._stems[position], self._content[position]
                blocks, words = self._blocks[position], self._words[position]
                if not content[first]:
                    continue
                for last in range(first + MIN_SPAN - 1, min(first + MAX_SPAN, len(row))):
                    if blocks[last] != blocks[first]:
                        break
                    if (
                        not content[last]
                        or row[last] not in stems
                        or not self._whole(position, first, last)
                        or content[first : last + 1].count(False) > MAX_INNER_STOP_WORDS
                    ):
                        continue
                    span = {row[k] for k in range(first, last + 1) if content[k]}
                    shared = len(span & stems)
                    sharing = {
                        words[k] for k in range(first, last + 1) if content[k] and row[k] in stems
                    }
                    jaccard = shared / len(span | stems)
                    if (
                        shared >= MIN_SHARED_STEMS
                        and jaccard >= threshold
                        and (
                            len(sharing) >= MIN_SHARED_STEMS
                            or stems <= span
                            or (strong and self._agrees(position, first, last, identifiers))
                        )
                    ):
                        spans.append((jaccard, position, first, last))
        return spans

    def span(self, position: int, first: int, last: int) -> tuple[int, int]:
        row = self._tokens[position]
        return row[first].start, row[last].end

    def _agrees(self, position: int, first: int, last: int, wanted: frozenset[str]) -> bool:
        """Whether the span holds exactly the keyword's identifiers."""
        span = slice(first, last + 1)
        found = _identifiers(
            self._folded[position][span], self._blocks[position][span], self.stems, self.brand
        )
        return found == wanted

    def phrase_spans(self, position: int) -> list[tuple[int, int, int]]:
        """(first token, last token, content tokens) of every span of a sentence the ladder
        accepts as a phrase, whatever the keyword: MIN_SPAN to MAX_SPAN tokens inside one block,
        content words at both edges, at most MAX_INNER_STOP_WORDS stop words between them and
        no compound cut; in token order."""
        content, blocks = self._content[position], self._blocks[position]
        spans: list[tuple[int, int, int]] = []
        for first in range(len(content)):
            if not content[first]:
                continue
            for last in range(first + MIN_SPAN - 1, min(first + MAX_SPAN, len(content))):
                if blocks[last] != blocks[first]:
                    break
                inner = content[first : last + 1].count(False)
                if (
                    content[last]
                    and inner <= MAX_INNER_STOP_WORDS
                    and self._whole(position, first, last)
                ):
                    spans.append((first, last, last - first + 1 - inner))
        return spans

    def find(
        self,
        keyword: str,
        *,
        existing: Sequence[tuple[int, int]],
        threshold: float,
        blocking: set[tuple[int, int]],
        mismatched: set[tuple[int, int]],
    ) -> _Found | None:
        """The ladder for one keyword: the lowest rung that finds it, its earliest free place
        there. The existing anchor spans that took a place go into ``blocking``. Below the
        exact rung a place must hold exactly the keyword's identifiers, so a modifier drop
        cannot lose one and a stem set cannot swap one; a place refused so goes into
        ``mismatched``."""

        def free(position: int, first: int, last: int) -> bool:
            start, end = self.span(position, first, last)
            taken = [(low, high) for low, high in existing if start < high and low < end]
            blocking.update(taken)
            return not taken

        def agrees(position: int, first: int, last: int) -> bool:
            if self._agrees(position, first, last, prepared.identifiers):
                return True
            mismatched.add(self.span(position, first, last))
            return False

        prepared = _keyword(keyword, self.stems, self.brand)
        for position, first, last in self._runs(prepared.exact, stemmed=False):
            if free(position, first, last):
                return _Found(AnchorRung.EXACT, position, first, last)

        runs = sorted(
            {run for variant in prepared.variants for run in self._runs(variant, stemmed=True)},
            key=lambda run: (run[0], run[1], run[1] - run[2]),
        )
        for position, first, last in runs:
            if agrees(position, first, last) and free(position, first, last):
                return _Found(AnchorRung.STEMMED, position, first, last)

        if len(prepared.stems) < MIN_SHARED_STEMS:
            return None
        ranked = sorted(
            self._stem_sets(prepared.stems, threshold, prepared.identifiers),
            key=lambda span: (-span[0], span[1], span[3] - span[2], span[2]),
        )
        for jaccard, position, first, last in ranked:
            if agrees(position, first, last) and free(position, first, last):
                return _Found(AnchorRung.STEM_SET, position, first, last, jaccard)
        return None


def extract(
    index: SourceIndex,
    target_url: str,
    keywords: Sequence[tuple[int, str, KeywordSource]],
    *,
    existing: Sequence[tuple[int, int]] = (),
    threshold: float = DEFAULT_THRESHOLD,
) -> tuple[list[AnchorMatch], frozenset[tuple[int, int]], frozenset[tuple[int, int]]]:
    """The target's ranked keywords, as (rank, text, source), found in the source page: one
    match per keyword found, the ``existing`` anchor spans that blocked a place, and the spans
    refused because their identifiers disagreed with the keyword's."""
    if not 0 < threshold <= 1:
        raise ValueError("threshold must be in (0, 1]")
    matches: list[AnchorMatch] = []
    blocking: set[tuple[int, int]] = set()
    mismatched: set[tuple[int, int]] = set()
    for rank, text, source in keywords:
        found = index.find(
            text, existing=existing, threshold=threshold, blocking=blocking, mismatched=mismatched
        )
        if found is None:
            continue
        sentence = index.sentences[found.sentence]
        start, end = index.span(found.sentence, found.first, found.last)
        matches.append(
            AnchorMatch(
                source_url=index.url,
                target_url=target_url,
                keyword=text,
                keyword_rank=rank,
                keyword_source=source,
                rung=found.rung,
                phrase=index.body[start:end],
                start=start,
                end=end,
                sentence=sentence.text,
                sentence_index=sentence.index,
                sentence_start=sentence.start,
                stem_jaccard=found.jaccard,
            )
        )
    return matches, frozenset(blocking), frozenset(mismatched)


def anchor_report(
    tenant_id: str,
    matches: Sequence[AnchorMatch],
    *,
    threshold: float,
    pairs: int,
    bridge_pairs: int,
    pairs_with_keywords: int,
    overlapping: int,
    identifier_mismatches: int = 0,
    located_anchors: int,
    unlocated_anchors: int,
    keywords: Iterable[tuple[str, str | None]],
    source_languages: Mapping[str, str | None],
    sources_without_body: int,
    missing_sources: int = 0,
    started: float,
) -> AnchorReport:
    """``overlapping`` counts the distinct existing anchor spans that blocked a phrase, per
    source page; ``keywords`` are the (text, source page language) of every keyword the run
    looked for; ``source_languages`` the language of every source page read, and
    ``missing_sources`` the source pages not stored at all, which count only as sources without
    body text. ``started`` is the run's ``time.perf_counter()``."""
    best: dict[tuple[str, str], int] = {}
    primary: set[tuple[str, str]] = set()
    for match in matches:
        pair = (match.source_url, match.target_url)
        best[pair] = min(best.get(pair, len(_LADDER)), _LADDER.index(match.rung))
        if match.keyword_rank == 1:
            primary.add(pair)
    rungs = Counter(match.rung for match in matches)
    rung_ranks = Counter((match.rung, match.keyword_rank) for match in matches)
    jaccards, _ = np.histogram(
        [match.stem_jaccard for match in matches if match.stem_jaccard is not None],
        bins=STEM_JACCARD_BINS,
        range=(0.0, 1.0),
    )
    positions = Counter(
        bisect_right(SENTENCE_INDEX_BINS, match.sentence_index) - 1 for match in matches
    )
    best_rungs = Counter(_LADDER[rung] for rung in best.values())
    stemmed: Counter[str] = Counter()
    unstemmed: Counter[str] = Counter()
    for language in source_languages.values():
        (stemmed if algorithm_for(language) else unstemmed)[language or NO_LANGUAGE] += 1
    return AnchorReport(
        tenant_id=tenant_id,
        stem_set_threshold=threshold,
        pairs=pairs,
        bridge_pairs=bridge_pairs,
        pairs_with_keywords=pairs_with_keywords,
        pairs_matched=len(best),
        primary_matched=len(primary),
        matches=len(matches),
        by_rung={rung: rungs[rung] for rung in _LADDER},
        best_rung={rung: best_rungs[rung] for rung in _LADDER},
        by_keyword_rank=dict(sorted(Counter(match.keyword_rank for match in matches).items())),
        by_rung_and_rank={
            rung: {rank: n for (found, rank), n in sorted(rung_ranks.items()) if found is rung}
            for rung in _LADDER
        },
        stem_jaccard_histogram=tuple(int(count) for count in jaccards),
        sentence_index_histogram=tuple(positions[i] for i in range(len(SENTENCE_INDEX_BINS))),
        overlapping_existing_anchors=overlapping,
        identifier_mismatches=identifier_mismatches,
        existing_anchors_located=located_anchors,
        existing_anchors_unlocated=unlocated_anchors,
        single_token_keywords=_short_keywords(keywords),
        source_pages=len(source_languages) + missing_sources,
        sources_without_body=sources_without_body,
        stemmed_languages=dict(sorted(stemmed.items())),
        unstemmed_languages=dict(sorted(unstemmed.items())),
        seconds=round(time.perf_counter() - started, 3),
        finished_at=datetime.now(UTC),
    )


def _short_keywords(keywords: Iterable[tuple[str, str | None]]) -> int:
    """Distinct keyword texts with fewer than two content stems in a language they were looked
    for in: the stem set rung skips them."""
    stems: dict[str | None, Stems] = {}
    short: set[str] = set()
    for text, language in set(keywords):
        if language not in stems:
            stems[language] = Stems(language)
        if len(content_stems(keyword_tokens(text), stems[language])) < MIN_SHARED_STEMS:
            short.add(text)
    return len(short)


def summarise_anchors(report: AnchorReport) -> str:
    """A short prose record of one extraction run, for the MLflow run description; counts only,
    never urls, phrases or sentences."""
    rungs = ", ".join(f"{rung.value.lower()} {count}" for rung, count in report.by_rung.items())
    best = ", ".join(f"{rung.value.lower()} {count}" for rung, count in report.best_rung.items())
    share = report.pairs_matched / report.pairs_with_keywords if report.pairs_with_keywords else 0.0
    languages = ", ".join(
        f"{language} {count}" for language, count in report.stemmed_languages.items()
    )
    fallback = ", ".join(
        f"{language} {count}" for language, count in report.unstemmed_languages.items()
    )
    return "\n".join(
        [
            f"Anchor extraction for tenant {report.tenant_id}: {report.pairs} pairs "
            f"({report.bridge_pairs} from hub bridges), {report.pairs_with_keywords} with a "
            f"ranked keyword set; stem set threshold {report.stem_set_threshold:g}.",
            f"{report.pairs_matched} pairs matched ({share:.1%}), {report.primary_matched} on "
            f"their primary keyword; {report.matches} matches by rung: {rungs}; pairs by best "
            f"rung: {best}.",
            f"{report.overlapping_existing_anchors} existing link anchors blocked a phrase "
            f"({report.existing_anchors_located} located, {report.existing_anchors_unlocated} "
            f"not found in the body); {report.single_token_keywords} keywords with fewer than "
            "two content stems skip the stem set rung.",
            f"{report.identifier_mismatches} stemmed or stem set places refused because their "
            "numbers (ids, versions, years) disagreed with the keyword's.",
            f"{report.source_pages} source pages, {report.sources_without_body} without body "
            f"text. Stemmed: {languages or 'none'}. Casefolded only: {fallback or 'none'}.",
            f"{report.seconds:.1f} s.",
        ]
    )
