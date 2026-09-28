"""The semantic rung (#22): a target's keyword said in other words in the source copy.

For a pair the lexical rungs missed, the source's sentences closest to the target's keywords
are searched for a short phrase whose vector is close to one of the keywords. Phrases follow
the ladder's rules, so the rung still extracts text that is on the page; it never writes any.
The cut-off is derived per tenant from phrase-to-keyword cosines between unrelated pages.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Literal

import numpy as np

from linking_engine.anchor.extraction import content_stems, identifiers, keyword_tokens
from linking_engine.anchor.scoring import unit
from linking_engine.models import AnchorMatch, AnchorRung, SemanticThreshold

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Mapping, Sequence

    import numpy.typing as npt

    from linking_engine.anchor.extraction import SourceIndex, Stems
    from linking_engine.models import KeywordSource

TOP_SENTENCES: Final = 3
MAX_PHRASES_PER_SENTENCE: Final = 10
# Unrelated phrase-to-keyword cosines the threshold is read from, and descriptive existing anchors
# its recall is reported on, each sampled up to this many with a fixed seed.
NEGATIVE_SAMPLE: Final = 5000
SAMPLE_SEED: Final = 42
THRESHOLD_QUANTILE: Final = 0.99
THRESHOLD_BOUNDS: Final = (0.35, 0.9)
MIN_NEGATIVES: Final = 300
DEFAULT_SEMANTIC_THRESHOLD: Final = 0.6
# Draws per wanted negative before the sample gives up on a tenant with few unrelated pages.
_DRAWS_PER_NEGATIVE: Final = 20
# Phrases per matrix product of the best-target check.
OTHER_TARGET_CHUNK: Final = 512
# A keyword shared by several pages ties across them; float error must not break the tie.
_TIE_TOLERANCE: Final = 1e-9


@dataclass(frozen=True, slots=True)
class Phrase:
    """A span of one sentence of a source page that the semantic rung may propose."""

    # The sentence's position among the index's sentences, and its first and last token.
    position: int
    first: int
    last: int
    # Character offsets in the body, and the text there.
    start: int
    end: int
    text: str
    content_tokens: int


def candidate_phrases(
    index: SourceIndex,
    position: int,
    *,
    existing: Sequence[tuple[int, int]] = (),
    limit: int = MAX_PHRASES_PER_SENTENCE,
) -> list[Phrase]:
    """A sentence's phrases the ladder accepts, none overlapping an ``existing`` anchor span:
    the ones with the most content tokens first, then the earliest, at most ``limit``."""
    phrases: list[Phrase] = []
    for first, last, content in index.phrase_spans(position):
        start, end = index.span(position, first, last)
        if any(start < high and low < end for low, high in existing):
            continue
        phrases.append(Phrase(position, first, last, start, end, index.body[start:end], content))
    phrases.sort(key=lambda phrase: (-phrase.content_tokens, phrase.first, phrase.last))
    return phrases[:limit]


def _unit(vectors: npt.ArrayLike) -> npt.NDArray[np.float64]:
    matrix = np.atleast_2d(np.asarray(vectors, dtype=np.float64))
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if (norms == 0).any() or not np.isfinite(matrix).all():
        raise ValueError("vectors must be finite and non-zero")
    unit: npt.NDArray[np.float64] = matrix / norms
    return unit


def cosines(rows: npt.ArrayLike, columns: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Cosine of every row vector to every column vector, clipped to [-1, 1]."""
    found: npt.NDArray[np.float64] = np.clip(_unit(rows) @ _unit(columns).T, -1.0, 1.0)
    return found


def relevance(first: npt.ArrayLike | None, second: npt.ArrayLike | None) -> float | None:
    """Two vectors' cosine as (1 + cosine) / 2, in [0, 1] like the existing links' scores
    (#16); None when either is missing."""
    if first is None or second is None:
        return None
    return unit(float(cosines(first, second)[0, 0]))


def top_sentences(
    sentences: npt.ArrayLike, keywords: npt.ArrayLike, count: int = TOP_SENTENCES
) -> list[int]:
    """Rows of the ``count`` sentence vectors with the highest cosine to any keyword vector,
    best first; a tie goes to the earlier sentence."""
    if count < 1:
        raise ValueError("count must be at least 1")
    best = cosines(sentences, keywords).max(axis=1)
    order = np.lexsort((np.arange(len(best)), -best))
    return [int(row) for row in order[:count]]


@dataclass(frozen=True, slots=True)
class SemanticOutcome:
    match: AnchorMatch | None
    # Why a pair with a phrase at or above the threshold got no match: every such phrase held
    # other identifiers than its keyword, or another target's keywords were closer to it.
    rejected: Literal["identifier", "other_target"] | None = None


def agrees(phrase: str, keyword: str, stems: Stems, brand: frozenset[str] = frozenset()) -> bool:
    """Whether the phrase holds exactly the keyword's identifiers: tokens with a digit, and
    month names next to one, the tenant's ``brand`` tokens aside (see `identifiers`)."""
    return identifiers(phrase, stems, brand) == identifiers(keyword, stems, brand)


def _candidates(
    phrases: Sequence[Phrase],
    keywords: Sequence[tuple[int, str, KeywordSource]],
    similarity: npt.NDArray[np.float64],
    threshold: float,
) -> list[tuple[int, int]]:
    """(phrase, keyword) rows at or above the threshold, best cosine first, then the earlier
    phrase, then the better-ranked keyword."""
    rows, columns = np.nonzero(similarity >= threshold)
    return sorted(
        zip(rows.tolist(), columns.tolist(), strict=True),
        key=lambda pair: (
            -similarity[pair],
            phrases[pair[0]].position,
            phrases[pair[0]].start,
            phrases[pair[0]].end,
            keywords[pair[1]][0],
        ),
    )


def eligible_phrases(
    phrases: Sequence[Phrase],
    phrase_vectors: npt.ArrayLike,
    keywords: Sequence[tuple[int, str, KeywordSource]],
    keyword_vectors: npt.ArrayLike,
    *,
    threshold: float,
    stems: Stems,
    brand: frozenset[str] = frozenset(),
) -> list[int]:
    """The phrases that reach the threshold on a keyword whose identifiers they agree with, in
    the source's language (``stems``): the only ones the best-target check has to look at."""
    if not phrases or not keywords:
        return []
    similarity = cosines(phrase_vectors, keyword_vectors)
    return sorted(
        {
            i
            for i, k in _candidates(phrases, keywords, similarity, threshold)
            if agrees(phrases[i].text, keywords[k][1], stems, brand)
        }
    )


def best_other_cosines(
    phrase_vectors: npt.ArrayLike,
    keyword_vectors: npt.ArrayLike,
    excluded: Sequence[Collection[int]],
    *,
    chunk: int = OTHER_TARGET_CHUNK,
) -> npt.NDArray[np.float64]:
    """Each phrase row's highest cosine to the keyword rows other than its ``excluded`` ones,
    -inf when none remain; one matrix product, ``chunk`` phrases at a time."""
    phrases = np.asarray(phrase_vectors, dtype=np.float64)
    found = np.full(len(phrases), -np.inf)
    if len(excluded) != len(phrases):
        raise ValueError("one excluded set per phrase")
    keys = np.asarray(keyword_vectors, dtype=np.float64)
    if not len(phrases) or not len(keys):
        return found
    for start in range(0, len(phrases), chunk):
        block = cosines(phrases[start : start + chunk], keys)
        for row, columns in enumerate(excluded[start : start + chunk]):
            block[row, list(columns)] = -np.inf
        found[start : start + len(block)] = block.max(axis=1)
    return found


def semantic_match(
    index: SourceIndex,
    target_url: str,
    phrases: Sequence[Phrase],
    phrase_vectors: npt.ArrayLike,
    keywords: Sequence[tuple[int, str, KeywordSource]],
    keyword_vectors: npt.ArrayLike,
    *,
    threshold: float,
    other_best: Mapping[int, float] | None = None,
) -> SemanticOutcome:
    """The best (phrase, keyword) at or above ``threshold`` whose identifiers agree, and whose
    phrase is at least as close to the target's keywords as to any other target's
    (``other_best``, phrase index to that cosine; a phrase absent from it, or no mapping, has
    no rival). Ties go to the earlier phrase, then the better-ranked keyword. Phrase and keyword
    vectors are rows aligned with ``phrases`` and ``keywords``."""
    if not phrases or not keywords:
        return SemanticOutcome(None)
    similarity = cosines(phrase_vectors, keyword_vectors)
    if similarity.shape != (len(phrases), len(keywords)):
        raise ValueError("one vector per phrase and per keyword")
    candidates = _candidates(phrases, keywords, similarity, threshold)
    agreeing = False
    for i, k in candidates:
        if not agrees(phrases[i].text, keywords[k][1], index.stems, index.brand):
            continue
        agreeing = True
        rival = (other_best or {}).get(i, -np.inf)
        if rival > similarity[i].max() + _TIE_TOLERANCE:
            continue
        phrase, (rank, keyword, source) = phrases[i], keywords[k]
        sentence = index.sentences[phrase.position]
        return SemanticOutcome(
            AnchorMatch(
                source_url=index.url,
                target_url=target_url,
                keyword=keyword,
                keyword_rank=rank,
                keyword_source=source,
                rung=AnchorRung.SEMANTIC,
                phrase=phrase.text,
                start=phrase.start,
                end=phrase.end,
                sentence=sentence.text,
                sentence_index=sentence.index,
                sentence_start=sentence.start,
                semantic_similarity=float(similarity[i, k]),
            )
        )
    if not candidates:
        return SemanticOutcome(None)
    return SemanticOutcome(None, "other_target" if agreeing else "identifier")


def shares_stem(phrase: str, keyword: str, stems: Stems) -> bool:
    """Whether the phrase and the keyword have a content stem in common."""
    return bool(
        content_stems(keyword_tokens(phrase), stems) & content_stems(keyword_tokens(keyword), stems)
    )


def sample[T](
    items: Sequence[T], count: int = NEGATIVE_SAMPLE, *, seed: int = SAMPLE_SEED
) -> list[T]:
    """At most ``count`` of the items, drawn with ``seed`` and kept in their order."""
    if len(items) <= count:
        return list(items)
    chosen = np.random.default_rng(seed).choice(len(items), size=count, replace=False)
    return [items[i] for i in sorted(chosen.tolist())]


def negative_phrases(
    indexes: Sequence[SourceIndex],
    targets: Sequence[str],
    unrelated: Callable[[str, str], bool],
    *,
    count: int = NEGATIVE_SAMPLE,
    seed: int = SAMPLE_SEED,
) -> list[tuple[str, str]]:
    """Up to ``count`` distinct (phrase, target) pairs, each a candidate phrase of a source page
    drawn at random for a target the page is ``unrelated`` to; the same inputs give the same
    draw. A repeat draw adds nothing, so a small tenant yields as many as it has."""
    rng = np.random.default_rng(seed)
    pages = [index for index in indexes if index.sentences]
    found: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    if not pages or not targets:
        return found
    for _ in range(count * _DRAWS_PER_NEGATIVE):
        if len(found) == count:
            break
        index = pages[int(rng.integers(len(pages)))]
        target = targets[int(rng.integers(len(targets)))]
        if index.url == target or not unrelated(index.url, target):
            continue
        phrases = candidate_phrases(index, int(rng.integers(len(index.sentences))))
        if not phrases:
            continue
        pair = (phrases[int(rng.integers(len(phrases)))].text, target)
        if pair not in seen:
            seen.add(pair)
            found.append(pair)
    return found


def derive_threshold(
    negatives: Sequence[float],
    positives: Sequence[float],
    *,
    override: float | None = None,
    quantile: float = THRESHOLD_QUANTILE,
    bounds: tuple[float, float] = THRESHOLD_BOUNDS,
    min_negatives: int = MIN_NEGATIVES,
    default: float = DEFAULT_SEMANTIC_THRESHOLD,
) -> SemanticThreshold:
    """The ``quantile`` of the unrelated phrase-to-keyword cosines, clipped to ``bounds``;
    ``default`` with fewer than ``min_negatives``; ``override`` as is when set, when nothing is
    derived and the negatives are ignored. Recall is the share of the positives, descriptive
    existing anchors against their target's keywords, at or above the value."""
    low, high = bounds
    bounded = fallback = False
    if override is not None:
        value, negatives = override, ()
    elif len(negatives) < min_negatives:
        value, fallback = default, True
    else:
        found = float(np.quantile(np.asarray(negatives, dtype=np.float64), quantile))
        value = min(high, max(low, found))
        bounded = value != found
    return SemanticThreshold(
        value=value,
        quantile=quantile,
        negatives=len(negatives),
        positives=len(positives),
        positive_recall=(
            sum(1 for cosine in positives if cosine >= value) / len(positives)
            if positives
            else None
        ),
        bounded=bounded,
        fallback=fallback,
        overridden=override is not None,
    )
