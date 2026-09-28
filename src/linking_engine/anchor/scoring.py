"""Anchor scoring and choice (#23): which of a pair's extracted phrases becomes its anchor.

Every candidate already exists in the source copy; scoring only chooses between them. Four parts
are weighed: closeness to the target page, closeness to the keyword, diversity against the
target's other anchors (word Jaccard, since over-optimisation is about repeated words), and
length. The keyword's rank weighs the sum, so the primary keyword gets most anchors, and the
tenant's type profile adds a small nudge towards under-represented types. The profile never
removes a candidate, so a good link is never skipped for it.
"""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import pairwise
from typing import TYPE_CHECKING, Final

import numpy as np

from linking_engine.anchor.extraction import content_stems, keyword_tokens, tokens
from linking_engine.anchor.keywords import brand_affixes
from linking_engine.models import (
    AnchorChoice,
    AnchorRung,
    AnchorScore,
    AnchorSelectionReport,
    AnchorType,
    UnanchoredPair,
    UnanchoredReason,
)
from linking_engine.models.anchors import SCORE_HISTOGRAM_BINS, UNANCHORED_ADVICE

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from linking_engine.anchor.extraction import Stems
    from linking_engine.models import AnchorMatch, AnchorTypeProfile, SemanticThreshold

STAGE: Final = "anchor-selection"
# Set from an evaluation against editors' anchors; see the #23 decision comment.
SEMANTIC_WEIGHT: Final = 0.40
KEYWORD_WEIGHT: Final = 0.40
DIVERSITY_WEIGHT: Final = 0.10
LENGTH_WEIGHT: Final = 0.10
# The keyword part: Jaccard of the stems, then the phrase-to-keyword cosine.
STEM_SHARE: Final = 0.7
COSINE_SHARE: Final = 0.3
SECONDARY_WEIGHT: Final = 0.7
PROFILE_BONUS: Final = 0.1
# Below this best total, phrases exist but would read badly as a link.
AWKWARD_FLOOR: Final = 0.4
ALTERNATIVES: Final = 2
_RUNGS: Final = tuple(AnchorRung)


@dataclass(frozen=True, slots=True)
class Candidate:
    """One match of a pair with what scoring needs of it."""

    match: AnchorMatch
    # Jaccard of the phrase's and the keyword's content stems.
    stem_jaccard: float
    # Phrase words, casefolded, for diversity.
    words: frozenset[str]
    word_count: int
    # Raw cosines of the phrase to the target's content vector and to the keyword's; None
    # without the vectors, when the part they feed drops out.
    target_cosine: float | None
    keyword_cosine: float | None


@dataclass(frozen=True, slots=True)
class PairCandidates:
    source_url: str
    target_url: str
    candidates: tuple[Candidate, ...]
    # Whether the target has ranked keywords to look for at all, whether the source page had
    # text to search, and whether the search by meaning ran for a pair the lexical rungs missed.
    has_keywords: bool = True
    has_text: bool = True
    meaning_searched: bool = True


def _unsearched(pair: PairCandidates) -> UnanchoredReason:
    """Why a pair without a single candidate has none: a gap in the target page, a pair not
    fully searched in this run, or, when every rung searched it, a topic the source never
    mentions."""
    if not pair.has_keywords:
        return UnanchoredReason.TARGET_PAGE_HAS_NO_KEYWORD
    if not pair.has_text:
        return UnanchoredReason.SOURCE_PAGE_TEXT_UNAVAILABLE
    if not pair.meaning_searched:
        return UnanchoredReason.MEANING_SEARCH_NOT_RUN
    return UnanchoredReason.SOURCE_DOES_NOT_MENTION_TOPIC


@dataclass(frozen=True, slots=True)
class ExistingAnchor:
    """A descriptive anchor already linking into a target, with its type."""

    words: frozenset[str]
    anchor_type: AnchorType


def unit(cosine: float) -> float:
    """A cosine mapped to [0, 1], as the existing links' scores are (#16)."""
    return min(max((1 + cosine) / 2, 0.0), 1.0)


def word_count(phrase: str) -> int:
    """Words of a phrase; a compound joined without whitespace, as e-commerce, is one."""
    found = tokens(phrase)
    joined = sum(
        1 for a, b in pairwise(found) if not any(char.isspace() for char in phrase[a.end : b.start])
    )
    return len(found) - joined


def stem_jaccard(phrase: str, keyword: str, stems: Stems) -> float:
    """Jaccard of the content stems of the phrase and the keyword; all their stems when either
    is only stop words."""
    phrase_tokens, keyword_token_list = keyword_tokens(phrase), keyword_tokens(keyword)
    left, right = content_stems(phrase_tokens, stems), content_stems(keyword_token_list, stems)
    if not left or not right:
        left = frozenset(stems(token) for token in phrase_tokens)
        right = frozenset(stems(token) for token in keyword_token_list)
    union = left | right
    return len(left & right) / len(union) if union else 0.0


Brand = tuple[tuple[str, ...], ...]


def brand_tokens(titles: Iterable[str | None]) -> Brand:
    """The token sequences of the brand affixes of the tenant's titles, detected as keyword
    resolution does; casefolded, apostrophes folded, as the ladder compares tokens."""
    prefix, suffix = brand_affixes(titles)
    found = (tuple(keyword_tokens(affix)) for affix in (prefix, suffix) if affix)
    return tuple(dict.fromkeys(sequence for sequence in found if sequence))


def brand_words(brand: Brand) -> frozenset[str]:
    """Every token of every brand sequence: the ones never counted as identifiers."""
    return frozenset(token for sequence in brand for token in sequence)


def branded(text: str, brand: Brand) -> bool:
    """Whether the text holds one brand affix's whole token sequence, contiguous."""
    held = keyword_tokens(text)
    return any(
        tuple(held[i : i + len(sequence)]) == sequence
        for sequence in brand
        for i in range(len(held) - len(sequence) + 1)
    )


def anchor_type(match: AnchorMatch, brand: Brand) -> AnchorType:
    """BRANDED when the phrase holds a whole brand affix, over the others; EXACT for the primary
    keyword verbatim; NATURAL for a semantic match; PARTIAL otherwise."""
    if branded(match.phrase, brand):
        return AnchorType.BRANDED
    if match.rung is AnchorRung.EXACT and match.keyword_rank == 1:
        return AnchorType.EXACT
    if match.rung is AnchorRung.SEMANTIC:
        return AnchorType.NATURAL
    return AnchorType.PARTIAL


def existing_type(text: str, keywords: Sequence[str], stems: Stems, brand: Brand) -> AnchorType:
    """The type of an existing anchor into a page with these ranked keywords, primary first:
    BRANDED when it holds a whole brand affix, EXACT when it is the primary keyword's words,
    PARTIAL when it shares a content stem with any keyword, NATURAL otherwise."""
    if branded(text, brand):
        return AnchorType.BRANDED
    found = keyword_tokens(text)
    if keywords and found == keyword_tokens(keywords[0]):
        return AnchorType.EXACT
    held = content_stems(found, stems)
    if any(held & content_stems(keyword_tokens(keyword), stems) for keyword in keywords):
        return AnchorType.PARTIAL
    return AnchorType.NATURAL


def _length(words: int) -> float:
    if 2 <= words <= 4:
        return 1.0
    return 0.6 if words in (1, 5) else 0.3


def _deficit(
    kind: AnchorType, counts: Mapping[AnchorType, int], profile: AnchorTypeProfile
) -> float:
    """How far the type's share of the target's anchors falls below the profile's."""
    wanted = {
        AnchorType.EXACT: profile.exact,
        AnchorType.PARTIAL: profile.partial,
        AnchorType.NATURAL: profile.natural,
        AnchorType.BRANDED: profile.branded,
    }[kind]
    total = sum(counts.values())
    return max(0.0, wanted - (counts.get(kind, 0) / total if total else 0.0))


def score_candidate(
    candidate: Candidate,
    kind: AnchorType,
    *,
    against: Iterable[frozenset[str]],
    counts: Mapping[AnchorType, int],
    profile: AnchorTypeProfile,
) -> AnchorScore:
    """The candidate's parts and total, with ``against`` the word sets of the target's existing
    and already chosen anchors and ``counts`` their types.

    Missing data is no penalty: without the phrase's cosine to the target the semantic part
    drops out and the other weights rescale, and without its cosine to the keyword the keyword
    part is the stem Jaccard alone.
    """
    semantic = None if candidate.target_cosine is None else unit(candidate.target_cosine)
    keyword = (
        candidate.stem_jaccard
        if candidate.keyword_cosine is None
        else STEM_SHARE * candidate.stem_jaccard + COSINE_SHARE * unit(candidate.keyword_cosine)
    )
    overlaps = [
        len(candidate.words & other) / len(candidate.words | other)
        for other in against
        if candidate.words | other
    ]
    diversity = 1.0 - max(overlaps, default=0.0)
    length = _length(candidate.word_count)
    weighted = KEYWORD_WEIGHT * keyword + DIVERSITY_WEIGHT * diversity + LENGTH_WEIGHT * length
    if semantic is None:
        weighted /= KEYWORD_WEIGHT + DIVERSITY_WEIGHT + LENGTH_WEIGHT
    else:
        weighted += SEMANTIC_WEIGHT * semantic
    rank_weight = 1.0 if candidate.match.keyword_rank == 1 else SECONDARY_WEIGHT
    bonus = PROFILE_BONUS * _deficit(kind, counts, profile)
    return AnchorScore(
        semantic=semantic,
        keyword=min(keyword, 1.0),
        diversity=diversity,
        length=length,
        rank_weight=rank_weight,
        profile_bonus=bonus,
        total=weighted * rank_weight + bonus,
    )


def choose(
    pairs: Iterable[PairCandidates],
    *,
    existing: Mapping[str, Sequence[ExistingAnchor]],
    profile: AnchorTypeProfile,
    brand: Brand,
) -> tuple[list[AnchorChoice], list[UnanchoredPair]]:
    """Every pair's anchor and up to ``ALTERNATIVES`` distinct alternatives, or why it has none:
    the target has no keyword to look for, the pair was not fully searched, the source never
    mentions it, or no phrase that does reaches ``AWKWARD_FLOOR``.

    The floor applies to a phrase's weighted parts times its rank weight, before the profile
    bonus: the profile only chooses between phrases good enough on their own, and never lifts a
    poor one over the floor.

    Pairs are taken in the given order, a target's in descending retrieval similarity: each
    target's chosen anchors join its existing ones for the diversity and the type counts of its
    later pairs. Ties go to the earlier sentence, then the earlier offset. The choices' placement
    features are left empty, for the caller to fill from the vectors.
    """
    chosen_words: dict[str, list[frozenset[str]]] = {}
    chosen_types: dict[str, Counter[AnchorType]] = {}
    choices: list[AnchorChoice] = []
    unanchored: list[UnanchoredPair] = []
    for pair in pairs:
        if not pair.candidates:
            unanchored.append(
                UnanchoredPair(
                    source_url=pair.source_url,
                    target_url=pair.target_url,
                    reason=_unsearched(pair),
                )
            )
            continue
        prior = existing.get(pair.target_url, ())
        against = [anchor.words for anchor in prior] + chosen_words.get(pair.target_url, [])
        counts = Counter(anchor.anchor_type for anchor in prior) + chosen_types.get(
            pair.target_url, Counter()
        )
        scored = []
        for candidate in pair.candidates:
            kind = anchor_type(candidate.match, brand)
            score = score_candidate(
                candidate, kind, against=against, counts=counts, profile=profile
            )
            scored.append((score, kind, candidate))
        scored.sort(key=lambda item: _order(item[0], item[2]))
        good = [item for item in scored if _base(item[0]) >= AWKWARD_FLOOR]
        if not good:
            unanchored.append(
                UnanchoredPair(
                    source_url=pair.source_url,
                    target_url=pair.target_url,
                    reason=UnanchoredReason.TOPIC_MENTIONED_BUT_NO_GOOD_PHRASE,
                    best_score=max(_base(score) for score, _, _ in scored),
                )
            )
            continue
        phrases: set[str] = set()
        for score, kind, candidate in good:
            phrase = " ".join(candidate.match.phrase.split()).casefold()
            if phrase in phrases:
                continue
            phrases.add(phrase)
            choices.append(
                AnchorChoice(
                    rank=len(phrases),
                    match=candidate.match,
                    anchor_type=kind,
                    score=score,
                )
            )
            if len(phrases) == 1:
                chosen_words.setdefault(pair.target_url, []).append(candidate.words)
                chosen_types.setdefault(pair.target_url, Counter())[kind] += 1
            if len(phrases) > ALTERNATIVES:
                break
    return choices, unanchored


def _base(score: AnchorScore) -> float:
    """The weighted parts times the rank weight: the total before the profile bonus."""
    return max(score.total - score.profile_bonus, 0.0)


def _order(score: AnchorScore, candidate: Candidate) -> tuple[float, int, int, int, int, str]:
    match = candidate.match
    return (
        -score.total,
        match.sentence_index,
        match.start,
        _RUNGS.index(match.rung),
        match.keyword_rank,
        match.phrase,
    )


def selection_report(
    tenant_id: str,
    choices: Sequence[AnchorChoice],
    unanchored: Sequence[UnanchoredPair],
    *,
    pairs: int,
    lexical_pairs: int,
    semantic_invocations: int,
    semantic_similarities: Sequence[float],
    zero_overlap_matches: int,
    semantic_rejected_identifier: int = 0,
    semantic_rejected_other_target: int = 0,
    threshold: SemanticThreshold,
    semantic_skipped_reason: str | None,
    embedding_skipped_reason: str | None,
    sentences_embedded: int,
    sentences_cached: int,
    phrases_embedded: int,
    phrases_cached: int,
    profile: AnchorTypeProfile,
    targets: int,
    started: float,
) -> AnchorSelectionReport:
    """``semantic_similarities`` are the semantic matches' phrase-to-keyword cosines, one per
    matched pair; ``started`` is the run's ``time.perf_counter()`` start."""
    chosen = [choice for choice in choices if choice.rank == 1]
    totals, _ = np.histogram(
        np.clip([choice.score.total for choice in chosen], 0.0, 1.0),
        bins=SCORE_HISTOGRAM_BINS,
        range=(0.0, 1.0),
    )
    semantic, _ = np.histogram(
        np.clip(semantic_similarities, 0.0, 1.0), bins=SCORE_HISTOGRAM_BINS, range=(0.0, 1.0)
    )
    reasons = Counter(pair.reason for pair in unanchored)
    types = Counter(choice.anchor_type for choice in chosen)
    return AnchorSelectionReport(
        tenant_id=tenant_id,
        pairs=pairs,
        lexical_pairs=lexical_pairs,
        semantic_invocations=semantic_invocations,
        semantic_matched=len(semantic_similarities),
        zero_overlap_matches=zero_overlap_matches,
        semantic_rejected_identifier=semantic_rejected_identifier,
        semantic_rejected_other_target=semantic_rejected_other_target,
        threshold=threshold,
        semantic_skipped_reason=semantic_skipped_reason,
        embedding_skipped_reason=embedding_skipped_reason,
        sentences_embedded=sentences_embedded,
        sentences_cached=sentences_cached,
        phrases_embedded=phrases_embedded,
        phrases_cached=phrases_cached,
        chosen=len(chosen),
        alternatives=len(choices) - len(chosen),
        unanchored={reason: reasons[reason] for reason in UnanchoredReason},
        chosen_types={kind: types[kind] for kind in AnchorType},
        chosen_ranks=dict(sorted(Counter(choice.match.keyword_rank for choice in chosen).items())),
        profile=profile,
        targets=targets,
        targets_with_anchor=len({choice.match.target_url for choice in chosen}),
        features_filled=sum(
            1
            for choice in chosen
            if choice.context_relevance is not None and choice.anchor_target_fit is not None
        ),
        score_histogram=tuple(int(count) for count in totals),
        semantic_histogram=tuple(int(count) for count in semantic),
        seconds=round(time.perf_counter() - started, 3),
        finished_at=datetime.now(UTC),
    )


def summarise_selection(report: AnchorSelectionReport) -> str:
    """A short prose record of one selection run, for the MLflow run description; counts only,
    never urls, phrases, sentences or keywords."""
    threshold = report.threshold
    if threshold.overridden:
        how = f"configured threshold {threshold.value:.3f}"
    elif threshold.fallback:
        how = f"threshold {threshold.value:.3f}, the default (too few unrelated-topic phrases)"
    else:
        how = (
            f"threshold {threshold.value:.3f}, the {threshold.quantile:.0%} quantile of "
            f"{threshold.negatives} unrelated-topic phrases"
            + (", bounded" if threshold.bounded else "")
        )
    recall = (
        f"{threshold.positive_recall:.1%} of {threshold.positives} descriptive existing anchors "
        "reach it"
        if threshold.positive_recall is not None
        else "no descriptive existing anchors to check it against"
    )
    semantic = (
        f"The semantic rung was skipped: {report.semantic_skipped_reason}."
        if report.semantic_skipped_reason
        else f"The semantic rung ran on {report.semantic_invocations} pairs and matched "
        f"{report.semantic_matched} ({report.zero_overlap_matches} sharing no word with their "
        f"keyword) at {how}; {recall}. It refused "
        f"{report.semantic_rejected_identifier} phrases whose numbers disagreed with the "
        f"keyword's and {report.semantic_rejected_other_target} closer to another target's "
        "keyword."
    )
    types = ", ".join(f"{kind.value.lower()} {n}" for kind, n in report.chosen_types.items())
    return "\n".join(
        [
            f"Anchor selection for tenant {report.tenant_id}: {report.pairs} pairs, "
            f"{report.lexical_pairs} with a lexical match.",
            semantic,
            f"{report.chosen} anchors chosen with {report.alternatives} alternatives, for "
            f"{report.targets_with_anchor} of {report.targets} targets; by type: {types}.",
            *(
                f"{count} pairs without an anchor, {reason.value.lower()}: "
                f"{UNANCHORED_ADVICE[reason]}"
                for reason, count in report.unanchored.items()
            ),
            f"Embedded {report.sentences_embedded} sentences ({report.sentences_cached} cached) "
            f"and {report.phrases_embedded} phrases ({report.phrases_cached} cached); "
            f"{report.features_filled} candidate pairs got their relevance features.",
            (
                f"Texts missing from the vector cache were not embedded: "
                f"{report.embedding_skipped_reason}; their semantic parts and placement "
                "features stay empty."
                if report.embedding_skipped_reason
                else "Every text needed was embedded or cached."
            ),
            f"{report.seconds:.1f} s.",
        ]
    )
