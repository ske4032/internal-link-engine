"""Anchor extraction: where a target's keyword is already written in a source page.

Anchor text is extracted, never generated. The ladder looks for each of the target's ranked
keywords in the source page's sentences: verbatim, as a stemmed variant, then as an overlapping
set of stemmed tokens. Each match returns the phrase with its sentence and character offsets.
"""

from datetime import datetime
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from linking_engine.models.enums import (
    AnchorRung,
    AnchorType,
    ContentGapFinding,
    KeywordSource,
    UnanchoredReason,
)
from linking_engine.models.tenant import AnchorTypeProfile

# The language key for pages without a language.
NO_LANGUAGE = "und"
# Equal-width bins of the stem set Jaccard over [0, 1].
STEM_JACCARD_BINS = 20
# Lower bounds of the sentence position bins: 0, 1, 2, 3-5, 6-10, 11-20, 21 and later.
SENTENCE_INDEX_BINS = (0, 1, 2, 3, 6, 11, 21)
# Equal-width bins over [0, 1] for anchor totals and semantic similarities.
SCORE_HISTOGRAM_BINS = 20

# What to do about a pair without an anchor, in plain words for the people reading the output.
UNANCHORED_ADVICE: dict[UnanchoredReason, str] = {
    UnanchoredReason.SOURCE_DOES_NOT_MENTION_TOPIC: (
        "The source page never mentions the target page's topic. To link, add a sentence about "
        "it to the source page."
    ),
    UnanchoredReason.TOPIC_MENTIONED_BUT_NO_GOOD_PHRASE: (
        "The source page mentions the topic, but no phrase there reads well as a link. "
        "Rewording one sentence would give it an anchor."
    ),
    UnanchoredReason.TARGET_PAGE_HAS_NO_KEYWORD: (
        "The target page has no usable keyword: its H1 and title are missing, generic or "
        "repeated, and it has no GSC or strategic keyword, so there is nothing to look for in "
        "the source. Give the target page a clear H1 or title, or a strategic keyword."
    ),
    UnanchoredReason.SOURCE_PAGE_TEXT_UNAVAILABLE: (
        "The source page's text was not available in this run (not crawled, or empty), so it "
        "could not be searched. Recrawl the source page; this says nothing about its content."
    ),
    UnanchoredReason.MEANING_SEARCH_NOT_RUN: (
        "Only exact and word-level matches were searched for this pair: the search by meaning "
        "did not run (the embedding service was unavailable or not configured). Run again once "
        "it is available before treating this as a content gap."
    ),
}


def content_gap_finding(reason: UnanchoredReason) -> ContentGapFinding | None:
    """The content-gap finding a reason becomes on a recommendation; None for a target-page gap,
    which is not a content gap in the source."""
    return {
        UnanchoredReason.SOURCE_DOES_NOT_MENTION_TOPIC: ContentGapFinding.NO_TOPICAL_MENTION,
        UnanchoredReason.TOPIC_MENTIONED_BUT_NO_GOOD_PHRASE: ContentGapFinding.AWKWARD_PHRASING,
    }.get(reason)


class ExtractionSettings(BaseModel):
    """A tenant's extraction settings, stored on its tenant config; None there means these."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    # The least Jaccard between the keyword's and a phrase's sets of stems on the stem set rung.
    stem_set_threshold: float = Field(default=0.6, gt=0, le=1)
    # The semantic rung's cut-off; None derives it from the tenant's own copy on every run.
    semantic_threshold: float | None = Field(default=None, ge=-1, le=1)


class AnchorMatch(BaseModel):
    """One keyword of a target found in a source page's copy, the best place for it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_url: str = Field(min_length=1)
    target_url: str = Field(min_length=1)
    keyword: str = Field(min_length=1)
    # The keyword's place in the target's ranked keyword set, 1 the resolved keyword.
    keyword_rank: int = Field(ge=1)
    keyword_source: KeywordSource
    rung: AnchorRung
    phrase: str = Field(min_length=1)
    # Character offsets of the phrase in the source page's body text.
    start: int = Field(ge=0)
    end: int = Field(ge=1)
    sentence: str = Field(min_length=1)
    # The sentence's place in the body, 0 first, and its character offset there.
    sentence_index: int = Field(ge=0)
    sentence_start: int = Field(ge=0)
    # Jaccard of the stemmed token sets, on the stem set rung only.
    stem_jaccard: float | None = Field(default=None, gt=0, le=1)
    # Cosine of the phrase's vector to the keyword's, on the semantic rung only.
    semantic_similarity: float | None = Field(default=None, ge=-1, le=1)

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if self.source_url == self.target_url:
            raise ValueError("a page cannot link to itself")
        if not self.sentence_start <= self.start < self.end:
            raise ValueError("the phrase lies inside its sentence, start before end")
        offset = self.start - self.sentence_start
        if self.sentence[offset : offset + len(self.phrase)] != self.phrase:
            raise ValueError("the phrase is not at its offsets in the sentence")
        if self.end - self.start != len(self.phrase):
            raise ValueError("the offsets do not span the phrase")
        if (self.stem_jaccard is None) == (self.rung is AnchorRung.STEM_SET):
            raise ValueError("stem_jaccard is set on the stem set rung only")
        if (self.semantic_similarity is None) == (self.rung is AnchorRung.SEMANTIC):
            raise ValueError("semantic_similarity is set on the semantic rung only")
        return self


class AnchorReport(BaseModel):
    """One extraction run over a tenant's candidate pairs."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    tenant_id: str = Field(min_length=1)
    stem_set_threshold: float = Field(gt=0, le=1)
    pairs: int = Field(ge=0)
    # Of those, pairs from the tenant's hub bridges file, 0 without one.
    bridge_pairs: int = Field(ge=0)
    # Pairs whose target has a ranked keyword set, and those with at least one keyword found.
    pairs_with_keywords: int = Field(ge=0)
    pairs_matched: int = Field(ge=0)
    # Pairs whose rank-1 keyword was found.
    primary_matched: int = Field(ge=0)
    matches: int = Field(ge=0)
    by_rung: dict[AnchorRung, int]
    # Matched pairs by the lowest rung any of their keywords reached.
    best_rung: dict[AnchorRung, int]
    by_keyword_rank: dict[int, int]
    # Matches by rung, then keyword rank.
    by_rung_and_rank: dict[AnchorRung, dict[int, int]]
    # STEM_JACCARD_BINS counts of the stem set matches' Jaccard, and SENTENCE_INDEX_BINS counts
    # of every match's sentence position.
    stem_jaccard_histogram: tuple[int, ...]
    sentence_index_histogram: tuple[int, ...]
    # Distinct existing anchor spans that blocked a found phrase, per source page.
    overlapping_existing_anchors: int = Field(ge=0)
    # Existing links of the source pages whose anchor was, or was not, located in the body; an
    # unlocated anchor cannot protect its place.
    existing_anchors_located: int = Field(ge=0)
    existing_anchors_unlocated: int = Field(ge=0)
    # Places a stemmed or stem set match was refused because its numbers (ids, versions, years)
    # disagreed with the keyword's.
    identifier_mismatches: int = Field(default=0, ge=0)
    # Distinct keywords with fewer than two content stems, which skip the stem set rung.
    single_token_keywords: int = Field(ge=0)
    source_pages: int = Field(ge=0)
    sources_without_body: int = Field(ge=0)
    # Source pages by language, split by whether a stemmer covers it (NO_LANGUAGE for none).
    stemmed_languages: dict[str, int]
    unstemmed_languages: dict[str, int]
    seconds: float = Field(ge=0)
    finished_at: datetime

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if not self.primary_matched <= self.pairs_matched <= self.pairs_with_keywords <= self.pairs:
            raise ValueError("primary <= matched <= with keywords <= pairs")
        if self.bridge_pairs > self.pairs:
            raise ValueError("more bridge pairs than pairs")
        if sum(self.by_rung.values()) != self.matches:
            raise ValueError("matches by rung must add up to the matches")
        if sum(self.best_rung.values()) != self.pairs_matched:
            raise ValueError("pairs by best rung must add up to the matched pairs")
        if sum(self.by_keyword_rank.values()) != self.matches:
            raise ValueError("matches by keyword rank must add up to the matches")
        for rung, ranks in self.by_rung_and_rank.items():
            if sum(ranks.values()) != self.by_rung.get(rung, 0):
                raise ValueError("matches by rung and rank must add up to the matches by rung")
        by_rank: dict[int, int] = {}
        for ranks in self.by_rung_and_rank.values():
            for rank, count in ranks.items():
                by_rank[rank] = by_rank.get(rank, 0) + count
        if {r: c for r, c in by_rank.items() if c} != {
            r: c for r, c in self.by_keyword_rank.items() if c
        }:
            raise ValueError("matches by rung and rank must add up to the matches by rank")
        if len(self.stem_jaccard_histogram) != STEM_JACCARD_BINS or sum(
            self.stem_jaccard_histogram
        ) != self.by_rung.get(AnchorRung.STEM_SET, 0):
            raise ValueError(
                "the Jaccard histogram has one count per bin over the stem set matches"
            )
        if (
            len(self.sentence_index_histogram) != len(SENTENCE_INDEX_BINS)
            or sum(self.sentence_index_histogram) != self.matches
        ):
            raise ValueError("the sentence histogram has one count per bin over every match")
        if self.matches < self.pairs_matched:
            raise ValueError("every matched pair has at least one match")
        if self.overlapping_existing_anchors > self.existing_anchors_located:
            raise ValueError("more blocking anchors than located anchors")
        if self.sources_without_body > self.source_pages:
            raise ValueError("more sources without a body than source pages")
        counts = [
            *self.by_rung.values(),
            *self.best_rung.values(),
            *self.by_keyword_rank.values(),
            *self.stemmed_languages.values(),
            *self.unstemmed_languages.values(),
            *self.stem_jaccard_histogram,
            *self.sentence_index_histogram,
            *(count for ranks in self.by_rung_and_rank.values() for count in ranks.values()),
        ]
        if any(count < 0 for count in counts) or any(rank < 1 for rank in self.by_keyword_rank):
            raise ValueError("counts cannot be negative and keyword ranks start at 1")
        return self


class SemanticThreshold(BaseModel):
    """The semantic rung's cut-off for one tenant and run, derived from its own copy: the
    ``quantile`` of phrase-to-keyword cosines between pages of unrelated topics, bounded."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    value: float = Field(ge=-1, le=1)
    quantile: float = Field(gt=0, lt=1)
    # Phrase-to-keyword cosines between unrelated pages the value was read from.
    negatives: int = Field(ge=0)
    # Descriptive existing anchors scored against their target's keywords, for reporting.
    positives: int = Field(ge=0)
    # Share of those positives at or above the value; None without positives.
    positive_recall: float | None = Field(default=None, ge=0, le=1)
    # Set when the derived value fell outside the bounds, or too few negatives existed and the
    # default was used instead.
    bounded: bool = False
    fallback: bool = False
    # Set when the tenant's configured value was used and nothing was derived.
    overridden: bool = False

    @model_validator(mode="after")
    def _one_source(self) -> Self:
        if self.overridden and (self.fallback or self.bounded or self.negatives):
            raise ValueError("an overridden threshold is neither derived, bounded nor a fallback")
        return self


class AnchorScore(BaseModel):
    """How one candidate phrase scores for its pair (#23); every part in [0, 1]. Without phrase
    vectors (no Voyage) the semantic part is None and drops out, the other weights rescale, and
    the keyword part is the stem Jaccard alone."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    # Phrase vector against the target page's content vector.
    semantic: float | None = Field(default=None, ge=0, le=1)
    # 0.7 x Jaccard of the stems + 0.3 x the phrase-to-keyword cosine.
    keyword: float = Field(ge=0, le=1)
    # 1 - the highest word Jaccard against the target's existing and already chosen anchors.
    diversity: float = Field(ge=0, le=1)
    length: float = Field(ge=0, le=1)
    # The keyword's rank weight and the type profile's nudge, applied to the weighted sum.
    rank_weight: float = Field(gt=0, le=1)
    profile_bonus: float = Field(ge=0, le=1)
    total: float = Field(ge=0)


class AnchorChoice(BaseModel):
    """The anchor proposed for one pair (rank 1) or an alternative (ranks 2-3), with the
    placement's two relevance features."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    rank: int = Field(ge=1, le=3)
    match: AnchorMatch
    anchor_type: AnchorType
    score: AnchorScore
    # The sentence and the phrase against the target page's content vector, in [0, 1] as the
    # existing links' scores are (#16); None without vectors, so the features stay empty rather
    # than invented.
    context_relevance: float | None = Field(default=None, ge=0, le=1)
    anchor_target_fit: float | None = Field(default=None, ge=0, le=1)


class UnanchoredPair(BaseModel):
    """A pair with no usable anchor, and why."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_url: str = Field(min_length=1)
    target_url: str = Field(min_length=1)
    reason: UnanchoredReason
    # The best candidate's total when phrases existed but none scored high enough.
    best_score: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if (self.best_score is None) == (
            self.reason is UnanchoredReason.TOPIC_MENTIONED_BUT_NO_GOOD_PHRASE
        ):
            raise ValueError(
                "best_score is set when the topic is mentioned without a good phrase only"
            )
        return self


class AnchorSelectionReport(BaseModel):
    """One run of the semantic rung and anchor selection over a tenant's pairs."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    tenant_id: str = Field(min_length=1)
    pairs: int = Field(ge=0)
    # Pairs with a lexical match (rungs 1 to 2.5), and those sent to the semantic rung.
    lexical_pairs: int = Field(ge=0)
    semantic_invocations: int = Field(ge=0)
    semantic_matched: int = Field(ge=0)
    # Semantic phrases refused because their numbers disagreed with the keyword's, or because
    # another target's keyword was closer to them (an anchor for another page).
    semantic_rejected_identifier: int = Field(default=0, ge=0)
    semantic_rejected_other_target: int = Field(default=0, ge=0)
    # Semantic matches sharing no word with their keyword, logged as the ones likeliest to read
    # oddly.
    zero_overlap_matches: int = Field(ge=0)
    threshold: SemanticThreshold
    # Why the semantic rung did not run (no Voyage key, Voyage unavailable after retries); the
    # lexical rungs still choose anchors.
    semantic_skipped_reason: str | None = None
    # Why texts missing from the vector cache could not be embedded (no Voyage key, Voyage
    # unavailable), leaving their scores' semantic parts and placement features empty.
    embedding_skipped_reason: str | None = None
    sentences_embedded: int = Field(ge=0)
    sentences_cached: int = Field(ge=0)
    phrases_embedded: int = Field(ge=0)
    phrases_cached: int = Field(ge=0)
    chosen: int = Field(ge=0)
    alternatives: int = Field(ge=0)
    unanchored: dict[UnanchoredReason, int]
    chosen_types: dict[AnchorType, int]
    chosen_ranks: dict[int, int]
    profile: AnchorTypeProfile
    targets: int = Field(ge=0)
    targets_with_anchor: int = Field(ge=0)
    # Candidate pairs whose two relevance features were filled from their chosen anchor.
    features_filled: int = Field(ge=0)
    # SCORE_HISTOGRAM_BINS counts of the chosen anchors' totals over [0, 1], and of the semantic
    # matches' phrase-to-keyword cosines over [0, 1].
    score_histogram: tuple[int, ...]
    semantic_histogram: tuple[int, ...]
    seconds: float = Field(ge=0)
    finished_at: datetime

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if self.lexical_pairs + self.semantic_invocations > self.pairs:
            raise ValueError("lexical and semantic pairs cannot exceed the pairs")
        if self.zero_overlap_matches > self.semantic_matched:
            raise ValueError("more zero-overlap matches than semantic matches")
        if self.semantic_matched > self.semantic_invocations:
            raise ValueError("more semantic matches than invocations")
        if self.chosen + sum(self.unanchored.values()) != self.pairs:
            raise ValueError("every pair gets an anchor or an unanchored reason")
        if (
            sum(self.chosen_types.values()) != self.chosen
            or sum(self.chosen_ranks.values()) != self.chosen
        ):
            raise ValueError("chosen anchors by type and by keyword rank must add up to the chosen")
        if self.targets_with_anchor > self.targets or self.features_filled > self.chosen:
            raise ValueError("targets with an anchor <= targets; features filled <= chosen")
        if self.alternatives > 2 * self.chosen:
            raise ValueError("at most two alternatives per chosen anchor")
        if (
            len(self.score_histogram) != SCORE_HISTOGRAM_BINS
            or sum(self.score_histogram) != self.chosen
        ):
            raise ValueError("the score histogram has one count per bin over the chosen anchors")
        if (
            len(self.semantic_histogram) != SCORE_HISTOGRAM_BINS
            or sum(self.semantic_histogram) != self.semantic_matched
        ):
            raise ValueError(
                "the semantic histogram has one count per bin over the semantic matches"
            )
        counts = [
            *self.unanchored.values(),
            *self.chosen_types.values(),
            *self.chosen_ranks.values(),
            *self.score_histogram,
            *self.semantic_histogram,
        ]
        if any(count < 0 for count in counts) or any(rank < 1 for rank in self.chosen_ranks):
            raise ValueError("counts cannot be negative and keyword ranks start at 1")
        return self
