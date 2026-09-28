"""Anchor extraction: where a target's keyword is already written in a source page.

Anchor text is extracted, never generated. The ladder looks for each of the target's ranked
keywords in the source page's sentences: verbatim, as a stemmed variant, then as an overlapping
set of stemmed tokens. Each match returns the phrase with its sentence and character offsets.
"""

from datetime import datetime
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from linking_engine.models.enums import AnchorRung, KeywordSource

# The language key for pages without a language.
NO_LANGUAGE = "und"
# Equal-width bins of the stem set Jaccard over [0, 1].
STEM_JACCARD_BINS = 20
# Lower bounds of the sentence position bins: 0, 1, 2, 3-5, 6-10, 11-20, 21 and later.
SENTENCE_INDEX_BINS = (0, 1, 2, 3, 6, 11, 21)


class ExtractionSettings(BaseModel):
    """A tenant's extraction settings, stored on its tenant config; None there means these."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    # The least Jaccard between the keyword's and a phrase's sets of stems on the stem set rung.
    stem_set_threshold: float = Field(default=0.6, gt=0, le=1)


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
