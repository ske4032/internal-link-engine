"""Continuous quality evaluation of one tenant: a report per run, a section per check.

A check the tenant cannot have (no body links, no keywords, no Voyage key) is named in
``not_applicable`` and its section is None: it is reported, never failed. Nothing here holds
a page url or a keyword text, so the whole report can be logged.
"""

import math
from collections import Counter
from datetime import datetime
from typing import Final, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from linking_engine.models.enums import KeywordRung
from linking_engine.models.relevance import ScoreDistribution
from linking_engine.models.scoring import SCORE_HISTOGRAM_BINS

CheckName = Literal[
    "retrieval",
    "feature_signal",
    "scorer",
    "keywords",
    "keyword_uniqueness",
    "keyword_extractability",
    "anchor_match",
    "keyword_relevance",
    "link_relevance",
]
# Sub-checks of the keyword section; named only when the section itself applies.
KEYWORD_CHECKS: Final[tuple[CheckName, ...]] = (
    "keyword_uniqueness",
    "keyword_extractability",
    "anchor_match",
    "keyword_relevance",
)
_LIFT_TOLERANCE: Final = 1e-9
# Float noise must not decide whether an AUC is exactly the margin from 0.5: 0.5 - 0.45 < 0.05.
_SIGNAL_TOLERANCE: Final = 1e-9


class QualityVersions(BaseModel):
    """What produced the run: the code, the feature code and the scorer weights."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    git_sha: str = Field(min_length=1)
    feature_digest: str = Field(min_length=1)
    weights_version: str = Field(min_length=1)
    weights_hash: str = Field(min_length=1)


class RecallAtK(BaseModel):
    """Held-out links recovered among the first ``k`` candidates of their target."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    k: int = Field(ge=1)
    recall: float = Field(ge=0, le=1)
    # Expected recall of a random order of each target's eligible sources.
    random: float = Field(ge=0, le=1)


class RetrievalCheck(BaseModel):
    """Held-out link recovery: some body links hidden, every link-derived input recomputed
    without them, retrieval rerun on that view."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    hide_share: float = Field(gt=0, lt=1)
    seed: int
    # Distinct (source, target) body links between crawled pages.
    body_link_pairs: int = Field(ge=1)
    hidden: int = Field(ge=1)
    # Hidden links whose target is a retrieval target and whose source is eligible for it.
    recoverable: int = Field(ge=1)
    # Candidate pairs retrieved on the held-out view.
    candidates: int = Field(ge=0)
    recall: tuple[RecallAtK, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if not self.recoverable <= self.hidden <= self.body_link_pairs:
            raise ValueError("recoverable <= hidden <= body_link_pairs")
        ks = [entry.k for entry in self.recall]
        if ks != sorted(set(ks)):
            raise ValueError("recall ks must be unique and ascending")
        recalls = [entry.recall for entry in self.recall]
        if recalls != sorted(recalls):
            raise ValueError("recall must not fall as k grows")
        return self


class FeatureAuc(BaseModel):
    """How well one feature column separates hidden links from the other candidates."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    column: str = Field(min_length=1)
    # Over the pairs with a value; None when they hold one value or one class only.
    auc: float | None = Field(default=None, ge=0, le=1)
    # Share of the candidate pairs with a value.
    coverage: float = Field(ge=0, le=1)
    # The column alone as a ranker over every pair, in its best order (either direction,
    # missing values first or last): comparable with the scorer's AUC.
    ranker_auc: float = Field(ge=0.5, le=1)

    def has_signal(self, margin: float) -> bool:
        """Whether the AUC is at least ``margin`` from 0.5, in either direction."""
        return self.auc is not None and abs(self.auc - 0.5) >= margin - _SIGNAL_TOLERANCE


class FeatureSignalCheck(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    pairs: int = Field(ge=2)
    # Candidate pairs that are hidden links.
    positives: int = Field(ge=1)
    # A column has signal when its AUC is at least this far from 0.5.
    margin: float = Field(gt=0, lt=0.5)
    columns: tuple[FeatureAuc, ...] = Field(min_length=1)
    features_with_signal: int = Field(ge=0)

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if self.positives >= self.pairs:
            raise ValueError("some candidate pairs must be other than hidden links")
        names = [entry.column for entry in self.columns]
        if len(set(names)) != len(names):
            raise ValueError("duplicate columns")
        signal = sum(1 for entry in self.columns if entry.has_signal(self.margin))
        if signal != self.features_with_signal:
            raise ValueError("features_with_signal must count the columns with signal")
        return self


class ScorerCheck(BaseModel):
    """The baseline scorer on the same held-out pairs, against its best single feature."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    weights_version: str = Field(min_length=1)
    score_auc: float = Field(ge=0, le=1)
    best_feature: str = Field(min_length=1)
    best_feature_auc: float = Field(ge=0.5, le=1)
    # score_auc - best_feature_auc: what the scorer adds over one feature.
    score_auc_lift: float = Field(ge=-1, le=1)
    # Share of the weight on link-derived columns, whose AUC the held-out protocol inflates
    # because hiding a link moves them by itself; score_auc carries that inflation.
    link_derived_weight_share: float = Field(ge=0, le=1)
    # The like-for-like bar: the scorer on the other columns, weights renormalised, against
    # the best of those columns. None when every weighted column is link-derived.
    score_auc_excl_link_counts: float | None = Field(default=None, ge=0, le=1)
    best_feature_excl_link_counts: str = Field(min_length=1)
    best_feature_auc_excl_link_counts: float = Field(ge=0.5, le=1)
    score_auc_lift_excl_link_counts: float | None = Field(default=None, ge=-1, le=1)
    # Counts of the 0-100 score in SCORE_HISTOGRAM_BINS equal bins.
    hidden_histogram: tuple[int, ...] = Field(
        min_length=SCORE_HISTOGRAM_BINS, max_length=SCORE_HISTOGRAM_BINS
    )
    other_histogram: tuple[int, ...] = Field(
        min_length=SCORE_HISTOGRAM_BINS, max_length=SCORE_HISTOGRAM_BINS
    )

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if abs(self.score_auc_lift - (self.score_auc - self.best_feature_auc)) > _LIFT_TOLERANCE:
            raise ValueError("score_auc_lift must be score_auc - best_feature_auc")
        excl, lift = self.score_auc_excl_link_counts, self.score_auc_lift_excl_link_counts
        if (excl is None) != (self.link_derived_weight_share == 1):
            raise ValueError("the excl score is None exactly when all the weight is link-derived")
        if (lift is None) != (excl is None):
            raise ValueError("the excl score and its lift are set together")
        if (
            excl is not None
            and lift is not None
            and abs(lift - (excl - self.best_feature_auc_excl_link_counts)) > _LIFT_TOLERANCE
        ):
            raise ValueError(
                "score_auc_lift_excl_link_counts must be "
                "score_auc_excl_link_counts - best_feature_auc_excl_link_counts"
            )
        if self.best_feature_auc_excl_link_counts > self.best_feature_auc:
            raise ValueError("the best of fewer columns cannot beat the best of all")
        if any(n < 0 for n in (*self.hidden_histogram, *self.other_histogram)):
            raise ValueError("histogram counts are non-negative")
        return self


class KeywordExtractability(BaseModel):
    """Candidate pairs whose source copy carries the target's keyword as the extraction ladder
    finds it, existing anchors aside: the primary keyword against any of its ranked set."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    pairs: int = Field(ge=1)
    # Found at any rung.
    found_primary: float = Field(ge=0, le=1)
    found_set: float = Field(ge=0, le=1)
    # By the best rung any keyword of the set was found at; they add up to found_set.
    exact_set: float = Field(ge=0, le=1)
    stemmed_set: float = Field(ge=0, le=1)
    stem_set_set: float = Field(ge=0, le=1)
    # Every word of the keyword anywhere in the copy, in any order: the literal upper bound
    # of the exact rung.
    words_primary: float = Field(ge=0, le=1)
    words_set: float = Field(ge=0, le=1)
    stem_set_threshold: float = Field(gt=0, le=1)

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if self.found_primary > self.found_set or self.words_primary > self.words_set:
            raise ValueError("the primary keyword cannot match more pairs than its set")
        rungs = self.exact_set + self.stemmed_set + self.stem_set_set
        if abs(rungs - self.found_set) > _LIFT_TOLERANCE:
            raise ValueError("the best rungs must add up to found_set")
        return self


class AnchorMatchCheck(BaseModel):
    """Descriptive anchors of existing links that match their target's keywords."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    anchors: int = Field(ge=1)
    primary: float = Field(ge=0, le=1)
    any_rank: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if self.primary > self.any_rank:
            raise ValueError("the primary keyword cannot match more anchors than the set")
        return self


class RankRelevance(BaseModel):
    """Cosine of the keywords at one rank to their page's content vector."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    rank: int = Field(ge=1)
    keywords: int = Field(ge=1)
    mean: float = Field(ge=-1, le=1)
    p10: float = Field(ge=-1, le=1)
    p50: float = Field(ge=-1, le=1)
    p90: float = Field(ge=-1, le=1)

    @model_validator(mode="after")
    def _ordered(self) -> Self:
        if not self.p10 <= self.p50 <= self.p90:
            raise ValueError("percentiles must not decrease")
        return self


class RelevanceGroup(BaseModel):
    """Cosine of one group of keywords to their pages' content vectors: an origin or a length."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    group: str = Field(min_length=1)
    keywords: int = Field(ge=1)
    mean: float = Field(ge=-1, le=1)
    p50: float = Field(ge=-1, le=1)


class KeywordRelevance(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    ranks: tuple[RankRelevance, ...] = Field(min_length=1)
    # Rank averages mix origins and lengths, which a cosine to a whole-page vector does not
    # score alike.
    by_origin: tuple[RelevanceGroup, ...] = Field(min_length=1)
    by_length: tuple[RelevanceGroup, ...] = Field(min_length=1)
    # Distinct keyword texts, and how many of them were embedded in this run or read back
    # from the tenant's cache.
    texts: int = Field(ge=1)
    embedded: int = Field(ge=0)
    cached: int = Field(ge=0)
    api_tokens: int = Field(ge=0)

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        ranks = [entry.rank for entry in self.ranks]
        if ranks != sorted(set(ranks)):
            raise ValueError("ranks must be unique and ascending")
        for groups in (self.by_origin, self.by_length):
            names = [entry.group for entry in groups]
            if len(set(names)) != len(names):
                raise ValueError("duplicate relevance groups")
        counts = {
            sum(entry.keywords for entry in groups)
            for groups in (self.ranks, self.by_origin, self.by_length)
        }
        if len(counts) != 1:
            raise ValueError("ranks, origins and lengths must group the same keywords")
        if self.embedded + self.cached != self.texts:
            raise ValueError("every text is either embedded or cached")
        return self


class KeywordCheck(BaseModel):
    """Keyword resolution replayed read-only, and how usable its keywords are."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    # Crawled 2xx pages, and those that resolved a keyword.
    pages: int = Field(ge=1)
    resolved: int = Field(ge=0)
    by_rung: dict[KeywordRung, int]
    fallbacks_rejected: dict[str, int]
    # Share of the distinct resolved keywords that exactly one page resolved.
    unique_share: float | None = Field(default=None, ge=0, le=1)
    extractability: KeywordExtractability | None = None
    anchors: AnchorMatchCheck | None = None
    relevance: KeywordRelevance | None = None
    # Why keyword relevance does not apply: no Voyage key, a Voyage outage, page vectors of
    # another model; None when it does.
    relevance_reason: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if self.resolved > self.pages:
            raise ValueError("more resolved pages than pages")
        if any(n < 0 for n in (*self.by_rung.values(), *self.fallbacks_rejected.values())):
            raise ValueError("counts are non-negative")
        if sum(self.by_rung.values()) != self.resolved:
            raise ValueError("rung counts must add up to the resolved pages")
        if (self.relevance is None) != (self.relevance_reason is not None):
            raise ValueError("relevance_reason is set exactly when relevance is None")
        return self


class LinkRelevanceCheck(BaseModel):
    """The stored relevance scores of the existing links."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    # Body links between crawled pages whose target has a content vector.
    links: int = Field(ge=1)
    context: ScoreDistribution
    anchor: ScoreDistribution | None = None

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if self.context.count > self.links or (self.anchor and self.anchor.count > self.links):
            raise ValueError("more scores than links")
        return self


class CoverageCheck(BaseModel):
    """Data gaps of the evaluated candidate pairs and of the keywords."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    pairs: int = Field(ge=0)
    # Share of pairs whose target has GSC data; None without pairs.
    gsc_pair_share: float | None = Field(default=None, ge=0, le=1)
    # Share of crawled 2xx pages with a resolved keyword; None without such pages.
    keyword_page_share: float | None = Field(default=None, ge=0, le=1)
    all_null_columns: tuple[str, ...] = ()
    constant_columns: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if (self.gsc_pair_share is None) != (self.pairs == 0):
            raise ValueError("the GSC share is set exactly when there are pairs")
        return self


class QualityAlert(BaseModel):
    """A headline metric that moved beyond its band since the previous run."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    metric: str = Field(min_length=1)
    previous: float = Field(allow_inf_nan=False)
    current: float = Field(allow_inf_nan=False)
    # Relative to the previous value for a relative band, else the difference; None when a
    # relative band meets a previous value of 0.
    change: float | None = Field(default=None, allow_inf_nan=False)
    band: float = Field(gt=0)
    relative: bool

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if self.change is None and not (self.relative and self.previous == 0):
            raise ValueError("change is None only for a relative band from 0")
        return self


class QualityBaseline(BaseModel):
    """The tenant's previous finished quality-eval run, which alerts compare against."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: str = Field(min_length=1)
    metrics: dict[str, float]

    @model_validator(mode="after")
    def _finite(self) -> Self:
        if not all(math.isfinite(value) for value in self.metrics.values()):
            raise ValueError("baseline metrics must be finite")
        return self


class QualityReport(BaseModel):
    """One quality evaluation of a tenant."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    tenant_id: str = Field(min_length=1)
    versions: QualityVersions
    retrieval: RetrievalCheck | None = None
    feature_signal: FeatureSignalCheck | None = None
    scorer: ScorerCheck | None = None
    keywords: KeywordCheck | None = None
    link_relevance: LinkRelevanceCheck | None = None
    coverage: CoverageCheck
    not_applicable: tuple[CheckName, ...] = ()
    # None on the tenant's first run: nothing to compare against.
    baseline_run_id: str | None = Field(default=None, min_length=1)
    alerts: tuple[QualityAlert, ...] = ()
    seconds: float = Field(ge=0)
    finished_at: datetime

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        skipped = Counter(self.not_applicable)
        if any(count > 1 for count in skipped.values()):
            raise ValueError("a check is listed as not applicable twice")
        sections: dict[CheckName, object] = {
            "retrieval": self.retrieval,
            "feature_signal": self.feature_signal,
            "scorer": self.scorer,
            "keywords": self.keywords,
            "link_relevance": self.link_relevance,
        }
        for name, section in sections.items():
            if (section is None) != (name in skipped):
                raise ValueError(f"{name} is not applicable exactly when it has no section")
        keywords = self.keywords
        if keywords is None:
            if skipped.keys() & set(KEYWORD_CHECKS):
                raise ValueError("keyword sub-checks are listed only when keywords apply")
        else:
            parts: dict[CheckName, object] = {
                "keyword_uniqueness": keywords.unique_share,
                "keyword_extractability": keywords.extractability,
                "anchor_match": keywords.anchors,
                "keyword_relevance": keywords.relevance,
            }
            for name, part in parts.items():
                if (part is None) != (name in skipped):
                    raise ValueError(f"{name} is not applicable exactly when it is None")
        metrics = [alert.metric for alert in self.alerts]
        if len(set(metrics)) != len(metrics):
            raise ValueError("a metric alerts twice")
        if self.alerts and self.baseline_run_id is None:
            raise ValueError("alerts need a baseline run")
        return self
