"""The learned ranker: its settings, the report of a training run and of a ranking run.

Labels are proxies: some existing body links are hidden and a candidate pair is positive when
it is one of them. Nothing here holds a page url, so a whole report can be logged.
"""

import math
from typing import Final, Self

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

from linking_engine.models.enums import ScorerName

NDCG_HISTOGRAM_BINS: Final = 10
_SHARE_TOLERANCE: Final = 1e-9
# NDCG@10 over a set of test groups, then the low and high end of its 95% interval.
NdcgInterval = tuple[float, float, float]


class RankerParams(BaseModel):
    """LightGBM lambdarank hyperparameters."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    learning_rate: float = Field(default=0.05, gt=0)
    num_leaves: int = Field(default=31, ge=2)
    min_data_in_leaf: int = Field(default=20, ge=1)
    feature_fraction: float = Field(default=0.8, gt=0, le=1)
    max_rounds: int = Field(default=1000, ge=1)
    early_stopping_rounds: int = Field(default=50, ge=1)
    eval_at: int = Field(default=10, ge=1)
    seed: int = 42
    # Columns a pair's score never falls with, the others fixed; empty turns constraints off.
    monotone_increasing: tuple[str, ...] = (
        "content_cosine",
        "anchor_target_fit",
        "context_relevance",
    )

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        names = self.monotone_increasing
        if len(set(names)) != len(names) or not all(name.strip() for name in names):
            raise ValueError("monotone columns are distinct names")
        return self


class HeldOutSettings(BaseModel):
    """How the proxy labels are drawn and the source pages split.

    Round r hides the body links whose pair hash falls in ``[r * share, (r + 1) * share)`` of
    the hash range, so rounds are disjoint and round 0 is the quality evaluation's hidden set.
    With rounds * share = 1, the default, every body link is a positive in exactly one round.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    rounds: int = Field(default=10, ge=1)
    share: float = Field(default=0.10, gt=0, lt=1)
    # The quality evaluation's hide seed.
    seed: int = 42
    # Share of source pages held out for the test, then of the rest for early stopping.
    test_share: float = Field(default=0.2, gt=0, lt=1)
    valid_share: float = Field(default=0.1, gt=0, lt=1)
    # The split every model is trained on and the promotion gate compares on.
    split_seed: int = 7
    # Split seeds the held-out evaluation is repeated over, since one split is noisy.
    evaluation_seeds: tuple[int, ...] = (7, 11, 23, 42, 99)

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if self.rounds * self.share > 1 + _SHARE_TOLERANCE:
            raise ValueError("rounds * share must not exceed 1")
        if len(set(self.evaluation_seeds)) != len(self.evaluation_seeds):
            raise ValueError("evaluation seeds are distinct")
        if self.split_seed not in self.evaluation_seeds:
            raise ValueError("evaluation seeds must include the split seed")
        return self


class RoundSummary(BaseModel):
    """One held-out round: what was hidden and what retrieval returned on its view."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    round: int = Field(ge=0)
    hidden: int = Field(ge=0)
    # Hidden links whose target is a retrieval target and whose source is eligible for it.
    recoverable: int = Field(ge=0)
    # Candidate pairs on the view; positives are the hidden links among them.
    pairs: int = Field(ge=0)
    positives: int = Field(ge=0)
    groups_with_positive: int = Field(ge=0)
    # Share of the positive, then of the other pairs with an anchor placement value; None
    # without such pairs. A gap between them is the protocol handing anchors to positives.
    positive_placement_share: float | None = Field(ge=0, le=1)
    negative_placement_share: float | None = Field(ge=0, le=1)

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if not self.positives <= self.recoverable <= self.hidden:
            raise ValueError("positives <= recoverable <= hidden")
        if self.positives > self.pairs:
            raise ValueError("more positives than pairs")
        if self.groups_with_positive > self.positives:
            raise ValueError("more groups with a positive than positives")
        if (self.positive_placement_share is None) != (self.positives == 0):
            raise ValueError("positive_placement_share is None exactly without positives")
        if (self.negative_placement_share is None) != (self.positives == self.pairs):
            raise ValueError("negative_placement_share is None exactly without other pairs")
        return self


class RankingMetrics(BaseModel):
    """One scorer on the test source pages, over the groups (round, source page) that hold a
    hidden link."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    scorer: ScorerName
    # Mean per-group NDCG@10 and its bootstrap 95% interval, resampling source pages with all
    # their groups: a source page is in up to one group per round, and those are correlated.
    ndcg_at_10: float = Field(ge=0, le=1)
    ci_low: float = Field(ge=0, le=1)
    ci_high: float = Field(ge=0, le=1)
    precision_at_5: float = Field(ge=0, le=1)
    groups: int = Field(ge=1)
    # Of all test groups, before those without a positive were dropped.
    groups_with_positive_share: float = Field(gt=0, le=1)
    # Pairs labelled positive in the evaluated groups.
    n_labelled_pairs: int = Field(ge=1)
    # Mean NDCG@10 by round, over the rounds with an evaluated group.
    per_round: dict[int, float]
    # Per-group NDCG@10 in NDCG_HISTOGRAM_BINS equal bins over [0, 1].
    histogram: tuple[int, ...] = Field(
        min_length=NDCG_HISTOGRAM_BINS, max_length=NDCG_HISTOGRAM_BINS
    )

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if self.ci_low > self.ci_high:
            raise ValueError("ci_low must not exceed ci_high")
        if self.n_labelled_pairs < self.groups:
            raise ValueError("every evaluated group holds a positive")
        if not self.per_round or any(r < 0 for r in self.per_round):
            raise ValueError("per_round is keyed by the evaluated rounds")
        if any(not 0 <= v <= 1 for v in self.per_round.values()):
            raise ValueError("per_round values are in [0, 1]")
        if any(n < 0 for n in self.histogram) or sum(self.histogram) != self.groups:
            raise ValueError("histogram counts must add up to the groups")
        return self


class ImportanceEntry(BaseModel):
    """Total split gain of one model column."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    column: str = Field(min_length=1)
    gain: float = Field(ge=0)
    gain_share: float = Field(ge=0, le=1)


class PromotionDecision(BaseModel):
    """The new model against the production holder, else against the baseline scorer, on the
    same test groups."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    rival: ScorerName
    holder_version: str | None = None
    # Mean paired difference of per-group NDCG@10, new - rival, and its bootstrap 95% interval
    # over source pages.
    delta: float = Field(ge=-1, le=1)
    delta_ci_low: float = Field(ge=-1, le=1)
    delta_ci_high: float = Field(ge=-1, le=1)
    would_promote: bool
    # would_promote and promotion allowed in this environment.
    promoted: bool
    reason: str = Field(min_length=1)

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if self.rival not in (ScorerName.HOLDER, ScorerName.BASELINE):
            raise ValueError("the rival is the holder or the baseline")
        if (self.holder_version is None) != (self.rival is ScorerName.BASELINE):
            raise ValueError("holder_version is set exactly when the holder is the rival")
        if self.delta_ci_low > self.delta_ci_high:
            raise ValueError("delta_ci_low must not exceed delta_ci_high")
        if self.would_promote != (self.delta_ci_low > 0):
            raise ValueError("would_promote exactly when delta_ci_low > 0")
        if self.promoted and not self.would_promote:
            raise ValueError("only a model that would promote is promoted")
        return self


def _interval_ok(found: NdcgInterval) -> bool:
    _, low, high = found
    return all(0 <= x <= 1 for x in found) and low <= high


class SeedResult(BaseModel):
    """The held-out evaluation on one split seed: the test sources re-drawn, the learned and
    plain models retrained, every scorer on that seed's test groups without orphan targets."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    seed: int
    test_pages: int = Field(ge=1)
    learned: NdcgInterval
    plain: NdcgInterval
    baseline: NdcgInterval
    # learned - plain NDCG@10, paired per group, and its 95% interval over source pages.
    delta: float = Field(ge=-1, le=1)
    delta_ci_low: float = Field(ge=-1, le=1)
    delta_ci_high: float = Field(ge=-1, le=1)

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if not all(_interval_ok(found) for found in (self.learned, self.plain, self.baseline)):
            raise ValueError("NDCG@10 and its interval are in [0, 1], low <= high")
        if self.delta_ci_low > self.delta_ci_high:
            raise ValueError("delta_ci_low must not exceed delta_ci_high")
        return self

    @property
    def significantly_worse(self) -> bool:
        return self.delta_ci_high < 0

    @property
    def significantly_better(self) -> bool:
        return self.delta_ci_low > 0


class ProductMeasures(BaseModel):
    """One scorer on the tenant's production candidate pairs, over the first ``k`` pairs of
    every source page: what an editor would be shown."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    scorer: ScorerName
    k: int = Field(ge=1)
    # Mean content cosine of those pairs; None when none has one.
    top_relevance: float | None = Field(ge=-1, le=1)
    # Share of those pairs whose two pages share a hub.
    same_hub_share: float = Field(ge=0, le=1)
    # Share of those pairs whose target is an orphan, against the orphans' share of the
    # candidate target pages.
    orphan_slot_share: float = Field(ge=0, le=1)
    orphan_page_share: float = Field(ge=0, le=1)
    # Share of the orphan target pages in some source page's first k; None without any.
    orphans_reached: float | None = Field(ge=0, le=1)

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if (self.orphans_reached is None) != (self.orphan_page_share == 0):
            raise ValueError("orphans_reached is None exactly without orphan target pages")
        return self


class RankerReport(BaseModel):
    """One training run of a tenant's ranker. A skipped run names its reason and carries no
    model, metrics or promotion."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    tenant_id: str = Field(min_length=1)
    # Crawled pages, and the distinct body links between them that the rounds hide from.
    corpus_pages: int = Field(ge=0)
    body_links: int = Field(ge=0)
    # The feature code digest and the checkout that trained.
    feature_set_version: str = Field(min_length=1)
    git_sha: str = Field(min_length=1)
    settings: HeldOutSettings
    params: RankerParams
    rounds: tuple[RoundSummary, ...]
    columns: tuple[str, ...]
    # Feature columns kept out of the model, with the reason.
    excluded_columns: dict[str, str]
    train_groups: int = Field(ge=0)
    valid_groups: int = Field(ge=0)
    test_groups: int = Field(ge=0)
    # Positive pairs over all rounds.
    positives: int = Field(ge=0)
    best_iteration: int | None = Field(default=None, ge=1)
    metrics: tuple[RankingMetrics, ...] = ()
    importance: tuple[ImportanceEntry, ...] = ()
    # A column holding more than the dominant share of the gain.
    dominant_feature: str | None = None
    promotion: PromotionDecision | None = None
    model_version: str | None = None
    skipped_reason: str | None = Field(default=None, min_length=1)
    # Orphan targets, candidate targets without an inbound body link on the full graph, are
    # negatives by construction: their rows are left out of training and evaluation.
    unlabelable_targets: int = Field(ge=0)
    unlabelable_rows: int = Field(ge=0)
    seed_results: tuple[SeedResult, ...] = ()
    # Evaluation seeds whose split could not be evaluated, with the reason.
    skipped_seeds: dict[int, str] = Field(default_factory=dict)
    product_measures: tuple[ProductMeasures, ...] = ()
    product_skipped_reason: str | None = Field(default=None, min_length=1)
    seconds: float = Field(ge=0)
    finished_at: AwareDatetime

    @property
    def seeds_worse(self) -> int:
        """Seeds where the learned model is significantly worse than the plain one."""
        return sum(result.significantly_worse for result in self.seed_results)

    @property
    def seeds_better(self) -> int:
        """Seeds where the learned model is significantly better than the plain one."""
        return sum(result.significantly_better for result in self.seed_results)

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        seeds = [result.seed for result in self.seed_results]
        if len(set(seeds)) != len(seeds) or not set(seeds) <= set(self.settings.evaluation_seeds):
            raise ValueError("seed results are for distinct evaluation seeds")
        if not set(self.skipped_seeds) <= set(self.settings.evaluation_seeds) or set(
            self.skipped_seeds
        ) & set(seeds):
            raise ValueError("skipped seeds are evaluation seeds without a result")
        if not all(reason.strip() for reason in self.skipped_seeds.values()):
            raise ValueError("a skipped seed names its reason")
        measured = [entry.scorer for entry in self.product_measures]
        if len(set(measured)) != len(measured):
            raise ValueError("a scorer is measured twice")
        ids = [summary.round for summary in self.rounds]
        if ids != sorted(set(ids)) or any(r >= self.settings.rounds for r in ids):
            raise ValueError("rounds are unique, ascending and within the settings")
        if any(summary.hidden > self.body_links for summary in self.rounds):
            raise ValueError("a round hides more links than there are")
        if self.positives != sum(summary.positives for summary in self.rounds):
            raise ValueError("positives must add up over the rounds")
        if len(set(self.columns)) != len(self.columns):
            raise ValueError("duplicate columns")
        if set(self.excluded_columns) & set(self.columns):
            raise ValueError("an excluded column is a model column")
        scorers = [entry.scorer for entry in self.metrics]
        if len(set(scorers)) != len(scorers):
            raise ValueError("a scorer is evaluated twice")
        importance = [entry.column for entry in self.importance]
        if len(set(importance)) != len(importance) or not set(importance) <= set(self.columns):
            raise ValueError("importance covers model columns, once each")
        shares = sum(entry.gain_share for entry in self.importance)
        if any(entry.gain > 0 for entry in self.importance) and not math.isclose(
            shares, 1, abs_tol=_SHARE_TOLERANCE
        ):
            raise ValueError("gain shares must add up to 1")
        if self.dominant_feature is not None and self.dominant_feature not in importance:
            raise ValueError("the dominant feature is an importance entry")
        if self.skipped_reason is not None:
            model = (self.best_iteration, self.promotion, self.model_version, self.dominant_feature)
            if any(value is not None for value in model) or self.metrics or self.importance:
                raise ValueError("a skipped run has no model, metrics or promotion")
            if self.seed_results or self.skipped_seeds or self.product_measures:
                raise ValueError(
                    "a skipped run has no seed results, skipped seeds or product measures"
                )
        elif self.best_iteration is None or not self.metrics:
            raise ValueError("a trained run has a best iteration and metrics")
        elif {*seeds, *self.skipped_seeds} != set(self.settings.evaluation_seeds):
            raise ValueError("a trained run has a result or a skip reason for every seed")
        elif (not self.product_measures) != (self.product_skipped_reason is not None):
            raise ValueError("a trained run has product measures or the reason it has none")
        return self


class RankReport(BaseModel):
    """One ranking run: the learned model's scores, else the baseline scorer's with the reason."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    tenant_id: str = Field(min_length=1)
    pairs: int = Field(ge=0)
    scorer: ScorerName
    model_version: str | None = None
    # Why the baseline ranked: no promoted model, registry unreachable, columns missing.
    fallback_reason: str | None = Field(default=None, min_length=1)
    seconds: float = Field(ge=0)

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if self.scorer not in (ScorerName.LEARNED, ScorerName.BASELINE):
            raise ValueError("pairs are ranked by the learned model or the baseline")
        learned = self.scorer is ScorerName.LEARNED
        if (self.model_version is not None) != learned:
            raise ValueError("model_version is set exactly when the learned model ranked")
        if (self.fallback_reason is not None) != (not learned):
            raise ValueError("fallback_reason is set exactly when the baseline ranked")
        return self
