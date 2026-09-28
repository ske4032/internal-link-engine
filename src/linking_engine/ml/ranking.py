"""The maths of the learned ranker: the source split, query groups, LambdaMART training,
chunked prediction, NDCG@10 with bootstrap intervals, gain importance and the promotion gate.
No I/O.

A query group is one source page in one held-out round; its candidate pairs are ranked against
each other, never across sources.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Final

import lightgbm
import numpy as np

from linking_engine.discovery.features import FEATURE_COLUMNS
from linking_engine.ml.quality import LINK_DERIVED_COLUMNS
from linking_engine.models import ImportanceEntry, PromotionDecision, RankingMetrics, ScorerName
from linking_engine.models.ranking import NDCG_HISTOGRAM_BINS

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Mapping, Sequence

    import numpy.typing as npt
    from pandas import DataFrame

    from linking_engine.models import RankerParams, RankerReport

EXCLUDED_COLUMNS: Final[Mapping[str, str]] = MappingProxyType(
    {"target_crawl_depth": "stored BFS over links that still include the hidden ones"}
)
MODEL_COLUMNS: Final = tuple(column for column in FEATURE_COLUMNS if column not in EXCLUDED_COLUMNS)
# The like-for-like model: without the columns that hiding a link moves by itself.
LIKE_FOR_LIKE_COLUMNS: Final = tuple(
    column for column in MODEL_COLUMNS if column not in LINK_DERIVED_COLUMNS
)
# Set only for pairs with a chosen anchor. A hidden link's anchor phrase stays in the source copy,
# so nearly every positive has one, while a new pair has one only when the copy names the target.
PLACEMENT_COLUMNS: Final = ("context_relevance", "anchor_target_fit")
# The like-for-like model for anchors: without the placement columns.
EXCL_PLACEMENT_COLUMNS: Final = tuple(
    column for column in MODEL_COLUMNS if column not in PLACEMENT_COLUMNS
)
GROUP_COLUMNS: Final = ("round", "source_url")
LABEL_COLUMN: Final = "label"
DOMINANT_SHARE: Final = 0.6
MIN_TRAIN_GROUPS: Final = 100
MIN_TEST_GROUPS: Final = 30
PREDICT_CHUNK: Final = 50_000
BOOTSTRAP_SAMPLES: Final = 1000
NDCG_K: Final = 10
PRECISION_K: Final = 5
# Fixed so that training is reproducible whatever the machine's core count.
TRAIN_THREADS: Final = 4
_HASH_RANGE: Final = 1 << 256
_COVERAGE_TOLERANCE: Final = 1e-9
# Resampled values held at once by the bootstrap.
_BOOTSTRAP_BLOCK: Final = 2_000_000


@dataclass(frozen=True)
class Trained:
    """A booster cut to its best iteration and the columns it reads, in order."""

    booster: lightgbm.Booster
    columns: tuple[str, ...]
    best_iteration: int


def _source_hash(seed: int, source: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}\t{source}".encode()).digest())


def split_sources(sources: Iterable[str], *, share: float, seed: int) -> frozenset[str]:
    """The sources whose sha256 of (seed, source) falls below ``share`` of the hash range: a
    source stays on its side from run to run and round to round."""
    if not 0 < share < 1:
        raise ValueError("share must be in (0, 1)")
    cut = int(share * _HASH_RANGE)
    return frozenset(source for source in set(sources) if _source_hash(seed, source) < cut)


def with_groups(frame: DataFrame) -> tuple[DataFrame, npt.NDArray[np.int64]]:
    """The rows sorted by round, source and target, and the size of each group in that order."""
    missing = {*GROUP_COLUMNS, "target_url"} - set(frame.columns)
    if missing:
        raise ValueError(f"missing group columns: {sorted(missing)}")
    ordered = frame.sort_values([*GROUP_COLUMNS, "target_url"], ignore_index=True)
    if ordered.empty:
        return ordered, np.zeros(0, dtype=np.int64)
    rounds = ordered["round"].to_numpy()
    sources = ordered["source_url"].to_numpy()
    starts = np.flatnonzero((rounds[1:] != rounds[:-1]) | (sources[1:] != sources[:-1])) + 1
    bounds = np.concatenate(([0], starts, [len(ordered)]))
    return ordered, np.diff(bounds).astype(np.int64)


def positive_groups(frame: DataFrame) -> DataFrame:
    """The rows of the groups with at least one positive label, in their order."""
    best = frame.groupby(list(GROUP_COLUMNS), sort=False, dropna=False)[LABEL_COLUMN]
    return frame.loc[(best.transform("max") > 0).to_numpy()]


def _matrix(frame: DataFrame, columns: Sequence[str]) -> npt.NDArray[np.float32]:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"missing feature columns: {missing}")
    matrix: npt.NDArray[np.float32] = frame.loc[:, list(columns)].to_numpy(
        dtype=np.float32, na_value=np.nan
    )
    return matrix


def _dataset(
    frame: DataFrame, columns: Sequence[str], reference: lightgbm.Dataset | None = None
) -> lightgbm.Dataset:
    ordered, sizes = with_groups(frame)
    if not len(sizes):
        raise ValueError("no query groups")
    return lightgbm.Dataset(
        _matrix(ordered, columns),
        label=ordered[LABEL_COLUMN].to_numpy(dtype=np.float64),
        group=sizes,
        feature_name=list(columns),
        reference=reference,
        free_raw_data=True,
    )


def train(
    train: DataFrame, valid: DataFrame, columns: Sequence[str], params: RankerParams
) -> Trained:
    """LambdaMART on the training groups, stopped early on NDCG at ``eval_at`` of the validation
    groups; the same data, columns and params give the same model."""
    if not columns or len(set(columns)) != len(columns):
        raise ValueError("columns must be distinct and not empty")
    train_set = _dataset(train, columns)
    valid_set = _dataset(valid, columns, reference=train_set)
    booster = lightgbm.train(
        {
            "objective": "lambdarank",
            "metric": "ndcg",
            "eval_at": [params.eval_at],
            "learning_rate": params.learning_rate,
            "num_leaves": params.num_leaves,
            "min_data_in_leaf": params.min_data_in_leaf,
            "feature_fraction": params.feature_fraction,
            "seed": params.seed,
            "deterministic": True,
            "force_row_wise": True,
            "num_threads": TRAIN_THREADS,
            "verbosity": -1,
        },
        train_set,
        num_boost_round=params.max_rounds,
        valid_sets=[valid_set],
        valid_names=["valid"],
        callbacks=[
            lightgbm.early_stopping(
                params.early_stopping_rounds, first_metric_only=True, verbose=False
            )
        ],
    )
    best = booster.best_iteration if booster.best_iteration > 0 else booster.current_iteration()
    cut = lightgbm.Booster(model_str=booster.model_to_string(num_iteration=best))
    return Trained(booster=cut, columns=tuple(columns), best_iteration=best)


def predict(
    model: Trained | lightgbm.Booster, columns: Sequence[str], chunks: Iterable[DataFrame]
) -> Iterator[npt.NDArray[np.float64]]:
    """Scores for each chunk in turn, never holding more than one chunk's matrix; a chunk
    lacking a column raises ValueError."""
    booster = model.booster if isinstance(model, Trained) else model
    if list(columns) != booster.feature_name():
        raise ValueError("columns differ from the model's features")
    return _predict(booster, tuple(columns), chunks)


def _predict(
    booster: lightgbm.Booster, columns: tuple[str, ...], chunks: Iterable[DataFrame]
) -> Iterator[npt.NDArray[np.float64]]:
    for chunk in chunks:
        if chunk.empty:
            yield np.zeros(0, dtype=np.float64)
            continue
        yield np.asarray(booster.predict(_matrix(chunk, columns)), dtype=np.float64)


def _offsets(sizes: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
    return (np.cumsum(sizes) - sizes).astype(np.int64)


def _order(keys: npt.NDArray[np.float64], sizes: npt.NDArray[np.int64]) -> npt.NDArray[np.intp]:
    """Rows by descending key within each group, ties by row order, NaN last; groups stay put."""
    group = np.repeat(np.arange(len(sizes)), sizes)
    clean = np.where(np.isnan(keys), -np.inf, keys)
    return np.lexsort((np.arange(len(keys)), -clean, group))


def _ranks(sizes: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
    return np.arange(int(sizes.sum()), dtype=np.int64) - np.repeat(_offsets(sizes), sizes)


def _dcg(
    labels: npt.NDArray[np.float64], sizes: npt.NDArray[np.int64], k: int
) -> npt.NDArray[np.float64]:
    ranks = _ranks(sizes)
    gains = np.where(ranks < k, (np.exp2(labels) - 1) / np.log2(ranks + 2), 0.0)
    return np.add.reduceat(gains, _offsets(sizes)).astype(np.float64)


def _ndcg(
    labels: npt.NDArray[np.float64],
    scores: npt.NDArray[np.float64],
    sizes: npt.NDArray[np.int64],
    k: int,
) -> npt.NDArray[np.float64]:
    if k < 1:
        raise ValueError("k must be at least 1")
    if not len(sizes):
        return np.zeros(0, dtype=np.float64)
    dcg = _dcg(labels[_order(scores, sizes)], sizes, k)
    ideal = _dcg(labels[_order(labels, sizes)], sizes, k)
    ratio = np.divide(dcg, ideal, out=np.zeros_like(dcg), where=ideal > 0)
    return np.clip(ratio, 0.0, 1.0)


def ndcg_at(labels: npt.ArrayLike, scores: npt.ArrayLike, k: int) -> float:
    """NDCG@k of one ranked list: gain 2**label - 1, log2 discount, ties in input order; 0 when
    no item is relevant."""
    label_array = np.asarray(labels, dtype=np.float64)
    score_array = np.asarray(scores, dtype=np.float64)
    if label_array.shape != score_array.shape or label_array.ndim != 1:
        raise ValueError("labels and scores must be 1-d and of one length")
    if not len(label_array):
        return 0.0
    sizes = np.array([len(label_array)], dtype=np.int64)
    return float(_ndcg(label_array, score_array, sizes, k)[0])


def _group_arrays(
    frame: DataFrame, score_column: str
) -> tuple[DataFrame, npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.int64]]:
    ordered, sizes = with_groups(frame)
    labels = ordered[LABEL_COLUMN].to_numpy(dtype=np.float64)
    scores = ordered[score_column].to_numpy(dtype=np.float64, na_value=np.nan)
    return ordered, labels, scores, sizes


def _sources(ordered: DataFrame, sizes: npt.NDArray[np.int64]) -> npt.NDArray[np.object_]:
    sources: npt.NDArray[np.object_] = ordered["source_url"].to_numpy(dtype=object)
    return sources[_offsets(sizes)]


def group_sources(frame: DataFrame) -> npt.NDArray[np.object_]:
    """The source page of every group, in group order: the unit the bootstrap resamples."""
    return _sources(*with_groups(frame))


def group_ndcg(frame: DataFrame, score_column: str, k: int) -> npt.NDArray[np.float64]:
    """NDCG@k of every group, in group order (by round, then source)."""
    _, labels, scores, sizes = _group_arrays(frame, score_column)
    return _ndcg(labels, scores, sizes, k)


def _precision(
    labels: npt.NDArray[np.float64],
    scores: npt.NDArray[np.float64],
    sizes: npt.NDArray[np.int64],
    k: int,
) -> npt.NDArray[np.float64]:
    if k < 1:
        raise ValueError("k must be at least 1")
    if not len(sizes):
        return np.zeros(0, dtype=np.float64)
    hits = np.where(_ranks(sizes) < k, labels[_order(scores, sizes)] > 0, False)
    return np.add.reduceat(hits.astype(np.float64), _offsets(sizes)) / k


def precision_at(frame: DataFrame, score_column: str, k: int) -> float:
    """Mean over groups of the positives among the first ``k`` pairs, divided by ``k`` even
    when a group holds fewer pairs."""
    _, labels, scores, sizes = _group_arrays(frame, score_column)
    per_group = _precision(labels, scores, sizes, k)
    return float(per_group.mean()) if len(per_group) else 0.0


def bootstrap_ci(
    values: npt.ArrayLike,
    *,
    clusters: npt.ArrayLike | None = None,
    samples: int = BOOTSTRAP_SAMPLES,
    seed: int,
    level: float = 0.95,
) -> tuple[float, float]:
    """Percentile bootstrap interval of the mean of ``values``. With ``clusters``, one label
    per value, whole clusters are resampled with all their values, which keeps correlated
    values together; without, every value is its own cluster. The same inputs and seed give
    the same interval."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not len(array):
        raise ValueError("values must be a non-empty 1-d array")
    if samples < 1 or not 0 < level < 1:
        raise ValueError("samples must be positive and level in (0, 1)")
    if clusters is None:
        sums, counts = array, np.ones_like(array)
    else:
        labels = np.asarray(clusters)
        if labels.shape != array.shape:
            raise ValueError("clusters must hold one label per value")
        _, index = np.unique(labels, return_inverse=True)
        sums = np.bincount(index, weights=array)
        counts = np.bincount(index).astype(np.float64)
    rng = np.random.default_rng(seed)
    block = max(1, _BOOTSTRAP_BLOCK // len(sums))
    means = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, block):
        count = min(block, samples - start)
        draws = rng.integers(0, len(sums), (count, len(sums)))
        means[start : start + count] = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    tail = (1 - level) / 2
    low, high = np.quantile(means, [tail, 1 - tail])
    floor, ceiling = float(array.min()), float(array.max())
    return min(max(float(low), floor), ceiling), min(max(float(high), floor), ceiling)


def _mean(values: npt.NDArray[np.float64]) -> float:
    return min(max(float(values.mean()), float(values.min())), float(values.max()))


def ranking_metrics(
    frame: DataFrame, score_column: str, scorer: ScorerName, *, all_groups: int, seed: int
) -> RankingMetrics:
    """One scorer on the evaluated groups, every one of which holds a positive; ``all_groups``
    counts the test groups before those without a positive were dropped. The interval resamples
    source pages, the split unit, with all their groups."""
    ordered, labels, scores, sizes = _group_arrays(frame, score_column)
    if not len(sizes):
        raise ValueError("no groups to evaluate")
    if all_groups < len(sizes):
        raise ValueError("all_groups is fewer than the evaluated groups")
    positives = np.add.reduceat((labels > 0).astype(np.int64), _offsets(sizes))
    if (positives == 0).any():
        raise ValueError("a group without a positive; keep positive_groups only")
    per_group = _ndcg(labels, scores, sizes, NDCG_K)
    low, high = bootstrap_ci(per_group, clusters=_sources(ordered, sizes), seed=seed)
    rounds = ordered["round"].to_numpy()[_offsets(sizes)]
    histogram, _ = np.histogram(per_group, bins=NDCG_HISTOGRAM_BINS, range=(0.0, 1.0))
    return RankingMetrics(
        scorer=scorer,
        ndcg_at_10=_mean(per_group),
        ci_low=low,
        ci_high=high,
        precision_at_5=float(_precision(labels, scores, sizes, PRECISION_K).mean()),
        groups=len(sizes),
        groups_with_positive_share=len(sizes) / all_groups,
        n_labelled_pairs=int(positives.sum()),
        per_round={int(r): _mean(per_group[rounds == r]) for r in np.unique(rounds).tolist()},
        histogram=tuple(int(n) for n in histogram),
    )


def importance(trained: Trained) -> tuple[ImportanceEntry, ...]:
    """Total split gain of every model column, largest first, ties in column order."""
    gains = np.asarray(trained.booster.feature_importance(importance_type="gain"), np.float64)
    total = float(gains.sum())
    order = sorted(range(len(gains)), key=lambda i: (-gains[i], i))
    return tuple(
        ImportanceEntry(
            column=trained.columns[i],
            gain=float(gains[i]),
            gain_share=float(gains[i]) / total if total > 0 else 0.0,
        )
        for i in order
    )


def dominant_feature(entries: Iterable[ImportanceEntry]) -> str | None:
    """The column holding more than DOMINANT_SHARE of the gain, if any."""
    return next((entry.column for entry in entries if entry.gain_share > DOMINANT_SHARE), None)


def placement_gain_share(entries: Iterable[ImportanceEntry]) -> float:
    """The share of the gain held by the placement columns together."""
    return sum(entry.gain_share for entry in entries if entry.column in PLACEMENT_COLUMNS)


def placement_shares(frame: DataFrame) -> tuple[float | None, float | None]:
    """The share of the positive, then of the other pairs with a value in any placement column;
    None without such pairs."""
    placed = frame.loc[:, list(PLACEMENT_COLUMNS)].notna().any(axis=1).to_numpy(dtype=bool)
    positive = frame[LABEL_COLUMN].to_numpy() > 0
    return _share(placed, positive), _share(placed, ~positive)


def _share(placed: npt.NDArray[np.bool_], mask: npt.NDArray[np.bool_]) -> float | None:
    return float(placed[mask].mean()) if mask.any() else None


def _rival_words(rival_name: ScorerName, holder_version: str | None) -> str:
    if rival_name is ScorerName.HOLDER:
        return f"the production model (version {holder_version})"
    return "the baseline scorer"


def promotion(
    new: npt.ArrayLike,
    rival: npt.ArrayLike,
    *,
    sources: npt.ArrayLike,
    rival_name: ScorerName,
    holder_version: str | None,
    allowed: bool,
    seed: int,
) -> PromotionDecision:
    """The paired bootstrap of new - rival NDCG@10 per test group, resampling source pages
    (``sources``, one per group) with all their groups: the new model would promote only when
    the whole 95% interval lies above zero, and is promoted when that is also allowed here (the
    caller moves the alias)."""
    new_array = np.asarray(new, dtype=np.float64)
    rival_array = np.asarray(rival, dtype=np.float64)
    if new_array.shape != rival_array.shape or new_array.ndim != 1 or not len(new_array):
        raise ValueError("new and rival must be non-empty per-group arrays of one length")
    difference = new_array - rival_array
    delta = _mean(difference)
    low, high = bootstrap_ci(difference, clusters=sources, seed=seed)
    would = low > 0
    promoted = would and allowed
    against = _rival_words(rival_name, holder_version)
    measured = (
        f"NDCG@10 {delta:+.4f} against {against} over {len(difference)} test groups from "
        f"{len(np.unique(np.asarray(sources)))} source pages, 95% interval "
        f"[{low:+.4f}, {high:+.4f}] with source pages resampled"
    )
    if promoted:
        reason = f"{measured}, above zero: the production alias moves to this model."
    elif would:
        reason = f"{measured}, above zero, but promotion is not allowed here: the alias stays."
    else:
        reason = f"{measured}, not above zero: {against} stays."
    return PromotionDecision(
        rival=rival_name,
        holder_version=holder_version,
        delta=delta,
        delta_ci_low=low,
        delta_ci_high=high,
        would_promote=would,
        promoted=promoted,
        reason=reason,
    )


def _percent(share: float | None) -> str:
    return "n/a" if share is None else f"{share:.1%}"


LIMITATIONS: Final = (
    "Limitations: the labels are proxies, the site's own body links hidden and recovered, so the "
    "model learns what this site's existing links look like, not which suggestions an editor "
    "accepts. Hiding a link moves the link-count columns by itself, which flatters any scorer "
    "leaning on them; the learned_excl_link_counts row is the like-for-like comparison. A hidden "
    "link's anchor phrase stays in the source copy, so nearly every positive gets an anchor and "
    "the placement columns (context_relevance, anchor_target_fit), while a new pair gets them only "
    "when its source copy names the target; the placement shares by label show the gap and the "
    "learned_excl_placement row is the like-for-like comparison. With binary labels NDCG behaves "
    "close to mean average precision."
)


def summarise_ranker(report: RankerReport) -> str:
    """What ran, what it achieved and what it cannot show, for the MLflow run description; no
    page urls."""
    settings, params = report.settings, report.params
    lines = [
        f"Ranker training for tenant {report.tenant_id}: git {report.git_sha[:12]}, feature "
        f"code {report.feature_set_version[:12]}; {report.corpus_pages} crawled pages, "
        f"{report.body_links} body links between them; {report.seconds:.1f} s.",
        f"{settings.rounds} held-out rounds of {settings.share:.0%} of the body links each (seed "
        f"{settings.seed}), "
        + (
            "every body link hidden in exactly one round"
            if settings.rounds * settings.share >= 1 - _COVERAGE_TOLERANCE
            else f"{settings.rounds * settings.share:.0%} of the body links hidden once, the rest "
            "never"
        )
        + f"; source pages split {settings.test_share:.0%} test, then "
        f"{settings.valid_share:.0%} of the rest for early stopping (split seed "
        f"{settings.split_seed}), a page on one side in every round.",
    ]
    if report.rounds:
        lines.append(
            "Rounds: "
            + "; ".join(
                f"{r.round}: {r.hidden} hidden, {r.recoverable} recoverable, {r.positives} of "
                f"{r.pairs} candidate pairs positive in {r.groups_with_positive} source pages, "
                f"anchored {_percent(r.positive_placement_share)} of positives and "
                f"{_percent(r.negative_placement_share)} of the others"
                for r in report.rounds
            )
            + "."
        )
    excluded = ", ".join(f"{name} ({why})" for name, why in report.excluded_columns.items())
    lines.append(
        f"Groups (round, source page) with a positive: {report.train_groups} training, "
        f"{report.valid_groups} validation, {report.test_groups} test; {report.positives} "
        f"positive pairs. {len(report.columns)} model columns; excluded: {excluded or 'none'}."
    )
    if report.skipped_reason is not None:
        lines.append(f"Skipped: {report.skipped_reason}. No model trained, registered or promoted.")
        return "\n".join([*lines, LIMITATIONS])
    lines.append(
        f"LightGBM lambdarank, learning rate {params.learning_rate:g}, {params.num_leaves} "
        f"leaves, at least {params.min_data_in_leaf} rows a leaf, feature fraction "
        f"{params.feature_fraction:g}, seed {params.seed}; best iteration "
        f"{report.best_iteration} of at most {params.max_rounds}, stopped on validation "
        f"NDCG@{params.eval_at}."
    )
    for entry in report.metrics:
        lines.append(
            f"{entry.scorer.value}: NDCG@10 {entry.ndcg_at_10:.4f} "
            f"[{entry.ci_low:.4f}, {entry.ci_high:.4f}], P@5 {entry.precision_at_5:.4f} over "
            f"{entry.groups} test groups ({entry.groups_with_positive_share:.0%} of the test "
            f"groups hold a hidden link)."
        )
    if report.importance:
        lines.append(
            "Largest gain: "
            + ", ".join(f"{e.column} {e.gain_share:.1%}" for e in report.importance[:5])
            + "."
        )
    if report.importance:
        placement = placement_gain_share(report.importance)
        lines.append(
            f"The placement columns ({', '.join(PLACEMENT_COLUMNS)}) hold {placement:.1%} of the "
            "gain"
            + (
                f", more than {DOMINANT_SHARE:.0%}: the ranking rests mostly on whether a pair "
                "has an anchor, which the held-out protocol favours for positives."
                if placement > DOMINANT_SHARE
                else "."
            )
        )
    if report.dominant_feature is not None:
        lines.append(
            f"{report.dominant_feature} holds more than {DOMINANT_SHARE:.0%} of the gain: the "
            "model leans on one column; check it is not a leak of the held-out protocol."
        )
    if report.promotion is not None:
        lines.append(f"Promotion: {report.promotion.reason}")
    if report.model_version is not None:
        lines.append(f"Registered as version {report.model_version}.")
    return "\n".join([*lines, LIMITATIONS])
