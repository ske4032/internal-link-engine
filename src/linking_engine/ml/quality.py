"""The maths of the quality evaluation: held-out links, AUC, recall, keyword matching, the
flat metrics of a report and soft alerts against the previous run. No I/O."""

from __future__ import annotations

import hashlib
import math
import re
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, NamedTuple

import numpy as np

from linking_engine.anchor.keywords import MIN_TOKEN_LENGTH, tokens
from linking_engine.discovery.features import FEATURE_COLUMNS
from linking_engine.gsc import normalise_term
from linking_engine.models import (
    FeatureAuc,
    KeywordRung,
    KeywordSource,
    QualityAlert,
    RelevanceGroup,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    import numpy.typing as npt

    from linking_engine.models import QualityReport, ScoreDistribution

QUALITY_STAGE: Final = "quality-eval"
HIDE_SHARE: Final = 0.10
HIDE_SEED: Final = 42
RECALL_KS: Final = (10, 20, 50)
SIGNAL_MARGIN: Final = 0.05
# An anchor matches a keyword when either's words hold the other's, or they overlap this much.
ANCHOR_JACCARD: Final = 0.5
# Keyword relevance by origin: rank 1 by the rung that resolved it, the rest by edge source.
KEYWORD_ORIGINS: Final = (
    *(f"primary_{rung.value.lower()}" for rung in KeywordRung),
    *(f"secondary_{source.value.lower()}" for source in KeywordSource),
)
# Keyword relevance by length: (label, fewest words, most words or None).
LENGTH_BINS: Final = (("1_2", 1, 2), ("3_4", 3, 4), ("5_7", 5, 7), ("8plus", 8, None))
ALERT_BAND: Final = 0.2
# A move of exactly the band is not beyond it, whatever the float error: 0.80 - 0.75 > 0.05.
_BAND_TOLERANCE: Final = 1e-9
_WORD: Final = re.compile(r"\w+")
_HASH_RANGE: Final = 1 << 256
# Ten folds of 0.1 fit the hash range, whatever the float error of 10 * 0.1.
_FOLD_TOLERANCE: Final = 1e-9
# Hiding a link moves these by itself (the held-out view recomputes them), inflating their AUC;
# crawl depth is a stored BFS over links that still include the hidden ones, so it leaks them.
_LINK_DERIVED_FIELDS: Final = (
    "target_inbound_count",
    "target_is_orphan",
    "target_crawl_depth",
    "target_page_rank_percentile",
    "target_saturation_ratio",
    "source_outbound_count",
    "source_outbound_density",
    "source_link_equity_share",
    "source_page_rank_percentile",
    "source_is_hub_pillar",
    "target_is_hub_pillar",
    "target_hub_coverage",
    "same_link_community",
    "cluster_agreement",
    "content_agreement",
)
LINK_DERIVED_COLUMNS: Final = tuple(
    column
    for column in FEATURE_COLUMNS
    if any(column == field or column.startswith(f"{field}_") for field in _LINK_DERIVED_FIELDS)
)


class Band(NamedTuple):
    """How far a metric may move from the previous run before it alerts."""

    width: float
    # Relative to the previous value; otherwise the absolute difference.
    relative: bool = True


# AUCs and cosines sit in a narrow range, where a relative band is far too loose.
ALERT_BANDS: Final[Mapping[str, Band]] = MappingProxyType(
    {
        "score_auc": Band(0.05, relative=False),
        "best_feature_auc": Band(0.05, relative=False),
        "score_auc_lift": Band(0.05, relative=False),
        "score_auc_excl_link_counts": Band(0.05, relative=False),
        "best_feature_auc_excl_link_counts": Band(0.05, relative=False),
        "score_auc_lift_excl_link_counts": Band(0.05, relative=False),
        "keyword_relevance_rank_1_mean": Band(0.05, relative=False),
        "context_relevance_mean": Band(0.05, relative=False),
        "anchor_target_fit_mean": Band(0.05, relative=False),
        # Any check that starts or stops applying.
        "not_applicable_checks": Band(0.5, relative=False),
    }
)
HEADLINE_METRICS: Final = (
    *(f"recall_at_{k}" for k in RECALL_KS),
    "features_with_signal",
    "score_auc",
    "best_feature_auc",
    "score_auc_lift",
    "score_auc_excl_link_counts",
    "best_feature_auc_excl_link_counts",
    "score_auc_lift_excl_link_counts",
    "keyword_unique_share",
    "extract_found_primary",
    "extract_found_set",
    "extract_words_primary",
    "extract_words_set",
    "anchor_match_primary",
    "anchor_match_set",
    "keyword_relevance_rank_1_mean",
    "context_relevance_mean",
    "anchor_target_fit_mean",
    "gsc_pair_share",
    "keyword_page_share",
    "all_null_columns",
    "constant_columns",
    "not_applicable_checks",
)


def _pair_hash(seed: int, source: str, target: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}\t{source}\t{target}".encode()).digest())


def hide_links(
    pairs: Iterable[tuple[str, str]],
    *,
    share: float = HIDE_SHARE,
    seed: int = HIDE_SEED,
    fold: int = 0,
) -> frozenset[tuple[str, str]]:
    """The pairs whose sha256 of (seed, source, target) falls in ``[fold * share,
    (fold + 1) * share)`` of the hash range, so a pair keeps its hidden or visible state from
    run to run whatever else changes and folds never overlap. Fold 0 takes the lowest hash when
    no pair falls in it and there are pairs; no other fold then holds that pair."""
    if not 0 < share < 1:
        raise ValueError("share must be in (0, 1)")
    if fold < 0 or (fold + 1) * share > 1 + _FOLD_TOLERANCE:
        raise ValueError(f"fold must be in [0, {int((1 + _FOLD_TOLERANCE) / share) - 1}]")
    hashes = {pair: _pair_hash(seed, *pair) for pair in set(pairs)}
    if not hashes:
        return frozenset()
    low, high = int(fold * share * _HASH_RANGE), int((fold + 1) * share * _HASH_RANGE)
    hidden = frozenset(pair for pair, value in hashes.items() if low <= value < high)
    lowest = min(hashes, key=hashes.__getitem__)
    if fold == 0:
        return hidden or frozenset({lowest})
    if hashes[lowest] >= int(share * _HASH_RANGE):
        return hidden - {lowest}
    return hidden


def auc(labels: npt.ArrayLike, scores: npt.ArrayLike) -> float | None:
    """The chance that a positive scores above a negative, ties counting half; None without
    both classes. Scores may be infinite, never NaN."""
    positive = np.asarray(labels, dtype=bool)
    values = np.asarray(scores, dtype=np.float64)
    if positive.ndim != 1 or positive.shape != values.shape:
        raise ValueError("labels and scores must be vectors of one length")
    if np.isnan(values).any():
        raise ValueError("scores must not be NaN")
    positives = int(positive.sum())
    negatives = len(positive) - positives
    if not positives or not negatives:
        return None
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    # Tied values share the mean of the 1-based ranks they span.
    ranks = (np.cumsum(counts) - (counts - 1) / 2)[inverse.reshape(-1)]
    found = (ranks[positive].sum() - positives * (positives + 1) / 2) / (positives * negatives)
    return float(found)


def feature_auc(column: str, labels: npt.ArrayLike, values: npt.ArrayLike) -> FeatureAuc:
    """One feature column's AUC over the pairs with a value, its coverage, and its AUC as a
    ranker of every pair in its best order: either direction, missing values first or last."""
    positive = np.asarray(labels, dtype=bool)
    found = np.asarray(values, dtype=np.float64)
    present = ~np.isnan(found)
    kept = found[present]
    separable = len(np.unique(kept)) > 1
    # Reversing an order turns its AUC a into 1 - a, so these two cover all four orders.
    up = auc(positive, np.where(present, found, -np.inf))
    down = auc(positive, np.where(present, -found, -np.inf))
    spread = 0.0 if up is None or down is None else max(abs(up - 0.5), abs(down - 0.5))
    return FeatureAuc(
        column=column,
        auc=auc(positive[present], kept) if separable else None,
        coverage=float(present.mean()) if len(found) else 0.0,
        ranker_auc=0.5 + spread,
    )


def has_signal(entry: FeatureAuc, margin: float = SIGNAL_MARGIN) -> bool:
    return entry.has_signal(margin)


def recall_at(ranks: Sequence[int | None], k: int) -> float:
    """Share of held-out links whose source is among the first ``k`` candidates of its
    target; ``ranks`` are 1-based, None for a source not retrieved."""
    if k < 1:
        raise ValueError("k must be at least 1")
    if not ranks:
        raise ValueError("no held-out links")
    return sum(1 for rank in ranks if rank is not None and rank <= k) / len(ranks)


def random_recall(eligible: Sequence[int], k: int) -> float:
    """Expected recall@k of a random order: per held-out link, ``k`` over the eligible
    sources of its target, at most 1."""
    if k < 1:
        raise ValueError("k must be at least 1")
    counts = np.asarray(eligible, dtype=np.float64)
    if not len(counts) or (counts < 1).any():
        raise ValueError("every held-out link has at least its own source eligible")
    return float(np.minimum(k / counts, 1.0).mean())


def keyword_words(text: str) -> frozenset[str]:
    """The words of ``text`` that can carry meaning, normalised."""
    return frozenset(word for word in tokens(text) if len(word) >= MIN_TOKEN_LENGTH)


def anchor_matches(anchor: frozenset[str], keyword: frozenset[str]) -> bool:
    """Whether an anchor names the keyword, both as their ``keyword_words``: one's words hold
    the other's, or they share at least ANCHOR_JACCARD of their words."""
    if not anchor or not keyword:
        return False
    return (
        keyword <= anchor
        or anchor <= keyword
        or len(anchor & keyword) / len(anchor | keyword) >= ANCHOR_JACCARD
    )


def copy_words(text: str) -> frozenset[str]:
    """Every normalised word of a page's copy, to check a keyword's ``keyword_words`` against."""
    return frozenset(_WORD.findall(normalise_term(text)))


def keyword_origin(rank: int, rung: KeywordRung | None, source: KeywordSource) -> str:
    """One of KEYWORD_ORIGINS: a resolved keyword by its rung, any other by its source."""
    if rank == 1 and rung is not None:
        return f"primary_{rung.value.lower()}"
    return f"secondary_{source.value.lower()}"


def length_bin(text: str) -> str:
    """The LENGTH_BINS label of a keyword's normalised word count."""
    words = max(1, len(normalise_term(text).split()))
    return next(
        label
        for label, low, high in LENGTH_BINS
        if low <= words and (high is None or words <= high)
    )


def relevance_groups(
    cosines: Mapping[str, Sequence[float]], order: Sequence[str]
) -> tuple[RelevanceGroup, ...]:
    """Count, mean and median of each non-empty group, in ``order``."""
    unknown = sorted(cosines.keys() - set(order))
    if unknown:
        raise ValueError(f"unknown relevance groups {unknown}")
    return tuple(
        RelevanceGroup(
            group=group,
            keywords=len(cosines[group]),
            mean=min(1.0, max(-1.0, float(np.mean(cosines[group])))),
            p50=float(np.median(cosines[group])),
        )
        for group in order
        if cosines.get(group)
    )


def _distribution(name: str, found: ScoreDistribution | None) -> dict[str, float]:
    if found is None:
        return {}
    return {
        f"{name}_{field}": float(value)
        for field, value in found.model_dump(exclude={"histogram"}).items()
        if value is not None
    }


def quality_metrics(report: QualityReport) -> dict[str, float]:
    """Every number of the report under a stable name, flat; checks that do not apply are
    left out, never logged as zero."""
    metrics: dict[str, float] = {
        "seconds": report.seconds,
        "not_applicable_checks": float(len(report.not_applicable)),
        "alerts": float(len(report.alerts)),
    }
    if (retrieval := report.retrieval) is not None:
        metrics.update(
            {
                "body_link_pairs": float(retrieval.body_link_pairs),
                "hidden_links": float(retrieval.hidden),
                "hidden_recoverable": float(retrieval.recoverable),
                "held_out_candidates": float(retrieval.candidates),
            }
        )
        for at_k in retrieval.recall:
            metrics[f"recall_at_{at_k.k}"] = at_k.recall
            metrics[f"random_recall_at_{at_k.k}"] = at_k.random
    if (signal := report.feature_signal) is not None:
        metrics["auc_pairs"] = float(signal.pairs)
        metrics["auc_positives"] = float(signal.positives)
        metrics["features_with_signal"] = float(signal.features_with_signal)
        metrics.update(
            {f"auc_{entry.column}": entry.auc for entry in signal.columns if entry.auc is not None}
        )
    if (scorer := report.scorer) is not None:
        metrics["score_auc"] = scorer.score_auc
        metrics["best_feature_auc"] = scorer.best_feature_auc
        metrics["score_auc_lift"] = scorer.score_auc_lift
        metrics["link_derived_weight_share"] = scorer.link_derived_weight_share
        if scorer.score_auc_excl_link_counts is not None:
            metrics["score_auc_excl_link_counts"] = scorer.score_auc_excl_link_counts
        metrics["best_feature_auc_excl_link_counts"] = scorer.best_feature_auc_excl_link_counts
        if scorer.score_auc_lift_excl_link_counts is not None:
            metrics["score_auc_lift_excl_link_counts"] = scorer.score_auc_lift_excl_link_counts
    if (keywords := report.keywords) is not None:
        metrics["keyword_pages"] = float(keywords.pages)
        metrics["keyword_resolved"] = float(keywords.resolved)
        metrics.update(
            {f"keyword_rung_{rung.value.lower()}": float(n) for rung, n in keywords.by_rung.items()}
        )
        metrics.update(
            {
                f"keyword_rejected_{reason}": float(n)
                for reason, n in keywords.fallbacks_rejected.items()
            }
        )
        if keywords.unique_share is not None:
            metrics["keyword_unique_share"] = keywords.unique_share
        if (extract := keywords.extractability) is not None:
            metrics.update(
                {
                    "extract_pairs": float(extract.pairs),
                    "extract_found_primary": extract.found_primary,
                    "extract_found_set": extract.found_set,
                    "extract_exact_set": extract.exact_set,
                    "extract_stemmed_set": extract.stemmed_set,
                    "extract_stem_set_set": extract.stem_set_set,
                    "extract_words_primary": extract.words_primary,
                    "extract_words_set": extract.words_set,
                    "extract_stem_set_threshold": extract.stem_set_threshold,
                }
            )
        if (anchors := keywords.anchors) is not None:
            metrics.update(
                {
                    "anchor_match_anchors": float(anchors.anchors),
                    "anchor_match_primary": anchors.primary,
                    "anchor_match_set": anchors.any_rank,
                }
            )
        if (relevance := keywords.relevance) is not None:
            metrics.update(
                {
                    "keyword_relevance_texts": float(relevance.texts),
                    "keyword_vectors_embedded": float(relevance.embedded),
                    "keyword_vectors_cached": float(relevance.cached),
                    "keyword_embed_tokens": float(relevance.api_tokens),
                }
            )
            for rank in relevance.ranks:
                metrics[f"keyword_relevance_rank_{rank.rank}_mean"] = rank.mean
                metrics[f"keyword_relevance_rank_{rank.rank}_keywords"] = float(rank.keywords)
            metrics.update(
                {
                    f"keyword_relevance_{group.group}_mean": group.mean
                    for group in relevance.by_origin
                }
            )
            metrics.update(
                {
                    f"keyword_relevance_len_{group.group}_mean": group.mean
                    for group in relevance.by_length
                }
            )
    if (links := report.link_relevance) is not None:
        metrics["link_relevance_links"] = float(links.links)
        metrics.update(_distribution("context_relevance", links.context))
        metrics.update(_distribution("anchor_target_fit", links.anchor))
    coverage = report.coverage
    metrics["coverage_pairs"] = float(coverage.pairs)
    if coverage.gsc_pair_share is not None:
        metrics["gsc_pair_share"] = coverage.gsc_pair_share
    if coverage.keyword_page_share is not None:
        metrics["keyword_page_share"] = coverage.keyword_page_share
    metrics["all_null_columns"] = float(len(coverage.all_null_columns))
    metrics["constant_columns"] = float(len(coverage.constant_columns))
    return metrics


def alerts(
    current: Mapping[str, float],
    previous: Mapping[str, float],
    *,
    metrics: Iterable[str] = HEADLINE_METRICS,
    band: float = ALERT_BAND,
    bands: Mapping[str, Band] = ALERT_BANDS,
) -> tuple[QualityAlert, ...]:
    """The headline metrics that moved beyond their band: ``bands`` per metric, else
    ``band`` relative to the previous value. A metric missing from either run is skipped; one
    that leaves 0 under a relative band always alerts."""
    if band <= 0 or any(rule.width <= 0 for rule in bands.values()):
        raise ValueError("bands must be positive")
    found: list[QualityAlert] = []
    for name in metrics:
        now, before = current.get(name), previous.get(name)
        if now is None or before is None or not (math.isfinite(now) and math.isfinite(before)):
            continue
        rule = bands.get(name, Band(band))
        change: float | None
        if not rule.relative:
            change = now - before
            moved = abs(change) > rule.width + _BAND_TOLERANCE
        elif before == 0:
            change, moved = None, now != 0
        else:
            change = (now - before) / abs(before)
            moved = abs(change) > rule.width + _BAND_TOLERANCE
        if moved:
            found.append(
                QualityAlert(
                    metric=name,
                    previous=before,
                    current=now,
                    change=change,
                    band=rule.width,
                    relative=rule.relative,
                )
            )
    return tuple(found)
