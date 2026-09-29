"""The hand-weighted baseline scorer: the bar every learned ranker has to beat.

Each weighted feature is normalised to [0, 1] with higher always better, and a pair's raw
score is the weighted mean over the features it has: a missing feature drops out and the
remaining weights rescale, so optional data is never a penalty. Scores and tiers are
relative to the run, and every tie breaks by url, so the same input gives the same output.
"""

from __future__ import annotations

import functools
import hashlib
import json
import math
import time
from datetime import UTC, datetime
from importlib import resources
from typing import TYPE_CHECKING, Final

import numpy as np
import pandas

from linking_engine.discovery.features import KEY_COLUMNS
from linking_engine.models import ScoreReport, ScorerWeights
from linking_engine.models.scoring import SCORE_HISTOGRAM_BINS

if TYPE_CHECKING:
    import numpy.typing as npt

STAGE: Final = "score-pairs"
TOP_CONTRIBUTIONS: Final = 3
TOP_COLUMNS: Final = tuple(
    f"top{rank}_{part}"
    for rank in range(1, TOP_CONTRIBUTIONS + 1)
    for part in ("feature", "contribution", "value")
)
SCORE_COLUMNS: Final = ("score", "tier", "raw_score", *TOP_COLUMNS)


@functools.cache
def default_weights() -> ScorerWeights:
    """The packaged baseline weights."""
    text = (
        resources.files("linking_engine.discovery")
        .joinpath("scorer_weights.json")
        .read_text(encoding="utf-8")
    )
    return ScorerWeights.model_validate_json(text)


def weights_hash(weights: ScorerWeights) -> str:
    """sha256 of the weights' canonical JSON."""
    canonical = json.dumps(weights.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def _require(frame: pandas.DataFrame, columns: tuple[str, ...]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"the frame has no column {', '.join(missing)}")


def _percentile(values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Rank among the non-null values, ties averaged, scaled to [0, 1]; a single distinct
    value sits at 0.5."""
    count = int(np.count_nonzero(~np.isnan(values)))
    ranks = pandas.Series(values).rank(method="average").to_numpy(dtype=np.float64)
    if count < 2:
        return np.where(np.isnan(values), np.nan, 0.5)
    scaled: npt.NDArray[np.float64] = (ranks - 1) / (count - 1)
    return scaled


def rank_tiers(
    values: npt.NDArray[np.float64],
    source_urls: pandas.Series[str],
    target_urls: pandas.Series[str],
    tier_shares: tuple[float, float],
) -> npt.NDArray[np.int8]:
    """Tier 1, 2 or 3 of every pair by the rank of its value, highest first and ties by url:
    ``tier_shares`` of the pairs to tier 1, then tier 2, the rest tier 3."""
    pairs = len(values)
    sources = pandas.factorize(source_urls, sort=True)[0]
    targets = pandas.factorize(target_urls, sort=True)[0]
    order = np.lexsort((targets, sources, -values))
    rank = np.empty(pairs, dtype=np.int64)
    rank[order] = np.arange(pairs)
    first = math.floor(tier_shares[0] * pairs + 0.5)
    second = math.floor(sum(tier_shares) * pairs + 0.5)
    return np.where(rank < first, 1, np.where(rank < second, 2, 3)).astype(np.int8)


def normalise(frame: pandas.DataFrame, weights: ScorerWeights) -> pandas.DataFrame:
    """Every weighted column in [0, 1], higher always better; NaN stays NaN."""
    columns = tuple(feature.column for feature in weights.features)
    _require(frame, columns)
    normalised: dict[str, npt.NDArray[np.float64]] = {}
    for feature in weights.features:
        values = frame[feature.column].to_numpy(dtype=np.float64)
        found = _percentile(values) if feature.normalisation == "percentile" else values.clip(0, 1)
        normalised[feature.column] = 1 - found if feature.direction == "lower" else found
    return pandas.DataFrame(normalised, index=frame.index, columns=list(columns))


def score_frame(frame: pandas.DataFrame, weights: ScorerWeights) -> pandas.DataFrame:
    """The key columns and SCORE_COLUMNS of every pair, in the frame's order.

    ``raw_score`` is the weighted mean over the features the pair has, ``score`` its min-max
    within the run on 0-100 (50 when all are equal). Tiers follow the rank of the raw score,
    ``tier_shares`` of the pairs to tier 1, then tier 2, the rest tier 3. A contribution is
    the feature's weighted share of the raw score, so a pair's contributions add up to it.
    A pair with none of the weighted features has no raw score: score 0, tier 3.
    """
    columns = tuple(feature.column for feature in weights.features)
    _require(frame, KEY_COLUMNS)
    values = normalise(frame, weights).to_numpy(dtype=np.float64)
    raw_values = frame.loc[:, list(columns)].to_numpy(dtype=np.float64)
    weight = np.array([feature.weight for feature in weights.features], dtype=np.float64)
    pairs = len(frame)

    present = ~np.isnan(values)
    total = present @ weight
    scored = total > 0
    contributions = np.where(
        present, values * weight / np.where(scored, total, 1.0)[:, None], np.nan
    )
    raw = np.where(scored, np.where(present, contributions, 0.0).sum(axis=1), np.nan)

    score = np.zeros(pairs, dtype=np.float64)
    if scored.any():
        low, high = raw[scored].min(), raw[scored].max()
        score[scored] = 50.0 if high == low else (raw[scored] - low) / (high - low) * 100

    # Best raw score first, pairs without one last, then by url.
    tier = rank_tiers(
        np.where(scored, raw, -np.inf),
        frame["source_url"],
        frame["target_url"],
        weights.tier_shares,
    )
    tier[~scored] = 3

    data: dict[str, object] = {
        "source_url": frame["source_url"].to_numpy(),
        "target_url": frame["target_url"].to_numpy(),
        "score": score,
        "tier": tier,
        "raw_score": raw,
    }
    # Largest contribution first; ties keep the weights' order.
    best = np.argsort(np.where(present, -contributions, np.inf), axis=1, kind="stable")
    names = np.array(columns, dtype=object)
    rows = np.arange(pairs)
    for position in range(TOP_CONTRIBUTIONS):
        if position < len(columns):
            chosen = best[:, position]
            valid = present[rows, chosen]
            feature = names[chosen]
            feature[~valid] = None
            contribution = np.where(valid, contributions[rows, chosen], np.nan)
            value = np.where(valid, raw_values[rows, chosen], np.nan)
        else:
            feature = np.full(pairs, None, dtype=object)
            contribution = value = np.full(pairs, np.nan)
        data[f"top{position + 1}_feature"] = feature
        data[f"top{position + 1}_contribution"] = contribution
        data[f"top{position + 1}_value"] = value
    return pandas.DataFrame(data, index=frame.index, columns=[*KEY_COLUMNS, *SCORE_COLUMNS])


def score_report(
    tenant_id: str,
    frame: pandas.DataFrame,
    scores: pandas.DataFrame,
    weights: ScorerWeights,
    *,
    feature_cache_key: str,
    started: float,
) -> ScoreReport:
    """``frame`` holds the weighted features and ``scores`` the output of `score_frame` for
    it; ``started`` is the run's ``time.perf_counter()`` start."""
    if len(frame) != len(scores):
        raise ValueError("scores must have one row per pair of the frame")
    pairs = len(scores)
    tiers = scores["tier"].to_numpy()
    values = scores["score"].to_numpy(dtype=np.float64)
    percentiles: list[float | None] = [None, None, None]
    if pairs:
        percentiles = [round(float(value), 3) for value in np.percentile(values, [10, 50, 90])]
    histogram, _ = np.histogram(values, bins=SCORE_HISTOGRAM_BINS, range=(0.0, 100.0))
    leaders = scores["top1_feature"].dropna().value_counts()
    return ScoreReport(
        tenant_id=tenant_id,
        pairs=pairs,
        weights=weights,
        weights_hash=weights_hash(weights),
        tiers={tier: int(np.count_nonzero(tiers == tier)) for tier in (1, 2, 3)},
        score_p10=percentiles[0],
        score_p50=percentiles[1],
        score_p90=percentiles[2],
        score_histogram=tuple(int(count) for count in histogram),
        top_contributors={str(name): int(count) for name, count in leaders.items()},
        missing_share=(
            {f.column: float(frame[f.column].isna().mean()) for f in weights.features}
            if pairs
            else {}
        ),
        feature_cache_key=feature_cache_key,
        seconds=round(time.perf_counter() - started, 3),
        finished_at=datetime.now(UTC),
    )


def summarise_scores(report: ScoreReport) -> str:
    """A short prose record of one scoring run, for the MLflow run description."""
    scope = (
        f"Baseline scoring for tenant {report.tenant_id} with weights {report.weights.version} "
        f"({report.weights_hash[:12]}) over the feature matrix {report.feature_cache_key[:12]}: "
        f"{report.pairs} pairs."
    )
    timing = f"{report.seconds:.1f} s."
    if not report.pairs:
        return "\n".join([scope, "No pairs, so no scores.", timing])
    lines = [
        scope,
        f"Tiers 1, 2 and 3: {report.tiers.get(1, 0)}, {report.tiers.get(2, 0)} and "
        f"{report.tiers.get(3, 0)} pairs. Score p10 {report.score_p10:.1f}, "
        f"p50 {report.score_p50:.1f}, p90 {report.score_p90:.1f}.",
    ]
    leaders = sorted(report.top_contributors.items(), key=lambda item: (-item[1], item[0]))
    if leaders:
        lines.append(
            "Largest contributor: "
            + ", ".join(f"{name} {count}" for name, count in leaders[:5])
            + "."
        )
    missing = sorted(
        ((share, name) for name, share in report.missing_share.items() if share > 0),
        reverse=True,
    )
    lines.append(
        "Missing and left out of the score: "
        + (", ".join(f"{name} {share:.1%}" for share, name in missing) or "none")
        + "."
    )
    return "\n".join([*lines, timing])
