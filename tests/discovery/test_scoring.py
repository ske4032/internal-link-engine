"""The hand-weighted baseline scorer: its default weights, normalisation, the weighted mean
over the features a pair has, the 0-100 scale, tiers and the top contributions."""

from __future__ import annotations

import json
from importlib import resources

import numpy as np
import pandas
import pytest

from linking_engine.discovery.features import FEATURE_COLUMNS, KEY_COLUMNS
from linking_engine.discovery.scoring import (
    SCORE_COLUMNS,
    default_weights,
    normalise,
    score_frame,
    score_report,
    summarise_scores,
    weights_hash,
)
from linking_engine.models import FeatureWeight, ScorerWeights

DEFAULT_TABLE = [
    ("content_cosine", 0.15, "higher", "percentile"),
    ("same_hub", 0.05, "higher", "identity"),
    ("content_agreement_same_topic_other_links", 0.05, "higher", "identity"),
    ("pair_query_overlap", 0.05, "higher", "identity"),
    ("pair_kw_overlap", 0.05, "higher", "identity"),
    ("target_is_orphan", 0.10, "higher", "identity"),
    ("target_saturation_ratio", 0.10, "lower", "percentile"),
    ("target_impressions_log", 0.05, "higher", "percentile"),
    ("target_ctr_gap", 0.05, "lower", "percentile"),
    ("target_keyword_gap", 0.05, "higher", "percentile"),
    ("source_link_equity_share", 0.10, "higher", "identity"),
    ("source_page_rank_percentile", 0.10, "higher", "identity"),
    ("target_hub_coverage", 0.05, "lower", "identity"),
    ("target_is_hub_pillar", 0.05, "higher", "identity"),
]
GSC = ("target_impressions_log", "target_ctr_gap", "target_keyword_gap")
TOPS = [f"top{i}_{part}" for i in (1, 2, 3) for part in ("feature", "contribution", "value")]


def weights(
    *features: tuple[str, float, str, str], shares: tuple[float, float] = (0.1, 0.3)
) -> ScorerWeights:
    return ScorerWeights(
        version="test-1",
        features=tuple(
            FeatureWeight(column=c, weight=w, direction=d, normalisation=n)  # type: ignore[arg-type]
            for c, w, d, n in features
        ),
        tier_shares=shares,
    )


def frame(columns: dict[str, list[float]], pairs: int | None = None) -> pandas.DataFrame:
    count = pairs if pairs is not None else len(next(iter(columns.values())))
    return pandas.DataFrame(
        {
            "source_url": [f"example.com/s{i:02d}" for i in range(count)],
            "target_url": [f"example.com/t{i:02d}" for i in range(count)],
            **{name: np.asarray(values, dtype=np.float64) for name, values in columns.items()},
        }
    )


# ── default weights ─────────────────────────────────────────────────────────


def test_the_default_weights_are_the_baseline_table() -> None:
    found = default_weights()

    assert found.version == "baseline-1"
    assert [(f.column, f.weight, f.direction, f.normalisation) for f in found.features] == (
        DEFAULT_TABLE
    )
    assert sum(f.weight for f in found.features) == pytest.approx(1.0)
    assert found.tier_shares == (0.10, 0.30)


def test_every_default_weight_names_a_matrix_column() -> None:
    columns = {f.column for f in default_weights().features}
    assert columns <= set(FEATURE_COLUMNS), sorted(columns - set(FEATURE_COLUMNS))


def test_the_packaged_json_is_the_default() -> None:
    files = resources.files("linking_engine.discovery")
    stored = json.loads((files / "scorer_weights.json").read_text(encoding="utf-8"))
    assert ScorerWeights.model_validate(stored) == default_weights()


def test_the_weights_hash_is_stable_and_moves_with_every_field() -> None:
    base = default_weights()
    key = weights_hash(base)

    assert len(key) == 64
    assert int(key, 16) >= 0
    assert weights_hash(ScorerWeights.model_validate_json(base.model_dump_json())) == key
    first = base.features[0]
    for changed in (
        base.model_copy(update={"version": "baseline-2"}),
        base.model_copy(update={"tier_shares": (0.2, 0.3)}),
        base.model_copy(
            update={"features": (first.model_copy(update={"weight": 0.2}), *base.features[1:])}
        ),
        base.model_copy(
            update={
                "features": (first.model_copy(update={"direction": "lower"}), *base.features[1:])
            }
        ),
    ):
        assert weights_hash(changed) != key


# ── normalise ───────────────────────────────────────────────────────────────


def test_identity_values_are_clipped_to_the_unit_interval() -> None:
    found = normalise(frame({"a": [-0.5, 0.3, 1.7]}), weights(("a", 1.0, "higher", "identity")))
    assert list(found["a"]) == pytest.approx([0.0, 0.3, 1.0])


def test_percentiles_rank_within_the_run_with_average_ties() -> None:
    found = normalise(
        frame({"a": [10.0, 20.0, 20.0, 40.0, 5.0]}), weights(("a", 1.0, "higher", "percentile"))
    )

    # Average ranks 2, 3.5, 3.5, 5, 1 over five values: (rank - 1) / 4.
    assert list(found["a"]) == [0.25, 0.625, 0.625, 1.0, 0.0]


def test_lower_is_better_becomes_one_minus_the_value() -> None:
    identity = normalise(frame({"a": [0.2, 0.9]}), weights(("a", 1.0, "lower", "identity")))
    percentile = normalise(
        frame({"a": [3.0, 1.0, 2.0]}), weights(("a", 1.0, "lower", "percentile"))
    )

    assert list(identity["a"]) == pytest.approx([0.8, 0.1])
    assert (percentile["a"].iloc[1], percentile["a"].iloc[0]) == (1.0, 0.0)


@pytest.mark.parametrize("method", ["identity", "percentile"])
def test_missing_values_stay_missing_and_are_not_ranked(method: str) -> None:
    found = normalise(
        frame({"a": [np.nan, 0.2, np.nan, 0.8]}), weights(("a", 1.0, "higher", method))
    )

    assert np.isnan(found["a"].iloc[0])
    assert np.isnan(found["a"].iloc[2])
    if method == "percentile":
        assert (found["a"].iloc[1], found["a"].iloc[3]) == (0.0, 1.0)


# ── score_frame ─────────────────────────────────────────────────────────────


def test_a_missing_feature_drops_out_and_the_rest_rescale() -> None:
    both = weights(("a", 0.75, "higher", "identity"), ("b", 0.25, "higher", "identity"))
    only_a = weights(("a", 0.75, "higher", "identity"))
    data = frame({"a": [0.8, 0.4], "b": [np.nan, 1.0]})

    with_b = score_frame(data, both)
    without_b = score_frame(data, only_a)

    # The first pair lacks b: its raw score is a alone, not a pulled down by a zero.
    assert with_b["raw_score"].iloc[0] == pytest.approx(0.8)
    assert with_b["raw_score"].iloc[0] == pytest.approx(without_b["raw_score"].iloc[0])
    assert with_b["raw_score"].iloc[1] == pytest.approx(0.75 * 0.4 + 0.25 * 1.0)


def default_frame(pairs: int, seed: int) -> pandas.DataFrame:
    rng = np.random.default_rng(seed)
    columns = {f.column: rng.random(pairs).tolist() for f in default_weights().features}
    for binary in (
        "same_hub",
        "content_agreement_same_topic_other_links",
        "target_is_orphan",
        "target_is_hub_pillar",
    ):
        columns[binary] = rng.integers(0, 2, pairs).astype(float).tolist()
    return frame(columns)


def test_a_pair_without_gsc_scores_as_if_the_gsc_weights_did_not_exist() -> None:
    data = default_frame(12, seed=3)
    data.loc[4, list(GSC)] = np.nan
    defaults = default_weights()
    without_gsc = defaults.model_copy(
        update={"features": tuple(f for f in defaults.features if f.column not in GSC)}
    )

    full = score_frame(data, defaults)
    reduced = score_frame(data, without_gsc)

    assert full["raw_score"].iloc[4] == pytest.approx(reduced["raw_score"].iloc[4])


def test_scores_span_zero_to_one_hundred() -> None:
    scored = score_frame(default_frame(30, seed=5), default_weights())

    assert (scored["score"].min(), scored["score"].max()) == (0.0, 100.0)
    ranks = scored["raw_score"].rank(method="first")
    assert list(scored["score"].rank(method="first")) == list(ranks)


def test_equal_raw_scores_all_score_fifty() -> None:
    scored = score_frame(frame({"a": [0.5, 0.5, 0.5]}), weights(("a", 1.0, "higher", "identity")))
    assert list(scored["score"]) == [50.0, 50.0, 50.0]


def test_tiers_follow_the_shares_of_the_run() -> None:
    data = frame({"a": [i / 20 for i in range(20)]})

    scored = score_frame(data, weights(("a", 1.0, "higher", "identity")))

    assert scored["tier"].value_counts().to_dict() == {1: 2, 2: 6, 3: 12}
    best = scored.sort_values("raw_score", ascending=False)
    assert list(best["tier"]) == [1] * 2 + [2] * 6 + [3] * 12


def test_ties_at_a_tier_boundary_break_by_source_then_target_url() -> None:
    sources = [
        "example.com/s00",
        "example.com/s00",
        *(f"example.com/s{i:02d}" for i in range(2, 10)),
    ]
    targets = [
        "example.com/t09",
        "example.com/t01",
        *(f"example.com/t{i:02d}" for i in range(2, 10)),
    ]
    data = pandas.DataFrame({"source_url": sources, "target_url": targets, "a": np.full(10, 0.5)})

    scored = score_frame(data, weights(("a", 1.0, "higher", "identity")))

    tier = {
        (s, t): int(n)
        for s, t, n in zip(scored["source_url"], scored["target_url"], scored["tier"], strict=True)
    }
    # One source url twice: its lower target url wins tier 1, the other comes next.
    assert [pair for pair, n in tier.items() if n == 1] == [("example.com/s00", "example.com/t01")]
    assert tier["example.com/s00", "example.com/t09"] == 2
    assert [pair[0] for pair, n in tier.items() if n == 2] == [
        "example.com/s00",
        "example.com/s02",
        "example.com/s03",
    ]


def test_the_same_input_in_any_row_order_scores_the_same() -> None:
    data = default_frame(25, seed=7)
    data.loc[[2, 9], "target_ctr_gap"] = np.nan
    data.loc[:, "content_cosine"] = data["content_cosine"].round(1)

    first = score_frame(data, default_weights())
    again = score_frame(data.sample(frac=1.0, random_state=1), default_weights())

    keys = list(KEY_COLUMNS)
    left = first.sort_values(keys).reset_index(drop=True)
    right = again.sort_values(keys).reset_index(drop=True)
    pandas.testing.assert_frame_equal(
        left[[*keys, "score", "tier", "raw_score", *TOPS]],
        right[[*keys, "score", "tier", "raw_score", *TOPS]],
        check_exact=True,
    )


def test_the_top_contributions_explain_the_score() -> None:
    data = default_frame(15, seed=11)
    data.loc[3, list(GSC)] = np.nan
    defaults = default_weights()
    weighted = {f.column for f in defaults.features}

    scored = score_frame(data, defaults)

    for _, row in scored.iterrows():
        names = [row[f"top{i}_feature"] for i in (1, 2, 3)]
        shares = [row[f"top{i}_contribution"] for i in (1, 2, 3)]
        assert set(names) <= weighted
        assert len(set(names)) == 3
        assert shares == sorted(shares, reverse=True)
        assert all(share >= 0 for share in shares)
        assert sum(shares) <= row["raw_score"] + 1e-9 <= 1 + 1e-9
    missing_pair = scored.set_index("source_url").loc["example.com/s03"]
    assert not {missing_pair[f"top{i}_feature"] for i in (1, 2, 3)} & set(GSC)


def test_a_contribution_is_the_features_weighted_share_and_its_value_is_the_raw_input() -> None:
    data = frame({"a": [0.9, 0.1], "b": [0.3, 0.2]})

    scored = score_frame(
        data, weights(("a", 3.0, "higher", "identity"), ("b", 1.0, "higher", "identity"))
    )

    first = scored.iloc[0]
    assert (first["top1_feature"], first["top2_feature"]) == ("a", "b")
    assert first["top1_contribution"] == pytest.approx(3 * 0.9 / 4)
    assert first["top2_contribution"] == pytest.approx(1 * 0.3 / 4)
    assert (first["top1_value"], first["top2_value"]) == (0.9, 0.3)


def test_a_weighted_column_missing_from_the_matrix_is_named() -> None:
    with pytest.raises(ValueError, match="target_hub_coverage"):
        score_frame(
            frame({"a": [0.5]}),
            weights(
                ("a", 1.0, "higher", "identity"), ("target_hub_coverage", 1.0, "lower", "identity")
            ),
        )


def test_a_single_value_sits_in_the_middle_of_its_percentile_range() -> None:
    found = normalise(
        frame({"a": [np.nan, 7.0, np.nan]}), weights(("a", 1.0, "higher", "percentile"))
    )
    assert found["a"].iloc[1] == 0.5


def test_a_pair_with_no_weighted_feature_scores_zero_in_tier_three() -> None:
    data = frame({"a": [0.2, np.nan, 0.6, 0.4]})

    scored = score_frame(data, weights(("a", 1.0, "higher", "identity"), shares=(0.3, 0.4)))

    blank = scored.iloc[1]
    assert np.isnan(blank["raw_score"])
    assert (blank["score"], blank["tier"]) == (0.0, 3)
    assert pandas.isna(blank["top1_feature"])
    # The others still span 0-100 among themselves.
    assert list(scored["score"].iloc[[0, 2, 3]]) == pytest.approx([0.0, 100.0, 50.0])


@pytest.mark.parametrize(
    ("pairs", "tiers"),
    [(7, {1: 1, 2: 2, 3: 4}), (5, {1: 1, 2: 1, 3: 3}), (1, {3: 1}), (10, {1: 1, 2: 3, 3: 6})],
)
def test_tier_sizes_round_half_up(pairs: int, tiers: dict[int, int]) -> None:
    data = frame({"a": [i / 10 for i in range(pairs)]})
    scored = score_frame(data, weights(("a", 1.0, "higher", "identity")))
    assert scored["tier"].value_counts().to_dict() == tiers


def test_fewer_features_than_top_slots_leave_the_rest_empty() -> None:
    scored = score_frame(frame({"a": [0.4, 0.9]}), weights(("a", 1.0, "higher", "identity")))

    assert list(scored["top1_feature"]) == ["a", "a"]
    assert scored["top2_feature"].isna().all()
    assert scored["top3_contribution"].isna().all()


def test_an_empty_matrix_scores_to_an_empty_frame_and_report() -> None:
    empty = frame({"a": []}, pairs=0)
    own = weights(("a", 1.0, "higher", "identity"))

    scored = score_frame(empty, own)
    report = score_report("acme", empty, scored, own, feature_cache_key="f" * 64, started=0.0)

    assert list(scored.columns) == [*KEY_COLUMNS, *SCORE_COLUMNS]
    assert (report.pairs, report.tiers, report.score_p50) == (0, {1: 0, 2: 0, 3: 0}, None)
    assert (report.missing_share, report.top_contributors) == ({}, {})
    assert "No pairs" in summarise_scores(report)


def test_the_report_counts_tiers_leaders_and_missing_shares() -> None:
    data = frame({"a": [0.9, 0.1, 0.5, 0.3], "b": [np.nan, 0.8, np.nan, 0.2]})
    own = weights(
        ("a", 1.0, "higher", "identity"), ("b", 1.0, "higher", "identity"), shares=(0.25, 0.25)
    )
    scored = score_frame(data, own)

    report = score_report("acme", data, scored, own, feature_cache_key="f" * 64, started=0.0)

    assert (report.pairs, report.tiers) == (4, {1: 1, 2: 1, 3: 2})
    assert (report.weights.version, report.weights_hash) == ("test-1", weights_hash(own))
    assert report.top_contributors == {"a": 3, "b": 1}
    assert report.missing_share == {"a": 0.0, "b": 0.5}
    assert (len(report.score_histogram), sum(report.score_histogram)) == (20, 4)
    assert report.score_histogram[0] >= 1, "the lowest pair scores 0, in the first bin"
    assert report.score_histogram[-1] >= 1, "the highest scores 100, in the last bin"
    assert report.weights == own
    assert report.score_p90 == pytest.approx(np.percentile(scored["score"], 90), abs=1e-3)
    summary = summarise_scores(report)
    for fact in ("acme", "test-1", "4 pairs", "Tiers 1, 2 and 3: 1, 1 and 2", "b 50.0%"):
        assert fact in summary, f"{fact!r} missing from:\n{summary}"


def test_scores_and_frame_must_describe_the_same_pairs() -> None:
    data = frame({"a": [0.1, 0.2]})
    own = weights(("a", 1.0, "higher", "identity"))
    with pytest.raises(ValueError, match="one row per pair"):
        score_report(
            "acme",
            data,
            score_frame(data.iloc[:1], own),
            own,
            feature_cache_key="f" * 64,
            started=0.0,
        )
