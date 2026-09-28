"""#94 on real Neo4j and Mongo with a throwaway sqlite MLflow store: orphan targets never reach
training or evaluation, every evaluation seed gets a learned-against-plain result, the product
measures need the tenant's anchor choices, and promotion keeps the fixed split."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from mlflow import MlflowClient
from ranking_seed import (
    KEYWORDS,
    URLS,
    choose_anchors,
    noise_url,
    orphan_targets,
    seed_ranking,
    voyage,
)
from structlog.testing import capture_logs
from test_ranker_stage import ENOUGH, local_mlflow, run_text
from voyage_fakes import client

from linking_engine.ml.ranking import (
    MIN_TEST_GROUPS,
    MODEL_COLUMNS,
    positive_groups,
    ranking_metrics,
    seed_result,
    seed_split,
    summarise_ranker,
    train,
)
from linking_engine.models import HeldOutSettings, Page, RankerParams, ScorerName
from linking_engine.pipeline import ranker
from linking_engine.pipeline.ranker import train_ranker
from linking_engine.pipeline.ranking_data import held_out_rounds, load_rounds

if TYPE_CHECKING:
    from pathlib import Path

    import pandas

    from linking_engine.graph.repo import GraphRepo
    from linking_engine.ingest.mongo_repo import MongoRepo
    from linking_engine.ml.ranking import Trained
    from linking_engine.models import FeatureReport, RankerReport, RankingMetrics, SeedResult

__all__ = ["local_mlflow"]

PAGES = len(URLS)
ORPHANS = orphan_targets()


async def trained(
    graph: GraphRepo,
    mongo: MongoRepo,
    tenant: str,
    cache: Path,
    settings: HeldOutSettings = ENOUGH,
) -> RankerReport:
    return await train_ranker(
        graph, mongo, tenant, cache_dir=cache, voyage=client(voyage()), settings=settings
    )


def rows_of(cache: Path, tenant: str) -> pandas.DataFrame:
    return load_rounds(sorted((cache / tenant / "ranker" / "rounds").iterdir()))


@pytest.mark.integration
async def test_orphan_target_rows_never_reach_training_or_evaluation(
    graph: GraphRepo,
    mongo: MongoRepo,
    tenant: str,
    tmp_path: Path,
    local_mlflow: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await seed_ranking(graph, mongo, tenant)
    fits: list[tuple[set[str], tuple[str, ...]]] = []
    models: list[Trained] = []
    judged: list[set[str]] = []

    def training(
        train_frame: pandas.DataFrame,
        valid_frame: pandas.DataFrame,
        columns: tuple[str, ...],
        params: RankerParams,
    ) -> Trained:
        seen = set(train_frame["target_url"]) | set(valid_frame["target_url"])
        fits.append((seen, params.monotone_increasing))
        models.append(train(train_frame, valid_frame, columns, params))
        return models[-1]

    def metrics(frame: pandas.DataFrame, *args: Any, **kwargs: Any) -> RankingMetrics:
        judged.append(set(frame["target_url"]))
        return ranking_metrics(frame, *args, **kwargs)

    def seeded(frame: pandas.DataFrame, **kwargs: Any) -> SeedResult:
        judged.append(set(frame["target_url"]))
        return seed_result(frame, **kwargs)

    monkeypatch.setattr(ranker, "train", training)
    monkeypatch.setattr(ranker, "ranking_metrics", metrics)
    monkeypatch.setattr(ranker, "seed_result", seeded)

    report = await trained(graph, mongo, tenant, tmp_path)

    assert report.skipped_reason is None, report.skipped_reason
    rows = rows_of(tmp_path, tenant)
    assert report.unlabelable_targets == len(ORPHANS) == 6
    assert report.unlabelable_rows == int(rows["target_url"].isin(ORPHANS).sum()) > 0
    # The plain model of every seed keeps the orphan rows and has no constraints; every other
    # model sees no orphan target.
    plain = [seen for seen, monotone in fits if monotone == ()]
    learned = [seen for seen, monotone in fits if monotone != ()]
    assert len(plain) == len(ENOUGH.evaluation_seeds)
    assert len(fits) == 2 + 2 * len(ENOUGH.evaluation_seeds)
    assert all(seen & ORPHANS for seen in plain), "the plain model lost the orphan rows"
    assert not [seen & ORPHANS for seen in learned if seen & ORPHANS], "a model trained on orphans"
    # Every scorer, on the fixed split and on every seed, is judged without orphan targets.
    assert len(judged) == len(report.metrics) + len(ENOUGH.evaluation_seeds)
    assert not [seen & ORPHANS for seen in judged if seen & ORPHANS], "judged on an orphan target"
    # Leaving orphan rows out drops no group: an orphan target is never a positive.
    kept = positive_groups(rows)
    test, _ = seed_split(set(rows["source_url"]), ENOUGH, ENOUGH.split_seed)
    in_test = kept[kept["source_url"].isin(test)]
    assert report.test_groups == len(in_test.drop_duplicates(["round", "source_url"]))

    # Disabling both restores the model of #24-#27: no constraints, the orphan rows kept.
    _, valid = seed_split(set(rows["source_url"]), ENOUGH, ENOUGH.split_seed)
    before = train(
        kept[~kept["source_url"].isin(test | valid)],
        kept[kept["source_url"].isin(valid)],
        MODEL_COLUMNS,
        RankerParams(monotone_increasing=()),
    )
    assert models[1].booster.model_to_string() == before.booster.model_to_string()

    # On the planted pairs, raising a constrained column never lowers the learned score.
    sample = in_test.iloc[:: max(1, len(in_test) // 60)].loc[:, list(MODEL_COLUMNS)]
    grid = np.linspace(0.0, 1.0, 21)
    for column in RankerParams().monotone_increasing:
        swept = sample.loc[sample.index.repeat(len(grid))].assign(
            **{column: np.tile(grid, len(sample))}
        )
        scores = models[0].booster.predict(swept.to_numpy(dtype=np.float64))
        steps = np.diff(scores.reshape(len(sample), len(grid)), axis=1)
        assert steps.min() >= -1e-12, f"raising {column} lowered a planted pair's score"


@pytest.mark.integration
async def test_seed_results_cover_every_seed_with_paired_intervals(
    graph: GraphRepo, mongo: MongoRepo, tenant: str, tmp_path: Path, local_mlflow: str
) -> None:
    await seed_ranking(graph, mongo, tenant)

    report = await trained(graph, mongo, tenant, tmp_path)

    results = {result.seed: result for result in report.seed_results}
    assert tuple(results) == ENOUGH.evaluation_seeds, "a seed is missing or out of order"
    kept = positive_groups(rows_of(tmp_path, tenant))
    sources = set(kept["source_url"])
    for seed, result in results.items():
        test, _ = seed_split(sources, ENOUGH, seed)
        assert result.test_pages == len(test & sources), f"seed {seed}: test pages"
        # Paired over the same groups, the mean difference is the difference of the means.
        assert result.delta == pytest.approx(result.learned[0] - result.plain[0]), seed
        assert result.delta_ci_low <= result.delta <= result.delta_ci_high, seed
        for name, found in (
            ("learned", result.learned),
            ("plain", result.plain),
            ("baseline", result.baseline),
        ):
            assert found[1] <= found[2], f"seed {seed}: {name} interval"
    # At the split seed the learned model and its test groups are the fixed split's.
    fixed = {entry.scorer: entry for entry in report.metrics}
    at_split = results[ENOUGH.split_seed]
    assert at_split.learned[0] == pytest.approx(fixed[ScorerName.LEARNED].ndcg_at_10)
    assert at_split.baseline[0] == pytest.approx(fixed[ScorerName.BASELINE].ndcg_at_10)
    assert report.seeds_worse + report.seeds_better <= len(results)
    assert report.seeds_worse == sum(r.delta_ci_high < 0 for r in results.values())
    assert report.seeds_better == sum(r.delta_ci_low > 0 for r in results.values())
    test_pages = [result.test_pages for result in results.values()]
    assert len(set(test_pages)) > 1, "every seed drew the same test pages"


@pytest.mark.integration
async def test_product_measures_skipped_without_anchor_choices(
    graph: GraphRepo, mongo: MongoRepo, tenant: str, tmp_path: Path, local_mlflow: str
) -> None:
    await seed_ranking(graph, mongo, tenant)

    without = await trained(graph, mongo, tenant, tmp_path)

    assert without.product_measures == ()
    assert without.product_skipped_reason == ranker.NO_ANCHOR_CHOICES
    assert without.seed_results, "the seeds need no anchor choices"

    path = await choose_anchors(graph, mongo, tenant, tmp_path)
    written = path.read_bytes()
    path.write_bytes(b"not parquet")
    unreadable = await trained(graph, mongo, tenant, tmp_path)

    assert unreadable.product_measures == ()
    assert unreadable.product_skipped_reason == ranker.UNREADABLE_ANCHOR_CHOICES

    path.write_bytes(written)
    measured = await trained(graph, mongo, tenant, tmp_path)

    assert measured.product_skipped_reason is None
    by_scorer = {entry.scorer: entry for entry in measured.product_measures}
    assert list(by_scorer) == [ScorerName.LEARNED, ScorerName.PLAIN, ScorerName.BASELINE]
    for scorer, entry in by_scorer.items():
        # Every page is a candidate target, the six orphans among them.
        assert entry.orphan_page_share == pytest.approx(len(ORPHANS) / PAGES), scorer
        assert entry.orphans_reached is not None, scorer
        assert entry.k == 10
        assert entry.top_relevance is not None, scorer
        assert 0 < entry.same_hub_share <= 1, scorer


@pytest.mark.integration
async def test_promotion_still_uses_the_fixed_split(
    graph: GraphRepo, mongo: MongoRepo, tenant: str, tmp_path: Path, local_mlflow: str
) -> None:
    """More or fewer evaluation seeds change nothing the gate decides on; another split seed
    changes its test groups."""
    await seed_ranking(graph, mongo, tenant)
    only_split = ENOUGH.model_copy(update={"evaluation_seeds": (ENOUGH.split_seed,)})

    every = await trained(graph, mongo, tenant, tmp_path)
    one = await trained(graph, mongo, tenant, tmp_path, only_split)
    other = await trained(
        graph,
        mongo,
        tenant,
        tmp_path,
        HeldOutSettings(rounds=5, share=0.2, test_share=0.25, split_seed=11),
    )

    assert every.promotion is not None
    assert one.promotion is not None
    assert other.promotion is not None
    assert [r.seed for r in one.seed_results] == [ENOUGH.split_seed]
    assert one.promotion == every.promotion, "the evaluation seeds moved the promotion gate"
    assert one.metrics == every.metrics
    assert one.test_groups == every.test_groups
    assert other.test_groups != every.test_groups
    assert other.promotion.delta != every.promotion.delta
    assert f"over {every.test_groups} test groups" in every.promotion.reason
    kept = positive_groups(rows_of(tmp_path, tenant))
    test, _ = seed_split(set(kept["source_url"]), ENOUGH, ENOUGH.split_seed)
    assert f"from {len(test & set(kept['source_url']))} source pages" in every.promotion.reason


@pytest.mark.integration
async def test_train_ranker_end_to_end_logs_seed_and_product_tables(
    graph: GraphRepo, mongo: MongoRepo, tenant: str, tmp_path: Path, local_mlflow: str
) -> None:
    await seed_ranking(graph, mongo, tenant)
    await choose_anchors(graph, mongo, tenant, tmp_path / "cache")

    report = await trained(graph, mongo, tenant, tmp_path / "cache")

    client_ = MlflowClient(local_mlflow)
    experiment = client_.get_experiment_by_name(f"ranker-{tenant}")
    assert experiment is not None
    [run] = client_.search_runs([experiment.experiment_id])
    params, metrics = run.data.params, run.data.metrics
    assert params["monotone_increasing"] == "content_cosine,anchor_target_fit,context_relevance"
    assert params["evaluation_seeds"] == ",".join(map(str, ENOUGH.evaluation_seeds))
    assert run.data.tags["product_skipped_reason"] == "none"
    assert (metrics["unlabelable_targets"], metrics["unlabelable_rows"]) == (
        report.unlabelable_targets,
        report.unlabelable_rows,
    )
    assert metrics["seeds_evaluated"] == len(ENOUGH.evaluation_seeds)
    assert (metrics["seeds_worse"], metrics["seeds_better"]) == (
        report.seeds_worse,
        report.seeds_better,
    )
    steps = sorted(
        (m.step, m.value) for m in client_.get_metric_history(run.info.run_id, "seed_value")
    )
    assert steps == [(i, float(seed)) for i, seed in enumerate(ENOUGH.evaluation_seeds)]
    for scorer in ("learned", "plain", "baseline"):
        assert metrics[f"product_{scorer}_orphan_page_share"] == pytest.approx(len(ORPHANS) / PAGES)
    artifacts = {a.path for a in client_.list_artifacts(run.info.run_id)}
    assert {"seed_results.json", "product_measures.json"} <= artifacts
    logged = run_text(run.info.run_id, tmp_path / "artifacts")
    assert [u for u in URLS if u in logged] == [], "page urls reached the MLflow run"
    folded = logged.casefold()
    assert [k for k in KEYWORDS if k.casefold() in folded] == [], "keywords reached the run"
    assert np.isfinite(metrics["seed_mean_delta"])
    assert report.columns == MODEL_COLUMNS


@pytest.mark.integration
async def test_an_orphan_that_is_never_a_candidate_target_is_not_unlabelable(
    graph: GraphRepo, mongo: MongoRepo, tenant: str, tmp_path: Path, local_mlflow: str
) -> None:
    """A noindex page without a vector is crawled and has no inbound link, yet retrieval never
    makes it a target: it has no row to leave out, so it is not counted."""
    await seed_ranking(graph, mongo, tenant)
    imprint = noise_url("imprint")
    await graph.upsert_pages(
        tenant,
        [Page(url=imprint, status_code=200, is_indexable=False, word_count=300, language="en")],
    )

    report = await trained(graph, mongo, tenant, tmp_path)

    rounds = await held_out_rounds(
        graph, mongo, tenant, settings=ENOUGH, cache_dir=tmp_path, voyage=client(voyage())
    )
    assert rounds.cache_hit == (True,) * ENOUGH.rounds
    assert imprint in rounds.orphan_targets, "a crawled page no body link reaches"
    assert imprint not in set(rows_of(tmp_path, tenant)["target_url"])
    assert report.unlabelable_targets == len(ORPHANS) == 6
    assert f"Orphan targets: {len(ORPHANS)} " in summarise_ranker(report)


@pytest.mark.integration
async def test_a_seed_that_fails_the_guards_is_reported_with_its_reason(
    graph: GraphRepo, mongo: MongoRepo, tenant: str, tmp_path: Path, local_mlflow: str
) -> None:
    """Split seed 130 leaves 24 test groups on the planted tenant, fewer than the 30 needed; the
    fixed split, seed 7, passes."""
    await seed_ranking(graph, mongo, tenant)
    settings = ENOUGH.model_copy(update={"evaluation_seeds": (7, 130)})

    with capture_logs() as logs:
        report = await trained(graph, mongo, tenant, tmp_path, settings)

    reason = f"24 test groups with a hidden link, fewer than {MIN_TEST_GROUPS}"
    assert report.skipped_reason is None
    assert [result.seed for result in report.seed_results] == [7]
    assert report.skipped_seeds == {130: reason}
    assert f"seed 130 skipped: {reason}" in summarise_ranker(report)
    [line] = [entry for entry in logs if entry["event"] == "ranker.seed_skipped"]
    assert (line["seed"], line["reason"]) == (130, reason)
    client_ = MlflowClient(local_mlflow)
    experiment = client_.get_experiment_by_name(f"ranker-{tenant}")
    assert experiment is not None
    [run] = client_.search_runs([experiment.experiment_id])
    logged = run_text(run.info.run_id, tmp_path / "artifacts")
    assert "130" in logged
    assert reason in logged, "the skipped seed's reason is not in the MLflow run"


@pytest.mark.integration
async def test_a_production_matrix_that_cannot_be_read_skips_the_measures_not_the_run(
    graph: GraphRepo,
    mongo: MongoRepo,
    tenant: str,
    tmp_path: Path,
    local_mlflow: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The matrix is gone between the feature stage and the measures, as a concurrent cleanup
    would leave it: the run completes, trained and logged, with the error type as the reason."""
    await seed_ranking(graph, mongo, tenant)
    await choose_anchors(graph, mongo, tenant, tmp_path)
    assembled = ranker.assemble_features

    async def vanishing(*args: Any, **kwargs: Any) -> tuple[FeatureReport, Path]:
        report, path = await assembled(*args, **kwargs)
        path.unlink()
        return report, path

    monkeypatch.setattr(ranker, "assemble_features", vanishing)

    with capture_logs() as logs:
        report = await trained(graph, mongo, tenant, tmp_path)

    assert report.skipped_reason is None
    assert report.metrics
    assert report.seed_results
    assert report.product_measures == ()
    assert report.product_skipped_reason is not None
    found = re.fullmatch(r"production pairs unreadable \((\w+)\)", report.product_skipped_reason)
    assert found is not None, report.product_skipped_reason
    [line] = [entry for entry in logs if entry["event"] == "ranker.product_skipped"]
    assert line["error"] == found.group(1)
    assert str(tmp_path) not in report.product_skipped_reason
    client_ = MlflowClient(local_mlflow)
    experiment = client_.get_experiment_by_name(f"ranker-{tenant}")
    assert experiment is not None
    [run] = client_.search_runs([experiment.experiment_id])
    assert run.data.tags["product_skipped_reason"] == report.product_skipped_reason
