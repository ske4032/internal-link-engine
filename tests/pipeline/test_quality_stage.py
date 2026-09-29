"""evaluate_quality: the held-out view, every check on a planted tenant, not-applicable
handling, alerts, tenant isolation, and read-only against both stores and the feature cache."""

from __future__ import annotations

import hashlib
import shutil
import subprocess
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest
import quality_factories as make
from quality_seed import (
    GSC_PAGES,
    KEYWORDS,
    OFFSETS,
    SIZE,
    TOPICS,
    URLS,
    context_score,
    seed_quality,
    url,
)
from quality_seed import voyage as keyword_voyage
from sklearn.metrics import adjusted_rand_score
from store_state import graph_state, mongo_state
from structlog.testing import capture_logs
from voyage_fakes import client, settings
from voyageai.error import AuthenticationError, ServiceUnavailableError

from linking_engine.discovery.candidates import candidate_report
from linking_engine.discovery.features import code_digest
from linking_engine.discovery.scoring import default_weights, weights_hash
from linking_engine.errors import DatabaseReadError
from linking_engine.ml.quality import (
    HIDE_SEED,
    LINK_DERIVED_COLUMNS,
    hide_links,
    quality_metrics,
)
from linking_engine.models import (
    AnchorRules,
    CandidateSet,
    CandidateTarget,
    CommunityContext,
    FeatureWeight,
    KeywordRung,
    KeywordSource,
    LinkGraphSnapshot,
    LinkRelevance,
    PageStructure,
    PillarFloor,
    QualityBaseline,
    ResolvedKeyword,
    ScorerWeights,
    SourceExtractability,
    TargetCandidates,
    TargetSelection,
)
from linking_engine.pipeline import quality
from linking_engine.pipeline.analytics import (
    compute_centrality,
    compute_communities,
    compute_hubs,
    load_link_graphs,
)
from linking_engine.pipeline.quality import (
    evaluate_quality,
    git_sha,
    held_out_view,
    link_relevance_check,
    retrieval_check,
    summarise_quality,
    unique_share,
)
from linking_engine.pipeline.scoring import score_pairs

if TYPE_CHECKING:
    from linking_engine.graph.repo import GraphRepo
    from linking_engine.ingest.mongo_repo import MongoRepo

SHA = "c" * 40
EVERY_SECTION = ("retrieval", "feature_signal", "scorer", "keywords", "link_relevance")
KEYWORD_SUB_CHECKS = (
    "keyword_uniqueness",
    "keyword_extractability",
    "anchor_match",
    "keyword_relevance",
)


# ── git sha ─────────────────────────────────────────────────────────────────


def test_the_git_sha_comes_from_the_environment_first(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GIT_SHA", f"  {SHA}\n")
    assert git_sha() == SHA


def test_without_the_environment_the_git_sha_is_the_checkouts_head(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GIT_SHA", raising=False)
    git = shutil.which("git")
    assert git is not None
    head = subprocess.run(  # noqa: S603 - fixed arguments
        [git, "rev-parse", "HEAD"],
        cwd=Path(quality.__file__).parent,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()

    assert git_sha() == head


@pytest.mark.parametrize("failure", ["no-git", "os-error", "timeout", "not-a-checkout"])
def test_the_git_sha_is_unknown_when_git_cannot_tell(
    monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    monkeypatch.setenv("GIT_SHA", "   ")

    def run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        if failure == "os-error":
            raise OSError("no such file")
        if failure == "timeout":
            raise subprocess.TimeoutExpired("git", 5)
        return subprocess.CompletedProcess(args, 128, stdout="", stderr="not a git repository")

    if failure == "no-git":
        monkeypatch.setattr(quality.shutil, "which", lambda _name: None)
    else:
        monkeypatch.setattr(quality.subprocess, "run", run)

    assert git_sha() == "unknown"


# ── the held-out view ───────────────────────────────────────────────────────

TOY = ("a", "b", "c", "d", "e")
GHOST = "example.com/toy/ghost"


def toy(name: str) -> str:
    return f"example.com/toy/{name}"


def toy_snapshot() -> LinkGraphSnapshot:
    links = [("a", "b"), ("a", "b"), ("b", "c"), ("c", "a"), ("a", "d"), ("e", "d"), ("d", "d")]
    return LinkGraphSnapshot(
        tenant_id="test-toy",
        pages=(*map(toy, TOY), GHOST),
        placeholders=(False,) * len(TOY) + (True,),
        links=(*((toy(s), toy(t)) for s, t in links), (toy("a"), GHOST)),
    )


def toy_structure(**overrides: object) -> list[PageStructure]:
    """Stored values the view must replace: stale counts, a wrong pillar, fixed labels."""
    return [
        PageStructure.model_validate(
            {
                "url": toy(name),
                "language": "en",
                "inbound": 9,
                "outbound": 9,
                "is_orphan": False,
                "page_rank_percentile": 0.5,
                "link_community_id": 7,
                "keyword_community_id": 3,
                "content_community_id": 4,
                "hub_id": 0 if name in "abc" else 1,
                "is_hub_pillar": name == "e",
                **overrides,
            }
        )
        for name in TOY
    ]


def toy_context() -> list[CommunityContext]:
    return [
        CommunityContext(
            url=toy(name),
            link_community_id=7,
            keyword_community_id=3,
            content_community_id=4,
            hub_id=0 if name in "abc" else 1,
        )
        for name in TOY
    ]


def toy_vectors() -> dict[str, np.ndarray]:
    return {
        toy("a"): np.array([1.0, 0.1, 0.0], dtype=np.float32),
        toy("b"): np.array([1.0, 0.3, 0.0], dtype=np.float32),
        toy("c"): np.array([1.0, 0.5, 0.0], dtype=np.float32),
        toy("d"): np.array([0.0, 0.0, 1.0], dtype=np.float32),
        toy("e"): np.array([0.0, 0.1, 1.0], dtype=np.float32),
    }


def by_url(view: quality.HeldOutView) -> dict[str, PageStructure]:
    return {page.url: page for page in view.structure}


def test_a_hidden_link_is_gone_at_every_position_and_every_link_count_is_recomputed() -> None:
    hidden = {(toy("a"), toy("b"))}

    view = held_out_view(toy_snapshot(), toy_structure(), toy_context(), toy_vectors(), hidden)

    assert (toy("a"), toy("b")) not in view.links
    assert len(view.links) == len(toy_snapshot().links) - 2
    pages = by_url(view)
    # Distinct crawled pages only: no self link, nothing to or from the placeholder.
    assert {name: pages[toy(name)].inbound for name in TOY} == {
        "a": 1,
        "b": 0,
        "c": 1,
        "d": 2,
        "e": 0,
    }
    assert {name: pages[toy(name)].outbound for name in TOY} == {
        "a": 1,
        "b": 1,
        "c": 1,
        "d": 0,
        "e": 1,
    }
    assert {name for name in TOY if pages[toy(name)].is_orphan} == {"b", "e"}


def test_pagerank_communities_and_pillars_are_recomputed_hubs_and_content_labels_kept() -> None:
    hidden = {(toy("a"), toy("b"))}

    view = held_out_view(toy_snapshot(), toy_structure(), toy_context(), toy_vectors(), hidden)
    whole = held_out_view(
        toy_snapshot(), toy_structure(), toy_context(), toy_vectors(), frozenset()
    )

    pages, before = by_url(view), by_url(whole)
    assert pages[toy("b")].page_rank_percentile < before[toy("b")].page_rank_percentile, (
        "b lost its only inbound link"
    )
    assert before[toy("b")].inbound == 1
    assert not before[toy("b")].is_orphan
    assert {p.hub_id for p in view.structure} == {0, 1}
    assert {(p.keyword_community_id, p.content_community_id) for p in view.structure} == {(3, 4)}
    # Hub 0 has three members, so one pillar; hub 1 is too small for one.
    assert sum(p.is_hub_pillar for p in view.structure if p.hub_id == 0) == 1
    assert not any(p.is_hub_pillar for p in view.structure if p.hub_id == 1), (
        "the stored pillar flag was kept"
    )
    assert all(p.link_community_id is not None for p in view.structure)
    assert {c.url: c.link_community_id for c in view.context} == {
        p.url: p.link_community_id for p in view.structure
    }
    assert {c.hub_id for c in view.context} == {0, 1}


@pytest.mark.parametrize("stray", [toy("zzz"), GHOST], ids=["unknown", "placeholder"])
def test_a_page_that_is_not_a_crawled_page_of_the_snapshot_is_refused(stray: str) -> None:
    structure = [*toy_structure(), toy_structure(url=stray)[0]]
    with pytest.raises(DatabaseReadError, match="not crawled pages"):
        held_out_view(toy_snapshot(), structure, toy_context(), toy_vectors(), frozenset())


# ── retrieval check ─────────────────────────────────────────────────────────


def candidates(
    targets: list[TargetCandidates], floors: dict[str, PillarFloor] | None = None
) -> CandidateSet:
    selection = TargetSelection(
        crawled_pages=len(targets),
        not_indexable=0,
        without_vector=0,
        targets=tuple(CandidateTarget(url=t.target_url, indexable_assumed=True) for t in targets),
    )
    report = candidate_report(
        "test-toy",
        "page_content",
        50,
        512,
        selection,
        10,
        targets,
        floors=floors,
        load_seconds=0.0,
        search_seconds=0.0,
        seconds=0.0,
    )
    return CandidateSet(report=report, targets=tuple(targets))


def target(name: str, sources: tuple[str, ...], eligible: int) -> TargetCandidates:
    return TargetCandidates(
        target_url=toy(name),
        sources=tuple(map(toy, sources)),
        similarities=tuple(1.0 - i / 10 for i in range(len(sources))),
        eligible=eligible,
        linked=0,
        linked_nearer=0,
    )


def test_recall_counts_only_hidden_links_a_retrieval_could_return() -> None:
    found = candidates([target("t1", ("s1", "s2", "s3"), 40), target("t2", ("s4",), 80)])
    hidden = {
        (toy("s2"), toy("t1")),  # retrieved at rank 2
        (toy("s5"), toy("t2")),  # recoverable, not retrieved
        (toy("s6"), toy("t3")),  # t3 is not a retrieval target
        (toy("s7"), toy("t1")),  # s7 has no vector, so is not in the pool
        (toy("s8"), toy("t1")),  # s8 is in another language
    }
    pool = {toy(name) for name in ("s1", "s2", "s3", "s4", "s5", "s6", "s8", "t1", "t2")}
    languages = {toy(name): "en" for name in ("s1", "s2", "s3", "s4", "s5", "s6", "s7", "t1", "t2")}
    languages[toy("s8")] = "de"

    check = retrieval_check(50, hidden, found, pool, languages)

    assert check is not None
    assert (check.body_link_pairs, check.hidden, check.recoverable, check.candidates) == (
        50,
        5,
        2,
        4,
    )
    assert (check.hide_share, check.seed) == (0.1, HIDE_SEED)
    assert [(r.k, r.recall) for r in check.recall] == [(10, 0.5), (20, 0.5), (50, 0.5)]
    # Mean over the two recoverable links of min(1, k / eligible), eligible 40 and 80.
    assert [r.random for r in check.recall] == pytest.approx([0.1875, 0.375, 0.8125])


def test_a_hidden_link_the_channel_adds_past_a_full_nearest_is_never_inside_k() -> None:
    nearest = tuple(f"n{i:02d}" for i in range(50))
    pillar = TargetCandidates(
        target_url=toy("pillar"),
        sources=(*map(toy, nearest), toy("hub")),
        similarities=(*(0.95 - i / 100 for i in range(50)), 0.45),
        eligible=80,
        linked=0,
        linked_nearer=0,
        pillar_pairs=1,
    )
    floor = PillarFloor(floor=0.4, basis="existing_links", links=60)
    found = candidates([pillar], {"en": floor})
    pool = {*pillar.sources, pillar.target_url}

    check = retrieval_check(
        60, {(toy("hub"), toy("pillar"))}, found, pool, dict.fromkeys(pool, "en")
    )

    assert check is not None
    assert (check.recoverable, check.candidates) == (1, 51)
    assert [(r.k, r.recall) for r in check.recall] == [(10, 0.0), (20, 0.0), (50, 0.0)], (
        "a channel pair ranks inside k although the target's nearest sources fill it"
    )
    assert [r.random for r in check.recall] == pytest.approx([10 / 80, 20 / 80, 50 / 80])


def test_retrieval_is_not_applicable_when_no_hidden_link_is_recoverable() -> None:
    found = candidates([target("t1", ("s1",), 5)])
    hidden = {(toy("s6"), toy("t3"))}

    assert retrieval_check(10, hidden, found, {toy("s6")}, {}) is None


# ── link relevance and keyword uniqueness ───────────────────────────────────


def relevance_row(position: int, context: float | None, fit: float | None) -> LinkRelevance:
    return LinkRelevance(
        source_url=toy("s"),
        position=position,
        target_url=toy("t"),
        context_relevance=context,
        anchor_target_fit=fit,
    )


def test_link_relevance_summarises_the_stored_scores() -> None:
    rows = [relevance_row(0, 0.8, 0.9), relevance_row(1, 0.6, None), relevance_row(2, None, None)]

    check = link_relevance_check(rows)

    assert check is not None
    assert check.links == 3
    assert (check.context.count, check.context.mean) == (2, pytest.approx(0.7))
    assert check.anchor is not None
    assert (check.anchor.count, check.anchor.mean) == (1, pytest.approx(0.9))


def test_link_relevance_without_context_scores_is_not_applicable_and_fits_are_optional() -> None:
    assert link_relevance_check([relevance_row(0, None, None)]) is None
    assert link_relevance_check([]) is None
    check = link_relevance_check([relevance_row(0, 0.5, None)])
    assert check is not None
    assert check.anchor is None


def resolved(page: str, text: str, language: str = "en") -> ResolvedKeyword:
    return ResolvedKeyword(url=toy(page), text=text, language=language, rung=KeywordRung.H1)


def test_uniqueness_is_the_share_of_distinct_keywords_that_one_page_resolved() -> None:
    keywords = {
        toy("a"): resolved("a", "Trail Shoes"),
        toy("b"): resolved("b", "trail  SHOES"),  # the same keyword as a's once normalised
        toy("c"): resolved("c", "Trail Shoes", "de"),  # another language, another keyword
        toy("d"): resolved("d", "Tents"),
    }

    assert unique_share(keywords) == pytest.approx(2 / 3)
    assert unique_share({toy("a"): resolved("a", "Tents")}) == 1.0
    assert unique_share({}) is None


def test_extractability_is_split_by_the_rung_of_the_targets_primary_keyword() -> None:
    shoes, jacket, pegs = toy("shoes"), toy("jacket"), toy("pegs")
    keywords = {
        target: [(rank, text, KeywordSource.INFERRED) for rank, text in enumerate(texts, 1)]
        for target, texts in (
            (shoes, ("trail shoes", "hiking boots")),
            (jacket, ("rain jacket",)),
            (pegs, ("tent pegs",)),
        )
    }
    sources = {
        toy("a"): quality._SourcePage("Our trail shoes, tent pegs and a rain jacket.", (), "en"),
        toy("b"): quality._SourcePage("Hiking boots for rocky ground.", (), "en"),
    }
    # Page c has no stored body: its pair counts, with nothing found.
    wanted = {toy("a"): [shoes, jacket, pegs], toy("b"): [shoes, jacket], toy("c"): [shoes]}

    extract = quality._extractability(
        wanted,
        keywords,
        sources,
        0.6,
        frozenset(),
        {shoes: KeywordRung.H1, jacket: KeywordRung.GSC},
    )

    assert extract is not None
    # The pegs target resolved no keyword: its one pair, found on a, is in the totals only.
    assert (extract.pairs, extract.found_primary, extract.found_set) == (6, 3 / 6, 4 / 6)
    assert extract.words_primary == 3 / 6
    # shoes: its primary on a, only its second keyword on b, nothing on c.
    assert extract.by_source == {
        KeywordRung.GSC: SourceExtractability(
            pairs=2, found_primary=1 / 2, found_set=1 / 2, words_primary=1 / 2
        ),
        KeywordRung.H1: SourceExtractability(
            pairs=3, found_primary=1 / 3, found_set=2 / 3, words_primary=1 / 3
        ),
    }


# ── the summary ─────────────────────────────────────────────────────────────


def test_the_summary_of_a_first_run_says_what_was_checked_and_that_there_is_no_baseline() -> None:
    summary = summarise_quality(make.report("acme"))

    assert summary.startswith("Quality evaluation for tenant acme: git aaaaaaaaaaaa")
    for expected in (
        "Recall 50.0% at 10, 75.0% at 20, 100.0% at 50 (random 25.0%, 50.0%, 100.0%)",
        "Feature signal: 1 of 3 columns",
        "Scorer (baseline-1): AUC 0.800",
        "Keywords: 4 of 10 crawled 2xx pages resolved",
        "Extractability over 8 candidate pairs, by the extraction ladder (stem set threshold "
        "0.6), existing anchors aside: the primary keyword is found in the source copy for "
        "50.0%, some keyword of the set for 75.0% (best rung exact 37.5%, stemmed 25.0%, "
        "stem set 12.5%)",
        "75.0% set. H1 keywords: 25.0% found, 50.0% ceiling, 4 pairs. STRATEGIC keywords: "
        "50.0% found, 100.0% ceiling, 2 pairs.",
        "Existing anchors: of 4 descriptive anchors",
        "rank 1 0.600 (n=3), rank 2 0.400 (n=2)",
        "By origin: primary_h1 0.600 (median 0.600, n=3), secondary_gsc_observed 0.400",
        "by words: 1_2 0.550 (median 0.600, n=4), 3_4 0.300 (median 0.300, n=1).",
        "50% of the scorer's weight sits on link-derived columns",
        "Like for like, without the link-derived columns: scorer AUC 0.700, weights "
        "renormalised, against content_cosine 0.900 (lift -0.200).",
        "Link relevance over 30 links",
        "Not applicable: none.",
        "no baseline",
    ):
        assert expected in summary, expected


def test_the_summary_says_when_no_like_for_like_score_exists() -> None:
    scorer = make.scorer(
        link_derived_weight_share=1.0,
        score_auc_excl_link_counts=None,
        score_auc_lift_excl_link_counts=None,
    )

    summary = summarise_quality(make.report("acme", scorer=scorer))

    assert "100% of the scorer's weight sits on link-derived columns" in summary
    assert "none, every weighted column is link-derived" in summary


def test_the_summary_gives_the_reason_keyword_relevance_does_not_apply() -> None:
    summary = summarise_quality(make.report("acme", not_applicable=("keyword_relevance",)))

    assert "Keyword relevance: not applicable, no Voyage API key." in summary


def test_the_summary_names_each_check_that_does_not_apply() -> None:
    summary = summarise_quality(
        make.report(
            "acme",
            not_applicable=("retrieval", "feature_signal", "scorer", "keywords", "link_relevance"),
        )
    )

    for expected in (
        "Retrieval: not applicable",
        "Feature signal and scorer: not applicable",
        "Keywords: not applicable",
        "Link relevance: not applicable",
        "Not applicable: retrieval, feature_signal, scorer, keywords, link_relevance.",
    ):
        assert expected in summary, expected
    assert "Scorer (" not in summary


def test_the_summary_lists_what_moved_since_the_baseline() -> None:
    quiet = summarise_quality(make.report("acme", baseline_run_id="run-0"))
    moved = summarise_quality(
        make.report(
            "acme",
            baseline_run_id="run-0",
            alerts=(
                make.alert(),
                make.alert(metric="gsc_pair_share", previous=0.0, current=0.25, change=None),
                make.alert(
                    metric="score_auc",
                    previous=0.8,
                    current=0.7,
                    change=-0.1,
                    band=0.05,
                    relative=False,
                ),
            ),
        )
    )

    assert "Against run run-0: no headline metric moved beyond its band" in quiet
    assert moved.splitlines()[-4:] == [
        "Moved since run run-0:",
        "- recall_at_10: 0.5 -> 0.25 (-50.0%, band 0.2 relative)",
        "- gsc_pair_share: 0 -> 0.25 (from 0, band 0.2 relative)",
        "- score_auc: 0.8 -> 0.7 (-0.1, band 0.05)",
    ]


# ── refused before any read ─────────────────────────────────────────────────


@pytest.mark.parametrize("tenant_id", [" ", "..", "../escape", "a/b"])
async def test_a_tenant_that_is_not_a_directory_name_is_refused_before_any_read(
    tmp_path: Path, tenant_id: str
) -> None:
    unused = object()
    with pytest.raises(ValueError, match="tenant_id"):
        await evaluate_quality(
            cast("GraphRepo", unused), cast("MongoRepo", unused), tenant_id, cache_dir=tmp_path
        )
    assert entries(tmp_path) == []


class WeightsOnly:
    """A Mongo stand-in that only has scorer weights; any other read fails the test."""

    def __init__(self, weights: ScorerWeights) -> None:
        self.weights = weights

    async def get_scorer_weights(self, tenant_id: str) -> ScorerWeights:
        return self.weights


async def test_weights_naming_an_unknown_column_are_refused_before_the_graph_is_read(
    tmp_path: Path,
) -> None:
    weights = ScorerWeights(
        version="custom-1", features=(FeatureWeight(column="no_such_column", weight=1.0),)
    )
    with pytest.raises(ValueError, match="no_such_column"):
        await evaluate_quality(
            cast("GraphRepo", object()),
            cast("MongoRepo", WeightsOnly(weights)),
            "test-weights",
            cache_dir=tmp_path,
        )


# ── the planted tenant ──────────────────────────────────────────────────────


def entries(directory: Path) -> list[str]:
    return sorted(p.name for p in directory.iterdir())


def files(root: Path) -> dict[str, tuple[int, str]]:
    # The tenant's text-vector lock file holds no data.
    return {
        str(path.relative_to(root)): (
            path.stat().st_mtime_ns,
            hashlib.sha256(path.read_bytes()).hexdigest(),
        )
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != ".lock"
    }


def planted_hidden() -> frozenset[tuple[str, str]]:
    return hide_links(
        (url(topic, i), url(topic, i + offset))
        for topic in TOPICS
        for i in range(SIZE)
        for offset in OFFSETS
    )


def planted_eligible() -> dict[tuple[str, str], int]:
    """Each target has 23 other pages; the three that link to it are not eligible, except
    those whose link is hidden."""
    into = Counter(target for _, target in planted_hidden())
    return {link: 20 + into[link[1]] for link in planted_hidden()}


@pytest.mark.integration
async def test_a_planted_tenant_gets_every_check_and_nothing_is_written(
    graph: GraphRepo, mongo: MongoRepo, tenant: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GIT_SHA", SHA)
    await seed_quality(graph, mongo, tenant)
    cache = tmp_path / "features"
    await score_pairs(graph, mongo, tenant, cache_dir=cache)
    matrices = files(cache)
    stored_graph, stored_mongo = await graph_state(graph, tenant), await mongo_state(mongo)
    voyage = keyword_voyage()

    with capture_logs() as logs:
        report = await evaluate_quality(
            graph, mongo, tenant, cache_dir=cache, voyage=client(voyage)
        )

    assert await graph_state(graph, tenant) == stored_graph, "the graph changed"
    assert await mongo_state(mongo) == stored_mongo, "Mongo changed"
    after = files(cache)
    assert {name: after[name] for name in matrices} == matrices, "a cached matrix changed"
    assert set(after) - set(matrices) == {f"{tenant}/keyword_vectors.parquet"}, (
        "held-out data reached the cache"
    )

    assert report.tenant_id == tenant
    assert report.not_applicable == ()
    assert (report.baseline_run_id, report.alerts) == (None, ())
    weights = default_weights()
    assert report.versions.model_dump() == {
        "git_sha": SHA,
        "feature_digest": code_digest(),
        "weights_version": weights.version,
        "weights_hash": weights_hash(weights),
    }

    retrieval = report.retrieval
    assert retrieval is not None
    eligible = planted_eligible()
    hidden = len(planted_hidden())
    assert hidden >= 1
    assert (retrieval.body_link_pairs, retrieval.hidden, retrieval.recoverable) == (
        72,
        hidden,
        hidden,
    )
    recall = {entry.k: entry for entry in retrieval.recall}
    assert recall[10].recall == 1.0, "a hidden source ranks among its topic's nine eligible pages"
    for k in (10, 20, 50):
        assert recall[k].random == pytest.approx(
            float(np.mean([min(1.0, k / n) for n in eligible.values()]))
        )
    assert recall[10].recall > recall[10].random
    # Every target keeps all its eligible sources: 20 each, plus one per link hidden into it.
    assert retrieval.candidates == 24 * 20 + hidden

    signal = report.feature_signal
    assert signal is not None
    assert (signal.pairs, signal.positives) == (retrieval.candidates, hidden)
    cosine = {entry.column: entry for entry in signal.columns}["content_cosine"]
    assert cosine.auc is not None
    assert cosine.auc > 0.55, "hidden links are same-topic pairs"
    assert signal.features_with_signal >= 1

    scorer = report.scorer
    assert scorer is not None
    assert scorer.weights_version == weights.version
    assert scorer.best_feature_excl_link_counts not in LINK_DERIVED_COLUMNS
    by_column = {entry.column: entry.ranker_auc for entry in signal.columns}
    assert scorer.best_feature_auc == max(by_column.values())
    assert scorer.best_feature_auc_excl_link_counts == max(
        auc for column, auc in by_column.items() if column not in LINK_DERIVED_COLUMNS
    )
    assert sum(scorer.hidden_histogram) == hidden
    assert sum(scorer.other_histogram) == retrieval.candidates - hidden
    link_weight = sum(f.weight for f in weights.features if f.column in LINK_DERIVED_COLUMNS)
    assert scorer.link_derived_weight_share == pytest.approx(
        link_weight / sum(f.weight for f in weights.features)
    )
    assert scorer.score_auc_excl_link_counts is not None
    assert scorer.score_auc_lift_excl_link_counts == pytest.approx(
        scorer.score_auc_excl_link_counts - scorer.best_feature_auc_excl_link_counts
    )

    keywords = report.keywords
    assert keywords is not None
    assert (keywords.pages, keywords.resolved) == (24, 24)
    assert keywords.by_rung == {
        KeywordRung.STRATEGIC: 1,
        KeywordRung.GSC: 0,
        KeywordRung.H1: 23,
        KeywordRung.TITLE: 0,
    }
    assert keywords.unique_share == 1.0
    # Production candidates: every target keeps its 20 eligible sources. For each of the 23 H1
    # targets, page i-4's copy holds its keyword verbatim ("trail elm": exact) and page i-5's
    # holds "trail <word> with elm", a span sharing both stems (Jaccard 2/3: stem set). The
    # strategic target's primary "alder trail guide" shares two stems with "trail alder" on
    # trail page 8, which is also its second keyword "trail alder boots" with the last
    # modifier dropped (stemmed), and with "Alder guide" on tent page 0 (stem set). Trail
    # pages 7 and 8 hold every word of the primary.
    extract = keywords.extractability
    assert extract is not None
    assert extract.pairs == 480
    assert extract.stem_set_threshold == 0.6
    assert (extract.found_primary, extract.found_set) == (48 / 480, 48 / 480)
    assert (extract.exact_set, extract.stemmed_set, extract.stem_set_set) == (
        23 / 480,
        1 / 480,
        24 / 480,
    )
    assert (extract.words_primary, extract.words_set) == (48 / 480, 48 / 480)
    # Split by the rung each target's primary keyword was resolved at, from the flow's plan.
    assert extract.by_source == {
        KeywordRung.H1: SourceExtractability(
            pairs=460, found_primary=46 / 460, found_set=46 / 460, words_primary=46 / 460
        ),
        KeywordRung.STRATEGIC: SourceExtractability(
            pairs=20, found_primary=2 / 20, found_set=2 / 20, words_primary=2 / 20
        ),
    }
    # Offset 1 anchors name the target's keyword, offset 2 are generic and skipped, offset 3
    # name only the topic.
    anchors = keywords.anchors
    assert anchors is not None
    assert (anchors.anchors, anchors.primary, anchors.any_rank) == (48, 0.5, 0.5)
    relevance = keywords.relevance
    assert relevance is not None
    assert (relevance.texts, relevance.embedded, relevance.cached) == (25, 25, 0)
    assert sorted(t for call in voyage.calls for t in call.texts) == sorted(
        k for k in KEYWORDS if k != "Trail Alder"
    )
    assert [(r.rank, r.keywords) for r in relevance.ranks] == [(1, 24), (2, 1)]
    assert [(g.group, g.keywords) for g in relevance.by_origin] == [
        ("primary_strategic", 1),
        ("primary_h1", 23),
        ("secondary_client_strategic", 1),
    ]
    # Every H1 keyword has two words; both strategic keywords have three.
    assert [(g.group, g.keywords) for g in relevance.by_length] == [("1_2", 23), ("3_4", 2)]
    assert relevance.ranks[0].mean > 0.8, "each keyword sits at its page's topic centre"

    links = report.link_relevance
    assert links is not None
    assert (links.links, links.context.count) == (72, 72)
    assert links.anchor is not None
    assert links.anchor.count == 24
    assert links.context.mean == pytest.approx(
        np.mean([context_score(i, o) for _ in TOPICS for i in range(SIZE) for o in OFFSETS])
    )

    coverage = report.coverage
    assert coverage.pairs == retrieval.candidates
    gsc_targets = {url("trail", i) for i in range(GSC_PAGES)}
    into = Counter(t for _, t in planted_hidden())
    gsc_pairs = sum(20 + into[t] for t in gsc_targets)
    assert coverage.gsc_pair_share == pytest.approx(gsc_pairs / retrieval.candidates)
    assert coverage.keyword_page_share == 1.0

    [line] = [entry for entry in logs if entry["event"] == "quality.evaluated"]
    assert (line["tenant_id"], line["hidden_links"], line["not_applicable"]) == (
        tenant,
        hidden,
        [],
    )
    # Every line the eval logs, its retrievals' and the keyword cache's included, is its own.
    staged = [entry for entry in logs if "stage" in entry]
    assert {entry["event"] for entry in staged} >= {
        "candidates.retrieved",
        "keywords.vectors",
        "quality.evaluated",
    }
    assert {entry["stage"] for entry in staged} == {"quality-eval"}
    assert {entry["tenant_id"] for entry in staged} == {tenant}
    logged = " ".join(str(value) for entry in logs for value in entry.values())
    assert [u for u in URLS if u in logged] == [], "page urls in the log"
    assert [k for k in KEYWORDS if k in logged] == [], "keyword texts in the log"


@pytest.mark.integration
async def test_a_rerun_reuses_the_keyword_vectors_and_compares_against_its_baseline(
    graph: GraphRepo, mongo: MongoRepo, tenant: str, tmp_path: Path
) -> None:
    await seed_quality(graph, mongo, tenant)
    first = await evaluate_quality(
        graph, mongo, tenant, cache_dir=tmp_path, voyage=client(keyword_voyage())
    )
    metrics = quality_metrics(first)
    voyage = keyword_voyage()

    again = await evaluate_quality(
        graph,
        mongo,
        tenant,
        cache_dir=tmp_path,
        voyage=client(voyage),
        baseline=QualityBaseline(run_id="run-1", metrics=metrics),
    )

    assert voyage.call_count == 0
    assert again.keywords is not None
    assert again.keywords.relevance is not None
    assert (again.keywords.relevance.embedded, again.keywords.relevance.cached) == (0, 25)
    assert (again.baseline_run_id, again.alerts) == ("run-1", ())
    same = {"seconds", "keyword_vectors_embedded", "keyword_vectors_cached", "keyword_embed_tokens"}
    assert {k: v for k, v in quality_metrics(again).items() if k not in same} == {
        k: v for k, v in metrics.items() if k not in same
    }, "the same data gave another result"

    doctored = {**metrics, "recall_at_10": 0.5, "score_auc": metrics["score_auc"] - 0.2}
    moved = await evaluate_quality(
        graph,
        mongo,
        tenant,
        cache_dir=tmp_path,
        voyage=client(keyword_voyage()),
        baseline=QualityBaseline(run_id="run-2", metrics={**doctored, "gsc_pair_share": 0.0}),
    )

    assert moved.baseline_run_id == "run-2"
    assert [a.metric for a in moved.alerts] == ["recall_at_10", "score_auc", "gsc_pair_share"]
    by_metric = {a.metric: a for a in moved.alerts}
    assert (by_metric["recall_at_10"].change, by_metric["recall_at_10"].relative) == (1.0, True)
    assert by_metric["score_auc"].relative is False
    assert by_metric["gsc_pair_share"].change is None

    dropped = await evaluate_quality(
        graph,
        mongo,
        tenant,
        cache_dir=tmp_path,
        baseline=QualityBaseline(run_id="run-1", metrics=metrics),
    )

    assert dropped.not_applicable == ("keyword_relevance",)
    [moved_check] = dropped.alerts
    assert (moved_check.metric, moved_check.previous, moved_check.current) == (
        "not_applicable_checks",
        0.0,
        1.0,
    )


@pytest.mark.integration
async def test_nothing_hidden_reproduces_what_the_analytics_stage_stored(
    graph: GraphRepo, mongo: MongoRepo, tenant: str
) -> None:
    """The held-out view recomputes with the production functions: hiding nothing gives back
    the graph-analytics stage's own values."""
    await seed_quality(graph, mongo, tenant)
    graphs = await load_link_graphs(graph, tenant)
    await compute_centrality(graph, tenant, graphs)
    await compute_communities(graph, tenant, graphs)
    await compute_hubs(graph, tenant, graphs)
    structure = await graph.page_structure(tenant)

    view = held_out_view(
        await graph.link_graph(tenant),
        structure,
        await graph.community_context(tenant),
        await graph.content_vectors(tenant),
        frozenset(),
    )

    recomputed = by_url(view)
    assert sum(page.is_hub_pillar for page in structure) == 2, "the analytics stage found hubs"
    for page in structure:
        mine = recomputed[page.url]
        assert (mine.inbound, mine.outbound, mine.is_orphan, mine.is_hub_pillar) == (
            page.inbound,
            page.outbound,
            page.is_orphan,
            page.is_hub_pillar,
        ), page.url
        assert mine.page_rank_percentile == pytest.approx(page.page_rank_percentile), page.url
    assert adjusted_rand_score(
        [page.link_community_id for page in structure],
        [recomputed[page.url].link_community_id for page in structure],
    ) == pytest.approx(1.0)


# ── not applicable ──────────────────────────────────────────────────────────


@pytest.mark.integration
async def test_a_tenant_without_gsc_or_keywords_gets_every_other_check(
    graph: GraphRepo, mongo: MongoRepo, tenant: str, tmp_path: Path
) -> None:
    await seed_quality(graph, mongo, tenant, gsc=False, keywords=False)

    voyage = keyword_voyage()

    report = await evaluate_quality(graph, mongo, tenant, cache_dir=tmp_path, voyage=client(voyage))

    assert report.not_applicable == KEYWORD_SUB_CHECKS
    assert report.keywords is not None
    assert (report.keywords.pages, report.keywords.resolved, report.keywords.unique_share) == (
        24,
        0,
        None,
    )
    assert all(getattr(report, name) is not None for name in EVERY_SECTION)
    assert report.keywords.relevance_reason == "no ranked keyword sets"
    assert voyage.call_count == 0
    assert report.coverage.gsc_pair_share == 0.0
    assert report.coverage.keyword_page_share == 0.0
    assert entries(tmp_path) == [], "nothing to embed, nothing cached"
    metrics = quality_metrics(report)
    assert "keyword_unique_share" not in metrics
    assert not [name for name in metrics if name.startswith(("extract_", "anchor_match"))]


@pytest.mark.integration
async def test_without_a_voyage_client_only_keyword_relevance_is_not_applicable(
    graph: GraphRepo, mongo: MongoRepo, tenant: str, tmp_path: Path
) -> None:
    await seed_quality(graph, mongo, tenant)
    # The tenant's own generic anchors: the topic-only anchors now say nothing.
    await mongo.set_anchor_rules(
        tenant, AnchorRules(generic_add=tuple(f"{topic} gear" for topic in TOPICS))
    )

    report = await evaluate_quality(graph, mongo, tenant, cache_dir=tmp_path)

    assert report.not_applicable == ("keyword_relevance",)
    assert report.keywords is not None
    assert report.keywords.relevance_reason == "no Voyage API key"
    anchors = report.keywords.anchors
    assert anchors is not None
    assert (anchors.anchors, anchors.primary) == (24, 1.0)
    assert entries(tmp_path) == []


@pytest.mark.integration
async def test_extractability_reads_the_stored_keyword_edges_not_the_dry_plan(
    graph: GraphRepo, mongo: MongoRepo, tenant: str, tmp_path: Path
) -> None:
    await seed_quality(graph, mongo, tenant)
    await graph._auto("MATCH (:Page {tenantId: $t})-[e:TARGETS_KEYWORD]->() DELETE e", t=tenant)

    report = await evaluate_quality(
        graph, mongo, tenant, cache_dir=tmp_path, voyage=client(keyword_voyage())
    )

    assert report.not_applicable == ("keyword_extractability",)
    assert report.keywords is not None
    assert report.keywords.resolved == 24, "the dry plan still resolves every page"
    assert report.keywords.anchors is not None


@pytest.mark.integration
async def test_pages_embedded_with_another_model_leave_keyword_relevance_not_applicable(
    graph: GraphRepo, mongo: MongoRepo, tenant: str, tmp_path: Path
) -> None:
    await seed_quality(graph, mongo, tenant)
    await graph._auto(
        "MATCH (p:Page {tenantId: $t}) SET p.embeddingModel = 'voyage-3-large'", t=tenant
    )
    voyage = keyword_voyage()

    with capture_logs() as logs:
        report = await evaluate_quality(
            graph, mongo, tenant, cache_dir=tmp_path, voyage=client(voyage)
        )

    reason = "page vectors are from voyage-3-large, not voyage-4-large"
    assert report.not_applicable == ("keyword_relevance",)
    assert report.keywords is not None
    assert report.keywords.relevance_reason == reason
    assert voyage.call_count == 0, "keywords were embedded for vectors of another model"
    [line] = [entry for entry in logs if entry["event"] == "quality.keyword_relevance_skipped"]
    assert (line["stage"], line["tenant_id"], line["reason"]) == ("quality-eval", tenant, reason)


@pytest.mark.integration
async def test_an_empty_tenant_is_reported_not_failed(
    graph: GraphRepo, mongo: MongoRepo, tenant: str, tmp_path: Path
) -> None:
    report = await evaluate_quality(
        graph, mongo, tenant, cache_dir=tmp_path, voyage=client(keyword_voyage())
    )

    assert report.not_applicable == EVERY_SECTION
    assert report.coverage.model_dump() == {
        "pairs": 0,
        "gsc_pair_share": None,
        "keyword_page_share": None,
        "all_null_columns": (),
        "constant_columns": (),
    }
    assert quality_metrics(report)["not_applicable_checks"] == 5.0
    assert entries(tmp_path) == []


@pytest.mark.integration
@pytest.mark.parametrize(
    ("failure", "reason"),
    [
        pytest.param(
            ServiceUnavailableError("down", http_status=503),
            "Voyage unavailable after retries (ServiceUnavailableError)",
            id="outage",
        ),
        pytest.param(
            AuthenticationError("Provided API key is invalid.", http_status=401),
            "Voyage refused the key (AuthenticationError)",
            id="bad-key",
        ),
    ],
)
async def test_a_voyage_failure_leaves_only_keyword_relevance_not_applicable(
    graph: GraphRepo,
    mongo: MongoRepo,
    tenant: str,
    tmp_path: Path,
    failure: Exception,
    reason: str,
) -> None:
    await seed_quality(graph, mongo, tenant)
    voyage = keyword_voyage()
    voyage.failures = [failure]

    with capture_logs() as logs:
        report = await evaluate_quality(
            graph,
            mongo,
            tenant,
            cache_dir=tmp_path,
            # One request at a time, so the failure is provably the only call made.
            voyage=client(voyage, settings(max_attempts=1, max_concurrent_requests=1)),
        )

    assert voyage.call_count == 1
    assert report.not_applicable == ("keyword_relevance",)
    assert report.keywords is not None
    assert report.keywords.relevance_reason == reason
    assert all(getattr(report, name) is not None for name in EVERY_SECTION)
    assert report.keywords.extractability is not None
    assert report.keywords.anchors is not None
    [line] = [entry for entry in logs if entry["event"] == "quality.keyword_relevance_skipped"]
    assert (line["stage"], line["reason"]) == ("quality-eval", reason)
    assert entries(tmp_path) == [], "a failed embedding wrote a cache"


@pytest.mark.integration
async def test_the_like_for_like_bar_scores_only_the_columns_hiding_a_link_leaves_alone(
    graph: GraphRepo, mongo: MongoRepo, tenant: str, tmp_path: Path
) -> None:
    await seed_quality(graph, mongo, tenant)
    content_only = ScorerWeights(
        version="content-only",
        features=(FeatureWeight(column="content_cosine", weight=1.0, normalisation="percentile"),),
    )
    await mongo.set_scorer_weights(tenant, content_only)

    clean = await evaluate_quality(graph, mongo, tenant, cache_dir=tmp_path)

    scorer, signal = clean.scorer, clean.feature_signal
    assert scorer is not None
    assert signal is not None
    cosine = {entry.column: entry for entry in signal.columns}["content_cosine"]
    # One column, no link-derived weight: the like-for-like scorer is the scorer, and a
    # percentile of one column orders the pairs as the column does.
    assert scorer.link_derived_weight_share == 0.0
    assert scorer.score_auc_excl_link_counts == pytest.approx(scorer.score_auc)
    assert scorer.score_auc == pytest.approx(cosine.ranker_auc)
    assert scorer.score_auc_lift_excl_link_counts == pytest.approx(
        scorer.score_auc - scorer.best_feature_auc_excl_link_counts
    )

    links_only = ScorerWeights(
        version="links-only",
        features=(
            FeatureWeight(column="target_inbound_count", weight=0.5, direction="lower"),
            FeatureWeight(column="source_outbound_count", weight=0.5, direction="lower"),
        ),
    )
    await mongo.set_scorer_weights(tenant, links_only)

    inflated = await evaluate_quality(graph, mongo, tenant, cache_dir=tmp_path)

    scorer = inflated.scorer
    assert scorer is not None
    assert scorer.link_derived_weight_share == 1.0
    assert (scorer.score_auc_excl_link_counts, scorer.score_auc_lift_excl_link_counts) == (
        None,
        None,
    )
    metrics = quality_metrics(inflated)
    assert "score_auc_excl_link_counts" not in metrics
    assert metrics["link_derived_weight_share"] == 1.0
    assert scorer.best_feature_excl_link_counts not in LINK_DERIVED_COLUMNS


# ── tenant isolation ────────────────────────────────────────────────────────


@pytest.mark.integration
async def test_tenants_never_share_data_or_keyword_vectors(
    graph: GraphRepo, mongo: MongoRepo, tenant: str, tmp_path: Path
) -> None:
    other = f"{tenant}-other"
    await seed_quality(graph, mongo, other, gsc=False)
    await seed_quality(graph, mongo, tenant)
    await evaluate_quality(
        graph, mongo, tenant, cache_dir=tmp_path, voyage=client(keyword_voyage())
    )
    voyage = keyword_voyage()

    theirs = await evaluate_quality(graph, mongo, other, cache_dir=tmp_path, voyage=client(voyage))

    assert theirs.keywords is not None
    assert theirs.keywords.relevance is not None
    # Its 24 H1 keywords: 23 of them are texts this tenant has cached.
    assert (theirs.keywords.relevance.embedded, theirs.keywords.relevance.cached) == (24, 0), (
        "the other tenant reused this tenant's keyword vectors"
    )
    assert entries(tmp_path) == sorted([tenant, other])
    assert theirs.coverage.gsc_pair_share == 0.0, "the other tenant saw this tenant's GSC data"
    assert theirs.keywords.by_rung[KeywordRung.STRATEGIC] == 0
    assert theirs.retrieval is not None
    assert theirs.retrieval.body_link_pairs == 72
