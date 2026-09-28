"""Feature assembly: per-page contexts, pair features, the matrix encoding and its report.

Expected values are worked out in the tests from small hand-built tenants: saturation from
decile medians, hub coverage from who links to whom, the CTR gap from a fixed curve.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError

from linking_engine import gsc
from linking_engine.discovery import features, signals
from linking_engine.discovery.features import (
    CHUNK_PAIRS,
    FEATURE_COLUMNS,
    KEY_COLUMNS,
    POSITION_BANDS,
    PageContext,
    cache_key,
    code_digest,
    feature_chunks,
    feature_report,
    missing_pages,
    page_contexts,
    pair_features,
    position_band,
    summarise_features,
    to_frame,
)
from linking_engine.discovery.signals import build_page_signals
from linking_engine.models import (
    ClusterAgreement,
    CommunityContext,
    CtrCurve,
    GscMetrics,
    PageSignals,
    PageStructure,
    PairFeatures,
    StrategicKeyword,
    TargetCandidates,
)

# A deliberately low-entropy stand-in for a sha256 cache key.
CACHE_KEY = "f" * 64

CURVE = CtrCurve(ctr=(0.3, 0.2, 0.1, 0.05), rows=100, impressions=10_000)
CLUSTER_FIELDS = ("link_community_id", "keyword_community_id", "content_community_id", "hub_id")


def url(name: str) -> str:
    return f"example.com/{name}"


def page(
    name: str,
    *,
    queries: tuple[str, ...] = (),
    keywords: tuple[str, ...] = (),
    gap: int = 0,
    **fields: Any,
) -> tuple[PageStructure, PageSignals]:
    """A page's structure and signals, agreeing on its cluster ids."""
    structure = PageStructure.model_validate(
        {"url": url(name), "inbound": 0, "outbound": 0, **fields}
    )
    signals = PageSignals(
        url=url(name),
        queries=frozenset(queries),
        keywords=frozenset(keywords),
        keyword_gap=gap,
        **{f: fields.get(f) for f in CLUSTER_FIELDS},
    )
    return structure, signals


def contexts(
    pages: list[tuple[PageStructure, PageSignals]],
    links: list[tuple[str, str]] | None = None,
    metrics: list[GscMetrics] | None = None,
    keywords: list[StrategicKeyword] | None = None,
    curve: CtrCurve | None = CURVE,
) -> dict[str, PageContext]:
    return page_contexts(
        [s for s, _ in pages],
        {s.url: signals for s, signals in pages},
        links or [],
        metrics or [],
        keywords or [],
        curve,
    )


def metric(
    name: str, impressions: int, clicks: int, position: float | None, queries: int = 3
) -> GscMetrics:
    return GscMetrics(
        url=url(name),
        impressions_28d=impressions,
        clicks_28d=clicks,
        avg_position=position,
        query_count=queries,
    )


def candidates(pairs: dict[str, list[tuple[str, float]]]) -> list[TargetCandidates]:
    return [
        TargetCandidates(
            target_url=url(target),
            sources=tuple(url(s) for s, _ in hits),
            similarities=tuple(sim for _, sim in hits),
            eligible=len(hits),
            linked=0,
            linked_nearer=0,
        )
        for target, hits in pairs.items()
    ]


# ── columns and bands ───────────────────────────────────────────────────────


def test_the_constants_are_the_contracts() -> None:
    assert (CHUNK_PAIRS, POSITION_BANDS, KEY_COLUMNS) == (
        50_000,
        (3, 10, 20, 50),
        ("source_url", "target_url"),
    )


def test_every_feature_has_a_column_and_cluster_ids_have_none() -> None:
    ids = {f"{side}_{field}" for side in ("source", "target") for field in CLUSTER_FIELDS}
    categorical = {"target_position_band", "cluster_agreement", "content_agreement"}
    plain = set(PairFeatures.model_fields) - ids - categorical - set(KEY_COLUMNS)

    assert len(set(FEATURE_COLUMNS)) == len(FEATURE_COLUMNS)
    assert not ids & set(FEATURE_COLUMNS), "per-run cluster labels must stay out of the matrix"
    assert not set(KEY_COLUMNS) & set(FEATURE_COLUMNS)
    assert plain <= set(FEATURE_COLUMNS), sorted(plain - set(FEATURE_COLUMNS))
    one_hot = set(FEATURE_COLUMNS) - plain
    assert one_hot == {
        *(f"target_position_band_{band}" for band in (0, 1, 2, 3, 4, "null")),
        *(f"cluster_agreement_{a.value.lower()}" for a in ClusterAgreement),
        *(f"content_agreement_{a.value.lower()}" for a in ClusterAgreement),
    }


def test_the_columns_follow_the_model_field_by_field() -> None:
    """A PairFeatures field added without a column, or a column without a field, fails here."""
    ids = [f"{side}_{field}" for side in ("source", "target") for field in CLUSTER_FIELDS]
    levels = {
        "target_position_band": [*range(len(POSITION_BANDS) + 1), "null"],
        "cluster_agreement": [a.value.lower() for a in ClusterAgreement],
        "content_agreement": [a.value.lower() for a in ClusterAgreement],
    }
    expected = [
        column
        for field in PairFeatures.model_fields
        if field not in (*KEY_COLUMNS, *ids)
        for column in (
            [f"{field}_{level}" for level in levels[field]] if field in levels else [field]
        )
    ]

    assert (len(ids), len(KEY_COLUMNS)) == (8, 2)
    assert tuple(expected) == FEATURE_COLUMNS
    assert len(FEATURE_COLUMNS) == 46


def test_the_code_digest_hashes_the_modules_that_compute_features() -> None:
    expected = hashlib.sha256()
    for module in (features, signals, gsc):
        assert module.__file__ is not None
        expected.update(Path(module.__file__).read_bytes())
    assert code_digest() == expected.hexdigest()


def test_the_cache_key_changes_with_the_feature_code(monkeypatch: pytest.MonkeyPatch) -> None:
    pages, targets = chain(3)
    before = cache_key("acme", targets, pages)

    monkeypatch.setattr(features, "code_digest", lambda: "0" * 64)

    assert cache_key("acme", targets, pages) != before


@pytest.mark.parametrize(
    ("position", "band"),
    [
        (None, None),
        (1, 0),
        (3, 0),
        (3.4, 0),
        (3.5, 1),
        (4, 1),
        (10, 1),
        (10.5, 2),
        (20, 2),
        (21, 3),
        (50, 3),
        (50.5, 4),
        (51, 4),
        (180, 4),
    ],
)
def test_positions_fall_into_bands_rounded_like_the_ctr_curve(
    position: float | None, band: int | None
) -> None:
    assert position_band(position) == band


def test_a_position_below_one_is_refused() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        position_band(0.5)


# ── page_contexts ───────────────────────────────────────────────────────────


def test_saturation_is_inbound_over_the_median_of_the_pagerank_decile() -> None:
    pages = [
        page(f"top{i}", inbound=n, page_rank_percentile=0.95)
        for i, n in enumerate((2, 4, 6, 8, 10))
    ]
    pages += [
        page(f"low{i}", inbound=n, page_rank_percentile=0.05) for i, n in enumerate((0, 0, 3))
    ]
    pages += [page("unranked", inbound=4)]

    found = contexts(pages)

    top = [found[url(f"top{i}")].saturation_ratio for i in range(5)]
    assert top == pytest.approx([2 / 6, 4 / 6, 1.0, 8 / 6, 10 / 6])
    # A decile whose median is 0 takes the inbound count itself.
    assert [found[url(f"low{i}")].saturation_ratio for i in range(3)] == [0.0, 0.0, 3.0]
    assert found[url("unranked")].saturation_ratio == 1.0


def test_hub_coverage_is_the_share_of_the_targets_other_hub_members_linking_to_it() -> None:
    pages = [page(f"h{i}", hub_id=1) for i in range(4)]
    pages += [page("lone", hub_id=2), page("noise", hub_id=-1), page("unhubbed")]
    links = [
        (url("h1"), url("h0")),
        (url("h1"), url("h0")),
        (url("h2"), url("h0")),
        (url("h0"), url("h1")),
        (url("h0"), url("h0")),
        (url("lone"), url("h0")),
        (url("noise"), url("h3")),
        (url("gone"), url("h3")),
    ]

    found = contexts(pages, links)

    coverage = {name: found[url(name)].hub_coverage for name in ("h0", "h1", "h2", "h3")}
    assert coverage == pytest.approx({"h0": 2 / 3, "h1": 1 / 3, "h2": 0.0, "h3": 0.0})
    assert {found[url(f"h{i}")].hub_size for i in range(4)} == {4}
    assert (found[url("lone")].hub_size, found[url("lone")].hub_coverage) == (1, 0.0)
    for outside in ("noise", "unhubbed"):
        assert (found[url(outside)].hub_size, found[url(outside)].hub_coverage) == (None, None)


def test_gsc_metrics_become_log_impressions_a_band_and_the_ctr_gap() -> None:
    pages = [page("ranked"), page("unranked"), page("no-position"), page("no-impressions")]
    metrics = [
        metric("ranked", 999, 150, 3.0, queries=7),
        metric("no-position", 500, 20, None),
        metric("no-impressions", 0, 0, 12.0),
        metric("elsewhere", 5000, 50, 2.0),
    ]

    found = contexts(pages, metrics=metrics)

    ranked = found[url("ranked")]
    assert ranked.has_gsc_data is True
    assert ranked.impressions_log == pytest.approx(math.log(1000))
    assert (ranked.position_band, ranked.query_count) == (0, 7)
    # 150 / 999 against the curve's 0.1 at position 3.
    assert ranked.ctr_gap == pytest.approx(150 / 999 - 0.1)
    unranked = found[url("unranked")]
    assert (unranked.has_gsc_data, unranked.impressions_log, unranked.position_band) == (
        False,
        None,
        None,
    )
    assert (unranked.ctr_gap, unranked.query_count) == (None, None)
    assert (found[url("no-position")].position_band, found[url("no-position")].ctr_gap) == (
        None,
        None,
    )
    assert found[url("no-impressions")].ctr_gap is None


def test_without_a_ctr_curve_there_is_no_ctr_gap() -> None:
    found = contexts([page("ranked")], metrics=[metric("ranked", 999, 150, 3.0)], curve=None)
    assert (found[url("ranked")].has_gsc_data, found[url("ranked")].ctr_gap) == (True, None)


def test_the_highest_strategic_priority_of_the_page_is_kept() -> None:
    def keyword(name: str, text: str, priority: int | None) -> StrategicKeyword:
        return StrategicKeyword(url=url(name), keyword=text, language="en", priority=priority)

    keywords = [
        keyword("a", "tents", 2),
        keyword("a", "stoves", 5),
        keyword("a", "boots", None),
        keyword("b", "boots", None),
        keyword("gone", "tents", 5),
    ]

    found = contexts([page("a"), page("b"), page("c")], keywords=keywords)

    assert [found[url(n)].max_priority for n in ("a", "b", "c")] == [5, None, None]


@pytest.mark.parametrize(
    ("stored", "inbound", "orphan"),
    [(None, 0, True), (None, 2, False), (True, 2, True), (False, 0, False)],
)
def test_an_unknown_orphan_flag_falls_back_to_having_no_inbound_link(
    stored: bool | None, inbound: int, orphan: bool
) -> None:
    found = contexts([page("p", is_orphan=stored, inbound=inbound)])
    assert found[url("p")].is_orphan is orphan


@pytest.mark.parametrize(
    ("outbound", "words", "density", "share"),
    [(3, 1500, 2.0, 0.25), (0, 800, 0.0, 1.0), (4, 0, 4000.0, 0.2)],
)
def test_outbound_density_is_per_thousand_words_and_equity_divides_by_outbound_plus_one(
    outbound: int, words: int, density: float, share: float
) -> None:
    found = contexts([page("p", outbound=outbound, word_count=words)])[url("p")]
    assert (found.outbound_density, found.link_equity_share) == pytest.approx((density, share))


def test_structure_and_signals_must_describe_the_same_pages() -> None:
    a, b = page("a"), page("b")
    with pytest.raises(ValueError, match="differ on 1 pages"):
        page_contexts([a[0], b[0]], {a[1].url: a[1]}, [], [], [], CURVE)


def test_structure_and_signals_must_agree_on_the_clusters() -> None:
    structure, _ = page("a", hub_id=3)
    _, other = page("a", hub_id=4)
    with pytest.raises(ValueError, match="disagree on the clusters"):
        page_contexts([structure], {other.url: other}, [], [], [], CURVE)


def test_duplicate_pages_and_metrics_are_refused() -> None:
    a = page("a")
    with pytest.raises(ValueError, match="duplicate page urls"):
        page_contexts([a[0], a[0]], {a[1].url: a[1]}, [], [], [], CURVE)
    with pytest.raises(ValueError, match="duplicate GSC metrics"):
        contexts([a], metrics=[metric("a", 10, 1, 2.0), metric("a", 20, 1, 3.0)])


# ── pair_features ───────────────────────────────────────────────────────────


def tenant() -> dict[str, PageContext]:
    pages = [
        page(
            "s",
            outbound=9,
            word_count=3000,
            hub_id=1,
            is_hub_pillar=True,
            link_community_id=0,
            keyword_community_id=2,
            content_community_id=4,
            queries=("tents", "stoves"),
            keywords=("tents",),
        ),
        page(
            "t",
            inbound=2,
            crawl_depth=3,
            page_rank_percentile=0.42,
            hub_id=1,
            link_community_id=1,
            keyword_community_id=2,
            content_community_id=5,
            queries=("tents",),
            keywords=("tents", "boots"),
            gap=1,
        ),
        page("noise", hub_id=-1),
        page("other-noise", hub_id=-1),
    ]
    metrics = [metric("t", 999, 150, 3.0, queries=7)]
    return contexts(pages, [(url("s"), url("t"))], metrics)


def test_a_pair_takes_target_features_from_the_target_and_source_ones_from_the_source() -> None:
    pages = tenant()

    found = pair_features(pages[url("s")], pages[url("t")], 0.81)

    assert (found.source_url, found.target_url, found.content_cosine) == (url("s"), url("t"), 0.81)
    assert (found.has_gsc_data, found.target_position_band, found.target_query_count) == (
        True,
        0,
        7,
    )
    assert found.target_impressions_log == pytest.approx(math.log(1000))
    assert found.source_impressions_log is None
    assert found.pair_query_overlap == pytest.approx(0.5)
    assert found.pair_kw_overlap == pytest.approx(0.5)
    assert (found.target_kw_count, found.target_keyword_gap) == (2, 1)
    assert (found.target_inbound_count, found.target_crawl_depth) == (2, 3)
    assert found.target_page_rank_percentile == 0.42
    assert (found.source_outbound_count, found.source_link_equity_share) == (9, 0.1)
    assert found.source_outbound_density == pytest.approx(3.0)
    assert (found.same_hub, found.source_is_hub_pillar, found.target_is_hub_pillar) == (
        True,
        True,
        False,
    )
    # Hub 1 is s and t; s links to t, so all of t's other members already do.
    assert (found.source_hub_size, found.target_hub_size, found.target_hub_coverage) == (2, 2, 1.0)
    assert (found.same_link_community, found.same_keyword_community) == (False, True)
    assert found.same_content_community is False
    assert found.cluster_agreement is ClusterAgreement.SAME_TOPIC_OTHER_LINKS
    assert found.content_agreement is ClusterAgreement.OTHER_TOPIC_OTHER_LINKS
    assert (found.source_hub_id, found.target_link_community_id) == (1, 1)
    assert (found.context_relevance, found.anchor_target_fit) == (None, None)


def test_two_noise_pages_are_never_in_the_same_hub() -> None:
    pages = tenant()
    found = pair_features(pages[url("noise")], pages[url("other-noise")], 0.2)
    assert (found.same_hub, found.target_hub_size, found.target_hub_coverage) == (False, None, None)


def signal_pair(queries: list[tuple[str, str]]) -> PairFeatures:
    """s -> t through build_page_signals; t targets the strategic keywords tents and boots."""
    structure = [PageStructure(url=url(n), inbound=0, outbound=0) for n in ("s", "t")]
    signals = build_page_signals(
        [CommunityContext(url=url(n)) for n in ("s", "t")],
        queries,
        [(url("t"), "Tents"), (url("t"), "boots")],
    )
    pages = page_contexts(structure, signals, [], [], [], None)
    return pair_features(pages[url("s")], pages[url("t")], 0.5)


@pytest.mark.parametrize(
    ("queries", "gap"),
    [
        pytest.param([], None, id="tenant-without-gsc-rows"),
        pytest.param([(url("s"), "tents")], 2, id="target-without-queries"),
        pytest.param([(url("t"), "TENTS")], 1, id="one-keyword-ranks"),
        pytest.param([(url("gone"), "tents")], 2, id="gsc-rows-only-elsewhere"),
    ],
)
def test_the_keyword_gap_is_unknown_only_when_the_tenant_has_no_gsc_rows(
    queries: list[tuple[str, str]], gap: int | None
) -> None:
    assert signal_pair(queries).target_keyword_gap == gap


# ── chunks, frame, cache key ────────────────────────────────────────────────


def chain(count: int) -> tuple[dict[str, PageContext], list[TargetCandidates]]:
    pages = contexts([page(f"p{i}") for i in range(count)])
    targets = candidates(
        {
            f"p{t}": [(f"p{s}", round(0.9 - s * 0.01, 2)) for s in range(count) if s != t]
            for t in range(count)
        }
    )
    return pages, targets


def test_pairs_come_in_chunks_in_candidate_order() -> None:
    pages, targets = chain(3)

    chunks = list(feature_chunks(targets, pages, chunk_pairs=4))

    assert [len(c) for c in chunks] == [4, 2]
    flat = [(f.source_url, f.target_url, f.content_cosine) for c in chunks for f in c]
    assert flat == [
        (s, t.target_url, sim)
        for t in targets
        for s, sim in zip(t.sources, t.similarities, strict=True)
    ]


def test_chunking_refuses_a_zero_size_and_unknown_urls() -> None:
    pages, targets = chain(3)
    with pytest.raises(ValueError, match="chunk_pairs"):
        list(feature_chunks(targets, pages, chunk_pairs=0))
    stray = candidates({"p0": [("ghost", 0.5)]})
    assert missing_pages([*targets, *stray], pages) == [url("ghost")]
    with pytest.raises(ValueError, match="ghost"):
        list(feature_chunks(stray, pages))


def test_the_frame_is_keys_then_features_as_floats() -> None:
    pages = tenant()
    rows = [
        pair_features(pages[url("s")], pages[url("t")], 0.81),
        pair_features(pages[url("noise")], pages[url("s")], 0.2),
    ]

    frame = to_frame(rows)

    assert tuple(frame.columns) == (*KEY_COLUMNS, *FEATURE_COLUMNS)
    assert list(frame["source_url"]) == [url("s"), url("noise")]
    assert all(frame[c].dtype == np.float64 for c in FEATURE_COLUMNS)
    assert list(frame["same_hub"]) == [1.0, 0.0]
    assert list(frame["has_gsc_data"]) == [1.0, 0.0]
    assert np.isnan(frame["target_ctr_gap"].iloc[1])
    assert np.isnan(frame["context_relevance"]).all()
    bands = frame[[c for c in FEATURE_COLUMNS if c.startswith("target_position_band_")]]
    assert list(bands.iloc[0]) == [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert list(bands.iloc[1]) == [0.0, 0.0, 0.0, 0.0, 0.0, 1.0], "no position is its own bucket"
    for prefix in ("cluster_agreement_", "content_agreement_"):
        one_hot = frame[[c for c in FEATURE_COLUMNS if c.startswith(prefix)]]
        assert list(one_hot.sum(axis=1)) == [1.0, 1.0]
    assert frame.loc[0, "cluster_agreement_same_topic_other_links"] == 1.0


def test_an_empty_frame_keeps_every_column() -> None:
    assert tuple(to_frame([]).columns) == (*KEY_COLUMNS, *FEATURE_COLUMNS)


def test_the_cache_key_is_stable_and_moves_with_every_input() -> None:
    pages, targets = chain(3)
    key = cache_key("acme", targets, pages)

    assert key == cache_key("acme", targets, dict(reversed(pages.items())))
    assert key != cache_key("globex", targets, pages), "one tenant's cache never serves another"
    assert key != cache_key("acme", targets[:2], pages)
    moved = [targets[0].model_copy(update={"similarities": (0.5, 0.4)}), *targets[1:]]
    assert key != cache_key("acme", moved, pages)
    changed = dict(pages)
    first = changed[url("p0")]
    changed[url("p0")] = first.model_copy(
        update={"signals": first.signals.model_copy(update={"queries": frozenset({"tents"})})}
    )
    assert key != cache_key("acme", targets, changed)


# ── feature_report and summary ──────────────────────────────────────────────


def test_the_report_names_all_null_constant_and_partly_null_columns() -> None:
    pages = tenant()
    frames = [
        to_frame([pair_features(pages[url("s")], pages[url("t")], 0.81)]),
        to_frame([pair_features(pages[url("noise")], pages[url("s")], 0.81)]),
        to_frame([]),
    ]

    report = feature_report("acme", frames, cache_key="abc123", cache_hit=False, started=0.0)

    assert (report.tenant_id, report.pairs, report.chunks, report.columns) == (
        "acme",
        2,
        3,
        FEATURE_COLUMNS,
    )
    assert {"context_relevance", "anchor_target_fit"} <= set(report.all_null_columns)
    assert "content_cosine" in report.constant_columns
    assert "source_outbound_count" not in report.constant_columns
    # One value and one null: not constant, because trees split on the null.
    assert "target_ctr_gap" not in report.constant_columns
    assert report.null_share["target_ctr_gap"] == 0.5
    assert report.null_share["content_cosine"] == 0.0
    assert report.has_gsc_data_share == 0.5
    assert (report.cache_key, report.cache_hit) == ("abc123", False)


def test_an_empty_run_reports_no_shares() -> None:
    report = feature_report("acme", [], cache_key="k", cache_hit=True, started=0.0)

    assert (report.pairs, report.chunks, report.null_share, report.has_gsc_data_share) == (
        0,
        0,
        {},
        None,
    )
    assert report.all_null_columns == report.constant_columns == ()
    assert "No candidate pairs" in summarise_features(report)


def test_the_summary_names_the_gaps() -> None:
    pages = tenant()
    rows = [pair_features(pages[url("s")], pages[url("t")], 0.81)]
    report = feature_report(
        "acme", [to_frame(rows)], cache_key=CACHE_KEY, cache_hit=True, started=0.0
    )

    summary = summarise_features(report)

    for fact in (
        "acme",
        "1 candidate pairs",
        "read from the cache",
        CACHE_KEY[:12],
        "context_relevance",
    ):
        assert fact in summary, f"{fact!r} missing from:\n{summary}"


def test_a_page_context_describes_one_page() -> None:
    structure, _ = page("a")
    _, other = page("b")
    with pytest.raises(ValidationError, match="the same page"):
        PageContext(
            structure=structure,
            signals=other,
            has_gsc_data=False,
            is_orphan=True,
            saturation_ratio=0.0,
            outbound_density=0.0,
            link_equity_share=1.0,
        )
