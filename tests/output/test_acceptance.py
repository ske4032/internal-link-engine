"""Acceptance of #89: the tenant's output is assembled, stored per run and served by the API,
one tenant at a time.

Two tenants share every url and differ in content, verdicts and hubs (`fixture`), so any
response that mixes them, or any write that reaches the other, shows. Needs Docker.
"""

from __future__ import annotations

import re
import uuid
from collections import Counter, defaultdict
from contextlib import suppress
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, cast

import pyarrow.parquet as pq
import pytest
from fixture import (
    ALPHA,
    AUDITED,
    BETA,
    BRIDGED,
    DATABASE,
    FULL,
    GAP_LIMIT,
    KEYWORDLESS,
    KIT_CANONICAL,
    KIT_COPY,
    KIT_DEAD_END,
    KIT_ORPHAN,
    KIT_URLS,
    LIMIT,
    LISTINGS,
    MIXED,
    NO_KEYWORD,
    NOT_FOUND,
    PREFIX,
    ROUTES,
    UNAUTHORISED,
    URLS,
    Params,
    Planted,
    ServedOutput,
    anchor_sentences,
    expected_state,
    fetch,
    finish_run,
    kit_hub,
    open_api,
    pillars,
    plan_inputs,
    plant_inputs,
    publish_run,
    served_output,
    served_state,
    stage_run,
    walk,
    write_served,
)
from mlflow import MlflowClient
from mlflow.artifacts import load_text
from pydantic import BaseModel

from linking_engine.discovery.features import code_digest
from linking_engine.discovery.scoring import TOP_CONTRIBUTIONS
from linking_engine.ml.tracking import log_recommendations
from linking_engine.models import (
    AUDIT_ACTIONS,
    NEW_LINK_ACTIONS,
    ActionType,
    AnchorMix,
    AnchorType,
    BridgeMark,
    BridgePair,
    ContentGapFinding,
    DuplicateGroup,
    ExclusionReason,
    HubSummary,
    KeywordRung,
    OrphanLabel,
    PageDetail,
    PageProfile,
    PageType,
    Recommendation,
    RecommendationReport,
    RunInfo,
    ScorerName,
    SiteSummary,
    TargetFix,
    UnanchoredOut,
    UnanchoredReason,
)
from linking_engine.models.anchors import UNANCHORED_ADVICE
from linking_engine.output.collections import (
    EXCLUDED_PAGES,
    RECOMMENDATIONS,
    RUN_SCOPED,
    RUNS,
)
from linking_engine.output.keys import KeyStore
from linking_engine.output.writer import OutputWriter
from linking_engine.pipeline.anchor_selection import UNANCHORED_FILE
from linking_engine.pipeline.bridges import BRIDGES_FILE, HUB_PAIRS_FILE
from linking_engine.pipeline.features import ANCHOR_CHOICES_FILE
from linking_engine.pipeline.ranker import RANKED_PAIRS_FILE
from linking_engine.pipeline.recommendations import (
    publish_recommendations,
    summarise_recommendations,
)
from linking_engine.urls import normalise_url

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator
    from pathlib import Path

    import pandas

    from linking_engine.graph.repo import GraphRepo
    from linking_engine.ingest.mongo_repo import MongoRepo

pytestmark = pytest.mark.integration


def new_tenant() -> str:
    return f"test-{uuid.uuid4().hex[:12]}"


def run_id() -> str:
    return uuid.uuid4().hex


@pytest.fixture
def other_tenant() -> str:
    return new_tenant()


def probe(route: str, output: ServedOutput) -> tuple[str, dict[str, str]]:
    """A request for the tenant's own data on the route, url filters included."""
    path = (PREFIX + route).format(
        tenant=output.tenant_id, recommendation_id=output.recommendations[0].id
    )
    params = {
        "/recommendations": {"source": URLS[0], "target": URLS[1]},
        "/page": {"url": URLS[0]},
        "/unanchored": {"target": KEYWORDLESS},
        "/pages": {"hub": str(output.pages[0].hub_id)},
        "/bridges": {"hub": str(output.bridges[0].hub_a)},
    }.get(route, {})
    return path, params


async def tenant_documents(mongo: MongoRepo, tenant: str) -> dict[str, list[dict[str, object]]]:
    """Every output document of the tenant, by collection, in insertion order."""
    db = mongo._db
    return {
        name: [doc async for doc in db[name].find({"tenantId": tenant}).sort("_id", 1)]
        for name in (*RUN_SCOPED, RUNS, EXCLUDED_PAGES)
    }


async def test_a_key_never_reaches_another_tenant(
    mongo: MongoRepo, mongo_uri: str, tenant: str, other_tenant: str
) -> None:
    a = served_output(tenant, run_id(), ALPHA)
    b = served_output(other_tenant, run_id(), BETA)
    assert {page.url for page in a.pages} == {page.url for page in b.pages}
    for output in (a, b):
        await write_served(mongo, output)
    keys = KeyStore(mongo._db)
    key_a, info_a = await keys.issue(a.tenant_id, "acceptance")
    key_b, _ = await keys.issue(b.tenant_id, "acceptance")
    revoked, revoked_info = await keys.issue(a.tenant_id, "revoked")
    assert await keys.revoke(a.tenant_id, revoked_info.key_id) is True
    # A key is revoked only through its own tenant.
    assert await keys.revoke(b.tenant_id, info_a.key_id) is False

    async with open_api(mongo_uri) as (app, client):
        served = {path for path in app.openapi()["paths"] if path.startswith(PREFIX)}
        assert served == {PREFIX + route for route in ROUTES}, "the tenant routes changed"

        own = PREFIX.format(tenant=a.tenant_id)
        missing = [
            (f"{own}/recommendations/{'0' * 16}", {}),
            (f"{own}/recommendations/{b.recommendations[0].id}", {}),
            (f"{own}/page", {"url": "example.com/gear/nowhere"}),
            (f"{PREFIX.format(tenant=new_tenant())}/summary", {}),
            (f"{PREFIX.format(tenant=new_tenant())}/excluded-pages", {}),
        ]
        for path, params in missing:
            response = await client.get(path, params=params, headers={"X-API-Key": key_a})
            assert (response.status_code, response.json()) == (404, NOT_FOUND), path

        for route in ROUTES:
            path, params = probe(route, b)
            response = await client.get(path, params=params, headers={"X-API-Key": key_a})
            assert (response.status_code, response.json()) == (404, NOT_FOUND), (
                f"A's key on B's {route} must look like a missing resource"
            )
            response = await client.get(path, params=params, headers={"X-API-Key": key_b})
            assert response.status_code == 200, (route, response.text)
            for headers in ({}, {"X-API-Key": "lek_" + "f" * 43}, {"X-API-Key": revoked}):
                response = await client.get(path, params=params, headers=headers)
                assert (response.status_code, response.json()) == (401, UNAUTHORISED), route

        health = await client.get("/health")
        assert (health.status_code, health.json()) == (200, {"status": "ok"})

        assert await served_state(client, a.tenant_id, key_a) == expected_state(a)
        assert await served_state(client, b.tenant_id, key_b) == expected_state(b)


async def test_a_rerun_replaces_the_output_atomically(
    mongo: MongoRepo, mongo_uri: str, tenant: str, other_tenant: str
) -> None:
    first = served_output(tenant, run_id(), ALPHA)
    second = served_output(tenant, run_id(), ALPHA, revision=2, sources=4)
    crashed = served_output(tenant, run_id(), ALPHA, revision=3)
    fourth = served_output(tenant, run_id(), ALPHA, revision=4, sources=5)
    other = served_output(other_tenant, run_id(), BETA)
    keys = KeyStore(mongo._db)
    key, _ = await keys.issue(tenant, "acceptance")
    db = mongo._db

    async def runs_left(name: str) -> set[str]:
        return set(await db[name].distinct("runId", {"tenantId": tenant}))

    writer = await OutputWriter.connect(mongo_uri, DATABASE)
    try:
        await writer.ensure_indexes()
        for output in (first, other):
            await publish_run(writer, output)
            await mongo.replace_excluded_pages(output.tenant_id, output.excluded)
        untouched = await tenant_documents(mongo, other_tenant)

        await stage_run(writer, second)
        assert await runs_left(RUNS) == {first.run_id, second.run_id}
        async with open_api(mongo_uri) as (_, client):
            assert await served_state(client, tenant, key) == expected_state(first), (
                "a run still writing must not be served"
            )

        await finish_run(writer, second)
        async with open_api(mongo_uri) as (_, client):
            assert await served_state(client, tenant, key) == expected_state(second)
        for name in (*RUN_SCOPED, RUNS):
            assert await runs_left(name) == {second.run_id}, f"{name} keeps another run"

        await stage_run(writer, crashed, only=RECOMMENDATIONS)
        async with open_api(mongo_uri) as (_, client):
            assert await served_state(client, tenant, key) == expected_state(second), (
                "a crashed run must leave the previous one served"
            )

        await publish_run(writer, fourth)
        async with open_api(mongo_uri) as (_, client):
            assert await served_state(client, tenant, key) == expected_state(fourth)
        for name in (*RUN_SCOPED, RUNS):
            assert await runs_left(name) == {fourth.run_id}, f"{name} keeps a crashed run"

        assert await tenant_documents(mongo, other_tenant) == untouched
    finally:
        await writer.close()


# ── The assembled output: the real stage over the planted stage inputs ──────────────────────

LABELS: Final = {
    ActionType.ADD_LINK: "add a link",
    ActionType.CONTENT_GAP: "content gap: add copy first",
    ActionType.FIX: "fix this link",
    ActionType.REANCHOR: "change the anchor text",
    ActionType.REMOVE: "review this link",
}
FINDINGS: Final = {
    UnanchoredReason.SOURCE_DOES_NOT_MENTION_TOPIC: ContentGapFinding.NO_TOPICAL_MENTION,
    UnanchoredReason.TOPIC_MENTIONED_BUT_NO_GOOD_PHRASE: ContentGapFinding.AWKWARD_PHRASING,
}
AUDIT_DIMENSIONS: Final = (
    "anchor_quality_score",
    "keyword_alignment",
    "context_relevance",
    "anchor_target_fit",
    "equity_efficiency",
)
STAGE_FILES: Final = frozenset(
    {RANKED_PAIRS_FILE, ANCHOR_CHOICES_FILE, UNANCHORED_FILE, HUB_PAIRS_FILE, BRIDGES_FILE}
)


@dataclass(frozen=True, slots=True)
class Published:
    """A planted tenant after one run of the stage, a key for it, and the ranked pairs the run
    read."""

    planted: Planted
    report: RecommendationReport
    key: str
    ranked: pandas.DataFrame

    @property
    def tenant_id(self) -> str:
        return self.planted.tenant_id

    def ranked_pairs(self) -> pandas.DataFrame:
        """The ranked pairs of pages still in the pipeline."""
        excluded = sorted(self.planted.excluded_urls)
        keep = ~(
            self.ranked["source_url"].isin(excluded) | self.ranked["target_url"].isin(excluded)
        )
        return self.ranked.loc[keep]


@pytest.fixture
def local_mlflow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """Runs and the model registry go to a throwaway local store, never the remote server."""
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    return uri


async def assemble(
    graph: GraphRepo, mongo: MongoRepo, mongo_uri: str, tenant: str, cache: Path
) -> RecommendationReport:
    async with await OutputWriter.connect(mongo_uri, DATABASE) as writer:
        return await publish_recommendations(graph, mongo, writer, tenant, cache_dir=cache)


@pytest.fixture
async def published(
    graph: GraphRepo,
    mongo: MongoRepo,
    mongo_uri: str,
    tenant: str,
    other_tenant: str,
    tmp_path: Path,
    local_mlflow: str,
) -> tuple[Published, Published]:
    """Both tenants planted over the same urls and assembled by the real stage, in turn."""
    found = []
    for name, variant in ((tenant, ALPHA), (other_tenant, BETA)):
        planted = plan_inputs(name, variant)
        folder = await plant_inputs(graph, mongo, planted, tmp_path / "cache")
        report = await assemble(graph, mongo, mongo_uri, name, tmp_path / "cache")
        key, _ = await KeyStore(mongo._db).issue(name, "acceptance")
        ranked = pq.read_table(folder / RANKED_PAIRS_FILE).to_pandas()
        found.append(Published(planted, report, key, ranked))
    return found[0], found[1]


def strings(value: object) -> Iterator[str]:
    """Every string inside a JSON value."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from strings(item)
    elif isinstance(value, list | tuple):
        for item in value:
            yield from strings(item)


def dumped(value: object) -> object:
    """A served value as JSON."""
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, tuple):
        return [dumped(item) for item in value]
    return value


def urls_in(texts: Iterable[str], urls: frozenset[str]) -> set[str]:
    """The ``urls`` that the texts hold, in any form a url normalises from, or as a bare path."""
    paths = {"/" + url.split("/", 1)[1]: url for url in urls}
    found = set()
    for text in texts:
        for token in re.split(r"[\s\"'(),;:{}\[\]]+", text):
            if token in paths:
                found.add(paths[token])
                continue
            with suppress(ValueError):
                if (key := normalise_url(token)) in urls:
                    found.add(key)
    return found


async def test_recommendations_are_served_paginated_and_deterministic(
    published: tuple[Published, Published],
    graph: GraphRepo,
    mongo: MongoRepo,
    mongo_uri: str,
    tmp_path: Path,
) -> None:
    a, _ = published
    planted = a.planted
    path = f"{PREFIX.format(tenant=a.tenant_id)}/recommendations"
    async with open_api(mongo_uri) as (_, client):
        paged, total = await walk(client, path, a.key, limit=4)
        whole = await fetch(client, path, a.key, limit="500")
        again = await fetch(client, path, a.key, limit="500")
    assert whole == again, "two calls disagree"
    assert isinstance(whole, dict)
    assert (whole["next_cursor"], whole["total"]) == (None, total)
    assert whole["items"] == paged, "walking page by page differs from one big page"
    first = [Recommendation.model_validate(item) for item in paged]
    assert len(first) == total == sum(a.report.summary.recommendations.values()) > 4

    rerun = await assemble(graph, mongo, mongo_uri, a.tenant_id, tmp_path / "cache")
    assert rerun.run_id != a.report.run_id
    async with open_api(mongo_uri) as (_, client):
        items, _ = await walk(client, path, a.key, limit=4)
    second = [Recommendation.model_validate(item) for item in items]
    assert {r.run_id for r in second} == {rerun.run_id}
    assert [r.id for r in second] == [r.id for r in first], "a rerun changed ids or order"
    kept = {"run_id", "created_at"}
    assert [r.model_dump(exclude=kept) for r in second] == [
        r.model_dump(exclude=kept) for r in first
    ]

    for record in first:
        assert record.label == LABELS[record.action_type]
        assert record.rationale.strip(), record.action_type
        assert "_" not in record.rationale, record.rationale
        assert not urls_in([record.rationale], planted.urls), record.rationale
        if record.action_type not in NEW_LINK_ACTIONS:
            continue
        contributions = [abs(value) for _, value in record.signals]
        assert 0 < len(contributions) <= TOP_CONTRIBUTIONS, record.signals
        assert contributions == sorted(contributions, reverse=True)
        pair = (record.source_url, record.target_url)
        if record.action_type is ActionType.CONTENT_GAP:
            reason = planted.unanchored[pair]
            assert (record.finding, record.advice) == (FINDINGS[reason], UNANCHORED_ADVICE[reason])
            continue
        chosen = planted.anchored[pair]
        anchors = record.proposed_anchors or ()
        assert [c.text for c in anchors] == [chosen, f"{chosen} range"], "chosen, then alternatives"
        assert [c.anchor_type for c in anchors] == [AnchorType.EXACT, AnchorType.PARTIAL]
        assert [c.score for c in anchors] == [0.82, 0.64]
        body = planted.body(record.source_url)
        for candidate, sentence in zip(anchors, anchor_sentences(chosen), strict=True):
            placement = candidate.placement
            assert placement is not None
            assert candidate.source == "EXTRACTED"
            assert candidate.keyword == planted.keyword_of(record.target_url)
            assert placement.sentence == sentence
            assert planted.copy[record.source_url][placement.sentence_index] == sentence
            assert body[placement.start : placement.end] == candidate.text, "not a body offset"


async def test_ten_new_links_per_source_and_every_audit_verdict(
    published: tuple[Published, Published], mongo_uri: str
) -> None:
    ids: list[dict[tuple[object, ...], str]] = []
    for pub in published:
        planted = pub.planted
        async with open_api(mongo_uri) as (_, client):
            state = await served_state(client, pub.tenant_id, pub.key)
            for url in sorted(planted.excluded_urls):
                response = await client.get(
                    f"{PREFIX.format(tenant=pub.tenant_id)}/page",
                    params={"url": url},
                    headers={"X-API-Key": pub.key},
                )
                assert (response.status_code, response.json()) == (404, NOT_FOUND), url
        records = cast("tuple[Recommendation, ...]", state["/recommendations"])

        listed: dict[ActionType, defaultdict[str, list[Recommendation]]] = {
            kind: defaultdict(list) for kind in NEW_LINK_ACTIONS
        }
        for record in records:
            if record.action_type in NEW_LINK_ACTIONS:
                listed[record.action_type][record.source_url].append(record)
        links, gaps = listed[ActionType.ADD_LINK], listed[ActionType.CONTENT_GAP]
        assert max(len(found) for found in links.values()) == LIMIT
        assert max(len(found) for found in gaps.values()) == GAP_LIMIT

        # Links first: a page's best-ranked anchored pairs up to the limit, then up to three
        # content gaps from those ranked above its last link.
        ranked = pub.ranked_pairs()
        rank = {
            (s, t): int(n)
            for s, t, n in zip(
                ranked["source_url"], ranked["target_url"], ranked["rank_in_source"], strict=True
            )
        }
        assert {*links, *gaps} <= set(ranked["source_url"])
        exercised: set[str] = set()
        for source, group in ranked.groupby("source_url"):
            order = list(group.sort_values("rank_in_source")["target_url"])
            anchored = [t for t in order if (source, t) in planted.anchored]
            emitted = anchored[:LIMIT]
            last = rank[(source, emitted[-1])] if emitted else None
            candidates = [t for t in order if planted.unanchored.get((source, t)) in FINDINGS]
            eligible = [t for t in candidates if last is None or rank[(source, t)] < last]
            assert [r.target_url for r in links[source]] == emitted, source
            assert [r.target_url for r in gaps[source]] == eligible[:GAP_LIMIT], source
            for kind in (links[source], gaps[source]):
                assert [r.rank_in_source for r in kind] == list(range(1, len(kind) + 1)), source
            if last is not None:
                assert all(rank[(source, r.target_url)] < last for r in gaps[source]), (
                    f"{source}: a content gap ranked below the page's last link"
                )
            exercised |= {
                name
                for name, seen in (
                    ("links capped", len(anchored) > LIMIT),
                    ("gaps capped", len(eligible) > GAP_LIMIT),
                    ("gap below the last link", len(candidates) > len(eligible)),
                    ("gaps without links", last is None and bool(gaps[source])),
                    ("gaps above a link", last is not None and bool(gaps[source])),
                )
                if seen
            }
        assert len(exercised) == 5, f"the fixture only exercises {sorted(exercised)}"

        excluded = planted.excluded_urls
        expected = {
            (result.source_url, result.position): result
            for result in planted.audit
            if result.verdict is not None and not {result.source_url, result.target_url} & excluded
        }
        stale = {*pub.ranked["source_url"], *pub.ranked["target_url"]}
        stale |= {url for result in planted.audit for url in (result.source_url, result.target_url)}
        assert excluded <= stale, "the stage read no excluded page, so none could leak"
        audits = {
            (r.source_url, r.position): r
            for r in records
            if r.action_type in AUDIT_ACTIONS and r.position is not None
        }
        assert audits.keys() == expected.keys()
        assert sum(1 for source, _ in audits if source == AUDITED) > LIMIT, "verdicts are capped"
        anchors = {(link.source, link.position): link.anchor for link in planted.links}
        for key, record in audits.items():
            result = expected[key]
            assert (record.action_type, record.target_url) == (result.verdict, result.target_url)
            assert record.issue_flags == tuple(sorted(result.issue_flags))
            assert record.current_anchor == anchors[key]
            assert record.fix_target == result.fix_target
            assert record.rationale == "; ".join(result.reasons)
            assert [name for name, _ in record.signals] == [
                name for name in AUDIT_DIMENSIONS if getattr(result, name) is not None
            ]
            proposed = [c.text for c in record.proposed_anchors or ()]
            assert proposed == ([result.proposed_anchor] if result.proposed_anchor else [])
            assert all(
                c.score is None and c.placement is None for c in record.proposed_anchors or ()
            )

        served = {
            text
            for route, value in state.items()
            if route != "/excluded-pages"
            for text in strings(dumped(value))
        }
        assert not served & excluded, f"excluded pages served: {sorted(served & excluded)}"
        assert state["/excluded-pages"] == tuple(
            sorted(planted.excluded, key=lambda page: page.url)
        )
        unassessed = planted.candidates() - planted.anchored.keys() - planted.unanchored.keys()
        # Every unassessed pair is on a page below the link limit, where all of them count.
        assert not {source for source, _ in unassessed} & {
            source for source, found in links.items() if len(found) == LIMIT
        }
        assert pub.report.pairs_not_assessed == len(unassessed) > 0
        ids.append({(r.action_type, r.source_url, r.target_url, r.position): r.id for r in records})

    alpha, beta = ids
    assert alpha.keys() & beta.keys(), "the tenants share no action on the same link"
    assert not set(alpha.values()) & set(beta.values()), "a recommendation id spans tenants"


async def test_every_section_is_served(
    published: tuple[Published, Published], mongo: MongoRepo, mongo_uri: str
) -> None:
    for pub in published:
        planted, variant = pub.planted, pub.planted.variant
        async with open_api(mongo_uri) as (_, client):
            state = await served_state(client, pub.tenant_id, pub.key)
        records = cast("tuple[Recommendation, ...]", state["/recommendations"])
        new = [r for r in records if r.action_type in NEW_LINK_ACTIONS]
        audits = [r for r in records if r.action_type in AUDIT_ACTIONS]
        unanchored = cast("tuple[UnanchoredOut, ...]", state["/unanchored"])
        ranked = pub.ranked_pairs()

        run = cast("RunInfo", state["/runs/latest"])
        audit_run = await mongo.latest_link_audit_run(pub.tenant_id)
        assert audit_run is not None
        assert (run.tenant_id, run.run_id, run.status) == (
            pub.tenant_id,
            pub.report.run_id,
            "complete",
        )
        assert (run.scorer, run.model_version) == (ScorerName.BASELINE, None)
        assert (run.limit_per_source, run.content_gap_limit) == (LIMIT, GAP_LIMIT)
        assert (run.link_audit_run_id, run.feature_code) == (planted.audit_run_id, code_digest())
        assert STAGE_FILES | {"link_audit"} <= run.inputs.keys()
        assert run.inputs["link_audit"] == audit_run[1]
        assert run.quality is None, "no quality evaluation was logged"

        summary = cast("SiteSummary", state["/summary"])
        assert summary == run.summary == pub.report.summary
        excluded = planted.excluded_urls
        audited = [r for r in planted.audit if not {r.source_url, r.target_url} & excluded]
        per_source = Counter(r.source_url for r in new)
        link_count = Counter(r.source_url for r in new if r.action_type is ActionType.ADD_LINK)
        sources = set(ranked["source_url"])
        assert summary == SiteSummary(
            pages=len(KIT_URLS),
            excluded_pages={variant.excluded_reason: 1, ExclusionReason.SITEMAP: 1},
            orphan_pages={variant.orphan_label: 1},
            dead_end_pages=1,
            duplicate_groups=1,
            duplicate_copies=1,
            hubs=2,
            bridge_pairs=1,
            # One per slot: alternatives are not counted.
            bridge_links=sum(1 for link in planted.bridge_links if link.rank == 1),
            recommendations=dict(Counter(r.action_type for r in records)),
            tiers=dict(Counter(r.tier for r in new if r.tier is not None)),
            sources_with_recommendations=len(per_source),
            sources_below_limit=sum(1 for source in sources if link_count[source] < LIMIT),
            links_audited=len(audited),
            unverified_links=sum(1 for r in audited if r.unverified),
            audit_flags=dict(Counter(flag for r in audits for flag in r.issue_flags)),
            unanchored=dict(Counter(u.reason for u in unanchored)),
            target_fixes=1,
        )

        linked = {(link.source, link.target) for link in planted.links}
        inbound = Counter(target for _, target in linked)
        outbound = Counter(source for source, _ in linked)
        main = pillars(variant)
        pages = cast("tuple[PageProfile, ...]", state["/pages"])
        assert [page.url for page in pages] == sorted(KIT_URLS)
        for page in pages:
            url = page.url
            grouped = url in {KIT_CANONICAL, KIT_COPY}
            assert page.model_dump(
                exclude={
                    "anchor_mix",
                    "word_count",
                    "crawl_depth",
                    "page_rank_percentile",
                    "page_type",
                }
            ) == PageProfile(
                url=url,
                title=planted.title(url),
                language="en",
                word_count=0,
                inbound=inbound[url],
                outbound=outbound[url],
                hub_id=kit_hub(variant, url),
                is_hub_pillar=url in main,
                is_orphan=url == KIT_ORPHAN,
                orphan_label=variant.orphan_label if url == KIT_ORPHAN else None,
                is_dead_end=url == KIT_DEAD_END,
                duplicate_group=0 if grouped else None,
                is_canonical=url == KIT_CANONICAL if grouped else None,
                target_keyword=planted.keyword_of(url),
                keyword_rung=None if url == NO_KEYWORD else KeywordRung.H1,
                recommendations_out=per_source[url],
                recommendations_in=sum(1 for r in new if r.target_url == url),
                audit_verdicts_out=sum(1 for r in audits if r.source_url == url),
            ).model_dump(
                exclude={
                    "anchor_mix",
                    "word_count",
                    "crawl_depth",
                    "page_rank_percentile",
                    "page_type",
                }
            ), url
            assert page.word_count == len(planted.body(url).split())
            assert page.page_type is (PageType.PILLAR if url in main else PageType.ARTICLE)
        assert state["/orphans"] == tuple(page for page in pages if page.url == KIT_ORPHAN)
        # Body links into the dead end: its keyword exactly from MIXED, "best" + it from AUDITED.
        dead_end = next(page for page in pages if page.url == KIT_DEAD_END)
        assert dead_end.anchor_mix == AnchorMix(exact=1, partial=1)

        details = cast("tuple[PageDetail, ...]", state["/page"])
        for detail in details:
            url = detail.profile.url
            assert detail.outgoing == tuple(r for r in records if r.source_url == url)
            assert detail.incoming_total == sum(1 for r in new if r.target_url == url)

        hubs = cast("tuple[HubSummary, ...]", state["/hubs"])
        assert [hub.hub_id for hub in hubs] == sorted(variant.hubs), "a retired hub is served"
        for hub in hubs:
            side = variant.hubs.index(hub.hub_id)
            members = [page for page in pages if page.hub_id == hub.hub_id]
            assert hub == HubSummary(
                hub_id=hub.hub_id,
                language="en",
                size=7,
                pillar_url=main[side],
                pillar_title=planted.title(main[side]),
                orphan_pages=sum(1 for page in members if page.is_orphan),
                dead_end_pages=sum(1 for page in members if page.is_dead_end),
                recommendations_in=sum(page.recommendations_in for page in members),
                bridge_hubs=(variant.hubs[1 - side],),
            )

        bridges = cast("tuple[BridgePair, ...]", state["/bridges"])
        assert len(bridges) == 1
        pair = bridges[0]
        assert (pair.hub_a, pair.hub_b, pair.reasons) == (
            planted.hub_pair.hub_a,
            planted.hub_pair.hub_b,
            planted.hub_pair.reasons,
        )
        added = {
            (r.source_url, r.target_url): r for r in new if r.action_type is ActionType.ADD_LINK
        }
        served_links = sorted(pair.links, key=lambda link: (link.slot, link.rank))
        assert [(x.slot, x.rank, x.source_url, x.target_url) for x in served_links] == list(BRIDGED)
        for link in served_links:
            record = added.get((link.source_url, link.target_url))
            assert link.recommendation_id == (None if record is None else record.id), link
        _, _, source, target = BRIDGED[0]
        marked = added[(source, target)]
        assert marked.bridge == BridgeMark(
            hub_from=kit_hub(variant, source),
            hub_to=kit_hub(variant, target),
            reasons=planted.bridge_links[0].reasons,
        )
        assert [r for r in new if r.bridge is not None] == [marked]

        assert state["/duplicates"] == (
            DuplicateGroup(group_id=0, canonical=KIT_CANONICAL, copies=(KIT_COPY,)),
        )

        rank = {
            (s, t): int(n)
            for s, t, n in zip(
                ranked["source_url"], ranked["target_url"], ranked["rank_in_source"], strict=True
            )
        }
        gaps = {
            (r.source_url, r.target_url) for r in new if r.action_type is ActionType.CONTENT_GAP
        }
        assert {(u.source_url, u.target_url, u.reason) for u in unanchored} == {
            (s, t, reason) for (s, t), reason in planted.unanchored.items() if not {s, t} & excluded
        }
        for u in unanchored:
            pair_key = (u.source_url, u.target_url)
            assert (u.rank_in_source, u.recommended) == (rank[pair_key], pair_key in gaps)
            assert u.advice == UNANCHORED_ADVICE[u.reason]

        waiting = ranked.loc[ranked["target_url"] == NO_KEYWORD].sort_values(
            ["score", "source_url"], ascending=[False, True]
        )
        assert state["/target-fixes"] == (
            TargetFix(
                target_url=NO_KEYWORD,
                title=None,
                fix=UNANCHORED_ADVICE[UnanchoredReason.TARGET_PAGE_HAS_NO_KEYWORD],
                waiting_sources=len(waiting),
                best_sources=tuple(waiting["source_url"][:5]),
            ),
        )
        assert state["/excluded-pages"] == tuple(
            sorted(planted.excluded, key=lambda page: page.url)
        )

        # Every filter matches exactly, urls in any form they normalise from.
        tier = new[0].tier
        assert tier is not None
        filtered: list[tuple[str, Params, tuple[BaseModel, ...]]] = [
            (
                "/recommendations",
                {"action_type": [ActionType.FIX.value, ActionType.REMOVE.value]},
                tuple(r for r in records if r.action_type in {ActionType.FIX, ActionType.REMOVE}),
            ),
            ("/recommendations", {"tier": tier}, tuple(r for r in records if r.tier == tier)),
            (
                "/recommendations",
                {"source": f"https://www.{FULL}/", "target": f"http://{KIT_ORPHAN}#top"},
                tuple(r for r in records if (r.source_url, r.target_url) == (FULL, KIT_ORPHAN)),
            ),
            (
                "/pages",
                {"hub": variant.hubs[1]},
                tuple(p for p in pages if p.hub_id == variant.hubs[1]),
            ),
            ("/pages", {"orphan": "true"}, tuple(p for p in pages if p.is_orphan)),
            ("/pages", {"dead_end": "true"}, tuple(p for p in pages if p.is_dead_end)),
            (
                "/pages",
                {"duplicate": "true"},
                tuple(p for p in pages if p.duplicate_group is not None),
            ),
            (
                "/pages",
                {"orphan_label": variant.orphan_label.value},
                tuple(p for p in pages if p.is_orphan),
            ),
            ("/orphans", {"orphan_label": OrphanLabel.FOOTER_ONLY.value}, ()),
            ("/bridges", {"hub": variant.hubs[1]}, bridges),
            ("/bridges", {"hub": variant.retired_hub}, ()),
            (
                "/unanchored",
                {"reason": UnanchoredReason.TARGET_PAGE_HAS_NO_KEYWORD.value},
                tuple(u for u in unanchored if u.target_url == NO_KEYWORD),
            ),
            (
                "/unanchored",
                {"source": MIXED},
                tuple(u for u in unanchored if u.source_url == MIXED),
            ),
            (
                "/excluded-pages",
                {"reason": ExclusionReason.SITEMAP.value},
                tuple(page for page in planted.excluded if page.reason is ExclusionReason.SITEMAP),
            ),
        ]
        base = PREFIX.format(tenant=pub.tenant_id)
        async with open_api(mongo_uri) as (_, client):
            for route, params, expected in filtered:
                items, total = await walk(client, base + route, pub.key, params, limit=5)
                listed = tuple(LISTINGS[route].model_validate(item) for item in items)
                assert (listed, total) == (expected, len(expected)), (route, params)
            other_form = await fetch(
                client, f"{base}/page", pub.key, url=f"https://WWW.{FULL}/index.html"
            )
            assert PageDetail.model_validate(other_form) == details[0]
            invalid = await client.get(
                f"{base}/page",
                params={"url": "ftp://example.com/kit"},
                headers={"X-API-Key": pub.key},
            )
            assert invalid.status_code == 422, invalid.text


async def test_the_run_log_has_no_urls(
    published: tuple[Published, Published], local_mlflow: str
) -> None:
    client = MlflowClient(local_mlflow)
    for pub in published:
        report = pub.report
        run_id = log_recommendations(report, summarise_recommendations(report))
        run = client.get_run(run_id)
        assert (run.data.tags["tenant_id"], run.data.tags["output_run_id"]) == (
            pub.tenant_id,
            report.run_id,
        )
        metrics = run.data.metrics
        assert metrics["action_add_link"] == report.summary.recommendations[ActionType.ADD_LINK]
        assert metrics["pairs_not_assessed"] == report.pairs_not_assessed > 0
        logged = [
            *run.data.params,
            *map(str, run.data.params.values()),
            *map(str, run.data.tags.values()),
            *metrics,
            *(
                load_text(f"runs:/{run_id}/{name}")
                for name in ("report.json", "metrics.json", "summary.md")
            ),
        ]
        urls = pub.planted.urls
        control = [f"see https://www.{KIT_URLS[0]}/ or /{KIT_URLS[1].split('/', 1)[1]} today"]
        assert urls_in(control, urls) == {KIT_URLS[0], KIT_URLS[1]}, "the url check finds nothing"
        assert not urls_in(logged, urls), f"urls logged: {sorted(urls_in(logged, urls))}"
