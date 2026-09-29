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

import numpy as np
import pandas
import pyarrow.parquet as pq
import pytest
from fixture import (
    ALPHA,
    AUDITED,
    BETA,
    BRIDGED,
    DATABASE,
    DIMENSION,
    EMBEDDING_MODEL,
    FULL,
    GAP_LIMIT,
    KEYWORDLESS,
    KIT_CANONICAL,
    KIT_COPIES,
    KIT_DEAD_END,
    KIT_ORPHAN,
    KIT_ORPHANS,
    KIT_URLS,
    LIMIT,
    LISTINGS,
    MIXED,
    NO_KEYWORD,
    NOT_FOUND,
    ORPHAN_COPY,
    POPULAR,
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
    gini,
    kit_hub,
    kit_page_hub,
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

from linking_engine.discovery.candidates import retrieve_candidates
from linking_engine.discovery.features import code_digest
from linking_engine.discovery.scoring import TOP_CONTRIBUTIONS, default_weights, rank_tiers
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
    Link,
    OrphanLabel,
    OrphanRescue,
    OrphanSlotReason,
    Page,
    PageDetail,
    PageProfile,
    PageType,
    Recommendation,
    RecommendationReport,
    RescueSource,
    RunInfo,
    ScorerName,
    SiteSummary,
    TargetFix,
    UnanchoredOut,
    UnanchoredReason,
)
from linking_engine.models.anchors import UNANCHORED_ADVICE
from linking_engine.models.tenant import TenantConfig
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
    effective_guarantee,
    publish_recommendations,
    summarise_recommendations,
)
from linking_engine.urls import normalise_url

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Sequence
    from pathlib import Path

    import httpx
    import numpy.typing as npt

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


def probes(route: str, output: ServedOutput) -> list[tuple[str, Params]]:
    """Requests for the tenant's own data on the route, url filters and the #90 parameters
    included."""
    path = (PREFIX + route).format(
        tenant=output.tenant_id, recommendation_id=output.recommendations[0].id
    )
    params: dict[str, list[Params]] = {
        "/recommendations": [
            {"source": URLS[0], "target": URLS[1]},
            {"order": "best", "suggested": "true"},
            {"orphan_slot": "true"},
        ],
        "/page": [{"url": URLS[0]}],
        "/unanchored": [{"target": KEYWORDLESS}],
        "/pages": [{"hub": str(output.pages[0].hub_id)}],
        "/bridges": [{"hub": str(output.bridges[0].hub_a)}],
        "/orphans": [{}, {"unmet": "true", "orphan_label": str(output.pages[-1].orphan_label)}],
    }
    return [(path, query) for query in params.get(route, [{}])]


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
            for path, query in probes(route, b):
                response = await client.get(path, params=query, headers={"X-API-Key": key_a})
                assert (response.status_code, response.json()) == (404, NOT_FOUND), (
                    f"A's key on B's {route} {query} must look like a missing resource"
                )
                response = await client.get(path, params=query, headers={"X-API-Key": key_b})
                assert response.status_code == 200, (route, query, response.text)
                assert response.json().get("total", 1) > 0, f"B's {route} {query} probes nothing"
                for headers in ({}, {"X-API-Key": "lek_" + "f" * 43}, {"X-API-Key": revoked}):
                    response = await client.get(path, params=query, headers=headers)
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


async def served_pair(
    mongo: MongoRepo, tenant: str, other_tenant: str
) -> tuple[ServedOutput, ServedOutput, str]:
    """Both tenants' output stored over the same urls, and a key for the first."""
    a = served_output(tenant, run_id(), ALPHA)
    b = served_output(other_tenant, run_id(), BETA)
    for output in (a, b):
        await write_served(mongo, output)
    key, _ = await KeyStore(mongo._db).issue(tenant, "acceptance")
    return a, b, key


async def listed(
    client: httpx.AsyncClient, path: str, key: str, params: Params, *, limit: int = 3
) -> list[Recommendation]:
    items, total = await walk(client, path, key, params, limit=limit)
    assert total == len(items), (params, total, len(items))
    return [Recommendation.model_validate(item) for item in items]


async def test_best_first_order_and_the_suggested_filter(
    mongo: MongoRepo, mongo_uri: str, tenant: str, other_tenant: str
) -> None:
    a, _, key = await served_pair(mongo, tenant, other_tenant)
    records = a.recommendations
    best = sorted(
        (r for r in records if r.action_type in NEW_LINK_ACTIONS), key=lambda r: r.best_rank or 0
    )
    assert [r.best_rank for r in best] == list(range(1, len(best) + 1))
    assert [r.source_url for r in best] != sorted(r.source_url for r in best), (
        "the fixture's best-first order must cross pages, or it is the page order"
    )
    suggested = [r for r in records if r.suggested]
    assert {r.action_type for r in suggested} == {ActionType.ADD_LINK}
    assert 0 < len(suggested) < sum(1 for r in records if r.action_type is ActionType.ADD_LINK)
    assert sum(1 for r in records if r.orphan_slot) == 1, "the fixture places one orphan slot"
    path = f"{PREFIX.format(tenant=tenant)}/recommendations"

    async with open_api(mongo_uri) as (_, client):
        walked = await listed(client, path, key, {"order": "best"})
        whole = await fetch(client, path, key, order="best", limit="500")
        again = await fetch(client, path, key, order="best", limit="500")
        by_page = await listed(client, path, key, {"order": "page"}, limit=5)
        default = await listed(client, path, key, {}, limit=5)
        cases: list[tuple[Params, list[Recommendation]]] = [
            ({"suggested": "true"}, suggested),
            ({"suggested": "false"}, [r for r in records if not r.suggested]),
            ({"orphan_slot": "true"}, [r for r in records if r.orphan_slot]),
            ({"suggested": "true", "order": "best"}, [r for r in best if r.suggested]),
            ({"order": "best", "source": URLS[1]}, [r for r in best if r.source_url == URLS[1]]),
            ({"order": "best", "tier": 2}, [r for r in best if r.tier == 2]),
            ({"order": "best", "action_type": ActionType.FIX.value}, []),
        ]
        found = [(params, await listed(client, path, key, params)) for params, _ in cases]
        invalid = await client.get(path, params={"order": "worst"}, headers={"X-API-Key": key})

    assert walked == best, "order=best must list every new-link record by best_rank"
    assert whole == again, "two calls disagree"
    assert isinstance(whole, dict)
    assert whole["next_cursor"] is None
    assert [Recommendation.model_validate(item) for item in whole["items"]] == walked, (
        "walking best-first page by page differs from one big page"
    )
    assert by_page == default == list(records), "order=page is the stored order, the default"
    for (params, served), (_, expected) in zip(found, cases, strict=True):
        assert served == expected, params
    assert invalid.status_code == 422, invalid.text


async def test_orphans_endpoint_serves_rescue_with_reasons(
    mongo: MongoRepo, mongo_uri: str, tenant: str, other_tenant: str
) -> None:
    a, _, key = await served_pair(mongo, tenant, other_tenant)
    base = PREFIX.format(tenant=tenant)
    stored = a.orphans
    orphan = next(page for page in a.pages if page.is_orphan)
    assert orphan.orphan_label is not None
    label = orphan.orphan_label.value
    cases: list[tuple[Params, list[OrphanRescue]]] = [
        ({"unmet": "true"}, [r for r in stored if r.unmet_reason is not None]),
        ({"unmet": "false"}, [r for r in stored if r.unmet_reason is None]),
        ({"orphan_label": label}, [r for r in stored if r.profile.url == orphan.url]),
        ({"orphan_label": OrphanLabel.FOOTER_ONLY.value}, []),
        ({"orphan_label": label, "unmet": "false"}, []),
    ]
    async with open_api(mongo_uri) as (_, client):
        items, total = await walk(client, f"{base}/orphans", key, limit=2)
        rescues = [OrphanRescue.model_validate(item) for item in items]
        filtered = []
        for params, _ in cases:
            found, _ = await walk(client, f"{base}/orphans", key, params, limit=1)
            filtered.append([OrphanRescue.model_validate(item) for item in found])
        suggested_into = {
            rescue.profile.url: await listed(
                client,
                f"{base}/recommendations",
                key,
                {"target": rescue.profile.url, "suggested": "true"},
            )
            for rescue in rescues
        }
        linked = {
            source.recommendation_id: Recommendation.model_validate(
                await fetch(client, f"{base}/recommendations/{source.recommendation_id}", key)
            )
            for rescue in rescues
            for source in rescue.sources
            if source.recommendation_id is not None
        }
        summary = SiteSummary.model_validate(await fetch(client, f"{base}/summary", key))

    assert rescues == list(stored), "the rescue view as stored, in url order"
    below = a.run.guaranteed_inbound_below
    assert [r.profile.url for r in rescues] == sorted(
        page.url for page in a.pages if page.inbound < below
    ), f"every page with fewer than {below} body links in, and no other"
    assert total == summary.guaranteed_pages == len(rescues)
    assert summary.guarantees_unmet == dict(
        Counter(r.unmet_reason for r in rescues if r.unmet_reason is not None)
    )
    reasons = {r.unmet_reason for r in rescues}
    assert reasons == {None, OrphanSlotReason.SOURCES_FULL, OrphanSlotReason.NO_ANCHOR}, (
        f"the fixture exercises a met guarantee and two reasons, not {reasons}"
    )
    assert max(len(r.sources) for r in rescues) > 1, "no rescue has sources to order"
    assert linked, "no rescue source is a suggested link"

    for rescue in rescues:
        url = rescue.profile.url
        assert rescue.guaranteed == effective_guarantee(
            a.run.guaranteed_inbound_links, a.run.max_suggested_inbound
        )
        assert rescue.suggested_in == len(suggested_into[url]), url
        assert (rescue.unmet_reason is None) == (rescue.suggested_in >= rescue.guaranteed), url
        assert list(rescue.sources) == sorted(
            rescue.sources,
            key=lambda s: (
                s.anchor is None,
                -s.score,
                -(s.source_page_rank_percentile or 0.0),
                s.source_url,
            ),
        ), f"{url}: sources out of order"
        for source in rescue.sources:
            if source.recommendation_id is None:
                continue
            record = linked[source.recommendation_id]
            assert (record.source_url, record.target_url) == (source.source_url, url)
            assert record.suggested, "a rescue source's id names a suggested link"

    for (params, expected), served in zip(cases, filtered, strict=True):
        assert served == expected, params


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
# The summary's page budget and guarantee counts, checked by the #90 tests.
BUDGET_FIELDS: Final = {
    "suggested_links",
    "reserve_links",
    "guaranteed_pages",
    "orphan_slots",
    "guarantees_unmet",
    "orphans_reached",
    "orphans_to_pillar",
    "inbound_gini",
    "pages_at_cap",
    "links_moved_by_cap",
    "links_dropped_by_cap",
    "top10_inbound_share",
}


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
        assert summary.model_dump(exclude=BUDGET_FIELDS) == SiteSummary(
            pages=len(KIT_URLS),
            excluded_pages={variant.excluded_reason: 1, ExclusionReason.SITEMAP: 1},
            orphan_pages={variant.orphan_label: len(KIT_ORPHANS)},
            dead_end_pages=1,
            duplicate_groups=1,
            duplicate_copies=len(KIT_COPIES),
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
        ).model_dump(exclude=BUDGET_FIELDS)

        linked = {(link.source, link.target) for link in planted.links}
        inbound = Counter(target for _, target in linked)
        outbound = Counter(source for source, _ in linked)
        main = pillars(variant)
        pages = cast("tuple[PageProfile, ...]", state["/pages"])
        assert [page.url for page in pages] == sorted(KIT_URLS)
        for page in pages:
            url = page.url
            grouped = url in {KIT_CANONICAL, *KIT_COPIES}
            assert page.model_dump(
                exclude={
                    "anchor_mix",
                    "word_count",
                    "crawl_depth",
                    "page_rank_percentile",
                    "page_type",
                    "link_budget",
                }
            ) == PageProfile(
                url=url,
                title=planted.title(url),
                language="en",
                word_count=0,
                inbound=inbound[url],
                outbound=outbound[url],
                hub_id=kit_page_hub(variant, url),
                is_hub_pillar=url in main,
                is_orphan=url in KIT_ORPHANS,
                orphan_label=variant.orphan_label if url in KIT_ORPHANS else None,
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
                    "link_budget",
                }
            ), url
            assert page.word_count == len(planted.body(url).split())
            assert page.page_type is (PageType.PILLAR if url in main else PageType.ARTICLE)
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
            DuplicateGroup(group_id=0, canonical=KIT_CANONICAL, copies=tuple(sorted(KIT_COPIES))),
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


# ── #90: the page budget and the guaranteed inbound links, over the planted stage inputs ────


def link_budget(planted: Planted, url: str, rules: TenantConfig) -> int:
    """A link per ``words_per_link`` body words, at least one, at most the limit, less the
    distinct pages the body already links to."""
    words = len(planted.body(url).split())
    outbound = len({link.target for link in planted.links if link.source == url})
    return max(0, min(LIMIT, max(1, words // rules.words_per_link)) - outbound)


def guaranteed_pages(pages: Iterable[PageProfile], rules: TenantConfig) -> list[str]:
    """The pages guaranteed inbound links, by url: fewer body links in than the cut-off, and a
    retrieval target. Every planted page is indexable with a vector, so that is every page but a
    duplicate copy."""
    return sorted(
        page.url
        for page in pages
        if page.inbound < rules.guaranteed_inbound_below and page.is_canonical is not False
    )


def scored(pub: Published) -> pandas.DataFrame:
    """The ranked pairs of the pages in the pipeline, each with the score and tier served for
    it: the score's percentile over every pair the run read, unrounded as the stage orders by
    it and rounded as served, and its tier by rank."""
    frame = pub.ranked.copy()
    values = frame["score"].to_numpy(dtype=np.float64)
    ranks = pandas.Series(values).rank(method="average").to_numpy(dtype=np.float64)
    frame["percentile"] = (ranks - 1) / (len(values) - 1)
    frame["served"] = [round(100 * float(p), 1) for p in frame["percentile"]]
    frame["tier"] = rank_tiers(
        values, frame["source_url"], frame["target_url"], default_weights().tier_shares
    )
    excluded = sorted(pub.planted.excluded_urls)
    keep = ~(frame["source_url"].isin(excluded) | frame["target_url"].isin(excluded))
    return frame.loc[keep]


@dataclass(frozen=True, slots=True)
class Expected:
    """Each source page's links as the walk should leave them, in ranker order, and the
    suggestions the inbound cap moved to a reserve or left unfilled."""

    order: dict[str, list[str]]
    anchored: dict[str, list[str]]
    links: dict[str, list[str]]
    suggested: dict[str, list[str]]
    moved: dict[str, list[tuple[str, str]]]
    dropped: dict[str, list[str]]
    # Suggested links into each page before the cap.
    wanted: Counter[str]


def expected_walk(
    pub: Published, records: Sequence[Recommendation], pages: Sequence[PageProfile]
) -> Expected:
    """The walk from the planted inputs: each page's first `limit` anchored pairs, the first
    budget of them suggested; then the cap, visiting the suggestions best first, moves one into a
    full page to its source's first reserve into a page not full, or leaves it unfilled; then the
    orphan slots the run placed each take the place of their source's weakest suggested link into
    a page without a guarantee."""
    planted = pub.planted
    rules = TenantConfig(tenant_id=pub.tenant_id)
    cap = rules.max_suggested_inbound
    exempt = {page.url for page in pages if page.is_hub_pillar}
    guaranteed = set(guaranteed_pages(pages, rules))
    frame = scored(pub)
    order = {
        str(source): list(group.sort_values("rank_in_source")["target_url"])
        for source, group in frame.groupby("source_url")
    }
    raw = {
        (s, t): float(x)
        for s, t, x in zip(frame["source_url"], frame["target_url"], frame["score"], strict=True)
    }
    anchored = {s: [t for t in found if (s, t) in planted.anchored] for s, found in order.items()}
    first = {s: found[:LIMIT] for s, found in anchored.items()}
    chosen = {s: set(found[: link_budget(planted, s, rules)]) for s, found in first.items()}
    wanted = Counter(t for found in chosen.values() for t in found)

    into: Counter[str] = Counter()
    moved: defaultdict[str, list[tuple[str, str]]] = defaultdict(list)
    dropped: defaultdict[str, list[str]] = defaultdict(list)

    def full(target: str) -> bool:
        return bool(cap) and target not in exempt and into[target] >= cap

    visits = sorted(
        ((s, t) for s, found in chosen.items() for t in found),
        key=lambda pair: (-raw[pair], pair[0], order[pair[0]].index(pair[1]), pair[1]),
    )
    for source, target in visits:
        if not full(target):
            into[target] += 1
            continue
        chosen[source].discard(target)
        spare = next((t for t in first[source] if t not in chosen[source] and not full(t)), None)
        if spare is None:
            dropped[source].append(target)
            continue
        chosen[source].add(spare)
        into[spare] += 1
        moved[source].append((target, spare))

    slots = {r.source_url: r.target_url for r in records if r.orphan_slot}
    links: dict[str, list[str]] = {}
    suggested: dict[str, list[str]] = {}
    for source, found in order.items():
        kept = [t for t in found if t in first[source] or t == slots.get(source)]
        picked = chosen[source]
        if source in slots:
            given = [t for t in kept if t in picked and t not in guaranteed]
            assert given, f"{source} gave a slot with no link to give up"
            picked = (picked - {given[-1]}) | {slots[source]}
        if len(kept) > LIMIT:
            kept.remove([t for t in kept if t not in picked][-1])
        links[source] = kept
        suggested[source] = [t for t in kept if t in picked]
    return Expected(order, anchored, links, suggested, dict(moved), dict(dropped), wanted)


def by_source(
    records: Iterable[Recommendation],
) -> tuple[defaultdict[str, list[Recommendation]], defaultdict[str, list[Recommendation]]]:
    """Each source page's ADD_LINKs and CONTENT_GAPs, in served order."""
    links: defaultdict[str, list[Recommendation]] = defaultdict(list)
    gaps: defaultdict[str, list[Recommendation]] = defaultdict(list)
    for record in records:
        if record.action_type is ActionType.ADD_LINK:
            links[record.source_url].append(record)
        elif record.action_type is ActionType.CONTENT_GAP:
            gaps[record.source_url].append(record)
    return links, gaps


async def test_suggested_links_follow_the_page_budget(
    published: tuple[Published, Published], mongo_uri: str
) -> None:
    exercised: set[str] = set()
    for pub in published:
        planted = pub.planted
        rules = TenantConfig(tenant_id=pub.tenant_id)
        async with open_api(mongo_uri) as (_, client):
            state = await served_state(client, pub.tenant_id, pub.key)
        records = cast("tuple[Recommendation, ...]", state["/recommendations"])
        pages = cast("tuple[PageProfile, ...]", state["/pages"])
        run = cast("RunInfo", state["/runs/latest"])
        settings = (
            rules.words_per_link,
            rules.guaranteed_inbound_links,
            rules.guaranteed_inbound_below,
            rules.max_suggested_inbound,
        )
        assert (
            run.words_per_link,
            run.guaranteed_inbound_links,
            run.guaranteed_inbound_below,
            run.max_suggested_inbound,
        ) == settings, "the run stamps other settings than the tenant's"
        assert (
            pub.report.words_per_link,
            pub.report.guaranteed_inbound_links,
            pub.report.guaranteed_inbound_below,
            pub.report.max_suggested_inbound,
        ) == settings, "the stage reports other settings than the tenant's"
        budgets = {url: link_budget(planted, url, rules) for url in KIT_URLS}
        assert {page.url: page.link_budget for page in pages} == budgets
        links, gaps = by_source(records)
        expected = expected_walk(pub, records, pages)
        frame = scored(pub)
        rank = {
            (s, t): int(n)
            for s, t, n in zip(
                frame["source_url"], frame["target_url"], frame["rank_in_source"], strict=True
            )
        }

        for source, order in expected.order.items():
            anchored = expected.anchored[source]
            budget = budgets[source]
            slots = [r.target_url for r in links[source] if r.orphan_slot]
            assert len(slots) <= 1, f"{source} gave {len(slots)} orphan slots"
            kept, suggested = expected.links[source], expected.suggested[source]
            served = links[source]
            assert [r.target_url for r in served] == kept, f"{source}: its links"
            assert [r.target_url for r in served if r.suggested] == suggested, (
                f"{source}: suggested must be its first {budget} links, reserves after them, "
                "but for the cap and its orphan slot"
            )
            assert [r.rank_in_source for r in served] == list(range(1, len(served) + 1))
            assert len(served) <= LIMIT

            # Content gaps only on a page with a budget, above its last suggested link.
            candidates = [t for t in order if planted.unanchored.get((source, t)) in FINDINGS]
            last = rank[(source, suggested[-1])] if suggested else None
            eligible = (
                [t for t in candidates if last is None or rank[(source, t)] < last]
                if budget
                else []
            )
            assert [r.target_url for r in gaps[source]] == eligible[:GAP_LIMIT], (
                f"{source}: its content gaps"
            )
            assert [r.rank_in_source for r in gaps[source]] == list(range(1, len(gaps[source]) + 1))
            words = len(planted.body(source).split())
            exercised |= {
                name
                for name, seen in (
                    ("budget capped at the limit", words // rules.words_per_link > LIMIT),
                    ("links capped", len(anchored) > LIMIT),
                    ("reserves after suggested links", 0 < len(suggested) < len(kept)),
                    ("links without a budget", budget == 0 and bool(kept)),
                    ("gaps held back without a budget", budget == 0 and bool(candidates)),
                    ("gaps capped", len(eligible) > GAP_LIMIT),
                    (
                        "a gap below the last suggested link",
                        bool(budget) and eligible != candidates,
                    ),
                    ("gaps without suggested links", last is None and bool(gaps[source])),
                    ("gaps above a suggested link", last is not None and bool(gaps[source])),
                    ("a link given up for a slot", bool(slots)),
                )
                if seen
            }

        added = [r for r in records if r.action_type is ActionType.ADD_LINK]
        summary = cast("SiteSummary", state["/summary"])
        assert (summary.suggested_links, summary.reserve_links) == (
            sum(1 for r in added if r.suggested),
            sum(1 for r in added if not r.suggested),
        )
        # Best first across pages: the unrounded score, then source, rank in source, action and
        # target. No two new links here share a served score, so the unit tests prove the order
        # of rounded ties.
        percentile = {
            (s, t): float(p)
            for s, t, p in zip(
                frame["source_url"], frame["target_url"], frame["percentile"], strict=True
            )
        }
        new = [r for r in records if r.action_type in NEW_LINK_ACTIONS]
        assert [r.score for r in new] == [
            round(100 * percentile[(r.source_url, r.target_url)], 1) for r in new
        ], "a served score is not the rounded percentile the best-first order is keyed on"
        best = sorted(
            new,
            key=lambda r: (
                -percentile[(r.source_url, r.target_url)],
                r.source_url,
                r.rank_in_source,
                r.action_type,
                r.target_url,
            ),
        )
        assert [r.best_rank for r in best] == list(range(1, len(best) + 1))
        assert all(r.best_rank is None for r in records if r.action_type in AUDIT_ACTIONS)
        assert state["/recommendations?order=best"] == tuple(best)

    assert exercised == {
        "budget capped at the limit",
        "links capped",
        "reserves after suggested links",
        "links without a budget",
        "gaps held back without a budget",
        "gaps capped",
        "a gap below the last suggested link",
        "gaps without suggested links",
        "gaps above a suggested link",
        "a link given up for a slot",
    }, f"the fixture only exercises {sorted(exercised)}"


async def guarantee_checks(pub: Published, mongo_uri: str) -> set[str]:
    """Check one tenant's orphan slots, rescue view and summary against its planted inputs;
    returns the cases its output exercises."""
    exercised: set[str] = set()
    planted = pub.planted
    rules = TenantConfig(tenant_id=pub.tenant_id)
    need = rules.guaranteed_inbound_links
    async with open_api(mongo_uri) as (_, client):
        state = await served_state(client, pub.tenant_id, pub.key)
    records = cast("tuple[Recommendation, ...]", state["/recommendations"])
    pages = cast("tuple[PageProfile, ...]", state["/pages"])
    rescues = cast("tuple[OrphanRescue, ...]", state["/orphans"])
    summary = cast("SiteSummary", state["/summary"])
    hubs = cast("tuple[HubSummary, ...]", state["/hubs"])
    budgets = {url: link_budget(planted, url, rules) for url in KIT_URLS}
    hub = {page.url: page.hub_id for page in pages}
    strength = {page.url: page.page_rank_percentile for page in pages}
    guaranteed = guaranteed_pages(pages, rules)
    assert set(guaranteed) == KIT_ORPHANS - {ORPHAN_COPY}
    # The orphan copy is no retrieval target: an orphan in its profile, guaranteed nothing.
    copy = next(page for page in pages if page.url == ORPHAN_COPY)
    assert copy.inbound < rules.guaranteed_inbound_below
    assert (copy.is_orphan, copy.orphan_label) == (True, planted.variant.orphan_label)
    assert ORPHAN_COPY not in {rescue.profile.url for rescue in rescues}
    assert not any(r.target_url == ORPHAN_COPY for r in records), "a recommendation into a copy"

    def may_link(source: str, target: str) -> bool:
        """The hub rule: a source in the target's hub, or any page for a hubless target."""
        return hub[target] is None or hub[source] == hub[target]

    added = [r for r in records if r.action_type is ActionType.ADD_LINK]
    suggested = {(r.source_url, r.target_url): r for r in added if r.suggested}
    slots = [r for r in added if r.orphan_slot]
    givers = [r.source_url for r in slots]
    assert len(set(givers)) == len(givers), "a source gave more than one orphan slot"
    for record in slots:
        source, target = record.source_url, record.target_url
        assert target in guaranteed, f"a slot into {target}, which has no guarantee"
        assert (source, target) in planted.anchored, f"an unanchored slot {source} -> {target}"
        assert record.tier is not None
        assert record.tier <= 2, f"a tier {record.tier} slot {source} -> {target}"
        assert may_link(source, target), f"{source} is outside the hub of {target}"
        assert sum(1 for s, _ in suggested if s == source) <= budgets[source], (
            f"{source} is over its budget of {budgets[source]}"
        )

    def can_give(source: str) -> bool:
        """A source with a budget, no slot given yet and a link it may give up."""
        return (
            budgets[source] >= 1
            and source not in givers
            and any(
                s == source and t not in guaranteed and not r.orphan_slot
                for (s, t), r in suggested.items()
            )
        )

    frame = scored(pub)
    reasons = []
    for target, rescue in zip(guaranteed, rescues, strict=True):
        assert rescue.profile == next(page for page in pages if page.url == target)
        regular = sum(1 for (_, t), r in suggested.items() if t == target and not r.orphan_slot)
        placed = sum(1 for r in slots if r.target_url == target)
        assert placed <= max(0, need - regular), f"{target}: more slots than it needed"
        rows = frame.loc[frame["target_url"] == target]
        # Each source with its unrounded score, which orders the sources, and the served one.
        universe = [
            (str(s), float(p), float(score), int(tier))
            for s, p, score, tier in zip(
                rows["source_url"], rows["percentile"], rows["served"], rows["tier"], strict=True
            )
            if may_link(str(s), target)
        ]
        relevant = [s for s, _, _, tier in universe if tier <= 2]
        anchored = [s for s in relevant if (s, target) in planted.anchored]
        if regular + placed >= need:
            reason = None
        elif not relevant:
            reason = OrphanSlotReason.NO_RELEVANT_SOURCE
        elif not anchored:
            reason = OrphanSlotReason.NO_ANCHOR
        else:
            reason = OrphanSlotReason.SOURCES_FULL
            spare = [s for s in anchored if (s, target) not in suggested and can_give(s)]
            assert not spare, f"{target} is short while {spare} could still give a slot"
        reasons.append(reason)
        assert (rescue.guaranteed, rescue.suggested_in, rescue.unmet_reason) == (
            need,
            regular + placed,
            reason,
        ), target
        sources = sorted(
            universe,
            key=lambda row: (
                (row[0], target) not in planted.anchored,
                -row[1],
                -(strength[row[0]] or 0.0),
                row[0],
            ),
        )[:5]
        assert rescue.sources == tuple(
            RescueSource(
                source_url=s,
                score=score,
                tier=tier,
                anchor=planted.anchored.get((s, target)),
                source_page_rank_percentile=strength[s],
                recommendation_id=suggested[(s, target)].id if (s, target) in suggested else None,
            )
            for s, _, score, tier in sources
        ), f"{target}: its rescue sources"
        exercised |= {
            name
            for name, seen in (
                ("an orphan slot", placed > 0),
                ("a guarantee met", reason is None),
                ("a guarantee unmet", reason is not None),
                (
                    "a hubless page reached from a hub",
                    hub[target] is None
                    and any(t == target and hub[s] is not None for s, t in suggested),
                ),
            )
            if seen
        }

    into = Counter(t for _, t in suggested)
    pillar = {h.hub_id: h.pillar_url for h in hubs}
    to_pillar = sum(
        1
        for page in pages
        if page.is_orphan
        and page.hub_id is not None
        and (page.url, pillar[page.hub_id]) in suggested
    )
    counts = {
        "guaranteed_pages": len(guaranteed),
        "orphan_slots": len(slots),
        "guarantees_unmet": dict(Counter(r for r in reasons if r is not None)),
        "orphans_reached": sum(1 for page in pages if page.is_orphan and into[page.url]),
        "orphans_to_pillar": to_pillar,
    }
    assert summary.model_dump(include=set(counts)) == counts
    targets = sorted(set(frame["target_url"]))
    assert summary.inbound_gini == pytest.approx(gini([into[t] for t in targets]))
    exercised |= {"an orphan linked to its main page"} if to_pillar else set()

    return exercised


async def test_orphans_get_their_guaranteed_links_within_the_budget(
    published: tuple[Published, Published], mongo_uri: str
) -> None:
    exercised: set[str] = set()
    for pub in published:
        exercised |= await guarantee_checks(pub, mongo_uri)
    assert exercised == {
        "an orphan slot",
        "a guarantee met",
        "a guarantee unmet",
        "a hubless page reached from a hub",
        "an orphan linked to its main page",
    }, f"the fixture only exercises {sorted(exercised)}"


async def test_no_page_takes_more_than_the_cap_but_hub_main_pages(
    published: tuple[Published, Published], mongo_uri: str
) -> None:
    exercised: set[str] = set()
    for pub in published:
        rules = TenantConfig(tenant_id=pub.tenant_id)
        cap = rules.max_suggested_inbound
        async with open_api(mongo_uri) as (_, client):
            state = await served_state(client, pub.tenant_id, pub.key)
        records = cast("tuple[Recommendation, ...]", state["/recommendations"])
        pages = cast("tuple[PageProfile, ...]", state["/pages"])
        summary = cast("SiteSummary", state["/summary"])
        assert cast("RunInfo", state["/runs/latest"]).max_suggested_inbound == cap
        main = {page.url for page in pages if page.is_hub_pillar}
        into = Counter(
            r.target_url for r in records if r.action_type is ActionType.ADD_LINK and r.suggested
        )
        over = {url: n for url, n in into.items() if url not in main and n > cap}
        assert not over, f"pages that are not a hub main page over the cap of {cap}: {over}"

        expected = expected_walk(pub, records, pages)
        links, _ = by_source(records)
        for source, found in links.items():
            suggested = {r.target_url for r in found if r.suggested}
            reserves = {r.target_url for r in found if not r.suggested}
            for capped, spare in expected.moved.get(source, []):
                assert capped in reserves, f"{source}: its link into full {capped} is suggested"
                assert spare in suggested, f"{source}: its slot did not move to {spare}"
            for capped in expected.dropped.get(source, []):
                assert capped in reserves, f"{source}: its link into full {capped} is suggested"
            assert [r.target_url for r in found if r.suggested] == expected.suggested[source], (
                f"{source}: its suggested links after the cap"
            )

        assert expected.wanted[POPULAR] > cap, "the fixture's popular page is not over the cap"
        if POPULAR in main:
            assert into[POPULAR] == expected.wanted[POPULAR], "a hub main page was capped"
            exercised.add("a hub main page over the cap")
        else:
            assert into[POPULAR] == cap, f"the popular page takes {into[POPULAR]}, not {cap}"
            exercised.add("a page held at the cap")
        moved = sum(len(found) for found in expected.moved.values())
        dropped = sum(len(found) for found in expected.dropped.values())
        exercised |= {"a suggestion moved to a reserve"} if moved else set()
        exercised |= {"a suggestion left unfilled"} if dropped else set()

        counts = {
            "pages_at_cap": sum(1 for url, n in into.items() if url not in main and n >= cap),
            "links_moved_by_cap": moved,
            "links_dropped_by_cap": dropped,
        }
        assert summary.model_dump(include=set(counts)) == counts
        top = sorted(into.values(), reverse=True)[:10]
        assert summary.top10_inbound_share == pytest.approx(sum(top) / sum(into.values()))

    assert exercised == {
        "a hub main page over the cap",
        "a page held at the cap",
        "a suggestion moved to a reserve",
        "a suggestion left unfilled",
    }, f"the fixture only exercises {sorted(exercised)}"


# ── #90: the hub-main-page channel, over three tenants of the same pages ───────────────────

# Cosine of each hub page to its pillar: ten close pages, then five on the fringe.
CLOSE: Final = tuple(round(0.99 - 0.01 * k, 2) for k in range(10))
FRINGE: Final = (0.7, 0.6, 0.5, 0.4, 0.3)
# Cosine between the two hubs' pillars.
BETWEEN_HUBS: Final = 0.2
PER_TARGET: Final = 3


@dataclass(frozen=True, slots=True)
class Site:
    """Two hubs, each a pillar, fifteen pages graded by their cosine to it and a page in
    another language as close to it as the closest."""

    vectors: dict[str, npt.NDArray[np.float64]]
    hubs: dict[str, int]
    pillars: tuple[str, str]
    languages: dict[str, str]

    def pages(self, hub: int) -> list[str]:
        return [url for url, found in self.hubs.items() if found == hub and url not in self.pillars]

    def close(self, hub: int) -> list[str]:
        return [f"example.com/topic-{hub}/page-{k:02d}" for k in range(1, len(CLOSE) + 1)]

    def cosine(self, a: str, b: str) -> float:
        u, v = self.vectors[a], self.vectors[b]
        return float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v)))


def channel_site() -> Site:
    axes = iter(np.eye(64, DIMENSION))
    first = next(axes)
    centres = (first, BETWEEN_HUBS * first + np.sqrt(1 - BETWEEN_HUBS**2) * next(axes))
    vectors: dict[str, npt.NDArray[np.float64]] = {}
    hubs: dict[str, int] = {}
    languages: dict[str, str] = {}
    pillars = []
    for hub, centre in enumerate(centres):
        pillar = f"example.com/topic-{hub}/main"
        pillars.append(pillar)
        pages = {pillar: (1.0, "en")}
        for k, cosine in enumerate((*CLOSE, *FRINGE), 1):
            pages[f"example.com/topic-{hub}/page-{k:02d}"] = (cosine, "en")
        pages[f"example.com/de/topic-{hub}/seite"] = (CLOSE[0], "de")
        for url, (cosine, language) in pages.items():
            vectors[url] = cosine * centre + np.sqrt(1 - cosine**2) * next(axes)
            hubs[url] = hub
            languages[url] = language
    return Site(vectors, hubs, (pillars[0], pillars[1]), languages)


async def plant_site(
    graph: GraphRepo, tenant: str, site: Site, links: list[tuple[str, str]]
) -> None:
    await graph.upsert_pages(
        tenant,
        [
            Page(
                url=url,
                status_code=200,
                is_indexable=True,
                word_count=400,
                language=site.languages[url],
                crawl_depth=1,
                page_type=PageType.PILLAR if url in site.pillars else PageType.ARTICLE,
            )
            for url in site.vectors
        ],
    )
    positions: Counter[str] = Counter()
    edges = []
    for source, target in links:
        edges.append(
            Link(
                source_url=source,
                target_url=target,
                position=positions[source],
                anchor_text="read on",
                surrounding_text="Read on here.",
            )
        )
        positions[source] += 1
    await graph.replace_links(tenant, sorted(positions), edges)
    await graph._auto(
        "UNWIND $rows AS row MATCH (p:Page {tenantId: $t, url: row.url}) "
        "SET p.content_embedding = row.vec, p.embeddingModel = $model, p.hubId = row.hub, "
        "p.isHubPillar = row.pillar",
        t=tenant,
        model=EMBEDDING_MODEL,
        rows=[
            {
                "url": url,
                "vec": vector.tolist(),
                "hub": site.hubs[url],
                "pillar": url in site.pillars,
            }
            for url, vector in site.vectors.items()
        ],
    )


async def test_the_pillar_channel_adds_relevant_page_to_main_page_pairs(
    graph: GraphRepo, tenant: str, other_tenant: str
) -> None:
    site = channel_site()
    rules = TenantConfig(tenant_id=tenant)
    # Each hub's fifth page already links to its pillar.
    linked = [(site.close(hub)[4], site.pillars[hub]) for hub in (0, 1)]
    close_links = [
        (a, b)
        for hub in (0, 1)
        for i, a in enumerate(site.close(hub))
        for b in site.close(hub)[i + 1 : i + 4]
    ]
    loose_links = [(a, b) for a in site.close(0) for b in site.close(1)[:5]]
    plans = {
        # Links between close pages of a hub, and links across the hubs.
        tenant: [*close_links, *linked],
        other_tenant: [*loose_links, *linked],
        # Fewer links than the floor needs.
        new_tenant(): [*linked, (site.close(0)[0], site.close(0)[1])],
    }
    assert min(len(plans[tenant]), len(plans[other_tenant])) >= rules.pillar_floor_min_links

    channels: dict[str, set[str]] = {}
    floors: dict[str, float] = {}
    for name, links in plans.items():
        await plant_site(graph, name, site, links)
        found = await retrieve_candidates(graph, name, per_target=PER_TARGET)
        report = found.report
        cosines = [site.cosine(a, b) for a, b in links]
        if len(links) >= rules.pillar_floor_min_links:
            key, basis, over = "en", "existing_links", cosines
        else:
            pairs = [x for t in found.targets for x in t.similarities[: t.nearest]]
            key, basis, over = "*", "candidate_pairs", pairs
        assert report.pillar_floor_basis == {key: basis}, f"{name}: the floor's basis"
        assert report.pillar_floor_links == {key: len(links)}
        floor = report.pillar_floors[key]
        assert floor == pytest.approx(
            float(np.quantile(over, rules.pillar_floor_quantile)), abs=1e-5
        ), f"{name}: the floor is the {rules.pillar_floor_quantile} quantile of its own {basis}"
        floors[name] = floor

        added: set[str] = set()
        for target in found.targets:
            assert target.nearest <= PER_TARGET, "the cap holds for the nearest sources"
            if target.target_url not in site.pillars:
                assert target.pillar_pairs == 0, f"{name}: pillar pairs into a page not a pillar"
                continue
            hub = site.hubs[target.target_url]
            nearest = set(target.sources[: target.nearest])
            assert nearest == set(site.close(hub)[:PER_TARGET]), "the nearest sources"
            channel = target.sources[target.nearest :]
            expected = [
                url
                for url in site.pages(hub)
                if site.languages[url] == "en"
                and (url, target.target_url) not in links
                and url not in nearest
                and site.cosine(url, target.target_url) >= floor
            ]
            assert set(channel) == set(expected), f"{name}: the channel into hub {hub}'s pillar"
            assert list(target.similarities[target.nearest :]) == pytest.approx(
                [site.cosine(url, target.target_url) for url in channel], abs=1e-5
            )
            assert site.close(hub)[4] not in target.sources, "a page already linking in"
            assert f"example.com/de/topic-{hub}/seite" not in target.sources, "another language"
            added |= set(channel)
        assert report.pillar_pairs == len(added)
        channels[name] = added

    tight, loose, sparse = (channels[name] for name in plans)
    fringe = {
        site.pages(hub)[k] for hub in (0, 1) for k in range(len(CLOSE), len(CLOSE) + len(FRINGE))
    }
    assert tight, "the close links' floor adds no pair"
    assert not tight & fringe, "the close links' floor keeps the close pages only"
    assert fringe <= loose, "the loose links' floor admits the fringe"
    assert floors[tenant] > floors[other_tenant], "the tenants' floors come from their own links"
    assert sparse, "the candidate-pair floor adds no pair"


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
