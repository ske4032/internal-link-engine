"""The rank of each TARGETS_KEYWORD edge in its page's keyword set, as written and rewritten."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from linking_engine.models import KeywordRung, KeywordSource, KeywordTarget, Page

if TYPE_CHECKING:
    from linking_engine.graph.repo import GraphRepo

STRATEGIC, OBSERVED = KeywordSource.CLIENT_STRATEGIC, KeywordSource.GSC_OBSERVED
PAGE = "example.com/shoes"


def target(
    text: str, source: KeywordSource, rank: int | None, rung: KeywordRung | None = None
) -> KeywordTarget:
    return KeywordTarget(url=PAGE, text=text, language="en", source=source, rank=rank, rung=rung)


async def ranks(graph: GraphRepo, tenant: str) -> dict[str, object]:
    rows = await graph._auto(
        "MATCH (:Page {tenantId: $t})-[r:TARGETS_KEYWORD]->(k:Keyword {tenantId: $t}) "
        "RETURN k.text AS text, r.rank AS rank",
        t=tenant,
    )
    return {str(row["text"]): row["rank"] for row in rows}


@pytest.mark.integration
async def test_every_edge_stores_its_rank_and_a_rewrite_replaces_it(
    graph: GraphRepo, tenant: str
) -> None:
    await graph.upsert_pages(tenant, [Page(url=PAGE, status_code=200)])
    await graph.replace_keyword_targets(
        tenant,
        STRATEGIC,
        [
            target("trail shoes", STRATEGIC, None, KeywordRung.STRATEGIC),
            target("rain jackets", STRATEGIC, 2),
        ],
    )
    await graph.replace_keyword_targets(tenant, OBSERVED, [target("waterproof boots", OBSERVED, 3)])

    assert await ranks(graph, tenant) == {
        "trail shoes": 1,
        "rain jackets": 2,
        "waterproof boots": 3,
    }

    await graph.replace_keyword_targets(tenant, OBSERVED, [target("waterproof boots", OBSERVED, 2)])
    await graph.replace_keyword_targets(
        tenant,
        STRATEGIC,
        [
            target("trail shoes", STRATEGIC, None, KeywordRung.STRATEGIC),
            target("rain jackets", STRATEGIC, None),
        ],
    )

    assert await ranks(graph, tenant) == {
        "trail shoes": 1,
        "rain jackets": None,
        "waterproof boots": 2,
    }
