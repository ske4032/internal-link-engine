from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from structlog.testing import capture_logs

from linking_engine.models import Link, Page
from linking_engine.pipeline.analytics import load_link_graphs

if TYPE_CHECKING:
    from linking_engine.graph.repo import GraphRepo


@pytest.mark.integration
async def test_load_link_graphs_builds_both_graphs_and_logs_the_cost(
    graph: GraphRepo, tenant: str
) -> None:
    urls = [f"example.com/{name}" for name in ("a", "b", "c", "orphan")]
    await graph.upsert_pages(tenant, [Page(url=u, status_code=200) for u in urls])
    links = [
        Link(
            source_url=urls[0], target_url=urls[1], position=0, anchor_text="b", surrounding_text=""
        ),
        Link(
            source_url=urls[1], target_url=urls[0], position=0, anchor_text="a", surrounding_text=""
        ),
        Link(
            source_url=urls[1], target_url=urls[2], position=1, anchor_text="c", surrounding_text=""
        ),
    ]
    await graph.replace_links(tenant, urls[:2], links)

    with capture_logs() as logs:
        graphs = await load_link_graphs(graph, tenant)

    assert graphs.urls == tuple(sorted(urls))
    assert (graphs.directed.ecount(), graphs.undirected.ecount()) == (3, 2)
    (event,) = [e for e in logs if e["event"] == "graph.build"]
    assert (event["pages"], event["link_rows"], event["isolated"]) == (4, 3, 1)
    assert event["pull_s"] >= 0
    assert event["build_s"] >= 0
