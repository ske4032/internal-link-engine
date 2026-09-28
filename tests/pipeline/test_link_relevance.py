"""Link relevance stage: the embedding-model guard, the report, the log line and the summary."""

from __future__ import annotations

import numpy as np
import pytest
from structlog.testing import capture_logs

from linking_engine.errors import DatabaseReadError, EmbeddingModelMismatchError
from linking_engine.models import EmbeddingModelCount, LinkRelevance
from linking_engine.pipeline.link_relevance import score_links, summarise_link_relevance

TENANT = "acme"
MODEL = "voyage-4-large"


def models(*names: str | None) -> tuple[EmbeddingModelCount, ...]:
    return tuple(EmbeddingModelCount(embedding_model=name, vectors=10) for name in names)


class FakeGraph:
    """Stands in for GraphRepo: stored vector models, the write counts and the rows read back.
    Every vector carries MODEL unless a test says otherwise."""

    def __init__(
        self,
        counts: tuple[int, int],
        rows: list[LinkRelevance],
        *,
        pages: tuple[EmbeddingModelCount, ...] = models(MODEL),
        anchors: tuple[EmbeddingModelCount, ...] = models(MODEL),
        sentences: tuple[EmbeddingModelCount, ...] = models(MODEL),
    ) -> None:
        self.counts = counts
        self.rows = rows
        self.models = {"pages": pages, "anchors": anchors, "sentences": sentences}
        self.calls: list[str] = []

    async def embedding_models(self, tenant_id: str) -> tuple[EmbeddingModelCount, ...]:
        self.calls.append(f"page models {tenant_id}")
        return self.models["pages"]

    async def anchor_embedding_models(self, tenant_id: str) -> tuple[EmbeddingModelCount, ...]:
        self.calls.append(f"anchor models {tenant_id}")
        return self.models["anchors"]

    async def surrounding_embedding_models(self, tenant_id: str) -> tuple[EmbeddingModelCount, ...]:
        self.calls.append(f"sentence models {tenant_id}")
        return self.models["sentences"]

    async def score_link_relevance(self, tenant_id: str) -> tuple[int, int]:
        self.calls.append(f"score {tenant_id}")
        return self.counts

    async def link_relevance(self, tenant_id: str) -> list[LinkRelevance]:
        self.calls.append(f"read {tenant_id}")
        return self.rows


def row(
    position: int, context: float | None, fit: float | None, *, generic: bool = False
) -> LinkRelevance:
    return LinkRelevance(
        source_url="example.com/source",
        position=position,
        target_url=f"example.com/target-{position}",
        context_relevance=context,
        anchor_target_fit=fit,
        anchor_generic=generic,
    )


async def test_the_report_counts_links_anchors_and_describes_both_scores() -> None:
    rows = [
        row(0, 0.8, 0.9),
        row(1, 0.6, None, generic=True),
        row(2, 0.4, None),
        row(3, None, 0.7),
    ]
    graph = FakeGraph((4, 2), rows)

    with capture_logs() as logs:
        report = await score_links(graph, TENANT)  # type: ignore[arg-type]

    assert graph.calls == [
        f"page models {TENANT}",
        f"anchor models {TENANT}",
        f"sentence models {TENANT}",
        f"score {TENANT}",
        f"read {TENANT}",
    ]
    assert (report.links, report.scored) == (6, 4)
    assert (report.generic_anchors, report.without_anchor_vector) == (1, 1)
    assert report.context is not None
    assert report.anchor is not None
    assert (report.context.count, report.anchor.count) == (3, 2)
    assert report.context.mean == pytest.approx(0.6)
    [line] = [entry for entry in logs if entry["event"] == "links.relevance"]
    assert (line["tenant_id"], line["stage"], line["scored"]) == (TENANT, "score-links", 4)
    logged = " ".join(map(str, line.values()))
    assert not any(r.source_url in logged or r.target_url in logged for r in rows)


@pytest.mark.parametrize(
    ("stored", "problem"),
    [
        ({"anchors": models("voyage-3-large")}, "different model than configured"),
        ({"sentences": models(MODEL, "voyage-3-large")}, "mix 2 embedding models"),
        ({"pages": models(MODEL, "voyage-3-large")}, "mix 2 embedding models"),
        ({"pages": models(None)}, "no embeddingModel"),
    ],
    ids=["other-anchor-model", "mixed-sentences", "mixed-pages", "page-without-model"],
)
async def test_vectors_of_another_model_stop_the_run_before_any_write(
    stored: dict[str, tuple[EmbeddingModelCount, ...]], problem: str
) -> None:
    graph = FakeGraph((1, 0), [row(0, 0.5, 0.5)], **stored)

    with pytest.raises(EmbeddingModelMismatchError, match=problem):
        await score_links(graph, TENANT)  # type: ignore[arg-type]
    assert not any(call.startswith("score") for call in graph.calls)


async def test_a_tenant_without_page_vectors_skips_the_model_check() -> None:
    graph = FakeGraph((0, 2), [], pages=(), anchors=models("voyage-3-large"))

    report = await score_links(graph, TENANT)  # type: ignore[arg-type]

    assert (report.links, report.scored) == (2, 0)
    assert f"anchor models {TENANT}" not in graph.calls


async def test_a_tenant_without_scored_links_has_no_distributions() -> None:
    report = await score_links(FakeGraph((0, 3), []), TENANT)  # type: ignore[arg-type]

    assert (report.links, report.scored, report.context, report.anchor) == (3, 0, None, None)


async def test_links_changing_between_write_and_read_fail_the_run() -> None:
    graph = FakeGraph((2, 0), [row(0, 0.5, 0.5)])

    with pytest.raises(DatabaseReadError, match="2 links scored but 1 read back"):
        await score_links(graph, TENANT)  # type: ignore[arg-type]


async def test_scored_links_without_any_sentence_vector_ask_for_embed_links() -> None:
    graph = FakeGraph((1, 0), [row(0, None, 0.5)])

    with pytest.raises(ValueError, match="run embed-links first"):
        await score_links(graph, TENANT)  # type: ignore[arg-type]


async def test_a_blank_tenant_is_rejected_before_any_read() -> None:
    graph = FakeGraph((0, 0), [])

    with pytest.raises(ValueError, match="tenant_id"):
        await score_links(graph, " ")  # type: ignore[arg-type]
    assert graph.calls == []


async def test_the_summary_names_the_counts_and_each_split() -> None:
    rng = np.random.default_rng(7)
    scores = np.clip(np.concatenate([rng.normal(0.3, 0.03, 200), rng.normal(0.7, 0.03, 100)]), 0, 1)
    rows = [row(i, float(v), None) for i, v in enumerate(scores)]

    report = await score_links(FakeGraph((len(rows), 0), rows), TENANT)  # type: ignore[arg-type]
    summary = summarise_link_relevance(report)

    assert summary.startswith(f"Link relevance for tenant {TENANT}: 300 of 300 body links")
    assert report.context is not None
    assert report.context.split is not None
    assert f"Split at {report.context.split:.3f}" in summary
    assert "Anchor-target fit (anchor vs target): no scores." in summary
