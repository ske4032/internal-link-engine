"""One valid instance per domain model, and the registry the parametrised tests drive.

The kwargs tables are written against the binding spec in issue #2 and the tenant config
block in wiki/Data-Model.md. They are the single place that changes when the
implementation deviates deliberately: the tests themselves iterate `model_fields`
wherever the exact field set is not the thing being asserted.

Where the spec does not pin a scalar's type, the value here is chosen to be valid under
every plausible annotation (`1` validates as int, float or bool; `"1"` additionally
validates as str) so that an ambiguity in the spec cannot masquerade as a model bug.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from linking_engine.models.audit import LinkAuditResult
from linking_engine.models.enums import (
    ActionType,
    AnchorType,
    IssueFlag,
    LifecycleStage,
    PageType,
    RecommendationStatus,
)
from linking_engine.models.features import PairFeatures
from linking_engine.models.page import Keyword, Link, Page
from linking_engine.models.recommendation import AnchorCandidate, Recommendation
from linking_engine.models.tenant import TenantConfig

if TYPE_CHECKING:
    from pydantic import BaseModel

SOURCE_URL = "https://example.com/guides/trail-running-shoes"
TARGET_URL = "https://example.com/shop/trail-shoes"
TIMESTAMP = datetime(2026, 1, 15, 12, 30, tzinfo=UTC)

# Short on purpose. The spec declares `tuple[float, ...] | None` with no length
# constraint, and a 2048-float literal would bury every assertion message.
VECTOR = (0.11, -0.42, 0.87, 0.03)

# The GSC block of PairFeatures, which has to distinguish missing from zero.
GSC_FIELDS = (
    "source_impressions_log",
    "target_impressions_log",
    "target_position_band",
    "target_ctr_gap",
    "target_query_count",
)

# The LinkAuditResult dimensions bounded to [0, 1]. `anchor_quality_score` is not one
# of them — it is 0-100, which is the distinction worth a test.
UNIT_INTERVAL_FIELDS = (
    "keyword_alignment",
    "context_relevance",
    "anchor_target_fit",
    "equity_efficiency",
)


@dataclass(frozen=True)
class ModelSpec:
    """A model, one set of kwargs that builds a valid instance, and one field change."""

    name: str
    model: type[BaseModel]
    kwargs: dict[str, object]
    mutation: dict[str, object]

    def build(self) -> BaseModel:
        return self.model(**self.kwargs)

    def kwargs_with(self, **overrides: object) -> dict[str, object]:
        return {**self.kwargs, **overrides}


PAGE_KWARGS: dict[str, object] = {
    "url": SOURCE_URL,
    "page_type": PageType.ARTICLE,
    "is_indexable": True,
    "status_code": 200,
    "is_placeholder": False,
    "content_hash": "ab" * 32,
    "body_hash": "cd" * 32,
    "word_count": 1450,
    "crawl_depth": 2,
    "language": "en",
    "freshness": 1,
    "published_at": "2024-06-01",
    "lifecycle_stage": LifecycleStage.ESTABLISHED,
    "page_rank": 0.0042,
    "betweenness": 0.13,
    "link_community_id": 3,
    "keyword_community_id": 7,
    "hub_id": 2,
    "is_chunked": False,
    "embedding_model": "voyage-4-large",
    "embedding_dimensions": 2048,
    # Differs from body_hash so a swapped mapping cannot round-trip.
    "embedded_body_hash": "9f" * 32,
    "embedded_at": TIMESTAMP,
    "content_embedding": VECTOR,
    "gnn_embedding": VECTOR,
}

KEYWORD_KWARGS: dict[str, object] = {
    "text": "trail running shoes",
    "language": "en",
    "search_volume": 1200,
    "difficulty": 1,
    "is_strategic": True,
}

LINK_KWARGS: dict[str, object] = {
    "source_url": SOURCE_URL,
    "target_url": TARGET_URL,
    "position": 0,
    "anchor_text": "trail running shoes",
    "anchor_type": AnchorType.PARTIAL,
    # Body links only (ADR-004): "body" is the only value that can reach the model.
    "link_position": "body",
    "weight": 1.0,
    "is_follow": True,
    "surrounding_text": "A good pair of trail running shoes matters more than the route.",
    "surrounding_embedding": VECTOR,
    "target_status_code": 200,
}

AUDIT_KWARGS: dict[str, object] = {
    "source_url": SOURCE_URL,
    "target_url": TARGET_URL,
    "anchor_quality_score": 72.5,
    "keyword_alignment": 0.81,
    "context_relevance": 0.64,
    "anchor_target_fit": 0.77,
    "equity_efficiency": 0.42,
    "issue_flags": frozenset({IssueFlag.GENERIC, IssueFlag.MISALIGNED}),
    "verdict": ActionType.REANCHOR,
    "audited_at": TIMESTAMP,
}

PAIR_KWARGS: dict[str, object] = {
    "source_url": SOURCE_URL,
    "target_url": TARGET_URL,
    # gsc
    "source_impressions_log": 9.21,
    "target_impressions_log": 8.15,
    "target_position_band": "1",
    "target_ctr_gap": 0.04,
    "target_query_count": 42,
    "has_gsc_data": True,
    # lifecycle
    "source_lifecycle_stage": LifecycleStage.MATURE,
    "target_lifecycle_stage": LifecycleStage.NEW,
    "source_page_age_days": 820,
    "target_page_age_days": 12,
    # strategic
    "target_kw_count": 5,
    "target_max_priority": 4,
    "target_keyword_gap": 2,
    "pair_kw_overlap": 0.25,
    # target structural
    "target_inbound_count": 3,
    "target_is_orphan": False,
    "target_crawl_depth": 2,
    "target_saturation_ratio": 0.45,
    # source structural
    "source_outbound_count": 12,
    "source_outbound_density": 0.008,
    "source_link_equity_share": 0.083,
    # cluster
    "source_link_community_id": 3,
    "target_link_community_id": 7,
    "source_keyword_community_id": 1,
    "target_keyword_community_id": 4,
    "source_hub_id": 2,
    "target_hub_id": 5,
    "cluster_agreement": 1,
    # semantic
    "content_cosine": 0.62,
    "context_relevance": 0.71,
    "anchor_target_fit": 0.55,
}

ANCHOR_KWARGS: dict[str, object] = {
    "text": "trail running shoes",
    "anchor_type": AnchorType.PARTIAL,
    "source": "EXTRACTED",
    "score": 0.78,
}

RECOMMENDATION_KWARGS: dict[str, object] = {
    "source_url": SOURCE_URL,
    "target_url": TARGET_URL,
    "action_type": ActionType.ADD_LINK,
    # null for anything that is not CONTENT_GAP
    "finding": None,
    "score": 64.0,
    # the spec does not pin tier's type; "1" validates as str, int or float
    "tier": "1",
    "status": RecommendationStatus.PENDING,
    # null for ADD_LINK — there is no existing anchor to replace
    "current_anchor": None,
    "proposed_anchors": (AnchorCandidate(**ANCHOR_KWARGS),),
    "rationale": "Target is orphaned and sits in the same hub as the source.",
    "signals": {"content_cosine": 0.62, "target_is_orphan": 1.0},
    "created_at": TIMESTAMP,
}

# Only the identifier. Every other value must come from the model's own defaults, which
# is what test_tenant.py asserts against the wiki.
TENANT_KWARGS: dict[str, object] = {"tenant_id": "client_abc"}


PAGE_SPEC = ModelSpec("Page", Page, PAGE_KWARGS, {"word_count": 999})
KEYWORD_SPEC = ModelSpec("Keyword", Keyword, KEYWORD_KWARGS, {"text": "road running shoes"})
LINK_SPEC = ModelSpec("Link", Link, LINK_KWARGS, {"anchor_text": "click here"})
AUDIT_SPEC = ModelSpec(
    "LinkAuditResult", LinkAuditResult, AUDIT_KWARGS, {"anchor_quality_score": 12.0}
)
PAIR_SPEC = ModelSpec("PairFeatures", PairFeatures, PAIR_KWARGS, {"content_cosine": 0.11})
ANCHOR_SPEC = ModelSpec("AnchorCandidate", AnchorCandidate, ANCHOR_KWARGS, {"score": 0.11})
RECOMMENDATION_SPEC = ModelSpec(
    "Recommendation", Recommendation, RECOMMENDATION_KWARGS, {"score": 11.0}
)
TENANT_SPEC = ModelSpec("TenantConfig", TenantConfig, TENANT_KWARGS, {"tenant_id": "client_xyz"})

MODEL_SPECS = (
    PAGE_SPEC,
    KEYWORD_SPEC,
    LINK_SPEC,
    AUDIT_SPEC,
    PAIR_SPEC,
    ANCHOR_SPEC,
    RECOMMENDATION_SPEC,
    TENANT_SPEC,
)

MODEL_IDS = [spec.name for spec in MODEL_SPECS]
