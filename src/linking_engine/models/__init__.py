"""Pydantic domain models - the only contract crossing a module boundary.

Bare dicts between modules are a lint failure. Every model here is frozen and
forbids extra fields, so a renamed field fails loudly at the boundary rather than
silently dropping data. Frozen means hashable, which is why containers are
``frozenset`` and ``tuple`` throughout and never ``set`` or ``list``.

The ``models-are-leaves`` import-linter contract keeps this package free of
neo4j, motor, voyageai, prefect, mlflow and igraph: models describe data, they do
not fetch it, compute it, or log it.
"""

from linking_engine.models.audit import LinkAuditResult
from linking_engine.models.corpus import (
    CleanedPage,
    CrawlPage,
    ExtractedLink,
    GraphLoadReport,
    Heading,
    LinkRecord,
    PageRecord,
    PageSummary,
)
from linking_engine.models.embedding import PageEmbedding, PageText
from linking_engine.models.enums import (
    ActionType,
    AnchorType,
    ContentGapFinding,
    IssueFlag,
    KeywordSource,
    LifecycleStage,
    PageType,
    RecommendationStatus,
)
from linking_engine.models.features import PairFeatures
from linking_engine.models.page import Keyword, Link, Page, TenantGraphCounts
from linking_engine.models.recommendation import AnchorCandidate, Recommendation
from linking_engine.models.tenant import AnchorTypeProfile, TenantConfig

__all__ = [
    "ActionType",
    "AnchorCandidate",
    "AnchorType",
    "AnchorTypeProfile",
    "CleanedPage",
    "ContentGapFinding",
    "CrawlPage",
    "ExtractedLink",
    "GraphLoadReport",
    "Heading",
    "IssueFlag",
    "Keyword",
    "KeywordSource",
    "LifecycleStage",
    "Link",
    "LinkAuditResult",
    "LinkRecord",
    "Page",
    "PageEmbedding",
    "PageRecord",
    "PageSummary",
    "PageText",
    "PageType",
    "PairFeatures",
    "Recommendation",
    "RecommendationStatus",
    "TenantConfig",
    "TenantGraphCounts",
]
