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
    PrepareReport,
    QueryParamEvidence,
    TemplateInlinks,
    TemplateLink,
)
from linking_engine.models.embedding import (
    AnchorKeyUpdate,
    EdgeRef,
    EmbeddingBatch,
    EmbeddingModelCount,
    EmbeddingSelection,
    EmbeddingTarget,
    EmbedRunReport,
    LinkEmbedReport,
    LinkText,
    PageEmbedding,
    PageText,
    SentenceTarget,
)
from linking_engine.models.enums import (
    ActionType,
    AnchorType,
    ContentGapFinding,
    IssueFlag,
    KeywordSource,
    LifecycleStage,
    OrphanLabel,
    PageType,
    RecommendationStatus,
)
from linking_engine.models.features import PairFeatures
from linking_engine.models.page import (
    CentralityReport,
    CommunityContext,
    CommunityReport,
    Keyword,
    Link,
    LinkGraphSnapshot,
    Page,
    PageCentrality,
    PageCommunities,
    PassReport,
    TenantGraphCounts,
)
from linking_engine.models.recommendation import AnchorCandidate, Recommendation
from linking_engine.models.tenant import AnchorRules, AnchorTypeProfile, TenantConfig

__all__ = [
    "ActionType",
    "AnchorCandidate",
    "AnchorKeyUpdate",
    "AnchorRules",
    "AnchorType",
    "AnchorTypeProfile",
    "CentralityReport",
    "CleanedPage",
    "CommunityContext",
    "CommunityReport",
    "ContentGapFinding",
    "CrawlPage",
    "EdgeRef",
    "EmbedRunReport",
    "EmbeddingBatch",
    "EmbeddingModelCount",
    "EmbeddingSelection",
    "EmbeddingTarget",
    "ExtractedLink",
    "GraphLoadReport",
    "Heading",
    "IssueFlag",
    "Keyword",
    "KeywordSource",
    "LifecycleStage",
    "Link",
    "LinkAuditResult",
    "LinkEmbedReport",
    "LinkGraphSnapshot",
    "LinkRecord",
    "LinkText",
    "OrphanLabel",
    "Page",
    "PageCentrality",
    "PageCommunities",
    "PageEmbedding",
    "PageRecord",
    "PageSummary",
    "PageText",
    "PageType",
    "PairFeatures",
    "PassReport",
    "PrepareReport",
    "QueryParamEvidence",
    "Recommendation",
    "RecommendationStatus",
    "SentenceTarget",
    "TemplateInlinks",
    "TemplateLink",
    "TenantConfig",
    "TenantGraphCounts",
]
