# Data Model

## Neo4j

### Nodes

```cypher
(:Page {
  url, pageType, isIndexable, httpStatus,
  wordCount, crawlDepth, freshness, language,
  publishedAt, lifecycleStage,
  pageRank, linkCommunityId, keywordCommunityId, betweenness,
  content_embedding,   // 2048d, HNSW cosine
  gnn_embedding,       // 2048d, HNSW cosine
  hubId,               // HDBSCAN cluster, -1 = noise      NEW v5.5
  pageRank,            // over body links only — nav/footer never captured
  crawlDepth,
  isChunked,           // false in MVP; reserved            NEW v5.5
  embeddingModel, embeddingDimensions, embeddedContentHash
})

(:Keyword { text, language, searchVolume, difficulty, isStrategic })
```

### Relationships

```cypher
-[:LINKS_TO {
    anchorText, anchorType, linkPosition, weight, isFollow,
    surroundingText,          // required by the audit
    surroundingEmbedding,     // 2048d, precomputed — no query-time embedding
    targetHttpStatus,
    anchorQualityScore, keywordAlignment, contextRelevance,
    equityEfficiency, issueFlags, auditedAt
}]->

-[:SUGGESTED_ACTION {
    actionType,     // ADD_LINK | REANCHOR | REMOVE | FIX | CONTENT_GAP
    finding,        // NO_TOPICAL_MENTION | AWKWARD_PHRASING  (CONTENT_GAP only)
    score, tier, status,
    currentAnchor,      // null for ADD_LINK
    proposedAnchors,    // null for REMOVE / FIX
    rationale, signals, createdAt
}]->

-[:TARGETS_KEYWORD {
    relevanceScore,
    source,          // CLIENT_STRATEGIC | GSC_OBSERVED | INFERRED
    priority,        // 1-5, strategic only
    currentPosition, isPrimary
}]->
```

**Only body links are stored.** The crawler discards nav, header, footer and
sidebar links at extraction, so `LINKS_TO` contains editorial links exclusively
and no filtering is needed at graph-build time.

`surroundingEmbedding` lives on the relationship. Neo4j cannot embed text at query time, so it is computed in the embedding stage and written back. Relationship properties hold vectors fine, but there is **no relationship vector index** in Community Edition — you cannot ANN-search over them. Acceptable, since the audit scans edges anyway.

The `source` discriminator on `TARGETS_KEYWORD` is what makes keyword gap computable:

```
keyword_gap = keywords the page TARGETS (strategic)
            − keywords the page RANKS FOR (observed)
```

### Indexes

```cypher
CREATE CONSTRAINT page_url IF NOT EXISTS
  FOR (p:Page) REQUIRE p.url IS UNIQUE;

CREATE VECTOR INDEX page_content IF NOT EXISTS
  FOR (p:Page) ON p.content_embedding
  OPTIONS { indexConfig: {
    `vector.dimensions`: 2048,
    `vector.similarity_function`: 'cosine' }};

CREATE VECTOR INDEX page_gnn IF NOT EXISTS
  FOR (p:Page) ON p.gnn_embedding
  OPTIONS { indexConfig: {
    `vector.dimensions`: 2048,
    `vector.similarity_function`: 'cosine' }};
```

**Dimension is fixed at creation.** Changing it means dropping the index,
re-embedding every page, and retraining the GNN.

2048 is chosen because Matryoshka makes it the reversible direction: 1024 is
literally the first 1024 dimensions of the 2048 vector, so downgrading later is a
truncation of data already held, while upgrading needs a full re-embed. Nobody has
measured whether 2048 helps on full-page document-to-document comparison with
voyage-4-large — the earlier 1024 test was voyage-4-nano, one article, three short
queries. See Measurement Backlog.

---

## MongoDB

| Collection | Contents |
|---|---|
| `pages` | title, h1, meta, bodyText, crawledAt, language, hreflangMap, publishedAt |
| `gsc_metrics` | impressions_28d, avg_position, ctr, ctr_gap, trend, query_count |
| `gsc_queries` | query, impressions, clicks, position, date_bucket, url |
| `strategic_keywords` | url, keyword, priority, isPrimary, language, searchVolume |
| `link_audit` | per-edge scores and issue history over time |
| `recommendations` | full payload: placements, anchors, signals, assumptions, model versions |
| `anchor_feedback` | anchor used, type, accepted/modified, actionType, anchor source |
| `tenant_config` | see below |
| `ctr_curves` | per-tenant CTR by position, derived from that tenant's GSC |

`anchor_feedback.actionType` is what answers the roadmap question empirically. If `REANCHOR` acceptance runs far above `ADD_LINK`, the audit is the product.

---

## Tenant config

```json
{
  "tenantId": "client_abc",
  "embeddingProvider": "VOYAGE_API",
  "embeddingModel": "voyage-4-large",
  "embeddingDimensions": 2048,
  "embeddingLocalFallbackModel": "voyage-4-nano",

  "contentGapModel": "qwen3.5:4b",
  "contentGapMinPriority": 4,
  "contentGapMinOpportunity": 500,
  "anchorTypeProfile": {"exact":0.15,"partial":0.20,"natural":0.50,"branded":0.15},

  "auditEnabled": true,
  "discoveryEnabled": false,

  "embeddingStrategy": "MULTILINGUAL_EMBED",
  "siteLanguages": ["en","de","fr"],
  "clientTier": "STARTER",
  "lifecycleBoostEnabled": true,
  "reservedNewPageSlotPct": 0.15,
  "diversityCapPct": 0.15,
  "maxRecommendationsPerSource": 10
}
```

`auditEnabled` and `discoveryEnabled` are independent — Stage 1 ships and runs alone.

---

## Pydantic contracts

Models are the only thing crossing module boundaries. Bare dicts are a lint failure.

```python
class ActionType(StrEnum):
    ADD_LINK = "ADD_LINK"
    REANCHOR = "REANCHOR"
    REMOVE = "REMOVE"
    FIX = "FIX"
    CONTENT_GAP = "CONTENT_GAP"

class AnchorCandidate(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    text: str = Field(min_length=1, max_length=120)
    anchor_type: AnchorType
    source: Literal["EXTRACTED", "GENERATED"]
    score: float = Field(ge=0, le=1)
```

`source` distinguishes existing page text from an insertion requiring a copy edit. Post-v5 it should always be `EXTRACTED`; the field stays so acceptance can be tracked if that ever changes.
