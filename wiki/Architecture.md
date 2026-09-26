# Architecture

Eleven stages. Stage 1 (audit) and Stage 2 (discovery) share the foundation and the anchor resolver.

| Stage | Input | Output |
|---|---|---|
| 00 Ingestion | Site URL, GSC property, keyword CSV | Page + link records, metrics, `TARGETS_KEYWORD` edges |
| 01 Embedding | Page body, link surrounding text | `content_embedding[2048]`, `surroundingEmbedding[2048]` |
| 02 Graph analytics | Link graph, keyword graph | `pageRank`, `betweenness`, two community ids |
| 02b Content clustering | `content_embedding` | `hubId`, `-1` noise label |
| **A1 Audit, no embeddings** | Every `LINKS_TO` edge | `FIX`, `REANCHOR`, saturation `REMOVE` |
| **A2 Audit, embeddings** | Edge + vectors | `contextRelevance`, `anchorTargetFit`, refined `REANCHOR` |
| 03 GNN encoding | ~270 scalars + content embedding | `gnn_embedding[2048]` |
| 04 Candidate retrieval | Target `gnn_embedding` | Top 50 candidate pairs |
| 05 Cross-attention | Two `gnn_embedding`s | 3105d pair vector |
| 06 LambdaMART | 3105d pair vector | `opportunity_score` 0–100 |
| 07 Anchor resolution | Target keyword + source body | Extracted phrase, or `CONTENT_GAP` |
| 08 Materialise | Scored actions | `SUGGESTED_ACTION` edges, Mongo payload, cache invalidation |
| 09 Retraining | `anchor_feedback` | Three versioned model artefacts |

---

## Trigger model

Event-driven. Nothing runs on a timer. The pipeline wakes on new pages, material content change, a GSC threshold shift, or a manual run. Between triggers the API serves entirely from Valkey.

---

## Anchor resolution

Shared by `ADD_LINK` and `REANCHOR` — one implementation, two applications.

```
keyword:  strategic primary
       →  GSC query by opportunity value: impressions × (CTR@1 − CTR@current)
       →  title / h1

ladder:   1  exact keyword verbatim in source body   → link it
          2  close variant (stem, plural, modifier)  → link it
          3  semantically related phrase             → link it
          4  nothing                                 → CONTENT_GAP (gated)

ladder:   1    exact keyword verbatim              → link it
          2    close variant (stem, plural)        → link it
          2.5  jaccard on stemmed token sets       → link it
          3    semantic phrase via cosine          → link it
          4    nothing suitable                    → CONTENT_GAP

score:    semantic .35  keyword .35  diversity .15  length .15
          keyword  = 0.7 jaccard(stemmed) + 0.3 cosine
          diversity = jaccard vs existing anchors — NOT cosine
```

Type distribution (15/20/50/15) is a **preference, not a constraint**. Extraction runs first; the profile only chooses among candidates that already exist in the copy. If the sole available phrase is a partial match, it is used even when the profile wants exact.

---

## Body links only

The crawler extracts links from body content. Nav, header, footer and sidebar
links are never captured, so nothing downstream sees them. Template links are an
SEO concern audited separately, per template rather than per page.

Justification is in [[Decisions]]: with template links included, 10/10 of the top
PageRank and betweenness pages are utility pages like `/terms`. Clustering is
unaffected either way.

Consequence: `REPOSITION` is not a verdict — with no footer links captured there
is nothing to move a link out of. Five action types, not six.

## Candidate eligibility

Hard constraints only:

```
isIndexable = true
AND source ≠ target
AND NOT (source)-[:LINKS_TO]->(target)
```

Everything else — impressions, position, page age, saturation, strategic priority — is a feature the ranker weighs. Eligibility is not priority.

---

## Serving

Cache-first. Two caps applied post-ranking, outside the pipeline:

```
diversity cap    no target in more than 15% of a site-wide result set
source cap       no source contributes more than 10 items
```

The source cap is the mirror of the diversity cap. Equity divides across a page's outbound links, so twenty new links on one page is not a useful work queue.

---

## What each stage costs

Projected for 25k pages, 8 cores, 32 GB. Replace with Prometheus data after the
first client.

```
Crawl                 2.2 h    ← network-bound, the real bottleneck
GSC pull              45 min   ← API quota
Embedding             28 min   ← 120K tokens/request, 3M TPM
Graph analytics        2-4 min ← igraph, exact betweenness
HDBSCAN                1-3 min
Audit pass A           8 min   ← runs during embedding
Audit pass B          20 min
GNN inference         12 min
Retrieval             10 min
Cross-attention       12 min
LambdaMART            35 min
Anchor resolution     18 min
Materialise           18 min
                     ─────────
Initial run          ~4.5 h
```

**Two stages run in parallel with embedding.** Graph analytics touches no
vectors, so it starts the moment the crawl ends. Audit pass A needs only
`pageRank` and string matching. Both fit inside the embedding window.

**GraphSAGE is a hard barrier.** It aggregates over neighbourhoods, so a node
whose neighbours are not yet embedded produces a wrong vector silently. Partial
HNSW is the same trap — querying a 60%-populated index returns plausible
candidates from an incomplete pool, with no error.

```
① embedding  ─┐
              ├─ parallel
② analytics  ─┘
A1 audit     ─┘
              ↓  BARRIER: all pages embedded
③ GraphSAGE → ④ → ⑤ → ⑥ → ⑦ → ⑧
```

One thing to build in from the start: **chunked LambdaMART inference**. At 2048d
embeddings the pair vector roughly doubles, so 1.25M pairs assembled in one pass
will not coexist with the Neo4j heap on a 32 GB node. Stream in ~50k chunks.

The crawler is still 40% of the run and none of this touches it. If you want the
run dramatically faster, that is where the hours are.
