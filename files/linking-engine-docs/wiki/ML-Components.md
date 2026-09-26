# ML Components

Nine models in pipeline order. Three of them learn from your data; the rest are pre-trained or deterministic.

| Model | Input | Output | Task type | Trained? |
|---|---|---|---|---|
| `voyage-4-large` | Page body text | `content_embedding[2048]` | Text → vector, bi-encoder | Pre-trained, frozen |
| PageRank | Weighted link graph | Scalar per page | Graph → scalar | No |
| Leiden A | Link graph | `linkCommunityId` | Graph → cluster id | Unsupervised, per run |
| Leiden B | Page–keyword bipartite | `keywordCommunityId` | Graph → cluster id | Unsupervised, per run |
| HDBSCAN | `content_embedding` | `hubId`, plus `-1` noise | Vectors → cluster id | Unsupervised, per run |
| Betweenness | Link graph | Scalar per page | Graph → scalar | No |
| GraphSAGE | Page features + neighbours | `gnn_embedding[2048]` | Node + neighbourhood → vector | **Yes, monthly** |
| HNSW | One query vector | Top 50 pages | Doc → doc search | No |
| Cross-attention | Two `gnn_embedding`s | 3105d pair vector | Doc ↔ doc pairwise | **Yes, monthly** |
| LambdaMART | 3105d pair vector | Score 0–100 | Pair → score, learning-to-rank | **Yes, monthly** |
| Qwen3.5-4B | Failed extraction context | `CONTENT_GAP` verdict | Text → classification | Pre-trained, frozen |

---

## Encoding modes

```
voyage        one doc in, one vector out, independently
              → indexable, but no pair awareness

HNSW          doc-to-doc search over those vectors
              → cheap, approximate, O(log N)

cross-attn    both docs in together, attention across them
              → expensive, accurate, no index possible
```

Standard two-stage retrieval: bi-encoder narrows, cross-encoder judges. **Both sides of every comparison are documents.** There is no query encoding anywhere in this system — which is why a query-document reranker is the wrong shape.

---

## Why two embeddings

```
content_embedding    what this page says
gnn_embedding        what it says AND where it sits
```

Two pages with identical copy — one a pillar linked from forty places, one an orphan six clicks deep — get identical `content_embedding` and very different `gnn_embedding`.

Both are kept because `content_embedding` is GraphSAGE's input, they have different update costs (content edit re-embeds one page; graph change shifts every neighbourhood), and retrieval runs on `gnn_embedding` while sentence matching uses content vectors.

If the GNN never beats the heuristic baseline, `gnn_embedding` goes away and raw graph scalars do the job directly. The second embedding is provisional until PR #53.

---

## Three clusterings, three questions

```
linkCommunityId      Leiden over LINKS_TO         what is connected
keywordCommunityId   Leiden over TARGETS_KEYWORD  what we say it is about
hubId                HDBSCAN over embeddings      what it is actually about
```

HDBSCAN's unique output is the `-1` label: a page belonging to no coherent
topic. Neither Leiden pass can express that — Leiden must assign every node to
a community, so a genuinely off-topic page with a few footer links gets placed
somewhere. That distinction is actionable:

```
noise + high strategic priority   content strategy problem
noise + low priority              thin page, probably fine
```

It also gives a real hierarchy through the condensed tree, and adapts to
per-cluster density — Leiden's resolution parameter is global, so it cannot
serve a tight product cluster and a diffuse blog cluster equally.

Two things only HDBSCAN enables:

**Hub bridges.** Rank cluster pairs by `0.4 × centroid similarity + 0.6 × query
Jaccard − link density`. Produces a handful of high-leverage structural links
rather than thousands of page-level suggestions.

**Pillar check.** Compare the declared pillar against the page nearest the
cluster centroid. Disagreement means the intended hub is not the most topically
central page.

Whether all three earn their place is Measurement Backlog §1. If Leiden and
HDBSCAN agree above ~0.85 ARI, one is a duplicate feature.

## Two Leiden passes

Pass A clusters by existing links, which is **circular for discovery** — a new section has no links, so no clusters, so no recommendations.

Pass B clusters by shared target keywords. Those relationships exist before any links do, so it is cold-start safe, and pillar/spoke roles fall out: pillar is the head term by aggregate volume, spokes are long-tail variants.

**The disagreement is the signal:**

```
same keyword cluster + same link cluster       → already well connected
same keyword cluster + different link cluster  → the missing link
different keyword    + same link cluster       → possibly a link to remove
```

Both ids feed the model as features.

---

## Where the learning lives

Everything the system knows about your team's judgement is in three artefacts, all trained on the same signal — which recommendations were accepted, edited, or dismissed.

```
GraphSAGE        contrastive loss: accepted pairs pulled together
cross-attention  trained jointly with the ranker
LambdaMART       lambdarank on graded labels, NDCG@10

promotion gate   new version ships only if it beats production
                 on holdout NDCG@10 (MLflow alias move)
```

This is the structural argument for shipping the audit first: it needs none of them, and it produces their labels.
