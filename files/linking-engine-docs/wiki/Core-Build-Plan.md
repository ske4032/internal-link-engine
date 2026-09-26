# Core Build Plan

Scope: **Voyage embedding through to a validated intelligence core.**

Excluded because they already exist or come later: crawler, GSC client, REST
API, Valkey, MinIO, multi-tenancy, Helm, ArgoCD. Those plug in around this once
the core is proven.

Runs against four datasets, each with a distinct job. Three synthetic corpora
carry planted ground truth, so "working" is a set of pass/fail checks rather than
a judgement call. The fourth is a real crawl, which settles the things synthetic
data structurally cannot.

| Dataset | Job |
|---|---|
| **600 pages** synthetic | Correctness. Fast enough to iterate many times a day. All seven ground-truth gates run here |
| **5,000** synthetic | Scale behaviour. Betweenness grows 70×, LambdaMART memory starts mattering, HNSW recall becomes measurable |
| **25,000** synthetic | Validates the runtime projections. Run occasionally, not per change |
| **1 real crawl** | Extraction hit rate, fixable-link rate, embedding behaviour on real text, and real acceptance labels |

Generate the synthetic three with the same seed so results compare across sizes.

**Estimate: 20–24 days with LLM assistance.** The spec is already written, which
is what makes that number reachable; most of these tasks are well-specified
transforms rather than design work.

---

## Phase 0 — Foundation · 2 days

| # | Task | Days |
|---|---|---|
| 1 | Repo scaffold: uv, ruff, mypy strict, pytest | 0.5 |
| 2 | Pydantic domain models: `Page`, `Link`, `Keyword`, `PairFeatures`, `Recommendation` | 1 |
| 3 | Neo4j async repo, Mongo repo, schema migrations, vector indexes @2048d | 0.5 |

Neo4j, MongoDB, Prefect and MLflow already run outside this repository
(ADR-014), so there is no stack to stand up. Schema, vector indexes and Mongo
collection indexes all come from task 3.

> **Vector index dimension is fixed at creation.** 2048 because MRL makes
> downgrading to 1024 a truncation of data already held, while upgrading needs a
> full re-embed and GNN retrain.

---

## Phase 1 — Embedding · 3 days

| # | Task | Days |
|---|---|---|
| 4 | Voyage client: token-aware batching, retry, rate-limit backoff | 1.5 |
| 5 | Write path: buffered flush, `embeddedContentHash` resume | 1 |
| 6 | Anchor + surrounding-sentence embeddings, deduplicated | 0.5 |

```python
model              voyage-4-large
output_dimension   2048
input_type         document        # everything here is a document
max per request    120,000 tokens  # NOT 1M — that is the lite models
max list length    1,000
rate limits        3M TPM / 2000 RPM
```

At ~3,300 tokens per page that is **~36 pages per request**. Batch by
`vo.count_tokens`, never by list length.

Decouple the API batch from the Neo4j write: accumulate ~500 pages, flush in one
`UNWIND`, set `embeddedContentHash` in the same transaction. A crash then leaves
a consistent prefix. Do not buffer the whole run — 25k pages at 2048d float32 is
~200 MB before anything is durable.

Anchor text repeats heavily; 250k edges might yield 8k unique strings. Dedupe
before sending.

**Gate:** every page embedded, no mixed `embeddingModel` values in the graph.

---

## Phase 2 — Graph analytics · 3 days

| # | Task | Days |
|---|---|---|
| 7 | igraph backend: edge pull, id mapping, graph construction | 1 |
| 8 | PageRank + betweenness (exact), write-back | 0.5 |
| 9 | Leiden ×2 via `leidenalg`: link graph, keyword graph | 1 |
| 10 | HDBSCAN over content embeddings → `hubId` + noise label | 0.5 |

```
edge pull       10-25s
build igraph     1-3s
PageRank         2-5s
Leiden ×2        5-15s
betweenness     60-190s   exact, all cores
write back      10-20s
─────────────────────────
                ~2-4 min
```

Neo4j internal ids are not stable across restarts. Build the mapping fresh each
run rather than caching it.

Only body links exist in the graph — the crawler discards nav, header, footer and
sidebar links at extraction, so no filtering is needed here. Measured
justification: with template links included, 10/10 of the top PageRank and
betweenness pages are utility pages. Clustering is unaffected either way.



**Gate — run this before writing Phase 3.** Measure ARI/NMI against planted
topics, noise precision and recall, per-topic purity, and label churn across
runs. The evaluation runs Leiden through `leidenalg`, exactly as the pipeline
does. The earlier GDS-based `make eval` harness is retired by the ADR-002
amendment.

```
ARI(leiden, hdbscan) 0.3-0.7    keep both, they disagree usefully
                     > 0.85     drop HDBSCAN, duplicate feature
noise_recall         < 0.5      HDBSCAN's main advantage failed
spearman(bc, pagerank) > 0.8    delete betweenness, saves 60-190s/run
```

Two of those can remove work from the rest of the plan.

---

## Phase 3 — Retrieval · 3 days

| # | Task | Days |
|---|---|---|
| 11 | Candidate retrieval: hard constraints, HNSW top-50 | 1.5 |
| 12 | Jaccard signals: GSC query overlap, strategic keyword overlap | 1 |
| 13 | Cluster membership signals from both Leiden passes + `hubId` | 0.5 |

Eligibility is three constraints and nothing else:

```
isIndexable = true
AND source ≠ target
AND NOT (source)-[:LINKS_TO]->(target)
```

Everything else is a signal, not a gate. Impressions, position, page age,
saturation — all become features. Eligibility is not priority.

**Gate:** the 3 planted orphan NEW pages appear as candidate targets. If they
don't, something is still filtering.

---

## Phase 4 — Features and heuristic scoring · 4 days

| # | Task | Days |
|---|---|---|
| 14 | Feature assembly: ~30 context features | 2 |
| 15 | Cosine dimensions: `contextRelevance`, `anchorTargetFit` | 0.5 |
| 16 | Hand-weighted linear scorer, `opportunity_score` 0–100 | 1 |
| 17 | Hub-to-hub bridge scoring | 0.5 |

```
GSC          impressions_log, position_band (with NULL bucket), ctr_gap,
             has_gsc_data, query_count
lifecycle    stage one-hot, page_age_days
strategic    kw_count, max_priority, keyword_gap, pair_kw_overlap
target       inbound_count, is_orphan, crawl_depth, saturation_ratio
source       outbound_count, outbound_density, link_equity_share
cluster      linkCommunityId, keywordCommunityId, hubId, cluster_agreement
semantic     cosine(source, target), contextRelevance, anchorTargetFit
```

`target_saturation_ratio` is actual inbound ÷ expected for its PageRank tier.
Source outbound matters because equity divides across a page's links — a 151st
link on a saturated page is near-worthless.

**Measure discipline.** Jaccard where set sizes are comparable, cosine where they
are not:

```
GSC query sets        ~40 vs ~40 tokens      jaccard
strategic kw sets     ~5 vs ~5               jaccard
anchor vs keyword     3 vs 3                 jaccard 0.7 + cosine 0.3
anchor diversity      3 vs 3                 jaccard
anchor vs page        3 vs 2,000             cosine
sentence vs page      20 vs 2,000            cosine
```

A 3-token set against a 2,000-token page maxes out near 0.0015 — every Jaccard
score collapses into noise. That is the whole rule.

**Gate:** the 2 planted bridge gaps rank top by hub-to-hub score.

---

## Phase 5 — Anchor resolution · 4 days

| # | Task | Days |
|---|---|---|
| 18 | Keyword resolution: strategic → GSC by opportunity value → title | 1 |
| 19 | Ladder rungs 1–2: exact match, stemmed variant | 1 |
| 20 | Rung 2.5: Jaccard on stemmed token sets | 0.5 |
| 21 | Rung 3: semantic phrase match, cached by `(url, content_hash)` | 1 |
| 22 | Anchor scoring + type distribution as preference | 0.5 |

```
opportunity_value = impressions × (CTR@1 − CTR@current)
```

No position band. A page at position 3 on a high-value term still needs links to
reach 1. The CTR curve comes from the tenant's own GSC data — the synthetic
corpus ships one.

```
ladder   1    exact keyword verbatim         → link it
         2    stemmed variant                → link it
         2.5  jaccard on stemmed sets ≥ 0.5  → link it
         3    semantic phrase via cosine     → link it
         4    nothing                        → CONTENT_GAP (out of scope here)

score    semantic .35  keyword .35  diversity .15  length .15
```

Type distribution is a **preference among candidates that exist in the copy**,
never a reason to skip a good link. If the only available phrase is a partial
match, use it even when the profile wants exact.

Rung 3 embeds candidate noun phrases per pair — cache hard by
`(source_url, content_hash)`, since the same source appears across many pairs.

**Gate:** extraction succeeds on the ~78% of pages containing their head term
and falls through on the ~22% that don't. Query the planted flag:

```javascript
db.pages.find({ _planted: "NO_HEAD_TERM" }, { url: 1 })
```

---

## Phase 6a — Learned ranking, synthetic · 4 days

| # | Task | Days |
|---|---|---|
| 23 | Proxy label generation from planted ground truth | 1 |
| 24 | LightGBM `lambdarank`, grouped by source page | 1.5 |
| 25 | NDCG@10 harness, heuristic baseline vs learned | 1 |
| 26 | MLflow tracking against your existing server | 0.5 |

### What proxy labels can and cannot do

LambdaMART trains on acceptance labels. On synthetic data nobody has accepted or
dismissed anything, so labels come from planted structure:

```python
label = 3  if pair spans a planted BRIDGE_GAP
        3  if target is a planted ORPHAN_NEW_STRATEGIC page
        2  if source and target share keywordCommunityId
        1  if target is TRUE_NOISE                    # should rank low
        0  otherwise
```

This validates **the plumbing and the feature engineering** — does the model
separate pairs you planted as good from ones you planted as bad? It does not
validate that your notion of good matches an SEO's. A model that learns these
labels perfectly has learned your priors.

**Feature importance is the real output.** It answers three measurement-backlog
questions at once:

```
betweenness low          → delete Phase 2 task 8, save 60-190s/run
hubId low                → HDBSCAN was a duplicate, drop it
keyword_gap high         → the strategic keyword work paid off
saturation_ratio high    → source/target outbound features earned their place
```

**Gate:** learned model beats the hand-weighted baseline on NDCG@10 against proxy
labels. If it doesn't, the features are the problem, not the model.

---

## Phase 6b — Learned ranking, real labels · 2 days

Where ranking *quality* actually gets tested. Mostly labelling time, not code.

| # | Task | Days |
|---|---|---|
| 27 | Real crawl through the full pipeline | 0.5 |
| 28 | Stratified sampling + blind label export | 0.5 |
| 29 | Retrain on real labels, holdout NDCG@10 vs baseline | 1 |

```
1  run the pipeline on the real crawl
2  sample ~200 candidate pairs, STRATIFIED BY SCORE DECILE
3  label each: accept / modify / dismiss
4  train on ~150, hold out ~50
5  NDCG@10 on holdout vs the heuristic baseline
```

**Stratify by decile, don't take the top 200.** Labelling only high scorers means
the model never sees negatives and learns nothing about what to reject. The bottom
deciles need labelling precisely to confirm they belong there.

**Label blind.** Export pairs without their scores, label, then join. Labelling
with the score visible anchors you to it and makes the evaluation circular.

200 labels is enough for a directional signal across ~30 features. It will not
support GraphSAGE — the extra capacity would overfit long before it helped.

**Gate:** learned model beats the heuristic baseline on held-out real labels.
That is the answer to "does the intelligence core work."

> Your labels are one person's judgement. Good ground truth for "does the ranking
> match what this SEO wants," but a sample of one. Worth stating in the write-up so
> a strong NDCG isn't read as validated against the profession.

---

## Phase 7 — Validation · 2 days

| # | Task | Days |
|---|---|---|
| 30 | End-to-end runs at 600 / 5k / 25k, timing per stage | 0.5 |
| 31 | Ground truth assertion suite | 1 |
| 32 | Feature importance report and decision write-up | 0.5 |

### Does it work? Seven checks

```
1  clusters recover 5 planted topics          ARI vs _topic > 0.7
2  24 noise pages excluded or flagged         noise_recall > 0.6
3  3 orphan NEW pages surface as ADD_LINK     all in top decile
4  2 bridge gaps rank top by hub score        both in top 3 pairs
5  2 pillar mismatches caught                 centroid-nearest ≠ declared
6  extraction hits ~78%, falls through ~22%   within ±5%
7  learned ranker beats heuristic baseline    NDCG@10 improvement
```

Seven pass/fail on synthetic. Plus three that only the real crawl can answer:

```
8   extraction hit rate on real content     the 78% above is a generator
                                             property, not a fact about the web
9   fixable-link rate                        < 10% discovery is the product
                                             > 30% the audit is
10  learned ranker beats baseline on
    real held-out labels                     Phase 6b gate
```

**Expected total runtime, 600-page corpus:**

```
embedding        ~1 min       real Voyage calls
graph analytics  ~10s         igraph, exact betweenness
HDBSCAN          ~5s
retrieval        ~15s
features         ~30s
anchors          ~1 min
ranking          ~20s
────────────────────────
                 ~3-4 min end to end
```

Fast enough to iterate many times a day, which is the point of staying synthetic.

---

## Totals

```
Phase 0  Foundation           2
Phase 1  Embedding            3
Phase 2  Graph analytics      3   ← gate can delete later work
Phase 3  Retrieval            3
Phase 4  Features + scoring   4
Phase 5  Anchor resolution    4
Phase 6a Learned ranking      4   synthetic, proxy labels
Phase 6b Real labels          2   the quality gate
Phase 7  Validation           2
                             ───
                             27 days
```

**20–24 with LLM assistance.** The compression is real on Pydantic models, Cypher,
client code and write paths. It is close to zero on threshold iteration and
integration debugging, and exactly zero on waiting for embedding runs.

---

## What this does not prove

Worth being explicit, so a green board isn't over-read.

**Generalisation beyond one site.** The real crawl settles extraction rate,
fixable-link rate and ranking quality *for that site*. A second client may behave
differently — different content style, different link density, different anchor
conventions.

**Labeller agreement.** 200 labels from one person. Directionally sound, not
externally validated.

**That 2048 beats 1024.** Untested for full-page document-to-document comparison
with voyage-4-large. The earlier 1024 result was voyage-4-nano, one article, three
short queries. Phase 6b gives you the harness to test it properly — same corpus,
both dimensions, NDCG on the same labels.

**Long-page dilution.** Chunking is deferred, so every page is one vector
regardless of length. Whether that hurts on 5,000-word pages is measurable once
real labels exist, using the same harness.

---

## What comes after

```
+ crawler and GSC          real corpora — you have these
+ real acceptance labels   retrain LambdaMART on human judgement
+ CONTENT_GAP              Ollama, priority gate
+ audit verdicts           dictionary, dispersion, technical health
+ REST API and Valkey      you have these
+ GraphSAGE, cross-attn    only if they beat the Phase 4 baseline
+ multi-tenancy            at client two
```

The GNN stays last and conditional. The embedding bake-off already showed that
changing the bottom of the stack made no measurable difference, which may equally
mean the layers above are doing less than assumed. Make it prove itself against
Phase 4 before spending the days.
