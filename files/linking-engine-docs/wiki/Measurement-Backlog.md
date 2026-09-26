# Measurement Backlog

Every unresolved assumption in the design, with a procedure for settling it.

Nothing here can be decided by reasoning. Each item states what is currently
assumed, why it matters, how to measure it, and what result changes what.

**Status key:** `BLOCKING` gates a build decision · `TUNING` affects quality but
not architecture · `WATCH` only becomes measurable with real usage.

---

## 1. Leiden vs HDBSCAN — do we need both? `BLOCKING`

**Current position.** Keep both. They cluster different things: Leiden over the
link and keyword graphs, HDBSCAN over content embeddings. Since Leiden runs
in-process via `leidenalg`, keeping both costs seconds rather than a second
database.

**Why it matters.** Determines whether `hubId` joins `linkCommunityId` and
`keywordCommunityId` as a model feature, and whether hub-to-hub bridge scoring
becomes a product output.

**How to measure.**

```bash
make seed
make eval              # ARI/NMI vs topic and subtopic, noise P/R, purity
make eval-umap         # HDBSCAN after UMAP to 10d
make eval-stability    # label reproducibility across 4 runs
```

The synthetic corpus plants five structures specifically to discriminate:

| Planted | Verified in corpus | What it tests |
|---|---|---|
| 5 topics → 12 subtopics | hierarchical embedding space | HDBSCAN condensed tree should expose both levels |
| spread 0.18 → 0.55 | mean cosine to centroid 0.54 vs 0.22 | Leiden's resolution is global |
| 24 noise pages **with inbound links** | noise max-cos 0.03 vs real 0.37 | Leiden must assign a community; HDBSCAN should say −1 |
| 2 bridge gaps | 1 and 3 cross-links vs ≥50 elsewhere | hub-to-hub scoring should rank them top |
| 2 pillar mismatches | declared pillar ranks #108/123, #88/90 | centroid-nearest should differ from declared |

**Decision rule.**

```
Keep both if     the two partitions disagree usefully:
                 ARI(leiden, hdbscan) roughly 0.3-0.7

Drop HDBSCAN if  ARI(leiden, hdbscan) > 0.85    — duplicate feature
Drop HDBSCAN if  noise_recall < 0.5             — its main advantage fails
Drop Leiden if   HDBSCAN wins on every metric AND label stability holds
```

**Watch:** label stability. Leiden with a fixed seed is reproducible; HDBSCAN's
labels churn. If pairwise ARI across runs is below ~0.8, the ids are too
volatile to feed the model directly and you need centroid matching between runs
to keep hub ids stable. That cost does not appear in any quality metric.

---

## 2. Does betweenness earn its runtime? `BLOCKING`

**Current position.** Included as a model feature, sampled. Costs 15–35 min of
a ~50 min graph analytics stage — the single most expensive algorithm in the
pipeline at O(V·E).

**Why it matters.** It may be redundant with features added later.
`keywordCommunityId` vs `linkCommunityId` disagreement already expresses "this
page connects two topic areas," and hub-to-hub bridge scoring expresses it at
the cluster level, which is arguably the more actionable form.

**What it uniquely gives.** The low-PageRank/high-betweenness quadrant: a page
that is structurally load-bearing but unlinked. Strengthening it unlocks flow
for a whole section.

**How to measure.**

*Redundancy check — can run today on the synthetic corpus:*

```python
# does it correlate with what we already have?
spearman(betweenness, pageRank)
spearman(betweenness, degree)
# > 0.8 with either → adding little
```

*Does it find the planted bridges?*

```python
# pages spanning the two planted BRIDGE_GAPS pairs
# should rank top-decile by betweenness
bridge_pages = pages_on_paths_between(gap_topic_a, gap_topic_b)
recall_at_decile(betweenness_ranking, bridge_pages)
```

*Feature importance — needs M6:*

```python
lgb.plot_importance(model, importance_type="gain")
# betweenness in the bottom third → delete the stage
```

**Decision rule.**

```
Keep     if spearman(bc, pagerank) < 0.7 AND it recalls planted bridges
Drop     if it ranks in the bottom third of LambdaMART gain importance
```

Deleting it is a bigger speedup than any optimisation of it.

---

## 3. Sampled vs exact betweenness `TUNING`

**Current position.** Sampled, k unspecified. Exact is O(V·E) — 6.25 billion
operations at 25k nodes.

**The sharp edge.** Sampling is weakest at exactly the thing betweenness is
for. A page bridging two *small* clusters lies on few shortest paths overall,
so it is the case sampling under-counts most — and the case you most wanted to
find.

**How to measure.**

```python
exact = g.betweenness()                       # once, on the 600-page corpus
for k in [50, 100, 250, 500, 1000]:
    sampled = g.betweenness(cutoff=None, sources=sample(k))
    log.info("betweenness.sampling",
        k=k,
        spearman=spearman(exact, sampled),
        top50_overlap=jaccard(top_n(exact,50), top_n(sampled,50)),
        planted_bridge_recall=recall(sampled, planted_bridge_pages),
        seconds=elapsed)
```

Find the knee. Expect Spearman > 0.9 somewhere in the low hundreds, but verify
`planted_bridge_recall` separately — aggregate correlation can look healthy
while the specific pages you care about are missed.

**Also test degree-weighted sampling.** Uniform sampling wastes most draws on
leaf nodes. Weighting toward high-degree sources catches more real paths per
sample — but biases against small-cluster bridges, which is the wrong direction
for this use case. Measure both.

**Mitigation regardless of k.** Store **percentile rank**, not raw value. Removes
cross-run magnitude drift and matches how trees use the feature anyway.

---

## 4. In-process algorithms vs GDS `TUNING`

**Current position.** GDS for PageRank, Leiden, betweenness. Neo4j Community
caps GDS at 4 cores and 3 in-memory projections per instance.

**The case for moving.** `leidenalg` and `igraph` are the reference
implementations — Neo4j's are ports. In-process removes the core cap, the
projection lifecycle, and JVM GC pauses mid-computation. Expect 5–20× on graph
analytics, though that is ~50 min of a ~5 h run.

**The cost.** Materialising 250k edges over Bolt into an igraph object, roughly
10–30s per run, amortised across all three algorithms. And you lose GDS's
guardrails on weighted PageRank, orientation, and disconnected components.

**How to measure.**

```python
# same graph, both paths, diff the outputs
gds_pr    = gds.pageRank.stream(g)
igraph_pr = ig_graph.pagerank(damping=0.85, weights=w)
spearman(gds_pr, igraph_pr)          # should be > 0.99

log.info("graph.backend.bench",
    backend=..., algorithm=..., nodes=n, edges=m,
    seconds=elapsed, cores_used=...)
```

Run on both the 600-page and 5000-page corpora — the gap should widen with
size as the core cap bites harder.

**Decision rule.** Move if outputs agree (Spearman > 0.99) and the speedup
holds at 5000 pages. Correctness first: a fast wrong PageRank is worse than a
slow right one.

---

## 5. Extraction hit rate `BLOCKING`

**Current position.** Unknown. Determines `CONTENT_GAP` volume and how much
linking opportunity is actionable without writing new content.

**Why it matters.** If the rate is low, the writer queue floods and the priority
gate needs tightening. It also determines whether the anchor stage is cheap
(mostly string matching) or expensive (mostly embedding calls at rung 3).

**How to measure.**

```python
for rung, matcher in [(1, exact), (2, variant), (2.5, jaccard_stemmed),
                      (3, semantic)]:
    hits = sum(matcher(target_keyword(t), body(s)) for s, t in candidate_pairs)
    log.info("extraction.ladder", rung=rung, hits=hits,
             pct=hits/len(candidate_pairs))
```

**Caveat.** The synthetic corpus plants ~22% of pages with no head term, so it
gives a *floor*, not a real rate. Synthetic body text has regular structure real
pages lack, so measured hit rates here will be optimistic. This number only
becomes meaningful on the first real corpus.

---

## 6. Rung 2.5 — Jaccard on stemmed tokens `TUNING`

**Current position.** Proposed, not yet in the ladder. Sits between stemmed
string matching and embedding cosine.

```
keyword:  "industrial hydraulic press"
phrase:   "hydraulic press systems"

rung 2   may miss — different token count
rung 3   catches it, costs an embedding call
rung 2.5 jaccard on stemmed sets = 0.5 → catches it free
```

**Why it matters.** Rung 3 embeds candidate noun phrases *per pair* — the
expensive step in the anchor stage. A cheap set-overlap rung resolves a chunk
of pairs before reaching it.

**How to measure.**

```python
# what fraction of rung-3 hits would rung 2.5 have caught?
rung3_hits = [p for p in pairs if semantic_match(p)]
caught_by_25 = [p for p in rung3_hits if jaccard_stemmed(p) >= threshold]
log.info("extraction.rung25", threshold=t,
         diverted_from_rung3=len(caught_by_25)/len(rung3_hits),
         false_positives=...)
```

Sweep threshold 0.3–0.7. Worth adding if it diverts more than ~20% of rung-3
traffic without introducing bad matches.

---

## 7. Cosine vs Jaccard for anchor diversity `TUNING`

**Current position.** Diversity dimension (weight 0.15) uses
`1 − max cosine vs existing anchors`. **This is probably wrong.**

Over-optimisation is about repeating the same *words*. Google reads literal
text, not meaning:

```
existing:  "industrial hydraulic press"
candidate: "commercial press machine"

cosine  0.85 → judged non-diverse   ✗
jaccard 0.20 → judged diverse       ✓
```

**How to measure.** Needs acceptance data. Score the same candidate set both
ways, serve alternating, and compare acceptance rates by diversity measure in
`anchor_feedback`.

Until then, the argument from first principles is strong enough to switch —
low cost, low risk.

---

## 8. Fixable-link rate `BLOCKING`

**Current position.** Unknown. Determines whether Stage 1 audit is a feature or
the product.

**How to measure.** Run M3 against one real client, then:

```
verdicts_emitted / total_edges
by actionType
```

**Decision rule.**

```
< 10%   audit is a useful feature, discovery is the product
> 30%   audit IS the product, discovery is the add-on
        → reweight the roadmap toward Stage 1 depth
```

Track `anchor_feedback.actionType` acceptance separately. If `REANCHOR`
acceptance runs far above `ADD_LINK`, that settles it empirically.

---

## 9. Does the GNN beat the heuristic baseline? `BLOCKING`

**Current position.** M7 is 19 engineer-days and conditional on this.

**Why the doubt is real.** The embedding bake-off found no measurable
difference between voyage-4-nano and Qwen3-Embedding-0.6B on real content. That
result cuts both ways: it may mean embedding quality is not the bottleneck, or
it may mean the layers above are doing less than assumed.

**How to measure.** PR #53:

```
baseline    hand-weighted linear score over ~30 tabular features
candidate   + gnn_embedding, cross-attention, 3105d pair vector
metric      NDCG@10 on held-out acceptance data
```

**Decision rule.** Ship M7 only if NDCG@10 improves meaningfully over the M5
baseline. A marginal gain does not justify 19 days plus ongoing training
infrastructure.

**Note on the pair vector.** LightGBM will find little in 3072 dense embedding
dimensions. In practice cross-attention should pass forward a compact
interaction summary, with the ranker working over that plus the ~30 tabular
features. "3105d into LightGBM" as written in the spec is not what will
actually happen — worth resolving in the M6/M7 design.

---

## 10. Configuration guesses `TUNING`

All defaults, none measured.

| Setting | Default | Failure mode if wrong |
|---|---|---|
| `contentGapMinPriority` | 4 | Too low floods the writer queue; too high hides real gaps |
| `anchorTypeProfile` | 15/20/50/15 | Conventional SEO heuristic, never validated |
| `reservedNewPageSlotPct` | 0.15 | Trades early perceived quality for feedback coverage |
| `maxRecommendationsPerSource` | 10 | Untested against real link-density distributions |
| `diversityCapPct` | 0.15 | Untested |
| Semantic fit threshold | 0.62 | **Derived against Qwen embeddings — does not port to Voyage** |
| HDBSCAN `min_cluster_size` | 15 | Too small fragments topics; too large merges them |

**Threshold portability is not a guess, it is a known problem.** Local testing
showed a 0.75 threshold flagging 2/3 test queries on Qwen and 0/3 on Voyage for
identical content. Absolute cosine values do not transfer between embedding
models. Re-derive before M3 completes.

**How to measure the rest.** Sweep against acceptance data once several hundred
actions exist. Before then, log the distribution rather than tuning blind:

```python
log.info("config.distribution", setting="contentGapMinPriority",
         would_emit_at_3=n3, at_4=n4, at_5=n5)
```

---

## 11. Splunk ingest volume `WATCH` — deferred to production

**Current position.** Not applicable to MVP — no Splunk deployment for MVP scope
(see [[ADRs]] ADR-009); `structlog` ships to the existing Loki via Promtail at no
marginal cost. Revisit if/when Splunk is adopted at production scope. Splunk
bills per GB ingested. Splunk Free caps at 500
MB/day and stops indexing beyond it.

**How to measure.** Run one full pipeline at INFO, then:

```
index=linking-engine | stats sum(len(_raw)) as bytes by stage
```

Multiply by tenant count and run frequency. **Then check licence headroom
before enabling per-tenant DEBUG.**

Never log page bodies, embeddings, or credentials — that is what turns a
manageable volume into an unmanageable one.

---

## 12. Voyage rate limits `WATCH`

Unverified against a 25k-page initial burst. Check before the first large-client
onboarding, not during it.

Also untested: **int8 quantisation.** Voyage 4 supports QAT int8, which would
cut Neo4j vector storage 4× at minimal quality cost. Test independently of any
provider decision.

---

## 13. Per-tenant idle footprint `WATCH`

**Current position.** ~2.5 GB per idle Neo4j instance, estimated from defaults.
Every tenancy capacity number derives from it.

**How to measure.** After the first tenant is provisioned:

```
kubectl top pod -l app=neo4j --containers
```

Take RSS at idle, during a run, and immediately after. If the real idle figure
is 1.5 GB the sizing table is pessimistic; if it is 4 GB the ceiling arrives far
sooner than planned.

**Related.** `projections_leaked` should be a logged metric from day one. A run
that dies without dropping its projection consumes memory until restart, and the
*next* run fails on the 3-projection limit with an error pointing nowhere near
the cause.

---

## 14. Position bias in feedback `WATCH`

If the team only reviews the top 5, ranks 6–10 always score `unseen=0` —
indistinguishable from actively worthless. The model then learns from an
incomplete picture and reinforces its own ranking.

**How to measure.** Log review depth per session:

```python
log.info("feedback.session", tenant_id=..., items_shown=n,
         max_rank_actioned=k, items_actioned=j)
```

If `max_rank_actioned` is consistently below 5, the labels beyond that are not
negatives and should be weighted or excluded rather than treated as `unseen=0`.

---

## 15. NDCG@10 cold start `WATCH`

With a few dozen labels the metric is noise. Grouped by source page, most
groups have one or two graded items and the rest zeros.

**Do not trust the promotion gate until several hundred actions have
accumulated.** Log label count alongside every NDCG figure so the number is
never read without its sample size:

```python
log.info("model.eval", ndcg_at_10=..., n_labelled_pairs=...,
         n_groups=..., mean_group_size=...)
```

---

## 16. Template link handling — SETTLED

**Resolved by design change.** The crawler now extracts body links only; nav,
header, footer and sidebar links are never captured.

Measured before the change, at a realistic 77% template ratio:

```
                     ARI    modularity   utility in top-10
                                         PageRank / betweenness
all links           0.917      0.235        10/10   10/10
body links only     0.910      0.578         0/10    0/10
```

PageRank and betweenness are destroyed by template links; clustering is not.
Leiden recovers the planted topics either way because utility pages form their own
dense community rather than blurring topic boundaries.

Note this contradicted the initial prediction that clustering would degrade toward
complete-bipartite. It does not.

---

## Priority order

```
Settle now, on the synthetic corpus
  16 Template handling                  SETTLED by design change
  1  Leiden vs HDBSCAN                  make eval
  2  Betweenness redundancy             spearman vs pagerank
  3  Sampled betweenness knee            sweep k
  4  igraph vs GDS correctness           spearman > 0.99

Settle on the first real corpus
  5  Extraction hit rate
  8  Fixable-link rate
  10 Threshold recalibration for Voyage

Settle once feedback accumulates
  9  GNN vs heuristic baseline
  6  Rung 2.5 value
  7  Anchor diversity measure
  10 Remaining config sweeps

Instrument now, read later
  11 Splunk volume
  13 Idle footprint, projections_leaked
  14 Position bias
  15 NDCG sample size
```

The first four cost hours, not days, and two of them can delete work from the
roadmap.
