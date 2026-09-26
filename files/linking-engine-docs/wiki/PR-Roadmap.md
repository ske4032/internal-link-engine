# PR Roadmap

Ordered by dependency. Sizes are engineer-days. `M0`–`M4` produce a shippable
product with no trained models.

**Ordering principle.** Stage 1 audit ships before any ML. It requires no model
and it generates the acceptance feedback every later model trains on. Building
discovery first means months of models with nothing to learn from.

---

## What changed in v5.5

| Change | Effect |
|---|---|
| **igraph replaces GDS** for PageRank, Leiden, betweenness | Graph analytics 50 min → 2–4 min. No 4-core cap, no projection limit, exact betweenness affordable |
| **HDBSCAN added** over content embeddings → `hubId` | Third clustering signal alongside the two Leiden passes. Enables noise detection, hub bridges, pillar check |
| **Audit split into two passes** | Pass A needs no embeddings and runs in parallel with Step ①. `FIX`, most `REANCHOR`, saturation-based `REMOVE` land within minutes of crawl |
| **`anchorTargetFit`** added as a sixth audit dimension | Catches links whose anchor matches the keyword but whose target is the wrong page. Lexical matching passes these clean |
| **Measure discipline: Jaccard vs cosine** | Jaccard where set sizes are comparable, cosine where they are not. Anchor diversity switches to Jaccard; keyword alignment becomes a 0.7/0.3 blend |
| **Rung 2.5 in the extraction ladder** | Jaccard on stemmed tokens, between variant matching and embedding cosine. Diverts traffic from the expensive rung |
| **voyage-4-large @ 2048d**, single vector, all pages | Chunking deferred. 2048 chosen because MRL makes downgrading to 1024 a truncation, while upgrading needs a full re-embed |
| **Token-based batching** | voyage-4-large caps at 120K tokens/request, not 1M. ~36 pages per call, batch by token count |
| **Embedding ∥ graph analytics** | Step ② needs no embeddings. Runs concurrently with Step ① |

---

## M0 — Foundation

| PR | Title | Days | Depends |
|---|---|---|---|
| #1 | Repo scaffold: uv, ruff, mypy strict, pytest, pre-commit | 1 | — |
| #2 | Pydantic domain models: `Page`, `Link`, `Keyword`, `Recommendation`, `TenantConfig` | 2 | #1 |
| #3 | Neo4j async repo + connection pooling + migration runner | 2 | #2 |
| #4 | Neo4j schema migrations: constraints, HNSW vector indexes @2048d | 1 | #3 |
| #5 | MongoDB repo (PyMongo async) + collection indexes | 1 | #2 |
| #6 | Valkey cache layer with tenant key prefixing | 1 | #2 |
| #7 | `structlog` → JSON stdout, `tenant_id`/`run_id`/`stage` binding (existing Promtail/Loki picks it up — see ADR-009) | 1 | #1 |
| #8 | Helm chart: Neo4j StatefulSet, Mongo, Valkey, app | 3 | #1 |
| #9 | ArgoCD app-of-apps + sealed secrets | 2 | #8 |
| #10 | Prometheus metrics module + Grafana dashboard skeleton | 1 | #7 |

**Subtotal: 15 days**

> #4 is worth care. Vector index dimension is set at creation and changing it later means a full re-embed plus GNN retrain. 2048d is settled, chosen for reversibility under MRL — see [[Decisions]].

---

## M1 — Ingestion

| PR | Title | Days | Depends |
|---|---|---|---|
| #11 | Crawler: async fetch, robots.txt, redirect chains, status capture | 4 | #2 |
| #12 | Crawler: JS rendering via Playwright, per-tenant rate limiting | 3 | #11 |
| #13 | Link extraction: anchor text, position class, `isFollow`, surrounding sentence | 2 | #11 |
| #14 | Delta detection: content hashing, change events | 2 | #11 |
| #15 | GSC client: OAuth, 28d rolling pull, quota backoff | 3 | #2 |
| #16 | Strategic keyword ingest: CSV upload + `TARGETS_KEYWORD` writes | 2 | #3 |
| #17 | Prefect flow skeleton wiring ingest tasks with retry policies | 2 | #11 #15 |

**Subtotal: 18 days**

> #13 is new relative to earlier specs — `surroundingText` is required by the audit and was not captured before. Do not skip it; retrofitting means a full recrawl.
>
> #11/#12 are the most commonly underestimated work in the project. Consider Scrapy + Playwright over hand-rolling.

---

## M2 — Embedding and graph

| PR | Title | Days | Depends |
|---|---|---|---|
| #18 | Voyage client: token-aware batching, retry, rate-limit backoff | 3 | #2 |
| #19 | Embedding write path: buffered flush, `embeddedContentHash` resume | 2 | #18 #4 |
| #20 | Anchor + surrounding-sentence embeddings, deduplicated | 2 | #19 |
| #21 | **igraph backend**: edge pull, id mapping, graph construction | 2 | #3 |
| #22 | PageRank + betweenness via igraph, write-back | 2 | #21 |
| #23 | Leiden via `leidenalg` — link graph → `linkCommunityId` | 1 | #21 |
| #24 | Bipartite keyword projection + Leiden → `keywordCommunityId` | 3 | #23 #16 |
| #25 | **HDBSCAN** over content embeddings → `hubId`, noise label | 3 | #19 |
| #26 | Correctness harness: closed-form fixtures and networkx reference | 1 | #22 |
| #27 | Parallel scheduling: Step ② concurrent with Step ① | 1 | #22 |

**Subtotal: 20 days**

> **#18** — voyage-4-large caps at **120K tokens per request**, max 1,000 inputs.
> At ~3,300 tokens per page that is roughly 36 pages per call, not 128. Batch by
> `vo.count_tokens`, not by list length. Rate limits are 3M TPM / 2000 RPM, so a
> 25k-page initial embed lands around 28 minutes.
>
> **#19** — decouple API batch size from the Neo4j write. Accumulate ~500 pages,
> flush in one `UNWIND`. Set `embeddedContentHash` in the same transaction so a
> crash leaves a consistent prefix. Do not buffer the whole run: 25k pages at
> 2048d float32 is ~200 MB before anything is durable.
>
> **#20** — anchor text repeats heavily. Deduplicate before embedding (250k edges
> might yield 8k unique strings) and skip anything the generic dictionary already
> flagged.
>
> **#21–#23** — igraph and `leidenalg` are the reference implementations; GDS is a
> port. In-process removes the 4-core Community cap, the 3-projection limit, and
> JVM cache-miss overhead on BFS. Betweenness at 25k nodes is ~190s single-threaded
> and less across cores, so **exact is affordable and sampling may be unnecessary**.
>
> **#24** — the cold-start fix. Link-graph clustering is circular: a new section has
> no links, so no cluster, so no recommendations. Keyword relationships exist first.
>
> **#25** — HDBSCAN clusters content embeddings, which is a different question from
> either Leiden pass. Its unique output is the `-1` noise label: a page belonging to
> no coherent topic. Neither Leiden pass can express that. Run the clustering evaluation on the
> synthetic corpus before building — the decision rule is in Measurement Backlog §1.
>
> **#26** — correctness before speed. Closed-form fixture graphs on every CI run,
> plus an independent networkx reference compared by URL after write-back; the
> thresholds are in the ADR-002 amendment. A fast wrong PageRank is worse than a
> slow right one.
>
> **#27** — Step ② touches no embeddings, so it can start the moment the crawl ends.
> With igraph the whole stage fits inside the first 5% of the embedding run.

---

## M3 — Stage 1 audit and anchors *(first shippable slice)*

### Pass A — no embeddings required, runs during Step ①

| PR | Title | Days | Depends |
|---|---|---|---|
| #28 | Audit query: edge scan, saturation counts, technical health | 3 | #13 |
| #29 | Generic-anchor dictionary + stemmed keyword alignment | 2 | #28 |
| #30 | Equity efficiency from `pageRank` × position weight | 1 | #22 #28 |
| #31 | Verdicts: `FIX`, `REANCHOR`, saturation-based `REMOVE` | 2 | #29 #30 |

### Pass B — needs embeddings

| PR | Title | Days | Depends |
|---|---|---|---|
| #32 | `contextRelevance`: surrounding sentence vs target vector | 2 | #20 #28 |
| #33 | **`anchorTargetFit`**: anchor vector vs target content vector | 2 | #20 #28 |
| #34 | Refined `REANCHOR` from cosine dimensions | 2 | #32 #33 |
| #35 | Batched audit write-back, chunked ~5k edges | 1 | #31 #34 |

### Anchor resolution

| PR | Title | Days | Depends |
|---|---|---|---|
| #36 | Per-tenant CTR-by-position curve from GSC | 2 | #15 |
| #37 | Keyword resolution: strategic → GSC by opportunity value → title | 3 | #16 #36 |
| #38 | Extraction ladder rungs 1–2: exact, stemmed variant | 3 | #37 |
| #39 | **Rung 2.5**: Jaccard on stemmed token sets | 1 | #38 |
| #40 | Rung 3: semantic phrase match, cached by `(url, content_hash)` | 3 | #38 #20 |
| #41 | Anchor scoring + type distribution as preference | 2 | #40 |
| #42 | `CONTENT_GAP` via Ollama, behind priority gate | 3 | #40 |
| #43 | Materialise: `SUGGESTED_ACTION`, Mongo payload, cache invalidation | 2 | #35 #41 |

**Subtotal: 34 days**

> **Pass A / Pass B split is the point.** Four of six audit dimensions need no
> embeddings — generic dictionary, stemmed keyword alignment, equity efficiency,
> technical health. Splitting them means actionable `FIX` and `REANCHOR` output
> lands within minutes of the crawl finishing, while embedding is still running.
>
> **#33 is new in v5.5 and is the real gap.** `keywordAlignment` asks whether the
> anchor contains the target's keyword. It cannot catch an anchor that matches the
> keyword while the target page is about something else. `cosine(anchor_emb,
> target.content_embedding)` catches exactly that, and lexical matching passes it
> clean.
>
> **Measure discipline.** Jaccard where set sizes are comparable, cosine where they
> are not:
> ```
> anchor vs keyword         3 vs 3 tokens        jaccard 0.7 + cosine 0.3
> anchor diversity          3 vs 3 tokens        jaccard   (was cosine — wrong)
> rung 2.5                  3 vs 3 tokens        jaccard
> anchor vs target page     3 vs 2,000 tokens    cosine
> sentence vs target page   20 vs 2,000 tokens   cosine
> ```
> Anchor diversity was cosine through v5. That is backwards: over-optimisation is
> about repeating *words*, and cosine scores "commercial press machine" as
> non-diverse against "industrial hydraulic press" at 0.85 when Jaccard correctly
> gives 0.20.
>
> **#39** sits between stemmed matching and embedding cosine. `"industrial hydraulic
> press"` vs `"hydraulic press systems"` scores 0.5 on stemmed Jaccard — caught for
> free, where rung 3 would have cost an embedding call. Rung 3 embeds candidate noun
> phrases *per pair*, so anything diverted before it is a direct saving.
>
> **#36 blocks #37.** Opportunity value is `impressions × (CTR@1 − CTR@current)` and
> the curve must come from the tenant's own GSC data — curves differ substantially
> by vertical and SERP feature mix.

---

## M4 — API and serving

| PR | Title | Days | Depends |
|---|---|---|---|
| #44 | FastAPI app, JWT auth, tenant resolution dependency | 3 | #2 |
| #45 | Recommendation endpoints with `actionType` filtering | 2 | #43 #44 |
| #46 | Cache-first read path + diversity cap + source cap | 2 | #6 #45 |
| #47 | Feedback endpoints (accept / dismiss / modify) → Mongo | 2 | #45 |
| #48 | Audit endpoints: link health, keyword gaps, cannibalisation, orphans | 3 | #45 |
| #49 | OpenAPI polish, error contracts, rate limiting | 2 | #45 |

**Subtotal: 14 days**

**→ Ship here. Product is usable and feedback starts accumulating.**

---

## M5 — Discovery, heuristic

| PR | Title | Days | Depends |
|---|---|---|---|
| #50 | Candidate retrieval: hard constraints, HNSW top-50, Jaccard signals | 4 | #19 #23 |
| #51 | Feature assembly: ~30 context features incl. source outbound | 3 | #50 |
| #52 | Hand-weighted linear scorer + `ADD_LINK` emission | 2 | #51 |
| #53 | Baseline NDCG@10 harness against accumulated feedback | 2 | #47 #52 |

**Subtotal: 11 days**

> **#53 is the measurement gate for everything after it.** No further ML ships without beating this baseline.

---

## M6 — Learned ranking

| PR | Title | Days | Depends |
|---|---|---|---|
| #54 | MLflow tracking server + MinIO backend, Helm-deployed | 2 | #8 |
| #55 | Training data assembly from `anchor_feedback` with graded labels | 2 | #47 |
| #56 | LightGBM `lambdarank` training, grouped by source page | 3 | #55 |
| #57 | MLflow registry + promotion gate as alias move | 2 | #54 #56 |
| #58 | Chunked inference (~50k pairs) with memory ceiling | 3 | #56 |

**Subtotal: 12 days**

> **#58 is not optional.** 1.25M pairs × 3105 features × 4 bytes ≈ 15.5 GB assembled
> in one pass, which will not coexist with the Neo4j heap on a 32 GB node. Stream in
> ~50k chunks. Note that at 2048d embeddings the pair vector roughly doubles, so this
> ceiling arrives sooner than the v5 figure suggested.

---

## M7 — Deep models *(only if M5 baseline is beaten)*

| PR | Title | Days | Depends |
|---|---|---|---|
| #59 | PyG graph loader from Neo4j, feature matrix construction | 3 | #51 |
| #60 | GraphSAGE 3-layer, MEAN aggregation, inference path | 5 | #59 |
| #61 | Contrastive fine-tuning on accept/dismiss pairs | 4 | #60 #55 |
| #62 | Cross-attention module → 3105d pair vector | 5 | #60 |
| #63 | A/B harness: GNN features vs M5 baseline on NDCG@10 | 2 | #62 #53 |

**Subtotal: 19 days**

> **#63 decides whether M7 ships at all.** The embedding bake-off already showed that changing the bottom of the stack made no measurable difference — that cuts both ways, and the GNN layer may be doing less than assumed.

---

## M8 — Retraining and multi-tenancy

| PR | Title | Days | Depends |
|---|---|---|---|
| #64 | Monthly retraining flow for all three artefacts | 3 | #57 #61 |
| #65 | Anchor weight update from acceptance by type and action | 2 | #47 |
| #66 | Tenant manager: registry, driver cache, config routing | 4 | #44 #21 |
| #67 | Helm-based tenant provisioning via K8s API | 4 | #66 #8 |
| #68 | Per-tenant resource quotas and namespace policies | 2 | #67 |

**Subtotal: 15 days**

---

## Totals

```
M0  Foundation             15
M1  Ingestion              18
M2  Embedding + graph      20   igraph, HDBSCAN, parallel scheduling
M3  Audit + anchors        34   two-pass split, anchorTargetFit, rung 2.5
M4  API + serving          14   ← SHIP
M5  Discovery heuristic    11
M6  Learned ranking        12
M7  Deep models            19   ← conditional on beating the M5 baseline
M8  Retrain + tenancy      15
                          ────
                          158 engineer-days
```

**To first ship (M0–M4): 101 days.** Roughly five months solo full-time.

Up from 141 in v5. M2 gained 5 days for igraph plus HDBSCAN; M3 gained 10 for
the two-pass audit split, `anchorTargetFit`, and rung 2.5. Both buy real things:
graph analytics drops from ~50 min to 2–4 min per run, and Pass A produces
actionable output while embedding is still going.

The Java estimate for the same scope was 136–180 days without any of this.

---

## Conventions

**Branches:** `feat/`, `fix/`, `chore/` + short slug.

**Every PR must:** pass `ruff`, `mypy --strict`, and `pytest`; include tests for new domain logic; update the relevant wiki page if it changes a contract.

**Never in a PR:** a Neo4j vector index dimension change, a Pydantic model field rename without a migration note, or a new dependency without a line in [[Stack]].

**Definition of done for M3 and M5:** a real client corpus runs end to end and
produces recommendations a human agrees with. Not unit tests alone.

**Before starting M2**, run the four cheap experiments in Measurement Backlog
§1–4 on the synthetic corpus. Two of them can delete work from this roadmap:
if betweenness correlates above 0.8 with PageRank it comes out of the pipeline,
and if Leiden and HDBSCAN agree above 0.85 ARI then #25 is a duplicate feature.
