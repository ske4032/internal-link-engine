# Architecture Decision Records

One log file rather than one-file-per-decision — easier to scan chronologically
and diff over time. Each entry is Status / Context / Decision / Consequences.

---

## ADR-001: Python over Java/Spring Boot for the SEO engine

**Status:** Accepted

**Context.** The original design assumed the Chirp stack (Java, Spring Boot,
Spring Batch) for consistency across projects. But the ML layer — PyTorch
Geometric, LightGBM, embedding clients — is Python-native regardless. A Java
pipeline meant a process boundary, duplicated domain models on both sides, and
serialising a multi-thousand-dimension feature vector across it per candidate
pair.

**Decision.** Python end to end. FastAPI, Pydantic, native ML libraries, no
cross-language boundary.

**Consequences.** Removes an estimated 10–14 engineer-days of integration work
and deletes the single highest-risk item in the original build estimate. Diverges
from Chirp's stack, so the two projects no longer share tooling or patterns —
accepted, since they were already independent systems.

---

## ADR-002: Graph algorithms run outside Neo4j (igraph/leidenalg, not GDS)

**Status:** Accepted

**Context.** Neo4j GDS Community Edition caps concurrency at 4 cores regardless
of host hardware and limits projections to 3 in memory per instance. Measured:
exact betweenness on 25k nodes took 15–35 minutes in GDS. The dominant cost was
not the algorithm but memory layout — GDS stores nodes as JVM heap objects,
so BFS (the core of betweenness) suffers constant cache misses; igraph uses
contiguous `int32` adjacency arrays.

**Decision.** PageRank, Leiden (×2), betweenness, and later HDBSCAN all run
in-process via igraph/leidenalg/scikit-learn, not Neo4j GDS. Neo4j is reduced to
an edge and vector store.

**Consequences.** Graph analytics dropped from ~50 min to ~2–4 min measured.
Removes the 4-core cap and the 3-projection limit entirely. Requires a
correctness gate (Spearman vs GDS output > 0.99) before trusting the swap —
implemented as PR #26/Phase-2 task #8. Neo4j internal ids are not stable across
restarts, so the id mapping must be rebuilt every run, never cached.

---

## ADR-003: Voyage API (voyage-4-large) for embeddings, at 2048 dimensions

**Status:** Accepted

**Context.** A local bake-off between `voyage-4-nano` and `Qwen3-Embedding-0.6B`
on real client content showed no measurable quality difference — `content_embedding`
is one input among ~30 downstream features, and both the GNN and the ranker train
on acceptance feedback, so two competent embedders converge. Dimension choice was
separately contested: local testing on `voyage-4-nano` (one article, three short
queries, query-to-document) showed 2048→1024 truncation cost ~0.01 cosine with
6/6 rank agreement — but that result does not transfer to `voyage-4-large` on
full-page document-to-document comparison, which was never tested.

**Decision.** `voyage-4-large` via API, 2048 dimensions, local `voyage-4-nano` kept
as the documented exit path (same embedding space, no re-index required to
switch).

**Consequences.** Removes the local TEI embedding service, ~2–3 GB RAM freed per
node, no CPU embedding bottleneck. Runtime and Neo4j vector storage are ~2× a
1024d design. Dimension is fixed at index creation — changing it later means
dropping the index, re-embedding every page, and retraining the GNN, so 2048 was
chosen specifically because it is the *reversible* direction under Matryoshka
(1024 is a truncation of 2048, not vice versa). Whether 2048 actually earns its
cost for `voyage-4-large` document-to-document is an open measurement, not yet
run — see Measurement Backlog.

---

## ADR-004: Graph is built from body links only

**Status:** Accepted

**Context.** Early design included all captured links (nav, header, footer,
sidebar, body) with position-based weighting (nav 0.4, footer 0.3, body 1.0).
Measured on the synthetic corpus at a realistic 77% template-link ratio: with
template links included, 10/10 of the top PageRank pages and 10/10 of the top
betweenness pages were utility pages (`/terms`, `/contact`, `/sitemap`).
Position weighting reduces per-link *value* but does nothing about link *count* —
a page in global nav receives one inbound edge per page on the site.

**Decision.** The crawler extracts and stores body links only. Nav, header,
footer, and sidebar links are never captured — not filtered post-hoc, simply
never extracted. Matches how SEOs actually work: template links are audited once
per template, not per page.

**Consequences.** PageRank, betweenness, and both Leiden clustering passes are
unaffected by template noise by construction. Measured (before the rule was
finalised): with templates included, clustering ARI barely changed (−0.007) —
the four-fold justification originally assumed (audit, clustering, PageRank,
betweenness all affected) narrowed to two (PageRank, betweenness). `REPOSITION`
is not a valid action type: with no footer links captured there is nothing to
move a link out of. Utility pages (nav/footer targets) exist in the corpus and
on real sites but appear as orphans in the graph — correct, not a bug.

---

## ADR-005: HDBSCAN as a third clustering signal, alongside Leiden

**Status:** Accepted

**Context.** Two Leiden passes exist: over the link graph (`linkCommunityId`) and
over the bipartite keyword projection (`keywordCommunityId`). Neither can express
"this page belongs to no coherent topic" — Leiden must assign every node to a
community. A third clustering, over `content_embedding` directly, can.

**Decision.** Add HDBSCAN → `hubId`, with `-1` as a genuine noise label. Enables
hub-to-hub bridge scoring (topic pairs with high query overlap but near-zero link
density) and a pillar-mismatch check (is the declared pillar actually the page
nearest its cluster's centroid?). HDBSCAN is retained regardless of the
evaluation harness outcome — a deliberate call made without waiting on the
`ARI(leiden, hdbscan)` / `noise_recall` gate described in the Measurement
Backlog.

**Consequences.** A third categorical feature and a third source of run-to-run
label instability (HDBSCAN labels churn between runs more than Leiden's do with a
fixed seed) — mitigated by centroid matching between runs before `hubId` is used
as a model feature. The evaluation harness (`make eval`) is still worth running,
but as a tuning input (e.g. `min_cluster_size`) rather than a keep/drop gate — if
it later shows `ARI > 0.85` against Leiden, that is a finding about redundancy to
note, not a trigger to remove the feature on its own.

## ADR-006: Anchor resolution is deterministic extraction, not LLM generation

**Status:** Accepted (supersedes an earlier LLM-based design)

**Context.** The original anchor pipeline used an instruction-tuned LLM (Qwen3.5-4B
local / Claude Haiku API) to generate anchor text when no suitable phrase existed
in the source page. That design traced back to a critique of an even earlier
T5-base implementation — the critique correctly identified that T5 was the wrong
model, but the underlying assumption that *generation* was the right operation was
never re-examined for several design iterations. Inserting an invented phrase
bolts a link onto copy that was never actually about the target topic.

**Decision.** Extraction only: exact keyword match → stemmed variant → Jaccard on
stemmed tokens → semantic phrase via embedding cosine. If none succeed, emit
`CONTENT_GAP` — a verdict that the source page lacks the content to support the
link, routed to a content queue rather than a link-implementation queue, gated by
target priority to avoid flooding that queue with every failed extraction.

**Consequences.** No hallucination surface on the primary anchor output. Anchor
generation runtime collapsed from an estimated 30–90 min (LLM path) to string and
vector operations. LLM involvement in the anchor pipeline is now limited to
generating the human-readable rationale for a `CONTENT_GAP` finding, not to
producing any anchor text that gets published.

---

## ADR-007: No reranking stage between retrieval and ranking

**Status:** Accepted, amended 2026-09-23 — see amendment below

**Context.** A cross-encoder reranking stage — `rerank-3` via API, or a
self-hosted `Qwen3-Reranker` — was evaluated for insertion between HNSW candidate
retrieval (top-50 per target) and LambdaMART ranking. Three separate reasons were
weighed, not just one.

**Reason 1 — a cross-encoder already exists in the pipeline.** The cross-attention
stage between source and target GNN embeddings is functionally a cross-encoder:
both sides attend to each other jointly before scoring. Unlike a general
reranker, it is fine-tuned monthly on this system's own acceptance feedback — it
is learning *this tenant's* notion of a good link. A generic reranker has never
seen an SEO linking decision.

**Reason 2 — the task shape is a mismatch.** Rerankers are built for
query-document scoring, where the two sides are structurally different (a short
query, a long document). This pipeline has no query side anywhere — Step ④ is
page-to-page, Step ⑦ extraction is phrase-to-page. Both sides of every comparison
are documents. Fitting a reranker in means synthesising a pseudo-query per
target, discarding most of what makes the comparison meaningful.

**Reason 3 — the economics collapse under their own mitigation.** `rerank-3`
bills as `(query tokens × num documents) + sum of document tokens`. At top-50
candidates with full page bodies, that is roughly 690M tokens (~$34) per client
per full run — the entire evaluated free tier gone on the first client. The
obvious fix, truncating to top-15 candidates before reranking, undercuts the
point: at top-15 the reranker is just reordering a list HNSW already narrowed
sharply, immediately before LambdaMART ranks it properly anyway. The cheaper the
reranking is made, the less work is left for it to usefully do.

**Decision.** Not adopted. Self-hosting removes reason 3 (no per-token cost) but
not reasons 1 or 2, which are the load-bearing ones — so self-hosting alone does
not change the conclusion.

**Consequences.** One fewer service, no per-pair reranking cost at Steps ④/⑤.
Revisit reranking at retrieval only if LambdaMART's NDCG@10 plateaus **and**
error analysis specifically attributes the misses to topical relevance rather
than authority or GSC-signal features (see ADR-011 for why upstream signals
should not gate candidates in the first place). If revisited there, the
reranker's score must be added as a ranking *feature*, never as a pre-ranking
filter — a hard cutoff would reintroduce the same eligibility-vs-priority
mistake that ADR-011 removed.

---

### Amendment — 2026-09-23: Reason 2 was scoped too broadly; approve a scoped experiment at anchor disambiguation

**Trigger.** Reason 2 as originally written claimed "no query side anywhere" as a
property of the whole pipeline. That is only true at retrieval and ranking
(Steps ④/⑤, page-to-page). It does not hold at **Step ⑦d, anchor scoring**: once
a pair has passed the `score > 45` gate and the extraction ladder has produced
candidate phrases, the anchor text is genuinely query-shaped against the target
page as the document. The original wording conflated two different locations in
the pipeline.

**Reason 3 revisited at this second location.** The ~690M-token / ~$34-per-client
figure was computed for pre-gate candidate-*pair* reranking — top-50 candidates ×
full page bodies × every eligible target on the site. Anchor disambiguation runs
*after* the ranking gate, over a small candidate set per link (typically single
digits, since the extraction ladder has already narrowed to literal or
near-literal phrase matches). Estimated at ~6 candidates × one target page
(~2,500 tokens) per link, ~5,000 links needing anchors on a mid-size client: ~75M
tokens, ~$3.75 at `rerank-3`'s confirmed $0.05/1M pricing — comfortably inside the
200M free tier for most tenants. The original cost objection does not transfer to
this location.

**What still holds.** Reason 1 (cross-attention is already a monthly-fine-tuned
cross-encoder) and Reason 3 as originally computed remain valid *for candidate-pair
retrieval at Steps ④/⑤* — no change there. The marginal value of a reranker at
Step ⑦d is also genuinely smaller than at Location A: candidates reaching anchor
scoring have already been filtered by the extraction ladder to relevant phrases,
so what remains is mostly a type/placement preference decision, not a relevance
decision. The case a reranker helps most — unfiltered candidates — is exactly the
case that stays unaffordable and shapeless.

**Amended decision.** The original conclusion (no reranker at Steps ④/⑤) stands,
on Reasons 1 and 3 as originally stated. Separately: **approved** — a scoped
experiment adding Voyage's `rerank-3` at Step ⑦d, reranking the small
candidate-anchor set against the target page, gated on a measured improvement in
`anchor_feedback` acceptance rate versus the current weighted jaccard/cosine
blend. Not a default; not promoted beyond experiment until measured.

**Consequences of the amendment.** Step ⑦d gains an optional reranking substep
behind a tenant/experiment flag — no change to Steps ④/⑤ or to the retrieval
economics. If the experiment is ever promoted, ADR-011's constraint still
applies: the rerank score becomes a *feature* feeding the existing ⑦d weighted
score, never a hard filter that drops or reorders candidates ahead of it.

## ADR-008: Orchestration — Prefect deployed for MVP

**Status:** Superseded (original decision reversed)

**Context.** Prefect 3 was originally selected as the orchestrator (self-hosted,
Apache 2.0 engine with a Community-licensed server component) to replace Spring
Batch's retry/skip/history properties. A subsequent MVP-scoping pass argued for
deferring orchestration entirely — plain async functions plus `tenacity` for
retry and `embeddedContentHash` for resume — on the grounds that a single-tenant,
manually-triggered, single-box pipeline does not need a scheduler or a run-history
UI yet.

**Decision.** That deferral did not hold — Prefect has already been deployed for
the MVP. This ADR is retained to record that the "no orchestration for MVP"
argument was made and then overridden by actual deployment, not quietly dropped.
`prefect` (and `prefect-client` for worker images) stay in the dependency
manifest.

**Consequences.** The MVP carries the container, Postgres database, and
dependency footprint that the deferral was meant to avoid — accepted, since the
deployment has already happened. Setup.md's Prefect section (Postgres database
setup, `+asyncpg` connection string, `@task` retry pattern) reflects the current,
deployed state and needs no further change on this account.

## ADR-009: No dedicated observability stack for the MVP; existing Promtail/Loki

**Status:** Accepted for MVP scope

**Context.** OpenTelemetry (collector + Tempo + Loki + Prometheus + Grafana) and
separately Splunk were both evaluated for logs, metrics, and traces. The server
already runs Promtail, which ships stdout logs to an existing Loki instance.

**Decision.** No OpenTelemetry collector, no Splunk. `structlog` emits structured
JSON to stdout; Promtail picks it up as-is. `tenant_id` / `run_id` / `stage` bound
to log context substitute for trace spans at MVP scale (single operator, single
box).

**Consequences.** Removes the OTel dependency set and the collector deployment
entirely for now. No distributed tracing and no first-class metrics backend until
reintroduced — acceptable because the MVP has no distributed call graph to trace.
Revisit if either becomes false: multiple services, or a need to correlate
duration/metrics rather than just read log lines.

---

## ADR-010: Apache Spark rejected for a post-MVP rearchitecture

**Status:** Rejected, not deferred to a specific trigger

**Context.** Proposed as a general-purpose rearchitecture after MVP validation.

**Decision.** Not adopted, and not treated as a default next step. Spark is
data-parallel; the pipeline's most expensive components (betweenness, Leiden) are
graph-shaped and specifically do *not* parallelise well under Spark's
partition-and-shuffle model — the same reasoning that motivated moving off Neo4j
GDS in ADR-002 applies against GraphX/GraphFrames too. LightGBM inference already
parallelises across cores without a cluster; HDBSCAN has no mature distributed
implementation; the embedding API calls are I/O-bound, not compute-bound.

**Consequences.** No general rewrite is planned. If a genuine bottleneck emerges
in production (candidate scenario named: feature-matrix assembly/joins across many
large concurrent tenants — the one piece of this pipeline that is actually
tabular and data-parallel), it should be measured first and addressed as a
scoped addition to that one stage, not as a platform-wide migration.

---

## ADR-011: GSC eligibility filter removed from candidate generation — mandatory

**Status:** Accepted, not conditional

**Context.** An earlier design gated which pairs could even become candidates
using a hard filter on GSC signals — `impressions > threshold AND position
within a band`. Two independent defects were identified. Statistically,
`avg_position` is a mean across every query a page ranks for; a page averaging
position 2.5 might be #1 on five branded queries and #28 on forty commercial
ones, and the filter cannot distinguish that from a page genuinely at position
2.5 throughout. Structurally, the filter excluded new pages (no GSC history yet),
orphan pages, striking-distance pages just outside the position band, and — most
consequentially — pillar pages, since a well-optimised pillar often already sits
in a position or impression range the filter treated as "done." That made the
standard hub-and-spoke internal linking pattern structurally unrepresentable:
the filter would refuse to recommend linking spokes to their own pillar.

**Decision.** Removed entirely and treated as a closed question, not a tunable
one. Candidate eligibility is now three hard constraints only —
`isIndexable = true`, `source != target`, `NOT already linked` — and nothing
GSC-derived gates eligibility. Every signal the old filter used (impressions,
position, position band, has-GSC-data) is instead passed to LambdaMART as a
**feature**, so the ranker itself decides how much any of it matters, informed
by acceptance feedback.

**Consequences.** This is the precedent the reranker decision (ADR-007) and any
future upstream-filtering proposal must be checked against: a hardcoded cutoff
before the ranker sees a candidate means the ranker can never learn that
candidate's true value, no matter how good the ranker later becomes. Any new
gate proposed on eligibility (as opposed to priority) should be treated as
presumptively wrong unless it excludes something genuinely never actionable
(e.g. `isIndexable = false`), not something merely currently unattractive by one
signal.
