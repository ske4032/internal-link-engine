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

**Status:** Accepted, amended 2026-09-26 — see amendment below

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

*The correctness gate in this paragraph is superseded; see the 2026-09-26 amendment below.*

### Amendment — 2026-09-26: GDS removed from every environment; the correctness gate is replaced

**Trigger.** Neo4j GDS is no longer installed anywhere. The deployed Neo4j is
5.26.30 Community with APOC only and zero `gds.*` procedures. igraph and
`leidenalg` are now the only implementations of PageRank, Leiden and betweenness
in every environment, not the preferred of two. The correctness gate in the
original consequences — Spearman vs GDS output > 0.99, assigned to PR-Roadmap
#26 and issue #8 — has nothing left to compare against.

**What the old gate was actually protecting.** igraph and `leidenalg` are the
reference implementations; the algorithms are not where the risk sits. The risk
is this project's code around them: the edge pull, the Neo4j-id-to-vertex-index
mapping rebuilt every run, edge orientation (PageRank directed, betweenness
undirected), and the write-back. A GDS comparison exercised those only
incidentally. The replacement targets them directly.

**Amended decision.** Two checks replace the GDS comparison, each covering what
the other cannot.

*Fixture graphs with closed-form expected values* — unit layer, every CI run.
Path graph P_n: betweenness of vertex i is i·(n−1−i). Star S_n: centre
betweenness (n−1)(n−2)/2, leaves 0. Directed cycle C_n: PageRank 1/n everywhere.
One small directed graph with a dangling node, hand-computed once with the
derivation committed beside it. Two cliques joined by a single edge: Leiden
returns exactly the two cliques, identically across two runs with a fixed seed.
Every fixture also runs with Neo4j ids shuffled, so the id mapping — not fixture
ordering — is what produces the answer. Tolerance: relative error ≤ 1e-9.

*An independent networkx reference* — repo layer (testcontainers), 600-page
synthetic corpus, at #8 and #9 acceptance and on any change under `graph/`. The
reference graph is built from a separate URL-keyed Cypher read, not from the
pipeline's id mapping, and compared by URL after the pipeline has written back
to Neo4j — so pull, mapping and write-back are covered end to end.
- PageRank (damping 0.85, unweighted, networkx run to `tol=1e-12`): Spearman
  ≥ 0.999 and max absolute difference ≤ 1e-6.
- Exact betweenness (networkx `normalized=False`, undirected): max absolute
  difference ≤ 1e-9 × max value. Issue #8 stores betweenness as a percentile
  rank, so this raw comparison runs on the igraph output before conversion,
  and the written-back ranks must equal the percentile ranks of the networkx
  values computed with the same function, ties aside.
- Leiden has no cross-implementation equivalent. networkx `modularity()`,
  recomputed on the written-back `linkCommunityId` labels, must match
  the `leidenalg` partition's `modularity` within 1e-9. Compare against
  `modularity`, not `quality()`: for `RBConfigurationVertexPartition`,
  `quality()` is unnormalised and equals modularity × 2m (verified 2026-09-26). Partition quality against planted
  topics remains gate #11's job, unchanged.

The thresholds are tighter than the old 0.99 deliberately: that allowance
absorbed a different implementation's iteration defaults. Two implementations of
the same definition, run to convergence on the same graph, agree to rounding;
anything looser hides a real bug. `networkx` is a dev-only dependency and `src/`
may not import it. `graphdatascience` joins the banned imports, and no `gds.*`
call is permitted anywhere, dev scripts included.

**What still holds.** The original decision and its reasoning are unchanged.
Neo4j remains an edge and vector store; internal ids are still rebuilt every
run, never cached. APOC is present on the server, nothing in the core build
depends on it, and its presence is not a route back to in-database graph
algorithms.

**Consequences of the amendment.** Issue #8's step e and its first acceptance
criterion ("Spearman vs GDS > 0.99") are replaced by the two checks above; issue
#9's acceptance gains the modularity cross-check. PR-Roadmap #26 becomes
"Correctness harness: closed-form fixtures and networkx reference" (PR-Roadmap
numbering — unrelated to Core Build Plan issue #26, the NDCG@10 harness).
Measurement Backlog §4 is settled by environment rather than measurement: its
current position, measurement snippet and decision rule are rewritten, and its
priority-order entry becomes "igraph vs networkx correctness"; the speed half
survives as #8's per-algorithm timings. `dev/scripts/eval_clustering.py` runs
Leiden through `graphdatascience`, so `make eval` — and therefore gate #11 — is
broken until `run_leiden()` moves to `leidenalg`, which also removes its silent
Louvain fallback; that fix blocks #11. `02-verify.cypher`,
`04-clustering-bench.cypher` and the Makefile targets `verify-db`,
`projections`, `drop-projections` and `bench` all call `gds.*` and are deleted
rather than ported: projection hygiene has no meaning without projections, and
Leiden vs Louvain is already decided. The one non-GDS check in `02-verify.cypher`,
`vector.similarity.cosine()`, already lives in issue #3's startup health check.

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

**Status:** Superseded (original decision reversed); Postgres consequence revised by ADR-014 (2026-09-26)

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

*Revised 2026-09-26 by ADR-014: the Prefect server and its Postgres run outside
this project, so Setup.md's Prefect section no longer describes anything the
project does.*

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

---

## ADR-012: PyMongo native async replaces Motor

**Status:** Accepted (supersedes the `motor` choice on Stack, Setup and PR-Roadmap #5)

**Context.** `motor` was chosen as the async MongoDB client, with a ruff rule
banning direct `pymongo` imports on the grounds that `pymongo` was a transitive
dependency. PyMongo's native async API went GA in 4.13, and Motor 3.7.1 was
deprecated on 14 May 2026, a year later. Motor wraps synchronous PyMongo and
dispatches each call to a thread pool; `AsyncMongoClient` is native asyncio.
Compared directly, the two collection classes share 41 identically named
methods — the migration is the import line and the client class name. No Mongo
application code exists yet.

**Decision.** `pymongo>=4.13` with `AsyncMongoClient` is the async MongoDB
client. `motor` leaves the dependency manifest and becomes the banned import; the
ruff rule is inverted.

**Consequences.** Switching now costs an import line; switching after issue #3
would mean touching every repository method. One fewer package and no thread-pool
hop per call. `pymongo` is now a direct dependency, so the DEP003 argument that
justified the old ban now applies to `motor` instead. Import-linter contracts
must forbid `pymongo` wherever they forbid `motor` — `models-are-leaves`
currently forbids `motor` but not `pymongo`, which after this change would leave
the models free to import the real driver. The dev scripts' synchronous
`MongoClient` comes from the same package and is unaffected.

---

## ADR-013: MongoDB runs standalone, with no replica set

**Status:** Accepted

**Context.** The deployed MongoDB 8.0.32 is a standalone `mongod`, shared with an
unrelated client crawl database. The project's compose file assumed an `rs0`
single-node replica set, and issue #3 says `directConnection=true` is required
because "Mongo needs the replica set for transactions". An audit of every
existing Mongo call found `insert_many` and `replace_one` only — no sessions, no
transactions, no change streams. Every "same transaction" reference in the build
plan (issue #5, Core Build Plan, PR-Roadmap #19) is the Neo4j write that sets
`embeddedContentHash`, not Mongo.

**Decision.** MongoDB stays standalone. Every Mongo write is single-document
atomic **and idempotent**: a re-run after partial failure converges to the same
state. In practice that means keyed upserts (`replace_one` / `update_one` with
`upsert=True`, or `bulk_write` of them) on a deterministic key backed by a unique
index leading with `tenantId`, never a bare `insert_many` into a collection a
re-run does not first clear. A multi-document change that must appear atomic —
publishing a run's recommendations — is written under its `runId` and made
visible by flipping one pointer document, which is a single-document write.
`directConnection=true` is dropped from connection strings. The project touches
only its own `linking_engine` database, named by `MONGO_DB` and never enumerated
or dropped by code. At MVP scope it connects with the instance's root
credentials, decided 2026-09-26 for a single-operator test build. A read-write
user scoped to `linking_engine` is the step to take before any second operator
or deployment shares this instance.

**Consequences.** No multi-document transactions, no change streams, no
`majority` read concern. Retryable writes are silently disabled — the driver does
not retry a failed write against a standalone — so `tenacity` at the repository
call is the retry, and that is safe only because of the idempotency rule. Crash
resume already rests on Neo4j's `embeddedContentHash`, not on Mongo. Stage 08
writes Neo4j edges and a Mongo payload together; no Mongo transaction could span
that anyway, so the `runId` pointer pattern is needed on any topology. Code
written to this rule runs unchanged on a replica set, which makes standalone the
reversible direction. Revisit when a feature on the plan needs multi-document
atomicity the pointer pattern cannot express, or change streams (e.g.
event-driven cache invalidation — out of scope until after #30); the first PR
that opens a session or calls `watch()` reopens this ADR rather than landing.
Converting then means restarting the shared `mongod` with `--replSet`, which
interrupts that database; a `keyFile` if authorization is on; and a member hostname
every client can resolve — the reason the old compose needed
`directConnection=true` was that it advertised `localhost:27017`. A dedicated
`mongod` for this project is the alternative to weigh at that point, not a
default conversion of the shared one.

---

## ADR-014: Runtime services are hosted outside this repository

**Status:** Accepted (revises ADR-008's consequence on the Prefect Postgres footprint)

**Context.** Neo4j (5.26.30 Community, APOC only), MongoDB (8.0.32 standalone,
shared — ADR-013) and the Prefect 3.8.6 server run in Docker from the user's own
compose project, outside this repository. The project's
`dev/docker-compose.yml` cannot start beside them: ports 7474, 7687 and 27017
are already bound, as is 4200 for the `prefect server start` that Development
tells readers to run, and the Makefile `docker exec`s into containers named
`neo4j` and `mongo` that do not exist. MLflow 3.16.1 runs on the user's k3s
server with built-in basic auth; its backend store and auth database are in
Postgres, and it proxies artifacts to MinIO (`mlflow-artifacts:/`). The client
therefore needs neither `boto3` nor MinIO credentials, and no Postgres
credentials exist on the workstation. The Prefect server's database likewise
lives outside the project.

**Decision.** The project runs no services of its own. It connects to
externally hosted Neo4j, MongoDB, Prefect and MLflow through endpoints and
credentials in `.env`, every one listed in `.env.example`. Client library
versions track the deployed server versions and never outrun them:
`mlflow>=3.16,<3.17` and `prefect>=3.8,<3.9`, raised only in the change that
records a server upgrade, with the startup health check (issue #3, step f)
asserting each client is no newer than its server so the rule fails loudly
rather than by memory. `asyncpg` and `boto3` are removed. The dev compose file,
`dev/mongo-init/`, and every Makefile target that shells into a container are
deleted once the collection indexes in `01-collections.js` are ported into issue
#3's Mongo repository as idempotent index creation at startup; `make schema` is
replaced by #3's migration runner, and `make sanity` survives rewritten to run
through the driver. Repo tests keep testcontainers, with images pinned to the
deployed versions: `neo4j:5.26-community` without GDS, `mongo:8.0` standalone.

**Consequences.** ADR-008's Postgres footprint still exists but is owned and
operated outside this project; the project carries only the Prefect client, and
Setup's Prefect section (database creation, `+asyncpg` URL) no longer describes
anything this project does. The MLflow client authenticates with
`MLFLOW_TRACKING_USERNAME` / `MLFLOW_TRACKING_PASSWORD`; `MLFLOW_S3_ENDPOINT_URL`
and the boto3 failure mode in issue #27 no longer apply. At a 3.16 floor, model
registry stages are deprecated in favour of aliases (deprecated since MLflow
2.9.0, verified against the installed 3.16.1 client), so the promotion gate
becomes an alias move rather than a stage transition — same gate, holdout
NDCG@10, different mechanism. Nobody without access to the user's services can
run the pipeline end to end; testcontainers still covers the repo layer —
accepted at single-operator MVP scope. Neo4j heap and page-cache sizing are no
longer declared in this repository, so the memory ceiling for chunked inference
must be measured against the deployed settings. PR-Roadmap #8 (Helm-deployed
Neo4j and Mongo) and #54 (MLflow and MinIO, Helm-deployed) describe services
that now exist outside the project; both stay out of scope until #30 and are
rewritten when the roadmap reopens, not now. Deletion is reversible through git
history; if a second developer or CI ever needs the full stack, the replacement
mirrors the deployed topology rather than restoring the old file.

---

## ADR-015: One Neo4j instance, every node scoped by `tenantId`

**Status:** Accepted (2026-09-26; supersedes "one Neo4j instance per tenant" in Decisions)

**Context.** Real crawl data, a client site's crawl collection, will sit
beside the synthetic `demo` corpus. MongoDB is already tenant-scoped. Neo4j was
not: the synthetic seeder wiped the whole graph, page URL uniqueness was global,
and vector search and the graph algorithms spanned every page. The documented
design was one Community instance per tenant; the alternative is one instance
with a tenant key on every node.

**Decision.** One Neo4j instance. Every node carries `tenantId`. Uniqueness is
composite: `(tenantId, url)` for `Page`, `(tenantId, text, language)` for
`Keyword`. Every repository read and write takes a tenant id and filters on it;
the graph pulls behind PageRank, Leiden and betweenness are per tenant; no query
may return or modify another tenant's nodes. Relationships are scoped through
their endpoints. Deletion is per tenant, `MATCH (n {tenantId: $t}) DETACH DELETE n`,
never a global wipe. The already-loaded synthetic corpus was stamped
`tenantId = 'demo'` on 2026-09-26.

**Consequences.** One driver and no per-tenant infrastructure, but isolation now
rests on the repository layer rather than on separate processes, so the repo
test suite must include a cross-tenant leak test. The shared vector index
returns neighbours from every tenant, so retrieval (issue #12) must isolate: an
exact per-tenant cosine scan, post-filtering with oversampling, or a per-tenant
label with its own index. That choice belongs to #12. The legacy seeder's global
`MATCH (n) DETACH DELETE n` must never run once a second tenant is loaded.
Revisit if tenants grow large enough that shared heap, page cache or noisy
neighbours matter; one instance per tenant remains the escape route.
