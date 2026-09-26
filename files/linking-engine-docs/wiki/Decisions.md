# Decisions

What was rejected, and why. Recorded so it isn't re-litigated.

---

## Stack

**Python over Java/Spring Boot.** *(changed after v5)*
PyTorch Geometric, LightGBM, and every embedding client are Python. A Java pipeline meant a process boundary, duplicated models on both sides, and serialising a 3105d feature vector per pair. Removing it saves roughly 10–14 engineer-days and deletes the riskiest integration seam in the build.

**Prefect over Spring Batch, Airflow, Argo Workflows.**
Needed step-level retry, skip policies, and persistent run history. Prefect gives all three in-process with typed Python objects between steps. Argo Workflows is container-DAG shaped, which fits the deployment but not the data flow.

**MLflow over hand-rolled MinIO versioning.**
Earlier specs versioned `gnn_encoder_vN.pt` by filename. MLflow makes the promotion gate a registry alias move rather than application logic, and rollback moving the alias back rather than a file copy. Registry stages were deprecated in MLflow 2.9; see ADR-014.

**No dedicated observability stack for MVP.** `structlog` JSON to stdout, picked up by the server's existing Promtail → Loki. OpenTelemetry and the Splunk-vs-Loki question are both deferred to production scope, where multiple concurrent tenants or a distributed call graph would justify the cost. See [[ADRs]] ADR-009.

**Body links only.** The crawler extracts links from page body content. Nav,
header, footer and sidebar links are never captured, so they never reach Neo4j and
no algorithm sees them. This matches how SEOs work: template links are audited
once per template, not per page, and they are a different discipline from
editorial linking.

Measured on the synthetic corpus at a realistic 77% template ratio, before the
rule was fixed: with template links included, 10/10 of the top PageRank and
betweenness pages were `/terms`, `/contact` and `/sitemap`. Clustering was
unaffected — Leiden recovered the planted topics either way, ARI delta −0.007,
because utility pages form their own dense community rather than blurring topic
boundaries.

So the rule matters for **authority and centrality, not clustering** — narrower
than it first appeared, and worth recording so nobody assumes Leiden needed it.

Consequence: `REPOSITION` is not a verdict. With no footer links captured there is
nothing to move a link out of.

**Graph algorithms outside the database.** igraph and `leidenalg` are the
reference implementations; GDS is a port. In-process removes the 4-core Community
cap, the 3-projection limit, and JVM cache-miss overhead on BFS — which dominates,
since betweenness is 6.25 billion edge visits at 25k nodes and BFS is pure pointer
chasing. Graph analytics drops from ~50 min to 2–4 min, and exact betweenness
becomes affordable, removing sampling instability from the design.

**Neo4j stays as the edge and vector store.** Replacing it is a separate decision
resting on per-tenant idle memory (~2.5 GB per JVM), not on algorithm speed.

**HDBSCAN alongside Leiden, not instead of it.** They cluster different things:
Leiden over graphs, HDBSCAN over content embeddings. HDBSCAN's `-1` noise label
expresses something neither Leiden pass can — a page belonging to no coherent
topic. Since Leiden now runs in-process, keeping both costs seconds rather than a
second database. Conditional on Measurement Backlog §1.

**Jaccard where set sizes are comparable, cosine where they are not.** Jaccard is
a ratio, so a 3-token anchor against a 2,000-token page maxes out near 0.0015 and
every score collapses into noise. Anchor diversity moves from cosine to Jaccard —
over-optimisation is about repeating words, and cosine rates "commercial press
machine" as non-diverse against "industrial hydraulic press" at 0.85 when Jaccard
correctly gives 0.20.

**`anchorTargetFit` added.** `keywordAlignment` cannot catch an anchor that matches
the target's keyword while the target page is about something else. Lexical
matching passes those clean; `cosine(anchor, target.content_embedding)` catches them.

**voyage-4-large at 2048d, chunking deferred.** MRL means 1024 is the first 1024
dimensions of the 2048 vector, so downgrading is a truncation of data already held
while upgrading needs a full re-embed and GNN retrain. On a deadline, take the
reversible direction. voyage-context-4 supports manual chunking
(`enable_auto_chunking=False`, nested lists), so the threshold approach remains
available — it is deferred, not blocked.

**Neo4j Community, one instance per tenant.** Enterprise multi-database is $10k–$50k+/year for the same isolation. No graph algorithms run inside Neo4j at all (ADR-002), so nothing edition-gated is needed.

**Valkey over Redis.** Redis moved to SSPL in 2024 — licence risk for commercial SaaS. Valkey is the BSD fork, API-identical.

**No Postgres/pgvector.** Collections are document-shaped with variable schema. Mongo for documents, Neo4j native HNSW for vectors, no third store.

**No Terraform.** Single static K3s cluster. Plain `hcloud` scripts, ArgoCD above the node.

---

## Models

**`voyage-4-large` API over local Qwen3-Embedding-0.6B.**
A local bake-off on real client content showed no meaningful quality difference. `content_embedding` is one input among ~270 GNN features, and downstream models train on acceptance feedback, so two competent embedders converge. With quality neutral the decision moved to operations: one less container, no CPU embedding bottleneck. Exit path is `voyage-4-nano` — all Voyage 4 models share an embedding space, so switching needs no re-index.

**2048 dimensions.** MRL makes this the reversible direction: 1024 is a truncation of the 2048 vector, so downgrading later costs nothing beyond re-truncating stored vectors, while upgrading from 1024 would need a full re-embed plus GNN retrain. Note the earlier local test (~0.01 cosine delta, 6/6 rank agreement, ~1024 vs 2048) was run on `voyage-4-nano` over one article with short queries — it does not establish that 2048 is unnecessary for `voyage-4-large` on full-page document-to-document comparison, which has not been tested. 2048 was chosen for reversibility, not because 1024 was shown sufficient.

**GraphSAGE over GCN.** GCN is transductive and cannot generalise to pages added after training. Clients publish constantly, and new pages are exactly the ones that need links.

**LambdaMART over neural regression.** Traffic delta is not attributable to a single link — confounded by algorithm updates, seasonality, content changes. Acceptance is directly observable, and ranking works well with hundreds of labels.

**Generative models cannot substitute for embedding models.** No bidirectional attention, no contrastive training. Qwen3.5-4B as an embedder would score roughly 45–52 MTEB against 64+ for a dedicated model.

---

## Rejected outright

**Reranking stage.**
Three reasons, cost being the weakest. Cross-attention is already a cross-encoder, fine-tuned monthly on acceptance feedback — a general reranker has never seen a linking decision. The shape is wrong: no queries exist in this pipeline, only page-to-page and sentence-to-page comparison, so adopting one means synthesising a pseudo-query and discarding most of the target page. And the economics collapse under their own fix — cheap enough to run means reordering a list the ANN already narrowed sharply, with the ranker about to order it properly.
*Revisit only if* NDCG@10 plateaus **and** error analysis shows topical-relevance misses. Then add as a feature in the pair vector, never as a filter.

**GSC eligibility filter.** *(removed v4)*
Required `impressions > 500 AND position 4–20`. Two defects: `avg_position` is a mean over a heavily skewed query distribution, so a page averaging 2.5 might rank #1 on five branded terms and #28 on forty commercial ones. And it structurally excluded new pages, orphans, striking-distance pages, strategic pages, and pillars — making hub-and-spoke unrepresentable. Eligibility is not priority.

**GSC position band for keyword selection.** *(removed v5)*
The same defect relocated. A page at position 3 on a high-value term still needs links to reach 1. Replaced by `impressions × (CTR@1 − CTR@current)`.

**LLM anchor generation.** *(removed v5)*
Present for four versions, inherited from a critique of a T5-base implementation. The debate was always *which model generates*, never *whether generation is the right operation*. Inserting an invented phrase bolts a link onto copy that was never about the topic. If no phrase exists, the honest finding is a content gap. Removing it collapsed anchor runtime from 30–90 minutes to string and vector operations.

**LLM-as-judge naturalness scoring.** *(removed v5)*
Carried self-preference bias — valid for ranking within a batch, not across strategies. Made redundant by extraction: a phrase pulled from the page already reads naturally, because that is what made it extractable.
