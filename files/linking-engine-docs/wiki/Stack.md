# Stack

Python throughout. The v1–v5 spec assumed Java/Spring Boot with a Python sidecar for ML; that split is gone.

---

## Why this matters more than it looks

Every prior build estimate carried a **Java ↔ Python boundary tax**. PyTorch Geometric, LightGBM, and every embedding client are Python; the pipeline and API were Java. That meant a REST or gRPC hop, duplicated Pydantic/POJO models on both sides, serialisation of a 3105-dimensional feature vector across a process boundary per pair, and two dependency trees.

It was flagged as one of the three highest-risk items in the build, at 3–5 engineer-days of pure integration cost plus ongoing drift.

Going Python-native removes it entirely. The feature vector is a NumPy array passed by reference. Pydantic models are the *only* contract, not one of two. Concretely: **roughly 10–14 engineer-days saved** and the riskiest integration seam deleted.

---

## Core

| Concern | Choice | Notes |
|---|---|---|
| Language | Python 3.13 | Free-threaded build not required; GIL is not the bottleneck here |
| API | FastAPI | Async, OpenAPI generated, Pydantic-native |
| Validation | Pydantic v2 | Rust core, fast enough to validate on hot paths |
| ASGI server | Uvicorn + Gunicorn | Workers sized to core count |
| Package manager | uv | Lockfile committed, ~10× faster than pip in CI |
| Lint / format | Ruff | Replaces flake8, isort, black |
| Types | mypy strict | Enforced in CI, no `Any` in domain modules |

---

## Pipeline orchestration

**Prefect 3** replaces Spring Batch.

Spring Batch was chosen for step-level retry, skip policies, and persistent job history across an 11-step pipeline. Prefect gives the same properties: `@task(retries=3, retry_delay_seconds=[10,60,300])`, run history, and a UI, with Apache 2.0 licensing and a self-hostable server.

```python
@flow(name="pipeline", task_runner=ThreadPoolTaskRunner(max_workers=4))
async def run_pipeline(tenant_id: str, mode: RunMode) -> RunResult:
    pages = await ingest(tenant_id)
    await embed(tenant_id, pages)
    await graph_analytics(tenant_id)
    audit = await link_audit(tenant_id)          # Stage 1
    if tenant.discovery_enabled:
        audit += await discover(tenant_id)        # Stage 2
    return await materialise(tenant_id, audit)
```

Argo Workflows was the alternative — already K8s-native given ArgoCD. Rejected because the pipeline is a data flow with typed Python objects between steps, not a container DAG, and Prefect keeps that in-process.

---

## Data

| Store | Client | Purpose |
|---|---|---|
| Neo4j 5.18+ | `neo4j` async driver + `graphdatascience` | Graph, vectors, GDS algorithms |
| MongoDB | `motor` (async) | Page text, metrics, recommendations, feedback |
| Valkey | `redis-py` (Valkey-compatible) | Recommendation cache, 24h TTL |
| MinIO | `boto3` | MLflow artefact backend |

Neo4j **5.18 or later is required** — `vector.similarity.cosine()` is used in the audit query. On older 5.x you would pull both vectors per edge and compute in Python, which is materially slower at 250k edges.

### Graph algorithms run outside the database

igraph and `leidenalg` are the reference implementations; Neo4j GDS is a port.
Running in-process removes three costs at once:

```
4-core concurrency cap    Community licensing, not the algorithm
3-projection limit        no build, no drop, no leak on failure
JVM cache misses          BFS is pure pointer chasing, and GDS stores
                          nodes as heap objects rather than contiguous arrays
```

That last one dominates. Betweenness is O(V·E) — 6.25 billion edge visits at
25k nodes — and BFS punishes scattered memory harder than almost any other
workload. Contiguous `int32` adjacency arrays keep the prefetcher fed; heap
pointers do not.

```
edge pull from Neo4j   10-25s
build igraph object     1-3s
PageRank                2-5s
Leiden × 2              5-15s
betweenness (exact)    60-190s
write back             10-20s
──────────────────────────────
total                 ~2-4 min     vs 15-35 min in GDS
```

**Exact betweenness becomes affordable**, which removes the sampling instability
from the design entirely. Neo4j internal ids are not stable across restarts, so
build the id mapping fresh each run rather than caching it.

---

## ML

| Concern | Choice |
|---|---|
| Embeddings | `voyageai` SDK, `voyage-4-large` @ **2048d** |
| Tokenizer | `tokenizers` — required for token-aware batching |
| Local embedding fallback | `sentence-transformers`, `voyage-4-nano` |
| Graph algorithms | **python-igraph + leidenalg** (not GDS) |
| Content clustering | **hdbscan** (+ umap-learn if reduction is needed) |
| GNN | PyTorch 2.x + PyTorch Geometric |
| Ranker | LightGBM (`lambdarank`) |
| Content gap verdicts | `ollama` client, Qwen3.5-4B |
| Text processing | spaCy (noun phrases), Lucene-style stemming via `nltk` |

### MLflow

Replaces hand-rolled MinIO artefact versioning from the earlier spec.

```
Tracking     every training run: params, metrics, NDCG@10 over time
Registry     gnn-encoder, lambdamart-ranker, anchor-weights
Stages       Staging → Production, with the promotion gate as a
             registry transition rather than application logic
Artifacts    MinIO as backing store (S3-compatible)
```

The promotion gate becomes a first-class concept: a model version only transitions to `Production` if its holdout NDCG@10 exceeds the current `Production` version. Rollback is a registry transition, not a file copy.

```python
if candidate_ndcg > production_ndcg:
    client.transition_model_version_stage(
        name="lambdamart-ranker", version=v,
        stage="Production", archive_existing_versions=True,
    )
```

---

## Observability

**MVP:** no dedicated observability stack. `structlog` emits structured JSON to
stdout; the server's existing Promtail instance ships it to the existing Loki,
with no new deployment required.

**Production (deferred):** OpenTelemetry (collector + Tempo + Prometheus,
Loki kept) once there are multiple services or concurrent tenants worth tracing.
Splunk was evaluated and rejected — ingest-volume billing is a real cost
(roughly $1,800/yr at 1GB/day) against a workload that Loki, already running,
covers for free.

Every log line carries `tenant_id`, `run_id`, and `stage` so a pipeline run is
queryable end to end via Loki/LogQL:

```
{job="linking-engine"} | json | run_id="..." | line_format "{{.stage}} {{.event}}"
```

**Never log page bodies or embeddings.** Log at INFO by default; DEBUG behind a
per-tenant flag once ingest volume is actually being billed (production phase).

---

## Infrastructure

Unchanged from the v5 spec: K3s single node on Hetzner, ArgoCD for GitOps, one Neo4j Community StatefulSet per tenant via Helm, Traefik ingress, sealed secrets.

---

## Repository layout

```
linking-engine/
├── src/linking_engine/
│   ├── api/              FastAPI routers, dependencies
│   ├── models/           Pydantic — the only cross-module contract
│   ├── ingest/           crawler, GSC, keyword upload
│   ├── graph/            Neo4j repo, GDS calls, Cypher
│   ├── embedding/        Voyage client, local fallback, caching
│   ├── audit/            Stage 1 scoring and verdicts
│   ├── anchor/           keyword resolution, extraction ladder
│   ├── discovery/        retrieval, GNN, cross-attention, ranker
│   ├── ml/               training, MLflow, promotion gate
│   ├── pipeline/         Prefect flows and tasks
│   └── tenancy/          config, provisioning, routing
├── tests/
├── charts/               Helm
├── migrations/           Neo4j constraints and indexes
└── pyproject.toml
```
