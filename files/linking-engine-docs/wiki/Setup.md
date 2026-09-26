# Setup

Everything needed to go from empty repo to a passing `uv run pytest`.

---

## Prerequisites

```
Python 3.12 or 3.13     3.14 not yet supported by the ML tree
uv                      curl -LsSf https://astral.sh/uv/install.sh | sh
Docker Desktop          8 GB memory minimum
```

Neo4j must be **5.18 or later** — the audit uses `vector.similarity.cosine()`,
which does not exist before that. The Compose file pins 5.26.

---

## First run

```bash
git clone <repo> && cd linking-engine
uv sync --all-extras
cp .env.example .env          # add VOYAGE_API_KEY
make up                       # neo4j + mongo, schema applied
make seed                     # 614-page synthetic corpus
make sanity                   # confirm it looks right
uv run pytest
```

Browser at http://localhost:7474, `neo4j` / `localdevpassword`.

---

## Dependencies

Full manifest is in `pyproject.toml`. The decisions worth knowing:

| Package | Why this one |
|---|---|
| `pydantic` | The **only** contract crossing module boundaries. Bare dicts between modules are a lint failure |
| `python-igraph` + `leidenalg` | Reference implementations. Not GDS: no 4-core cap, no projection limit, no JVM cache misses on BFS. `leidenalg` is a separate package |
| `tokenizers` | Required for `count_tokens`. Voyage batching is by **tokens**, not list length |
| `motor` | Async Mongo. Direct `pymongo` import is banned — it is transitive and will vanish on an upgrade |
| `tenacity` | Per-call backoff *inside* a task. Prefect retries whole tasks; different granularity |
| `structlog` | Binds `tenant_id` / `run_id` / `stage` to log context, propagates through async |
| `prefect` | Engine is Apache 2.0; the `/server` directory is Prefect Community License — free for any use except competing with Prefect. Fine here, but a conscious choice |

### Pinned deliberately

```toml
"numpy>=2.0,<3"
```

`hdbscan`, `umap-learn` and `lightgbm` have each lagged on NumPy major bumps. An
unpinned upgrade breaking three ML packages simultaneously is a bad morning.

### Optional extras, not dependencies

```bash
uv sync --extra umap    # only if HDBSCAN underperforms at raw 2048d
uv sync --extra nlp     # spacy, ~500MB with model, only for extraction rung 3
```

`umap-learn` pulls `numba`, the usual source of Apple Silicon build friction. Try
without it first. For `nlp`, check the rung 1–2.5 hit rate before installing — if
they handle most cases it is deferrable.

Post-install for the `nlp` extra:

```bash
uv run python -m spacy download en_core_web_sm
uv run python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

---

## Quality gates

All five run in CI. A PR that fails any of them does not merge.

```bash
uv run ruff check --fix        # lint
uv run ruff format             # format
uv run mypy --strict src/      # types
uv run deptry src/             # declared-vs-imported drift
uv run lint-imports            # architectural boundaries
uv run pytest                  # tests, 75% coverage floor
```

### What each catches that the others don't

**`deptry`** finds dependency drift. The one that bites is DEP003 — importing a
transitive package directly. `motor` pulls `pymongo`, `mlflow` pulls a large tree.
Import those directly and it works until someone upgrades.

**`lint-imports`** enforces boundaries between *our own* modules. Different problem
from deptry entirely. Contracts live in `.importlinter`:

```
layers                 api → pipeline → discovery → anchor → audit
                       → graph → embedding → models
domain-independence    audit, discovery, anchor talk through models/, never
                       to each other
algorithms-are-pure    graph.algorithms may not import neo4j, motor or pymongo
api-excludes-ml        api may not import lightgbm, hdbscan, mlflow, sklearn
models-are-leaves      models import nothing from the app
```

`algorithms-are-pure` is the one that matters most. It makes the igraph decision
**structural** rather than a convention — algorithms take edge lists and return
arrays, and only the repo layer speaks Bolt. Without it, someone adds a
convenience query inside an algorithm function and the separation dissolves.

Get these in at PR #1. Retrofitting contracts onto code that already violates them
means either a refactor or a pile of exemptions that defeat the purpose.

### Banned imports

```toml
"requests" → use httpx, the codebase is async
"pymongo"  → use motor, direct import is transitive
```

### mypy is stricter in the core

`models/`, `audit/` and `anchor/` additionally set `disallow_any_explicit`. No
`Any` in the domain core, where a renamed Pydantic field silently drops data at a
boundary.

---

## Repository layout

```
linking-engine/
├── src/linking_engine/
│   ├── api/              FastAPI routers, dependencies
│   ├── models/           Pydantic — the only cross-module contract
│   ├── ingest/           corpus loading, GSC, keyword upload
│   ├── embedding/        Voyage client, batching, write path
│   ├── graph/
│   │   ├── repo.py       Bolt queries — the ONLY place that talks to Neo4j
│   │   └── algorithms.py igraph, leidenalg, hdbscan — pure functions
│   ├── audit/            Stage 1 scoring and verdicts
│   ├── anchor/           keyword resolution, extraction ladder
│   ├── discovery/        retrieval, features, ranking
│   ├── ml/               training, MLflow, promotion gate
│   └── pipeline/         Prefect flows and tasks
├── tests/
├── scripts/              corpus generator, eval harnesses
├── .importlinter
└── pyproject.toml
```

The `graph/repo.py` versus `graph/algorithms.py` split is enforced by contract,
not convention.

---

## Pydantic conventions

```python
class LinkAuditResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    source_url: HttpUrl
    target_url: HttpUrl
    anchor_quality_score: float = Field(ge=0, le=100)
    keyword_alignment: float = Field(ge=0, le=1)
    context_relevance: float = Field(ge=0, le=1)
    anchor_target_fit: float = Field(ge=0, le=1)
    issue_flags: frozenset[IssueFlag]
    verdict: ActionType | None
```

`extra="forbid"` catches a renamed field at the boundary instead of silently
dropping it. `frozen=True` wherever a model represents a computed result.

---

## Logging

```python
log = structlog.get_logger()
structlog.contextvars.bind_contextvars(
    tenant_id=tenant.id, run_id=run.id, stage="audit"
)
log.info("audit.complete", edges_scored=n, verdicts=len(v))
```

Event names are dotted and stable — queries depend on them. `merge_contextvars`
propagates through async, so Prefect tasks inherit the binding without passing
anything down.

Never log page bodies, embeddings, or credentials.

---

## Prefect

Local runs need no server — flows execute in-process. Start the server only when
you want run history and the UI.

```bash
uvx prefect server start        # SQLite, ephemeral
```

For anything persistent, point it at the Postgres already running alongside
MLflow, using a **separate database** on the same instance:

```sql
CREATE DATABASE prefect;
```

```bash
PREFECT_API_DATABASE_CONNECTION_URL="postgresql+asyncpg://prefect:pass@host:5432/prefect"
```

Note `+asyncpg`. Prefect's server uses async SQLAlchemy and fails with a plain
`postgresql://` URL — the most common setup mistake.

Sharing one database between MLflow and Prefect means two tools issuing DDL
against the same schema. Separate databases cost nothing.

---

## Testing layers

| Layer | Tool | Rule |
|---|---|---|
| Unit | pytest | Pure functions: extraction ladder, scoring, verdict mapping |
| Repo | testcontainers | Real Neo4j and Mongo. Never mocked — graph logic is not unit-testable against a fake |
| Contract | schemathesis | OpenAPI fuzzing against the live app |
| Pipeline | Prefect test harness | Task retry and skip behaviour |

`asyncio_mode = "auto"` is set, so async tests need no decorator.
