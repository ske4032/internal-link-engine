# Development

## Prerequisites

```
Python 3.13
uv
Docker               (testcontainers only)
Neo4j 5.18+          (vector.similarity.cosine is required)
Ollama               (CONTENT_GAP verdicts only)
```

## Setup

```bash
git clone git@github.com:<org>/linking-engine.git
cd linking-engine
uv sync
cp .env.example .env          # service endpoints and credentials, VOYAGE_API_KEY
uv run pytest
```

Neo4j, MongoDB, Prefect and MLflow are hosted outside this repository (ADR-014).
Neo4j constraints and vector indexes are applied by the project's own migration
runner (issue #3). There is no separate migration tool.

## Running a pipeline locally

```bash
# PREFECT_API_URL in .env points at the hosted Prefect server and its UI
uv run --env-file .env python -m linking_engine.pipeline.run \
    --tenant demo --mode initial --limit 200
```

`--limit` caps pages so a local run finishes in minutes. Full runs belong on the cluster.

## Running the API

```bash
uv run uvicorn linking_engine.api.main:app --reload
# docs at http://localhost:8000/docs
```

## Testing layers

| Layer | Tool | Rule |
|---|---|---|
| Unit | pytest | Pure functions: extraction ladder, scoring, verdict mapping |
| Repo | testcontainers | Real Neo4j and Mongo, never mocked |
| Contract | Pydantic + schemathesis | OpenAPI fuzzing against the live app |
| Pipeline | Prefect test harness | Task retry and skip behaviour |

Graph logic is not unit-testable against a mock. Use `testcontainers-python` with a real Neo4j image, seeded from a fixture graph.

## Quality gates

```bash
uv run ruff check --fix
uv run ruff format
uv run mypy --strict src/
uv run pytest --cov=linking_engine --cov-fail-under=75
```

All four run in CI. `mypy --strict` has no exemptions in `models/`, `audit/`, or `anchor/`.

## Pydantic conventions

Models are the only contract crossing module boundaries. No bare dicts.

```python
class LinkAuditResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    source_url: HttpUrl
    target_url: HttpUrl
    anchor_quality_score: float = Field(ge=0, le=100)
    keyword_alignment: float = Field(ge=0, le=1)
    context_relevance: float = Field(ge=0, le=1)
    issue_flags: frozenset[IssueFlag]
    verdict: ActionType | None
```

`extra="forbid"` catches renamed fields at the boundary instead of silently dropping them. `frozen=True` everywhere a model represents a computed result.

## Logging

```python
log = structlog.get_logger()
log = log.bind(tenant_id=tenant.id, run_id=run.id, stage="audit")
log.info("audit.complete", edges_scored=n, verdicts=len(v))
```

Event names are dotted and stable — Loki/LogQL queries depend on them. Never log page bodies, embeddings, or credentials.

## Local model work

The MLflow tracking server is hosted (ADR-014). `MLFLOW_TRACKING_URI` and its basic-auth credentials in `.env` point training scripts at it, and its web UI is where runs are compared:

```bash
uv run --env-file .env python -m linking_engine.ml.train_ranker --tenant demo
```

Promotion is blocked locally. Only CI may move the `production` alias.
