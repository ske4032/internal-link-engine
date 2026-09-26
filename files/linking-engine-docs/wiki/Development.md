# Development

## Prerequisites

```
Python 3.13
uv
Docker + Docker Compose
Neo4j 5.18+          (vector.similarity.cosine is required)
Ollama               (CONTENT_GAP verdicts only)
```

## Setup

```bash
git clone git@github.com:<org>/linking-engine.git
cd linking-engine
uv sync --all-extras
cp .env.example .env          # add VOYAGE_API_KEY, GSC creds
docker compose up -d          # neo4j, mongo, valkey, minio, mlflow
uv run alembic-neo4j upgrade  # constraints + vector indexes
uv run pytest
```

## Running a pipeline locally

```bash
uv run prefect server start                    # UI on :4200
uv run python -m linking_engine.pipeline.run \
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

MLflow UI runs at `:5000` from Docker Compose. Training scripts log to it by default:

```bash
uv run python -m linking_engine.ml.train_ranker --tenant demo
mlflow ui   # compare NDCG@10 across runs
```

Promotion to `Production` is blocked locally. Only CI can transition registry stages.
