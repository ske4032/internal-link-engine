"""Fixtures shared by the whole suite.

There is deliberately no `sys.path` manipulation here. `uv sync` installs
`linking_engine` into the environment, so a failure to import it is a real packaging
failure and must not be papered over by the test harness.
"""

from __future__ import annotations

import os
import uuid
from typing import TYPE_CHECKING

import pytest

from linking_engine.graph.repo import GraphRepo
from linking_engine.ingest.mongo_repo import MongoRepo

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator
    from pathlib import Path

# Containers are stopped by the fixtures; Ryuk would only add an image pull.
os.environ.setdefault("TESTCONTAINERS_RYUK_DISABLED", "true")

NEO4J_IMAGE = "neo4j:5.26"
MONGO_IMAGE = "mongo:8.0"
NEO4J_PASSWORD = "test-password"

# Fallback list for the window in which models/tenant.py does not exist yet: the suite
# still has to isolate the environment for anything that does read it. Kept in sync with
# the tenant config block in wiki/Data-Model.md.
_TENANT_FIELD_NAMES = (
    "tenant_id",
    "embedding_provider",
    "embedding_model",
    "embedding_dimensions",
    "embedding_local_fallback_model",
    "content_gap_model",
    "content_gap_min_priority",
    "content_gap_min_opportunity",
    "anchor_type_profile",
    "audit_enabled",
    "discovery_enabled",
    "embedding_strategy",
    "site_languages",
    "client_tier",
    "diversity_cap_pct",
    "max_recommendations_per_source",
    "max_content_gaps_per_source",
)

# Prefixes a pydantic-settings model might plausibly be configured with.
_ENV_PREFIXES = ("", "tenant_", "linking_", "linking_engine_")


def _settings_field_names() -> frozenset[str]:
    """Every field name `TenantConfig` could pick up from the environment."""
    names = set(_TENANT_FIELD_NAMES)
    try:
        from linking_engine.models.tenant import TenantConfig
    except ImportError:
        return frozenset(names)
    names.update(TenantConfig.model_fields)
    return frozenset(names)


@pytest.fixture(autouse=True)
def isolated_settings_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Keep pydantic-settings away from the developer's own environment.

    `TenantConfig` reads the process environment and, when configured with one, a
    `.env` file resolved relative to the working directory. A defaults test that
    silently picks up either is a test of the machine it runs on. This drops every
    variable that could feed a settings field and runs each test from an empty
    directory so the repo's own `.env` is out of reach. The file is never read.
    """
    field_names = _settings_field_names()
    for var in list(os.environ):
        lowered = var.lower()
        for prefix in _ENV_PREFIXES:
            if lowered.startswith(prefix) and lowered.removeprefix(prefix) in field_names:
                monkeypatch.delenv(var, raising=False)
                break
    monkeypatch.chdir(tmp_path)


def _docker_available() -> bool:
    try:
        import docker
        from docker.errors import DockerException

        docker.from_env().ping()
    except (ImportError, DockerException, OSError):
        return False
    return True


@pytest.fixture(scope="session")
def neo4j_server() -> Iterator[tuple[str, str, str]]:
    if not _docker_available():
        pytest.skip("Docker is not available")
    from testcontainers.community.neo4j import Neo4jContainer

    with Neo4jContainer(NEO4J_IMAGE, password=NEO4J_PASSWORD) as container:
        yield container.get_connection_url(), "neo4j", NEO4J_PASSWORD


@pytest.fixture(scope="session")
def mongo_uri() -> Iterator[str]:
    if not _docker_available():
        pytest.skip("Docker is not available")
    from testcontainers.community.mongodb import MongoDbContainer

    with MongoDbContainer(MONGO_IMAGE) as container:
        yield container.get_connection_url()


@pytest.fixture
async def graph(neo4j_server: tuple[str, str, str]) -> AsyncIterator[GraphRepo]:
    repo = await GraphRepo.connect(*neo4j_server)
    await repo.migrate()
    yield repo
    await repo.close()


@pytest.fixture
async def mongo(mongo_uri: str) -> AsyncIterator[MongoRepo]:
    repo = await MongoRepo.connect(mongo_uri, "linking_engine_test")
    await repo.ensure_indexes()
    yield repo
    await repo.close()


@pytest.fixture
def tenant() -> str:
    return f"test-{uuid.uuid4().hex[:12]}"
