"""The read-only output API over each tenant's latest complete run.

    uv run --env-file .env uvicorn --factory linking_engine.api.app:create_app

Every route but ``/health`` needs an ``X-API-Key`` issued with ``scripts/api_keys.py``. The API
only reads: the recommendations stage writes the output and creates its indexes.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Final

import structlog
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict
from pymongo import AsyncMongoClient
from pymongo.errors import ConfigurationError

from linking_engine.api.routes import router, service
from linking_engine.errors import DatabaseError, DatabaseUnavailableError
from linking_engine.output.keys import KeyStore
from linking_engine.output.reader import OutputReader

if TYPE_CHECKING:
    from linking_engine.output.collections import Document

log = structlog.get_logger(__name__)

_SERVER_SELECTION_MS: Final = 5000


class ApiSettings(BaseSettings):
    """Where the output lives, from the ``MONGO_URI`` and ``MONGO_DB`` process env."""

    model_config = SettingsConfigDict(frozen=True, extra="forbid")

    mongo_uri: SecretStr
    mongo_db: str = Field(min_length=1)


def create_app(settings: ApiSettings | None = None) -> FastAPI:
    config = settings or ApiSettings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        try:
            client: AsyncMongoClient[Document] = AsyncMongoClient(
                config.mongo_uri.get_secret_value(),
                serverSelectionTimeoutMS=_SERVER_SELECTION_MS,
                tz_aware=True,
            )
        except (ConfigurationError, ValueError) as error:
            raise DatabaseUnavailableError(
                "mongodb", f"invalid connection settings: {error}"
            ) from error
        database = client[config.mongo_db]
        app.state.reader = OutputReader(database)
        app.state.keys = KeyStore(database)
        try:
            yield
        finally:
            await client.close()

    app = FastAPI(title="Internal Linking Engine API", version="1", lifespan=lifespan)
    app.include_router(service)
    app.include_router(router)
    app.add_exception_handler(DatabaseError, _store_unavailable)
    app.add_exception_handler(Exception, _internal_error)
    return app


async def _store_unavailable(request: Request, error: Exception) -> JSONResponse:
    log.warning("api.store_unavailable", path=request.url.path, error=type(error).__name__)
    return JSONResponse(
        {"detail": "store unavailable"}, status_code=status.HTTP_503_SERVICE_UNAVAILABLE
    )


async def _internal_error(request: Request, error: Exception) -> JSONResponse:
    # The server logs the traceback; this only keeps the body's shape.
    log.error("api.internal_error", path=request.url.path, error=type(error).__name__)
    return JSONResponse(
        {"detail": "internal error"}, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
    )
