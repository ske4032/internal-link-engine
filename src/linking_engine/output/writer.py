"""Writes one run of a tenant's served output: every document under a new run id while its run
is ``"writing"``, then the run marked complete and every other run of the tenant deleted.
Readers serve the latest complete run only, so a run that fails midway leaves the previous
one served; nothing is retried here.

A run's ``started_at`` and ``completed_at`` are stored as BSON dates, to the millisecond Mongo
keeps, so the latest complete run sorts exactly; every other field is the model's JSON form."""

from __future__ import annotations

from itertools import batched
from typing import TYPE_CHECKING, Final, Self

from pymongo import AsyncMongoClient
from pymongo.errors import ConfigurationError, ConnectionFailure, OperationFailure, PyMongoError

from linking_engine.errors import (
    DatabaseAuthError,
    DatabaseError,
    DatabaseReadError,
    DatabaseUnavailableError,
    DatabaseWriteError,
    SchemaError,
)
from linking_engine.models import RunInfo
from linking_engine.output.collections import (
    INDEXES,
    RUN_SCOPED,
    RUNS,
    Document,
    from_document,
    to_document,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Iterable
    from datetime import datetime
    from types import TracebackType

    from pydantic import BaseModel

    from linking_engine.models import SiteSummary

WRITE_BATCH: Final = 1000
_AUTH_CODES: Final = frozenset({13, 18})  # Unauthorized, AuthenticationFailed


class OutputWriter:
    """Tenant-scoped writes of the served output; every filter carries the tenant."""

    def __init__(self, client: AsyncMongoClient[Document], database: str) -> None:
        self._client = client
        self._db = client[database]

    @classmethod
    async def connect(cls, uri: str, database: str, *, timeout_ms: int = 5000) -> Self:
        try:
            client: AsyncMongoClient[Document] = AsyncMongoClient(
                uri, serverSelectionTimeoutMS=timeout_ms, tz_aware=True
            )
        except (ConfigurationError, ValueError) as error:
            raise DatabaseUnavailableError(
                "mongodb", f"invalid connection settings: {error}"
            ) from error
        try:
            await _call(client[database].command("ping"), write=False, what="connect")
        except DatabaseError:
            await client.close()
            raise
        return cls(client, database)

    async def close(self) -> None:
        await self._client.close()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.close()

    async def ensure_indexes(self) -> None:
        for name in (*RUN_SCOPED, RUNS):
            try:
                await _call(
                    self._db[name].create_indexes(list(INDEXES[name])),
                    write=True,
                    what=f"indexes on {name}",
                )
            except DatabaseWriteError as error:
                raise SchemaError("mongodb", f"cannot create indexes on {name}") from error

    async def begin(self, run: RunInfo) -> None:
        """Record a new run of the tenant as writing."""
        _require_tenant(run.tenant_id)
        if run.status != "writing":
            raise ValueError("a run begins as writing")
        await _call(
            self._db[RUNS].insert_one(_run_document(run)),
            write=True,
            what="begin output run",
        )

    async def write(
        self, collection: str, tenant_id: str, run_id: str, models: Iterable[BaseModel]
    ) -> int:
        """Insert ``models`` in their listing order, the ordinal counting from 0; returns the
        documents written."""
        _require_tenant(tenant_id)
        if collection not in RUN_SCOPED:
            raise ValueError(f"{collection!r} is not a run-scoped output collection")
        await self._writing(tenant_id, run_id)
        written = 0
        for chunk in batched(models, WRITE_BATCH):
            documents = [
                to_document(model, tenant_id=tenant_id, run_id=run_id, ordinal=written + offset)
                for offset, model in enumerate(chunk)
            ]
            result = await _call(
                self._db[collection].insert_many(documents, ordered=True),
                write=True,
                what=f"write {collection}",
            )
            if len(result.inserted_ids) != len(documents):
                raise DatabaseWriteError(
                    "mongodb",
                    f"{len(result.inserted_ids)} of {len(documents)} {collection} documents "
                    f"of run {run_id!r} written",
                )
            written += len(documents)
        return written

    async def complete(
        self, tenant_id: str, run_id: str, summary: SiteSummary, completed_at: datetime
    ) -> RunInfo:
        """Mark the writing run complete with its summary; from now on readers serve it. Returns
        the run as stored."""
        run = await self._writing(tenant_id, run_id)
        done = RunInfo.model_validate(
            {
                **run.model_dump(),
                "status": "complete",
                "completed_at": milliseconds(completed_at),
                "summary": summary,
            }
        )
        result = await _call(
            self._db[RUNS].replace_one(
                {"tenantId": tenant_id, "runId": run_id, "status": "writing"},
                _run_document(done),
            ),
            write=True,
            what="complete output run",
        )
        if result.matched_count != 1:
            raise DatabaseWriteError(
                "mongodb", f"run {run_id!r} of {tenant_id!r} stopped writing before completion"
            )
        return done

    async def prune(self, tenant_id: str, keep: str) -> int:
        """Delete every run of the tenant but ``keep``, which must be complete: its documents
        and its run. Returns the documents deleted."""
        _require_tenant(tenant_id)
        kept = await _call(
            self._db[RUNS].find_one(
                {"tenantId": tenant_id, "runId": keep, "status": "complete"}, {"_id": 1}
            ),
            write=False,
            what="read output run",
        )
        if kept is None:
            raise DatabaseWriteError(
                "mongodb", f"run {keep!r} of {tenant_id!r} is not complete; nothing pruned"
            )
        others: Document = {"tenantId": tenant_id, "runId": {"$ne": keep}}
        deleted = 0
        for name in (*RUN_SCOPED, RUNS):
            result = await _call(
                self._db[name].delete_many(others), write=True, what=f"prune {name}"
            )
            deleted += result.deleted_count
        return deleted

    async def _writing(self, tenant_id: str, run_id: str) -> RunInfo:
        _require_tenant(tenant_id)
        document = await _call(
            self._db[RUNS].find_one({"tenantId": tenant_id, "runId": run_id}),
            write=False,
            what="read output run",
        )
        if document is None:
            raise DatabaseWriteError("mongodb", f"run {run_id!r} of {tenant_id!r} was not begun")
        run = from_document(RunInfo, document)
        if run.status != "writing":
            raise DatabaseWriteError(
                "mongodb", f"run {run_id!r} of {tenant_id!r} is {run.status}, not writing"
            )
        return run


def milliseconds(moment: datetime) -> datetime:
    """The moment as Mongo stores a date."""
    return moment.replace(microsecond=moment.microsecond // 1000 * 1000)


def _run_document(run: RunInfo) -> Document:
    document = to_document(run, tenant_id=run.tenant_id, run_id=run.run_id, ordinal=None)
    document["started_at"] = milliseconds(run.started_at)
    document["completed_at"] = None if run.completed_at is None else milliseconds(run.completed_at)
    return document


async def _call[T](call: Awaitable[T], *, write: bool, what: str) -> T:
    try:
        return await call
    except PyMongoError as error:
        raise _translate(error, write=write, what=what) from error


def _translate(error: PyMongoError, *, write: bool, what: str) -> DatabaseError:
    if isinstance(error, OperationFailure) and error.code in _AUTH_CODES:
        return DatabaseAuthError("mongodb", f"{what}: {error}")
    if isinstance(error, ConnectionFailure):
        return DatabaseUnavailableError("mongodb", f"{what}: server unavailable: {error}")
    kind = DatabaseWriteError if write else DatabaseReadError
    return kind("mongodb", f"{what}: {error}")


def _require_tenant(tenant_id: str) -> None:
    if not tenant_id.strip():
        raise ValueError("tenant_id must be a non-empty string")
