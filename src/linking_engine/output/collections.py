"""Collections, indexes and document shape of the served output, shared by its writer and
its reader.

Every document carries ``tenantId``; run-scoped documents also carry ``runId`` and an
``ordinal``, their position in the listing's stable order, which the API pages on.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Final

from pydantic import BaseModel, ValidationError
from pymongo import ASCENDING, DESCENDING, IndexModel

from linking_engine.errors import DatabaseReadError

if TYPE_CHECKING:
    from collections.abc import Mapping

    from linking_engine.models import ActionType

Document = dict[str, object]

RECOMMENDATIONS: Final = "recommendations"
PAGES: Final = "output_pages"
HUBS: Final = "output_hubs"
BRIDGES: Final = "output_bridges"
DUPLICATES: Final = "output_duplicates"
UNANCHORED: Final = "output_unanchored"
TARGET_FIXES: Final = "output_target_fixes"
ORPHANS: Final = "output_orphans"
RUNS: Final = "output_runs"
API_KEYS: Final = "api_keys"
# Written by prepare-corpus, not by a run; served as it is.
EXCLUDED_PAGES: Final = "excluded_pages"

RUN_SCOPED: Final = (
    RECOMMENDATIONS,
    PAGES,
    HUBS,
    BRIDGES,
    DUPLICATES,
    UNANCHORED,
    TARGET_FIXES,
    ORPHANS,
)
STAMPS: Final = frozenset({"_id", "tenantId", "runId", "ordinal"})


def _run(*keys: str) -> list[tuple[str, int]]:
    return [("tenantId", ASCENDING), ("runId", ASCENDING), *((key, ASCENDING) for key in keys)]


INDEXES: Final[dict[str, tuple[IndexModel, ...]]] = {
    **{
        name: (IndexModel(_run("ordinal"), unique=True, name="tenant_run_ordinal"),)
        for name in RUN_SCOPED
    },
    RECOMMENDATIONS: (
        IndexModel(_run("ordinal"), unique=True, name="tenant_run_ordinal"),
        IndexModel(_run("id"), unique=True, name="tenant_run_id"),
        IndexModel(_run("source_url", "ordinal"), name="tenant_run_source"),
        IndexModel(_run("target_url", "ordinal"), name="tenant_run_target"),
        IndexModel(_run("action_type", "ordinal"), name="tenant_run_action"),
        IndexModel(_run("best_rank"), name="tenant_run_best"),
    ),
    PAGES: (
        IndexModel(_run("ordinal"), unique=True, name="tenant_run_ordinal"),
        IndexModel(_run("url"), unique=True, name="tenant_run_url"),
        IndexModel(_run("hub_id", "ordinal"), name="tenant_run_hub"),
        IndexModel(_run("is_orphan", "ordinal"), name="tenant_run_orphan"),
    ),
    UNANCHORED: (
        IndexModel(_run("ordinal"), unique=True, name="tenant_run_ordinal"),
        IndexModel(_run("reason", "ordinal"), name="tenant_run_reason"),
    ),
    RUNS: (
        IndexModel([("tenantId", ASCENDING), ("runId", ASCENDING)], unique=True, name="tenant_run"),
        IndexModel(
            [("tenantId", ASCENDING), ("status", ASCENDING), ("completed_at", DESCENDING)],
            name="tenant_status_completed",
        ),
    ),
    API_KEYS: (
        IndexModel([("keyHash", ASCENDING)], unique=True, name="key_hash"),
        IndexModel([("tenantId", ASCENDING), ("keyId", ASCENDING)], unique=True, name="tenant_key"),
    ),
}


def recommendation_id(
    tenant_id: str, action_type: ActionType, source_url: str, target_url: str, position: int | None
) -> str:
    """The same for the same tenant, action and link in every run."""
    parts = (
        tenant_id,
        str(action_type),
        source_url,
        target_url,
        "" if position is None else str(position),
    )
    return hashlib.sha256("\x1f".join(parts).encode()).hexdigest()[:16]


def to_document(
    model: BaseModel, *, tenant_id: str, run_id: str | None, ordinal: int | None
) -> Document:
    """The model's JSON form, snake_case, with the stamps; run stamps only when given."""
    document: Document = {**model.model_dump(mode="json"), "tenantId": tenant_id}
    if run_id is not None:
        document["runId"] = run_id
    if ordinal is not None:
        document["ordinal"] = ordinal
    return document


def from_document[M: BaseModel](model: type[M], document: Mapping[str, object]) -> M:
    """The model back from a document, stamps dropped."""
    try:
        return model.model_validate({k: v for k, v in document.items() if k not in STAMPS})
    except ValidationError as error:
        raise DatabaseReadError(
            "mongodb", f"output document does not fit {model.__name__}: {error}"
        ) from error
