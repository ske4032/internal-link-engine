"""Persistence and embedding exceptions. SDK errors are translated into these; the original is __cause__."""

from typing import Literal

Store = Literal["neo4j", "mongodb"]


class DatabaseError(Exception):
    """Base persistence error."""

    def __init__(self, store: Store, message: str) -> None:
        super().__init__(f"{store}: {message}")
        self.store = store


class DatabaseUnavailableError(DatabaseError):
    """Server unreachable or stopped answering."""


class DatabaseAuthError(DatabaseUnavailableError):
    """Credentials rejected."""


class DatabaseReadError(DatabaseError):
    """Read rejected, or stored data does not fit the model."""


class DatabaseWriteError(DatabaseError):
    """Write rejected, or fewer rows written than given."""


class SchemaError(DatabaseError):
    """Migration failed, wrong vector index dimension, or server too old."""


class ServiceError(Exception):
    """Prefect or MLflow is unreachable or incompatible."""


class EmbeddingError(Exception):
    """Base embedding error; status_code and error_type describe the provider failure."""

    def __init__(
        self, message: str, *, status_code: int | None = None, error_type: str | None = None
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.error_type = error_type or type(self).__name__


class EmbeddingUnavailableError(EmbeddingError):
    """Rate limited, server error, timeout or connection failure after retries."""


class EmbeddingRequestError(EmbeddingError):
    """Request rejected (bad input or credentials); never retried."""


class EmbeddingAuthError(EmbeddingRequestError):
    """Credentials rejected (HTTP 401 or 403)."""


class EmbeddingResponseError(EmbeddingError):
    """Wrong vector count or dimension, zero vector, not unit norm, or bad token count."""


class EmbeddingModelMismatchError(EmbeddingError):
    """Stored vectors mix models, lack a model, or differ from the configured model."""
