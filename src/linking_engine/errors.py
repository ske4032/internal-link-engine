"""Persistence exceptions. Driver errors are translated into these; the original is __cause__."""

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
