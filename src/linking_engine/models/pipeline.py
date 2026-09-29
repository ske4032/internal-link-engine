"""One tenant pipeline run: the outcome of each stage it planned and of the run as a whole."""

from datetime import datetime
from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StageStatus(StrEnum):
    OK = "ok"
    FAILED = "failed"
    # A stage it needs failed or was skipped.
    SKIPPED = "skipped"
    # Never started because another stage failed; nothing it needs did.
    NOT_RUN = "not_run"


class StageResult(BaseModel):
    """One planned stage; only a stage that started has its measures."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    stage: str = Field(min_length=1)
    status: StageStatus
    seconds: float | None = Field(default=None, ge=0)
    # The process's peak resident memory when the stage ended; concurrent stages share it.
    peak_mb: float | None = Field(default=None, ge=0)
    mlflow_run_id: str | None = None
    # The exception type of a failed stage, never its message.
    error: str | None = None

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        started = self.status in {StageStatus.OK, StageStatus.FAILED}
        if started != (self.seconds is not None and self.peak_mb is not None):
            raise ValueError("a stage has seconds and peak memory exactly when it started")
        if (self.status is StageStatus.FAILED) != (self.error is not None):
            raise ValueError("a stage has an error exactly when it failed")
        if self.mlflow_run_id is not None and self.status is not StageStatus.OK:
            raise ValueError("only a stage that finished has an MLflow run")
        return self


class PipelineReport(BaseModel):
    """One tenant pipeline run, its stages in the order they were planned."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    tenant_id: str = Field(min_length=1)
    # The Prefect root flow run, also tagged on every stage run it started.
    pipeline_run_id: str | None = None
    retrain: bool
    reports: bool
    from_stage: str | None = None
    started_at: datetime
    seconds: float = Field(ge=0)
    stages: tuple[StageResult, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique(self) -> Self:
        if len({result.stage for result in self.stages}) != len(self.stages):
            raise ValueError("each stage appears once")
        return self

    @property
    def failed(self) -> tuple[str, ...]:
        return tuple(r.stage for r in self.stages if r.status is StageStatus.FAILED)
