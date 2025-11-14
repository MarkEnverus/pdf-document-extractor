"""Stub models for ingestion status tracking."""

from enum import Enum
from uuid import UUID
from pydantic import BaseModel


class IngestionStepStatus(str, Enum):
    """Status of an ingestion step."""

    PENDING = "PENDING"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class IngestionStepFileStatus(BaseModel):
    """Status of a file in an ingestion step."""

    id: UUID
    file_id: UUID
    step_name: str
    status: IngestionStepStatus
    error_message: str | None = None


class IngestionStatus(BaseModel):
    """Overall ingestion status."""

    id: UUID
    status: IngestionStepStatus
    file_statuses: list[IngestionStepFileStatus] = []
