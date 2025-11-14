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


class IngestionStatus(str, Enum):
    """Ingestion pipeline steps/stages."""

    INGESTION = "INGESTION"
    CLASSIFICATION = "CLASSIFICATION"
    EXTRACTION = "EXTRACTION"
    TRANSFORMATION = "TRANSFORMATION"
    KB_SYNC = "KB_SYNC"


class IngestionStepFileStatus(BaseModel):
    """Status of a file in an ingestion step."""

    id: UUID
    file_id: UUID
    step_name: str
    status: IngestionStepStatus
    error_message: str | None = None


class IngestionStatusModel(BaseModel):
    """Overall ingestion status model."""

    id: UUID
    status: IngestionStepStatus
    file_statuses: list[IngestionStepFileStatus] = []
