"""
Status update models for external API integration.
"""

from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from uuid import UUID
from lib.models.ingestion_status import (
    IngestionStatus,
    IngestionStepStatus,
)
from pydantic import BaseModel, Field, ConfigDict, field_validator

# Re-export for type checking
__all__ = ["IngestionStepFileStatus", "KafkaCompletionMessage", "IngestionStepStatus"]


class IngestionStepFileStatus(BaseModel):
    """
    Model for sending status updates to external API.

    This model represents the structure expected by the external status API
    for tracking file processing status during ingestion.
    """

    project_id: UUID
    file_upload_id: UUID
    file_name: Optional[str] = Field(None, description="Name of the file being processed")
    step: IngestionStatus = Field(default=IngestionStatus.EXTRACTION, description="Current ingestion step")
    status: IngestionStepStatus = Field(description="Status of the current step")
    summary: Optional[Dict[str, Any]] = Field(None, description="Summary information about processing")
    errors: Optional[List[Dict[str, Any]]] = Field(None, description="List of errors encountered during processing")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Timestamp of last update")

    model_config = ConfigDict(json_encoders={datetime: lambda v: v.isoformat(), UUID: str})


class KafkaCompletionMessage(BaseModel):
    """
    Model for Kafka completion messages sent on successful document processing.

    This model ensures consistent message format across all SUCCESS notifications.
    """

    uuid: UUID = Field(description="Unique document identifier (same as upload_id)")
    project_id: UUID = Field(description="Project identifier")
    upload_id: UUID = Field(description="Upload identifier (same as uuid)")
    location: str = Field(description="S3 location of extracted results")
    file_type: str = Field(description="MIME type of processed file")
    status: str = Field(default="Success", description="Processing status")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Completion timestamp")

    @field_validator("location")
    @classmethod
    def validate_s3_location(cls, v: str) -> str:
        """Validate that location is a valid S3 path when not empty."""
        if v and not v.startswith("s3://"):
            raise ValueError("location must be a valid S3 path starting with 's3://'")
        return v

    model_config = ConfigDict(json_encoders={datetime: lambda v: v.isoformat(), UUID: str})
