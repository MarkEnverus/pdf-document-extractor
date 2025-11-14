"""Universal pipeline event model for all IDP pipeline stages."""

from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from idp_file_management.models.mime_type import IngestionMimeType
from pydantic import BaseModel, ConfigDict, Field, field_validator


class PipelineEvent(BaseModel):
    """
    Universal event published by IDP services when a pipeline stage completes.

    This message is used throughout the entire pipeline:
    - Ingestion → Extractor
    - Extractor → Transformer
    - Transformer → KB-Sync

    Only published on success (no status field needed).

    TODOs:
    - uuid and upload_id are currently identical - consider consolidating
    """

    uuid: UUID = Field(
        ...,
        description="Unique identifier for the pipeline operation (currently same as upload_id)",
    )
    project_id: UUID = Field(..., description="Project UUID")
    upload_id: UUID = Field(
        ..., description="File upload UUID (currently same as uuid)"
    )
    location: str = Field(
        ..., description="S3 URI to the data folder (s3://bucket/prefix/)"
    )
    file_type: IngestionMimeType = Field(
        default=IngestionMimeType.PDF,
        description="MIME type of the file (e.g., PDF, DOCX, DOC, PPTX, PNG, JPG)",
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat(),
        description="ISO 8601 timestamp of the event",
    )
    request_id: str | None = Field(
        default=None,
        description="Optional request ID from ingestion for correlation/tracing (passed through pipeline)",
    )
    metadata: dict[str, Any] | None = Field(
        default=None,
        description="Optional metadata (workflow name, classification, etc.)",
    )

    @field_validator("location")
    @classmethod
    def validate_location(cls, v: str) -> str:
        """Validate that location is a valid S3 URI."""
        if not v.startswith("s3://"):
            raise ValueError(
                f"location must be an S3 URI starting with s3://, got: {v}"
            )
        return v

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "uuid": "550e8400-e29b-41d4-a716-446655440000",
                "project_id": "123e4567-e89b-12d3-a456-426614174000",
                "upload_id": "550e8400-e29b-41d4-a716-446655440000",
                "location": "s3://genai-proprietary-data-extract-dev/projects/123e4567-e89b-12d3-a456-426614174000/uploads/550e8400-e29b-41d4-a716-446655440000/",
                "file_type": IngestionMimeType.PDF.value,  # Enum serializes to string
                "timestamp": "2025-01-23T10:30:00Z",
                "request_id": "req_abc123",
                "metadata": {
                    "workflow": "default",
                    "classification": "production_report",
                },
            }
        }
    )
