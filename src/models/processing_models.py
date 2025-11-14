# Copyright 2025 by Enverus. All rights reserved.

from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Literal
from pydantic import BaseModel, Field, field_validator, ConfigDict
from enum import Enum
from lib.models.mime_type import IngestionMimeType
from src.models.status_models import IngestionStepStatus

# Re-export for type checking
__all__ = ["IngestionStepStatus"]


class RequestSourceEnum(str, Enum):
    """Source enumeration for processing requests."""

    REST_API = "rest_api"
    KAFKA = "kafka"


class FileUploadRecord(BaseModel):
    """Model for file_upload table records in Postgres."""

    upload_id: str = Field(..., description="Unique identifier for the upload")
    file_name: str = Field(..., description="Name of the uploaded file")
    status: str = Field(default="uploaded", description="Upload status")
    bucket: str = Field(..., description="S3 bucket where file is stored")
    file_size: Optional[int] = Field(default=None, description="File size in bytes")
    mime_type: str = Field(default="application/pdf", description="MIME type of the file")
    product_id: str = Field(..., description="Product identifier")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Creation timestamp",
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Update timestamp",
    )

    model_config = ConfigDict(json_encoders={datetime: lambda dt: dt.isoformat()})


class ProcessingStatus(BaseModel):
    """Model for tracking document processing status with automatic datetime handling."""

    request_id: str
    status: IngestionStepStatus
    message: str
    document_urls: List[str] = Field(default_factory=list)
    extraction_s3_path: Optional[str] = None
    source: RequestSourceEnum = RequestSourceEnum.REST_API
    kafka_topic: Optional[str] = None
    kafka_partition: Optional[int] = None
    kafka_offset: Optional[int] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_config = ConfigDict(json_encoders={datetime: lambda dt: dt.isoformat()})

    def update_status(
        self,
        status: IngestionStepStatus,
        message: str = "",
        extraction_s3_path: Optional[str] = None,
    ) -> None:
        """Update status with automatic timestamp management."""
        self.status = status
        self.message = message
        self.updated_at = datetime.now(timezone.utc)
        if extraction_s3_path:
            self.extraction_s3_path = extraction_s3_path


# StatusCallback has been removed - now using centralized StatusTracker


class KafkaMessage(BaseModel):
    """Base model for Kafka messages with consistent structure."""

    request_id: str
    event_type: str = Field(default="", description="Event type for the message")
    metadata: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_config = ConfigDict(json_encoders={datetime: lambda dt: dt.isoformat()})


class DocumentsProcessingStartedMessage(KafkaMessage):
    """Kafka message for when document processing starts."""

    document_urls: List[str]
    document_count: Optional[int] = None

    def model_post_init(self, __context: Any) -> None:
        """Auto-calculate document_count if not provided and set event_type."""
        if self.document_count is None:
            object.__setattr__(self, "document_count", len(self.document_urls))
        # Set the specific event type for this message type
        object.__setattr__(self, "event_type", "documents_processing_started")


class DocumentsExtractionCompletedMessage(KafkaMessage):
    """Kafka message for when document extraction completes."""

    document_urls: List[str]
    document_count: Optional[int] = None
    extraction_s3_path: str
    extraction_result: Dict[str, Any]

    def model_post_init(self, __context: Any) -> None:
        """Auto-calculate document_count if not provided and set event_type."""
        if self.document_count is None:
            object.__setattr__(self, "document_count", len(self.document_urls))
        # Set the specific event type for this message type
        object.__setattr__(self, "event_type", "documents_extraction_completed")


class ExtractionError(BaseModel):
    """Model for extraction API errors."""

    status_code: int
    error_message: str
    response_body: Optional[str] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_config = ConfigDict(json_encoders={datetime: lambda dt: dt.isoformat()})


class ExtractionResult(BaseModel):
    """Model for extraction API results - success or failure."""

    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[ExtractionError] = None
    job_id: Optional[str] = None  # Job ID for polling status

    def model_post_init(self, __context: Any) -> None:
        """Ensure exactly one of data or error is set."""
        if self.success and self.error is not None:
            raise ValueError("Cannot have error when success=True")
        if not self.success and self.data is not None:
            raise ValueError("Cannot have data when success=False")


class JobStatusRequest(BaseModel):
    """Model for job status request."""

    job_id: str = Field(..., description="Job ID to check status for")


class JobStatusResponse(BaseModel):
    """Model for job status response from extraction API."""

    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None

    def get_job_status(self) -> Optional[str]:
        """Extract job status from response data."""
        if self.data and "job" in self.data:
            status = self.data["job"].get("status")
            return str(status) if status is not None else None
        return None

    def is_completed(self) -> bool:
        """Check if job is completed."""
        status = self.get_job_status()
        return status == "COMPLETED"

    def is_in_progress(self) -> bool:
        """Check if job is still in progress."""
        status = self.get_job_status()
        return status in ["PENDING", "IN_PROGRESS"]

    def has_failed(self) -> bool:
        """Check if job has failed."""
        return not self.success or (self.get_job_status() not in ["PENDING", "IN_PROGRESS", "COMPLETED"])

    def get_error_message(self) -> str:
        """Get error message from response."""
        if self.data and "job" in self.data:
            error_msg = self.data["job"].get("error_message")
            if error_msg:
                return str(error_msg)
        return self.message or "Unknown error"


class ProcessingResult(BaseModel):
    """Model for processing pipeline results."""

    request_id: str
    status: IngestionStepStatus
    document_urls: List[str]
    document_count: Optional[int] = None
    extraction_s3_path: Optional[str] = None
    extraction_result: Optional[ExtractionResult] = None
    error: Optional[ExtractionError] = None

    def model_post_init(self, __context: Any) -> None:
        """Auto-calculate document_count if not provided."""
        if self.document_count is None:
            object.__setattr__(self, "document_count", len(self.document_urls))


class IncomingDocumentProcessingRequest(BaseModel):
    """Model for incoming Kafka messages requesting document processing."""

    request_id: str = Field(..., description="Request identifier for tracking through pipeline")
    uuid: str = Field(..., description="Unique identifier for this processing request")
    project_id: str = Field(..., description="Project identifier")
    upload_id: str = Field(..., description="Upload identifier")
    location: str = Field(..., description="S3 location of the document to process")
    file_type: str = Field(..., description="Type of file to process")

    model_config = ConfigDict(json_encoders={datetime: lambda dt: dt.isoformat()})

    def get_extraction_s3_base_path(self) -> str:
        """Get the base S3 path where extracted content will be stored."""
        # Extract bucket from location (s3://bucket/path -> bucket)
        bucket = self.location.replace("s3://", "").split("/")[0]
        return f"s3://{bucket}/{self.project_id}/{self.uuid}/"


class DocumentRequest(BaseModel):
    document_urls: List[str] = Field(..., min_length=1, description="List of s3:// URLs to process")
    metadata: Optional[Dict[str, Any]] = None

    @field_validator("document_urls")
    @classmethod
    def validate_s3_urls(cls, values: List[str]) -> List[str]:
        if not isinstance(values, list) or len(values) == 0:
            raise ValueError("document_urls must be a non-empty list")

        for url in values:
            if not isinstance(url, str) or not url.startswith("s3://"):
                raise ValueError("All document_urls must be valid s3:// URLs")
        return values
