# Copyright 2025 by Enverus. All rights reserved.

from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any
from uuid import UUID

from pydantic import BaseModel

"""Those models are used to define the structure of the extraction-service job response"""


class JobDetailStatusEnum(str, Enum):
    PENDING = "PENDING"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class JobDetailExtractedData(BaseModel):
    status: str
    structured_output: Optional[Dict[str, Any]] = None


class JobDetails(BaseModel):
    job_details_id: UUID

    job_id: UUID

    filename: Optional[str] = None

    file_type: Optional[str] = None

    file_path: Optional[str] = None

    file_size: Optional[str] = None

    number_of_pages: Optional[int] = None

    extracted_data: Optional[JobDetailExtractedData] = None

    error_message: Optional[str] = None

    job_detail_status: JobDetailStatusEnum

    created_at: Optional[datetime] = None

    updated_at: Optional[datetime] = None

    duration: Optional[int] = None

    confidence_score: Optional[float] = None


class JobStatusEnum(str, Enum):
    """Status enumeration for extraction jobs."""

    PENDING = "PENDING"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    PARTIALLY_COMPLETED = "PARTIALLY_COMPLETED"


class ExtractionModeEnum(str, Enum):
    """Extraction mode enumeration for processing strategy."""

    SINGLE = "single"
    MULTI = "multi"


class Job(BaseModel):
    job_id: UUID

    status: JobStatusEnum

    created_at: Optional[datetime] = None

    updated_at: Optional[datetime] = None

    extraction_mode: Optional[ExtractionModeEnum] = None

    extraction_configs: Optional[Dict[str, Any]] = None

    output_schema: Optional[Dict[str, Any]] = None

    duration: Optional[int] = None

    user_id: Optional[str] = None

    org_id: Optional[str] = None

    error_message: Optional[str] = None


class JobResponse(BaseModel):
    job: Optional[Job] = None
    job_details: Optional[list[JobDetails]] = None


class ExtractionResponse(BaseModel):
    success: bool
    message: str
    data: JobResponse
