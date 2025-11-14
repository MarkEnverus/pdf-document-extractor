# Copyright 2025 by Enverus. All rights reserved.

"""
Docling processing models and configuration.
"""

from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field, ConfigDict
from enum import Enum

from src.models.extraction_models import ExtractedFigure, ExtractedTable
from src.models.processing_models import DocumentRequest


class DoclingProviderEnum(str, Enum):
    """Docling processing provider options."""

    LOCAL = "local"


class DoclingOutputFormatEnum(str, Enum):
    """Output format options for Docling processing."""

    MARKDOWN = "markdown"
    HTML = "html"
    TEXT = "text"
    JSON = "json"


class DoclingProcessRequest(DocumentRequest):
    provider: DoclingProviderEnum = Field(DoclingProviderEnum.LOCAL, description="Docling processing provider")
    output_format: DoclingOutputFormatEnum = Field(DoclingOutputFormatEnum.MARKDOWN, description="Output format")

    # Local processing options
    enable_ocr: bool = Field(True, description="Enable OCR for local processing")
    enable_table_structure: bool = Field(True, description="Enable table structure detection")
    enable_figure_extraction: bool = Field(True, description="Enable figure extraction")


class DoclingConfig(BaseModel):
    """Configuration for Docling document processing."""

    provider: DoclingProviderEnum = DoclingProviderEnum.LOCAL
    output_format: DoclingOutputFormatEnum = DoclingOutputFormatEnum.MARKDOWN

    # Local processing options
    enable_ocr: bool = True
    enable_table_structure: bool = True
    enable_figure_extraction: bool = True

    # Processing options
    chunk_size: Optional[int] = Field(None, description="Chunk size for large documents")
    overlap_size: int = Field(100, description="Overlap size between chunks")

    # Page extraction options
    extract_pages: str = Field("all", description="Page extraction mode: 'all', 'single', or specific page ranges")
    single_page_mode: bool = Field(False, description="Extract each page as separate document")


class DoclingRequest(BaseModel):
    """Request model for Docling document processing."""

    request_id: str
    project_id: str
    upload_id: str
    document_urls: List[str] = Field(..., min_length=1)
    config: DoclingConfig = Field(default_factory=lambda: DoclingConfig.model_validate({}))
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class DoclingDocumentResult(BaseModel):
    """Result for a single document processed by Docling."""

    document_url: str
    success: bool
    content: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    bounding_box: Optional[List[Dict[str, Any]]] = None
    error_message: Optional[str] = None
    processing_time_seconds: Optional[float] = None
    page_count: Optional[int] = None
    word_count: Optional[int] = None
    extracted_figures: Optional[List[ExtractedFigure]] = None
    extracted_tables: Optional[List[ExtractedTable]] = None
    doc: Optional[Any] = Field(
        default=None, description="Actual Docling document object for asset extraction", exclude=True
    )


class DoclingBatchResult(BaseModel):
    """Result for batch Docling processing."""

    request_id: str
    success: bool
    total_documents: int
    successful_documents: int
    failed_documents: int
    results: List[DoclingDocumentResult]
    total_processing_time_seconds: float
    config_used: DoclingConfig
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_config = ConfigDict(json_encoders={datetime: lambda dt: dt.isoformat()})


class DoclingError(BaseModel):
    """Error information for Docling processing failures."""

    error_type: str
    error_message: str
    document_url: Optional[str] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_config = ConfigDict(json_encoders={datetime: lambda dt: dt.isoformat()})


class ProcessingResult(BaseModel):
    """
    Generic result model for document processing operations.

    This model provides a consistent structure for results from any processing
    strategy (Docling, API extraction, etc.) used by DocumentProcessingService.
    """

    request_id: str = Field(..., description="Unique identifier for the processing request")
    strategy: str = Field(..., description="Processing strategy used ('docling', 'api', etc.)")
    success: bool = Field(..., description="Whether the processing completed successfully")

    # Document processing metrics
    total_documents: int = Field(0, description="Total number of documents processed")
    successful_documents: int = Field(0, description="Number of successfully processed documents")
    failed_documents: int = Field(0, description="Number of documents that failed processing")

    # Processing metadata
    project_id: Optional[str] = Field(None, description="Project identifier")
    upload_id: Optional[str] = Field(None, description="Upload identifier")
    processing_time_seconds: Optional[float] = Field(None, description="Total processing time in seconds")

    # Results and outputs
    s3_locations: List[str] = Field(default_factory=list, description="S3 paths to processing results")
    extraction_results: Optional[Dict[str, Any]] = Field(None, description="Raw extraction results")
    docling_results: Optional[DoclingBatchResult] = Field(None, description="Docling-specific batch results")

    # Error information
    error_message: Optional[str] = Field(None, description="Error message if processing failed")
    error_type: Optional[str] = Field(None, description="Type of error if processing failed")

    # Additional metadata
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional processing metadata")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_config = ConfigDict(json_encoders={datetime: lambda dt: dt.isoformat()})

    @property
    def failure_rate(self) -> float:
        """Calculate the failure rate as a percentage."""
        if self.total_documents == 0:
            return 0.0
        return (self.failed_documents / self.total_documents) * 100

    @property
    def success_rate(self) -> float:
        """Calculate the success rate as a percentage."""
        if self.total_documents == 0:
            return 0.0
        return (self.successful_documents / self.total_documents) * 100
