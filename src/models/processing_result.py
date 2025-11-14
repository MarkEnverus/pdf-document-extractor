"""
Processing result models for per-page document storage.
"""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field


class PageResult(BaseModel):
    """Result for a single page from a processed document."""

    page_number: int = Field(..., description="Page number (0-indexed)")
    s3_path: str = Field(..., description="S3 path to the stored page results JSON")
    content: Optional[str] = Field(None, description="Page content")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Page metadata")
    figure_count: int = Field(0, description="Number of figures on this page")
    table_count: int = Field(0, description="Number of tables on this page")


class DocumentProcessingResult(BaseModel):
    """Complete processing result for a document with per-page storage."""

    document_url: str = Field(..., description="Source document URL")
    success: bool = Field(..., description="Overall success status")
    total_pages: int = Field(..., description="Total number of pages processed")
    page_results: List[PageResult] = Field(default_factory=list, description="Per-page results")
    error_message: Optional[str] = Field(None, description="Error message if processing failed")
    processing_time_seconds: Optional[float] = Field(None, description="Total processing time")
