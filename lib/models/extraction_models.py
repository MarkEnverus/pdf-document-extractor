"""Pydantic models for document extraction results.

These models represent the structure of extraction outputs stored in S3:
- PageExtractionResult: Per-page extraction results (results.json)
- FigureMetadata: Figure metadata with LLM analysis (_metadata.json for figures)
- TableMetadata: Table metadata with structure info (_metadata.json for tables)
"""

from datetime import datetime
from enum import Enum
from typing import Any

from idp_file_management.models.mime_type import IngestionMimeType
from pydantic import BaseModel, Field


class ExtractionType(str, Enum):
    """Type of extraction strategy used."""
    DOCLING = "docling"
    API = "api"


class BoundingBox(BaseModel):
    """Bounding box coordinates for document elements."""

    left: float = Field(..., description="Left coordinate")
    top: float = Field(..., description="Top coordinate")
    right: float = Field(..., description="Right coordinate")
    bottom: float = Field(..., description="Bottom coordinate")
    coord_origin: str = Field(
        default="TOP_LEFT",
        description="Coordinate system origin (e.g., TOP_LEFT, BOTTOM_LEFT)",
    )


class ImageSize(BaseModel):
    """Image dimensions."""

    width: int = Field(..., description="Image width in pixels")
    height: int = Field(..., description="Image height in pixels")


class LLMAnalysisResult(BaseModel):
    """LLM analysis result for images."""

    model_used: str = Field(..., description="Model ID used for analysis")
    analysis_summary: str = Field(..., description="Summary of analysis")
    analysis_timestamp: str = Field(..., description="ISO timestamp of analysis")
    analysis_error: str | None = Field(
        None, description="Error message if analysis failed"
    )


class FigureReference(BaseModel):
    """Reference to a figure with metadata (stored in page results)."""

    figure_id: str = Field(..., description="Unique figure identifier")
    s3_image_path: str | None = Field(None, description="S3 path to figure image (PNG)")
    s3_metadata_path: str | None = Field(
        None, description="S3 path to figure metadata JSON"
    )
    page_number: int | None = Field(
        None, description="Page number where figure appears"
    )


class TableReference(BaseModel):
    """Reference to a table with metadata (stored in page results)."""

    table_id: str = Field(..., description="Unique table identifier")
    s3_csv_path: str | None = Field(None, description="S3 path to table CSV export")
    s3_metadata_path: str | None = Field(
        None, description="S3 path to table metadata JSON"
    )
    title: str | None = Field(None, description="Table title/caption")
    row_count: int | None = Field(None, description="Number of rows")
    column_count: int | None = Field(None, description="Number of columns")
    page_number: int | None = Field(None, description="Page number where table appears")


class PageMetadata(BaseModel):
    """Metadata for page extraction results."""

    page_count: int = Field(..., description="Total number of pages in document")
    word_count: int = Field(..., description="Word count for this page")
    figure_count: int = Field(default=0, description="Number of figures on this page")
    table_count: int = Field(default=0, description="Number of tables on this page")
    extraction_config: dict[str, Any] = Field(
        ..., description="Extraction configuration used"
    )


class PageExtractionResult(BaseModel):
    """Per-page extraction result stored as results.json.

    This model represents the structure of the results.json file stored for each page
    at: s3://{bucket}/{project_id}/{upload_id}/{page_number}/results.json
    """

    page_number: int = Field(..., description="Page number (1-indexed)")
    document_url: str = Field(..., description="Source document S3 URL")
    content: str = Field(..., description="Extracted text content for this page")
    metadata: PageMetadata = Field(..., description="Page metadata")
    bounding_boxes: list[dict[str, Any]] = Field(
        default_factory=list, description="Bounding box data for text elements"
    )
    figures: list[FigureReference] = Field(
        default_factory=list, description="References to figures on this page"
    )
    tables: list[TableReference] = Field(
        default_factory=list, description="References to tables on this page"
    )
    processing_time_seconds: float | None = Field(
        None, description="Processing time for this document"
    )
    extraction_type: ExtractionType | None = Field(
        None, description="Type of extraction strategy used"
    )


class FigureMetadata(BaseModel):
    """Figure metadata stored as {figure_id}_metadata.json.

    This model represents the detailed metadata stored for each figure at:
    s3://{bucket}/{project_id}/{upload_id}/{page_number}/figures/{figure_id}_metadata.json
    """

    figure_id: str = Field(..., description="Unique figure identifier")
    source_document: str = Field(..., description="Source document S3 path")
    page_number: int = Field(
        ..., description="Page number where figure appears (1-indexed)"
    )
    extraction_timestamp: datetime = Field(..., description="When figure was extracted")
    caption: str | None = Field(None, description="Figure caption")
    alt_text: str | None = Field(None, description="Alternative text description")
    image_format: IngestionMimeType | None = Field(
        None, description="Image MIME type (e.g., image/png, image/jpeg)"
    )
    image_size: ImageSize | None = Field(None, description="Image dimensions")
    file_size_bytes: int | None = Field(None, description="File size in bytes")
    bbox: BoundingBox | None = Field(None, description="Bounding box coordinates")
    docling_label: str | None = Field(None, description="Docling classification label")
    extraction_method: str = Field(
        ..., description="Extraction method used (e.g., docling_get_image)"
    )
    llm_analysis: dict[str, Any] | None = Field(
        None, description="LLM analysis results from Bedrock"
    )


class TableMetadata(BaseModel):
    """Table metadata stored as {table_id}_metadata.json.

    This model represents the detailed metadata stored for each table at:
    s3://{bucket}/{project_id}/{upload_id}/{page_number}/tables/{table_id}_metadata.json
    """

    table_id: str = Field(..., description="Unique table identifier")
    source_document: str = Field(..., description="Source document S3 path")
    page_number: int = Field(
        ..., description="Page number where table appears (1-indexed)"
    )
    extraction_timestamp: datetime = Field(..., description="When table was extracted")
    caption: str | None = Field(None, description="Table caption/title")
    num_rows: int | None = Field(None, description="Number of rows")
    num_cols: int | None = Field(None, description="Number of columns")
    headers: list[str] | None = Field(None, description="Column headers")
    cell_count: int | None = Field(None, description="Total number of cells")
    has_merged_cells: bool = Field(
        default=False, description="Whether table has merged cells"
    )
    csv_available: bool = Field(
        default=False, description="Whether CSV export is available"
    )
    bbox: BoundingBox | None = Field(None, description="Bounding box coordinates")
    docling_label: str | None = Field(None, description="Docling classification label")
    extraction_method: str = Field(
        ..., description="Extraction method used (e.g., docling_export)"
    )
