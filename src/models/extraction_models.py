# Copyright 2025 by Enverus. All rights reserved.

"""
Models for extracted figures and tables from documents.
"""

from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field, ConfigDict

from lib.models.mime_type import IngestionMimeType
from src.models.image_analysis_models import ImageAnalysisResult


class ExtractedFigure(BaseModel):
    """Model representing an extracted figure/image from a document."""

    figure_id: str = Field(..., description="Unique identifier for this figure")
    page_number: int = Field(..., description="Page number where figure appears")
    caption: Optional[str] = Field(None, description="Figure caption text")
    alt_text: Optional[str] = Field(None, description="Alt text for accessibility")
    label: str = Field(..., description="Docling label type (e.g. 'picture')")

    # Bounding box information
    bbox: Optional[Dict[str, Any]] = Field(None, description="Bounding box coordinates")

    # S3 storage paths
    s3_image_path: Optional[str] = Field(None, description="S3 path to extracted image file")
    s3_metadata_path: str = Field(..., description="S3 path to detailed metadata JSON")

    # Image metadata
    image_format: Optional[IngestionMimeType] = Field(None, description="Image MIME type (e.g., image/png)")
    image_size: Optional[Dict[str, int]] = Field(None, description="Image dimensions {width, height}")

    # Processing metadata
    extraction_method: str = Field("docling_get_image", description="Method used for extraction")
    extracted_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_config = ConfigDict(json_encoders={datetime: lambda dt: dt.isoformat()})


class ExtractedTable(BaseModel):
    """Model representing an extracted table from a document."""

    table_id: str = Field(..., description="Unique identifier for this table")
    page_number: int = Field(..., description="Page number where table appears")
    caption: Optional[str] = Field(None, description="Table caption text")
    label: str = Field(..., description="Docling label type (e.g. 'table')")

    # Table structure information
    num_rows: int = Field(..., description="Number of rows in the table")
    num_cols: int = Field(..., description="Number of columns in the table")
    headers: List[str] = Field(default_factory=list, description="Column headers")

    # Bounding box information
    bbox: Optional[Dict[str, Any]] = Field(None, description="Bounding box coordinates")

    # S3 storage paths
    s3_csv_path: Optional[str] = Field(None, description="S3 path to CSV export")
    s3_metadata_path: str = Field(..., description="S3 path to detailed metadata JSON")

    # Table data summary
    has_merged_cells: bool = Field(False, description="Whether table contains merged cells")
    cell_count: int = Field(..., description="Total number of cells")

    # Processing metadata
    extraction_method: str = Field("docling_export", description="Method used for extraction")
    extracted_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_config = ConfigDict(json_encoders={datetime: lambda dt: dt.isoformat()})


# FigureMetadata and TableMetadata removed - now imported from idp_pipeline_events
# These models are shared across services for consistent serialization/deserialization
