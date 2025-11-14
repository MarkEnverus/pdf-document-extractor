"""Shared event/message models for IDP pipeline services."""

from .mime_type import IngestionMimeType
from .extraction_models import (
    BoundingBox,
    ExtractionType,
    FigureMetadata,
    FigureReference,
    ImageSize,
    LLMAnalysisResult,
    PageExtractionResult,
    PageMetadata,
    TableMetadata,
    TableReference,
)
from .mime_type_utils import MimeTypeUtils, determine_mime_type
from .pipeline_event import PipelineEvent
from .utils import safe_uuid_conversion

__all__ = [
    "PipelineEvent",
    "IngestionMimeType",
    "safe_uuid_conversion",
    "MimeTypeUtils",
    "determine_mime_type",
    # Extraction models
    "PageExtractionResult",
    "FigureMetadata",
    "TableMetadata",
    "BoundingBox",
    "ImageSize",
    "PageMetadata",
    "FigureReference",
    "TableReference",
    "LLMAnalysisResult",
    "ExtractionType",
]
