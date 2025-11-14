"""Centralized S3 storage utilities for consistent path generation and file storage.

This module provides shared functions for S3 path generation and text file storage
to eliminate code duplication across extraction processors and asset storage service.
"""

from typing import Any

from lib.logger import Logger
from src.interfaces.storage import StorageService


def generate_page_text_s3_path(project_id: str, upload_id: str, page_number: int) -> str:
    """Generate standardized S3 key for page text files.

    Args:
        project_id: Project identifier
        upload_id: Upload identifier
        page_number: Page number (1-indexed)

    Returns:
        S3 key in format: {project_id}/{upload_id}/{page_number}/page_text.txt

    Example:
        >>> generate_page_text_s3_path("proj-123", "upload-456", 1)
        'proj-123/upload-456/1/page_text.txt'
    """
    return f"{project_id}/{upload_id}/{page_number}/page_text.txt"


def generate_results_json_s3_path(project_id: str, upload_id: str, page_number: int) -> str:
    """Generate standardized S3 key for results JSON files.

    Args:
        project_id: Project identifier
        upload_id: Upload identifier
        page_number: Page number (1-indexed)

    Returns:
        S3 key in format: {project_id}/{upload_id}/{page_number}/results.json

    Example:
        >>> generate_results_json_s3_path("proj-123", "upload-456", 1)
        'proj-123/upload-456/1/results.json'
    """
    return f"{project_id}/{upload_id}/{page_number}/results.json"


def store_text_file_to_s3(
    storage_service: StorageService,
    logger: Logger,
    text_content: str,
    bucket: str,
    s3_key: str,
    project_id: str,
    upload_id: str,
    log_context: dict[str, Any] | None = None,
) -> str:
    """Store text content to S3 with consistent encoding and logging.

    This utility function encapsulates the common pattern of:
    1. Encoding text content to UTF-8
    2. Uploading to S3
    3. Logging the operation with context

    Args:
        storage_service: Storage service for uploading content
        logger: Logger instance for operation tracking
        text_content: Text content to store
        bucket: S3 bucket name
        s3_key: S3 key (path) for the file
        project_id: Project identifier for logging and metadata
        upload_id: Upload identifier for logging and metadata
        log_context: Optional additional context for logging (e.g., page_number, figure_id)

    Returns:
        S3 path (s3://bucket/key) of the stored file

    Example:
        >>> s3_path = store_text_file_to_s3(
        ...     storage_service=storage,
        ...     logger=logger,
        ...     text_content="Page 1 content...",
        ...     bucket="my-bucket",
        ...     s3_key="proj/upload/1/page_text.txt",
        ...     project_id="proj",
        ...     upload_id="upload",
        ...     log_context={"page_number": 1}
        ... )
    """
    # Encode text content to UTF-8 bytes
    encoded_content = text_content.encode("utf-8")

    # Upload to S3
    s3_path = storage_service.upload_content(
        content=encoded_content,
        bucket=bucket,
        key=s3_key,
        project_id=project_id,
        upload_id=upload_id,
    )

    # Build log context
    log_data: dict[str, Any] = {
        "s3_path": s3_path,
        "text_length": len(text_content),
        "project_id": project_id,
        "upload_id": upload_id,
    }

    # Merge additional context if provided
    if log_context:
        log_data.update(log_context)

    # Log the operation
    logger.debug("Stored text file to S3", **log_data)

    return s3_path
