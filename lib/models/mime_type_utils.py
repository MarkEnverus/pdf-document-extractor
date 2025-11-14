# Copyright 2025 by Enverus. All rights reserved.

"""
Centralized MIME type utilities for document processing.

This module provides consistent MIME type detection across all services,
eliminating code duplication and ensuring uniform behavior.
"""

import pathlib

from lib.models.mime_type import IngestionMimeType


class MimeTypeUtils:
    """Utility class for MIME type detection and conversion."""

    # File extension to MIME type mapping - single source of truth
    _EXTENSION_TO_MIME = {
        "pdf": IngestionMimeType.PDF.value,
        ".pdf": IngestionMimeType.PDF.value,
        "docx": IngestionMimeType.DOCX.value,
        ".docx": IngestionMimeType.DOCX.value,
        "doc": IngestionMimeType.DOC.value,
        ".doc": IngestionMimeType.DOC.value,
        "pptx": IngestionMimeType.PPTX.value,
        ".pptx": IngestionMimeType.PPTX.value,
        "png": IngestionMimeType.PNG.value,
        ".png": IngestionMimeType.PNG.value,
        "jpg": IngestionMimeType.JPG.value,
        "jpeg": IngestionMimeType.JPG.value,
        ".jpg": IngestionMimeType.JPG.value,
        ".jpeg": IngestionMimeType.JPG.value,
    }

    @classmethod
    def determine_mime_type(
        cls, document_url: str, provided_file_type: str | None = None
    ) -> str:
        """
        Determine the MIME type from URL or provided file type.

        This is the single, authoritative method for MIME type detection
        used across all services.

        Args:
            document_url: S3 URL or path of the document
            provided_file_type: File type from metadata (can be None, "unknown", or a specific type)

        Returns:
            MIME type string from MimeType enum, or "unknown" if cannot be determined
        """
        # If file_type is provided and it's a known IngestionMimeType value, use it
        if provided_file_type and provided_file_type.lower() != "unknown":
            # Check if it's already a MIME type
            for mime_type in IngestionMimeType:
                if mime_type.value == provided_file_type:
                    return mime_type.value

            # Check if it's a file extension that we can map to MIME type
            mapped_type = cls._EXTENSION_TO_MIME.get(provided_file_type.lower())
            if mapped_type:
                return mapped_type

        # Extract file extension from URL and determine MIME type
        try:
            file_path = pathlib.Path(document_url)
            extension = file_path.suffix.lower()
            mapped_type = cls._EXTENSION_TO_MIME.get(extension)
            if mapped_type:
                return mapped_type
        except Exception:
            # Silently handle URL parsing errors and fall through to unknown
            pass

        # Default to unknown if we can't determine the type
        return "unknown"

    @classmethod
    def get_supported_extensions(cls) -> list[str]:
        """
        Get list of supported file extensions.

        Returns:
            List of supported file extensions (with and without dots)
        """
        return list(cls._EXTENSION_TO_MIME.keys())

    @classmethod
    def get_supported_mime_types(cls) -> list[str]:
        """
        Get list of supported MIME types.

        Returns:
            List of unique MIME types supported
        """
        return list(set(cls._EXTENSION_TO_MIME.values()))

    @classmethod
    def is_supported_extension(cls, extension: str) -> bool:
        """
        Check if a file extension is supported.

        Args:
            extension: File extension (with or without dot)

        Returns:
            True if the extension is supported, False otherwise
        """
        return extension.lower() in cls._EXTENSION_TO_MIME

    @classmethod
    def is_supported_mime_type(cls, mime_type: str) -> bool:
        """
        Check if a MIME type is supported.

        Args:
            mime_type: MIME type string

        Returns:
            True if the MIME type is supported, False otherwise
        """
        return mime_type in cls._EXTENSION_TO_MIME.values()


# Convenience function for backward compatibility and ease of use
def determine_mime_type(
    document_url: str, provided_file_type: str | None = None
) -> str:
    """
    Convenience function to determine MIME type.

    This is a direct alias to MimeTypeUtils.determine_mime_type() for easier importing.

    Args:
        document_url: S3 URL or path of the document
        provided_file_type: File type from metadata

    Returns:
        MIME type string from MimeType enum, or "unknown" if cannot be determined
    """
    return MimeTypeUtils.determine_mime_type(document_url, provided_file_type)
