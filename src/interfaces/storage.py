"""Storage service interface."""

from typing import Protocol, runtime_checkable, Optional
from abc import ABC, abstractmethod


@runtime_checkable
class StorageService(Protocol):
    """Protocol for storage operations."""

    def upload_content(
        self,
        content: bytes,
        bucket: str,
        key: str,
        request_id: Optional[str] = None,
        upload_id: Optional[str] = None,
        project_id: Optional[str] = None,
    ) -> str:
        """
        Upload content to storage.

        Args:
            content: Content to upload
            bucket: Storage bucket name
            key: Storage key/path
            request_id: Optional request ID for tracking
            upload_id: Optional upload ID for tracking
            project_id: Optional project ID for tracking

        Returns:
            Storage URI of uploaded content

        Raises:
            StorageError: If upload fails
        """
        ...

    def download_content(
        self,
        s3_uri: str,
        request_id: Optional[str] = None,
        upload_id: Optional[str] = None,
        project_id: Optional[str] = None,
    ) -> bytes:
        """
        Download content from storage.

        Args:
            s3_uri: Storage URI to download from
            request_id: Optional request ID for tracking
            upload_id: Optional upload ID for tracking
            project_id: Optional project ID for tracking

        Returns:
            Downloaded content as bytes

        Raises:
            StorageError: If download fails
        """
        ...

    def parse_s3_uri(self, s3_uri: str) -> tuple[str, str]:
        """
        Parse S3 URI into bucket and key components.

        Args:
            s3_uri: S3 URI to parse

        Returns:
            Tuple of (bucket, key)

        Raises:
            ValueError: If URI is invalid
        """
        ...

    def health_check(self) -> dict[str, str]:
        """
        Check the health status of the storage service.

        Returns:
            Dictionary with health status information
        """
        ...


class AbstractStorageService(ABC):
    """Abstract base class for storage services."""

    @abstractmethod
    def upload_content(
        self,
        content: bytes,
        bucket: str,
        key: str,
        request_id: Optional[str] = None,
        upload_id: Optional[str] = None,
        project_id: Optional[str] = None,
    ) -> str:
        """Upload content to storage."""
        pass

    @abstractmethod
    def download_content(
        self,
        s3_uri: str,
        request_id: Optional[str] = None,
        upload_id: Optional[str] = None,
        project_id: Optional[str] = None,
    ) -> bytes:
        """Download content from storage."""
        pass

    @abstractmethod
    def parse_s3_uri(self, s3_uri: str) -> tuple[str, str]:
        """Parse S3 URI into bucket and key."""
        pass

    @abstractmethod
    def health_check(self) -> dict[str, str]:
        """Check the health status of the storage service."""
        pass
