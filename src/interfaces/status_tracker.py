"""Status tracking service interface."""

from typing import Protocol, runtime_checkable, Optional, Dict, Awaitable, Any
from abc import ABC, abstractmethod

from src.models.status_models import IngestionStepStatus


@runtime_checkable
class StatusTracker(Protocol):
    """
    Protocol for status tracking operations.

    This interface defines the contract for status tracking services that handle
    document processing status updates and Kafka publishing. Kafka publishing
    happens automatically on SUCCESS status only.

    Future Changes:
    - Phase 2: Add batch status update methods for concurrent processing
    - Consider adding status retrieval methods if needed for monitoring
    """

    async def update_status(
        self,
        upload_id: str,
        status: IngestionStepStatus,
        project_id: str = "default",
        s3_location: Optional[str] = None,
        file_type: str = "unknown",
        message: Optional[str] = None,
        request_id: Optional[str] = None,
        document_url: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Update status for document processing with database storage and Kafka publishing.

        Args:
            upload_id: Primary key for status tracking (required)
            status: Processing status (PROCESSING, SUCCESS, FAILURE, etc.)
            project_id: Project identifier
            s3_location: S3 path to results/content (for SUCCESS status)
            file_type: MIME type of processed content
            message: Human-readable status message
            request_id: Internal correlation UUID (for pipeline tracking)
            document_url: Original document URL (for MIME type detection)
            metadata: Additional structured data

        Returns:
            True if status update succeeded, False otherwise
        """
        ...

    def health_check(self) -> Dict[str, str]:
        """
        Perform health check on the status tracker.

        Returns:
            Dictionary with health status information
        """
        ...


class AbstractStatusTracker(ABC):
    """
    Abstract base class for status trackers.

    Provides a base implementation for status tracking services. Concrete
    implementations should focus on the specific storage backend (API, database, etc.)
    and ensure critical service failures cause the application to fail-fast.
    """

    @abstractmethod
    async def update_status(
        self,
        upload_id: str,
        status: IngestionStepStatus,
        project_id: str = "default",
        s3_location: Optional[str] = None,
        file_type: str = "unknown",
        message: Optional[str] = None,
        request_id: Optional[str] = None,
        document_url: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Update processing status."""
        pass

    @abstractmethod
    def health_check(self) -> Dict[str, str]:
        """Perform health check on the status tracker."""
        pass
