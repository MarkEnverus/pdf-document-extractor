"""Document processing service interfaces and strategy pattern protocols."""

from typing import Protocol, runtime_checkable, Any, Dict, List, Optional
from abc import ABC, abstractmethod

from src.models.docling_models import DoclingRequest, DoclingBatchResult


# Strategy pattern interface for different processing approaches
@runtime_checkable
class DocumentProcessingStrategy(Protocol):
    """
    Protocol defining the strategy pattern for document processing.

    This enables switching between different processing approaches:
    - Docling local processing
    - API-based extraction
    - Cloud services
    - Hybrid approaches
    """

    async def process_documents(
        self,
        request_id: str,
        project_id: str,
        upload_id: str,
        document_urls: List[str],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Process documents using this strategy.

        Args:
            request_id: Unique identifier for this processing request
            project_id: Project identifier (required for all logs)
            upload_id: Upload identifier - this is the key for tracking (required for all logs)
            document_urls: List of document URLs to process
            metadata: Optional metadata containing processing hints and parameters

        Returns:
            Dictionary containing processing results in a standardized format
        """
        ...

    def health_check(self) -> Dict[str, str]:
        """
        Check the health of this processing strategy.

        Returns:
            Dictionary with health status information
        """
        ...

    def get_strategy_name(self) -> str:
        """
        Get the name/identifier of this processing strategy.

        Returns:
            String identifier for this strategy (e.g., "docling", "api", "cloud")
        """
        ...

    def supports_file_type(self, file_type: str) -> bool:
        """
        Check if this strategy supports the given file type.

        Args:
            file_type: MIME type string to check

        Returns:
            True if this strategy can process the file type, False otherwise
        """
        ...

    def get_processing_capabilities(self) -> Dict[str, Any]:
        """
        Get information about what this processor can do.

        Returns:
            Dictionary describing processing capabilities
        """
        ...


class AbstractDocumentProcessor(ABC):
    """Abstract base class for Docling document processors (legacy)."""

    @abstractmethod
    def process_documents(self, request: DoclingRequest) -> DoclingBatchResult:
        """Process documents through processing pipeline."""
        pass


class AbstractProcessingStrategy(ABC):
    """
    Abstract base class for document processing strategies.

    This provides a concrete base class that implements the DocumentProcessingStrategy protocol
    with common functionality and requires subclasses to implement the core methods.
    """

    def __init__(self, strategy_name: str):
        """
        Initialize the abstract processor.

        Args:
            strategy_name: Name identifier for this strategy
        """
        self.strategy_name = strategy_name

    @abstractmethod
    async def process_documents(
        self,
        request_id: str,
        project_id: str,
        upload_id: str,
        document_urls: List[str],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Abstract method for document processing.

        Args:
            request_id: Unique identifier for this processing request
            project_id: Project identifier (required for all logs)
            upload_id: Upload identifier - this is the key for tracking (required for all logs)
            document_urls: List of document URLs to process
            metadata: Optional additional metadata

        Returns:
            Dictionary containing processing results
        """
        pass

    @abstractmethod
    def health_check(self) -> Dict[str, str]:
        """Abstract method for health checking."""
        pass

    def get_strategy_name(self) -> str:
        """Return the strategy name."""
        return self.strategy_name

    @abstractmethod
    def supports_file_type(self, file_type: str) -> bool:
        """Abstract method for file type support checking."""
        pass

    @abstractmethod
    def get_processing_capabilities(self) -> Dict[str, Any]:
        """Abstract method for capability reporting."""
        pass

    def _create_standard_result_format(
        self,
        request_id: str,
        success: bool,
        document_urls: List[str],
        processing_data: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Create a standardized result format for consistency across processors.

        Args:
            request_id: Request identifier
            success: Whether processing was successful
            document_urls: List of processed document URLs
            processing_data: Strategy-specific processing data
            error_message: Error message if processing failed

        Returns:
            Standardized result dictionary
        """
        result = {
            "request_id": request_id,
            "success": success,
            "strategy": self.strategy_name,
            "document_urls": document_urls,
            "total_documents": len(document_urls),
            "processing_timestamp": self._get_current_timestamp(),
        }

        if success and processing_data:
            result.update(processing_data)
        elif not success and error_message:
            result["error_message"] = error_message

        return result

    def _get_current_timestamp(self) -> str:
        """Get current timestamp in ISO format."""
        from datetime import datetime

        return datetime.utcnow().isoformat() + "Z"


DocumentProcessingStrategyType = DocumentProcessingStrategy
