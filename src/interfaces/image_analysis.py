"""Image analysis service interface for LLM-based image processing."""

from typing import Protocol, runtime_checkable, Dict, Any, Optional
from abc import ABC, abstractmethod


@runtime_checkable
class ImageAnalysisService(Protocol):
    """Protocol for image analysis operations using LLM services."""

    def analyze_image(
        self, image_data: bytes, image_format: str, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Analyze an image using LLM and return description/analysis.

        Args:
            image_data: Raw image bytes
            image_format: Image format (PNG, JPEG, etc.)
            context: Optional context about the image (caption, source document, etc.)

        Returns:
            Dictionary containing analysis results:
            {
                "analysis_summary": str,
                "model_used": str,
                "analysis_timestamp": str (ISO format),
                "analysis_confidence": Optional[float],
                "analysis_error": Optional[str]
            }

        Raises:
            ImageAnalysisError: If analysis fails critically
        """
        ...

    def health_check(self) -> Dict[str, str]:
        """
        Check health of image analysis service.

        Returns:
            Dictionary with health status information
        """
        ...


class AbstractImageAnalysisService(ABC):
    """Abstract base class for image analysis services."""

    @abstractmethod
    def analyze_image(
        self, image_data: bytes, image_format: str, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Analyze an image using LLM and return description/analysis."""
        pass

    @abstractmethod
    def health_check(self) -> Dict[str, str]:
        """Check health of image analysis service."""
        pass


class ImageAnalysisError(Exception):
    """Exception raised when image analysis fails."""

    def __init__(self, message: str, model_used: Optional[str] = None, original_error: Optional[Exception] = None):
        """
        Initialize ImageAnalysisError.

        Args:
            message: Error message
            model_used: Model that was being used when error occurred
            original_error: Original exception that caused this error
        """
        super().__init__(message)
        self.model_used = model_used
        self.original_error = original_error
