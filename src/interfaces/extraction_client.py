"""Extraction API client interface."""

from typing import Protocol, runtime_checkable, Dict, Any
from abc import ABC, abstractmethod

from src.models.extraction_request import ExtractorRequest
from src.models.processing_models import ExtractionResult


@runtime_checkable
class ExtractionApiClient(Protocol):
    """Protocol for extraction API operations."""

    def extract(self, request: ExtractorRequest) -> ExtractionResult:
        """
        Submit extraction request to API.

        Args:
            request: Extraction request

        Returns:
            Extraction result

        Raises:
            ExtractionError: If extraction fails
        """
        ...

    def check_job_status(self, job_id: str) -> Dict[str, Any]:
        """
        Check status of extraction job.

        Args:
            job_id: Job identifier

        Returns:
            Job status information

        Raises:
            ExtractionError: If status check fails
        """
        ...

    def poll_job_status(self, job_id: str, poll_interval: int = 30, max_polls: int = 120) -> Dict[str, Any]:
        """
        Poll job status until completion.

        Args:
            job_id: Job identifier
            poll_interval: Polling interval in seconds
            max_polls: Maximum number of polls

        Returns:
            Final job status

        Raises:
            ExtractionError: If polling fails or times out
        """
        ...


class AbstractExtractionApiClient(ABC):
    """Abstract base class for extraction API clients."""

    @abstractmethod
    def extract(self, request: ExtractorRequest) -> ExtractionResult:
        """Submit extraction request to API."""
        pass

    @abstractmethod
    def check_job_status(self, job_id: str) -> Dict[str, Any]:
        """Check status of extraction job."""
        pass

    @abstractmethod
    def poll_job_status(self, job_id: str, poll_interval: int = 30, max_polls: int = 120) -> Dict[str, Any]:
        """Poll job status until completion."""
        pass
