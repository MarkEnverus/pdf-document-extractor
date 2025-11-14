"""HTTP client for extraction API operations."""

from typing import Dict, Any, Optional
import time
import requests

from src.interfaces.extraction_client import AbstractExtractionApiClient
from src.models.extraction_request import ExtractorRequest
from src.models.processing_models import ExtractionResult, ExtractionError
from src.exceptions.domain_exceptions import ExtractionError as DomainExtractionError
from lib.logger import Logger

logger = Logger.get_logger(__name__)


class ExtractionApiClient(AbstractExtractionApiClient):
    """HTTP client for extraction API with proper separation of concerns."""

    def __init__(self, endpoint: str, timeout: int = 300, logger_instance: Optional[Logger] = None):
        """
        Initialize extraction API client.

        Args:
            endpoint: API endpoint URL
            timeout: Request timeout in seconds
            logger_instance: Optional logger for dependency injection
        """
        self.endpoint = endpoint
        self.timeout = timeout
        self.logger = logger_instance or logger

    def _get_headers(self) -> Dict[str, str]:
        """Get HTTP headers for API requests."""
        return {"Content-Type": "application/json"}

    def extract(
        self,
        request: ExtractorRequest,
        request_id: Optional[str] = None,
        upload_id: Optional[str] = None,
        project_id: Optional[str] = None,
    ) -> ExtractionResult:
        """
        Submit extraction request to API.

        Args:
            request: Extraction request
            request_id: Optional request ID for tracking
            upload_id: Optional upload ID for tracking
            project_id: Optional project ID for tracking

        Returns:
            Extraction result

        Raises:
            DomainExtractionError: If extraction fails
        """
        try:
            url = f"{self.endpoint}/extract/"
            payload = request.model_dump(exclude_none=True)
            headers = self._get_headers()

            # Log context for structured logging
            log_context: dict[str, Any] = {
                "service": "extraction_api_client",
                "operation": "extract",
                "endpoint": url,
                "document_count": len(request.document_urls),
                "extraction_mode": request.extraction_mode,
            }
            if request_id:
                log_context["request_id"] = request_id
            if upload_id:
                log_context["upload_id"] = upload_id
            if project_id:
                log_context["project_id"] = project_id

            self.logger.info("Submitting extraction request to API", **log_context)
            response = requests.post(url, json=payload, headers=headers, timeout=self.timeout)

            if response.status_code == 200:
                response_data = response.json()
                self.logger.info(
                    "Extraction request submitted successfully", status_code=response.status_code, **log_context
                )
                job_id = response_data.get("processing_data", {}).get("extraction_result", {}).get("job_id")
                if job_id:
                    logger.info("Extraction job started", request_id=request_id, job_id=job_id)

                    # Poll for status synchronously in this thread
                    polling_result = self.poll_job_status(
                        job_id=job_id, request_id=request_id, upload_id=upload_id, project_id=project_id
                    )

                    success = polling_result.get("status") == "completed"
                    logger.info("Polling completed for job", job_id=job_id, success=success)
                return ExtractionResult(success=True, data=response_data)
            else:
                error = ExtractionError(
                    status_code=response.status_code,
                    error_message=f"Extraction API returned status {response.status_code}",
                    response_body=response.text,
                )
                self.logger.error(
                    "Extraction failed with HTTP error status",
                    status_code=response.status_code,
                    error_message=error.error_message,
                    **log_context,
                )
                return ExtractionResult(success=False, error=error)

        except requests.Timeout as e:
            error = ExtractionError(
                status_code=408,
                error_message=f"Extraction request timed out after {self.timeout}s",
                response_body=str(e),
            )
            self.logger.error(
                "Extraction request timeout",
                timeout_seconds=self.timeout,
                error_type="Timeout",
                error_message=str(e),
                **log_context,
            )
            return ExtractionResult(success=False, error=error)

        except requests.RequestException as e:
            error = ExtractionError(
                status_code=0, error_message=f"Extraction request failed: {str(e)}", response_body=str(e)
            )
            self.logger.error(
                "Extraction request network error", error_type=type(e).__name__, error_message=str(e), **log_context
            )
            return ExtractionResult(success=False, error=error)

        except Exception as e:
            self.logger.error(
                "Unexpected extraction error", error_type=type(e).__name__, error_message=str(e), **log_context
            )
            raise DomainExtractionError(message=f"Extraction failed: {str(e)}", cause=e)

    def check_job_status(
        self,
        job_id: str,
        request_id: Optional[str] = None,
        upload_id: Optional[str] = None,
        project_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Check status of extraction job.

        Args:
            job_id: Job identifier
            request_id: Optional request ID for tracking
            upload_id: Optional upload ID for tracking
            project_id: Optional project ID for tracking

        Returns:
            Job status information

        Raises:
            DomainExtractionError: If status check fails
        """
        try:
            url = f"{self.endpoint}/job-status/{job_id}/"
            headers = self._get_headers()

            # Log context for structured logging
            log_context: dict[str, Any] = {
                "service": "extraction_api_client",
                "operation": "check_job_status",
                "job_id": job_id,
                "endpoint": url,
            }
            if request_id:
                log_context["request_id"] = request_id
            if upload_id:
                log_context["upload_id"] = upload_id
            if project_id:
                log_context["project_id"] = project_id

            response = requests.get(url, headers=headers, timeout=30)

            if response.status_code == 200:
                status_data = response.json()
                self.logger.debug(
                    "Job status check successful",
                    status_code=response.status_code,
                    job_status=status_data.get("status", "unknown"),
                    **log_context,
                )
                return status_data
            else:
                self.logger.error("Job status check failed", status_code=response.status_code, **log_context)
                raise DomainExtractionError(
                    message=f"Job status check failed with status {response.status_code}",
                    context={"job_id": job_id, "status_code": response.status_code},
                )

        except requests.RequestException as e:
            self.logger.error(
                "Job status check request failed", error_type=type(e).__name__, error_message=str(e), **log_context
            )
            raise DomainExtractionError(
                message=f"Job status check request failed: {str(e)}", context={"job_id": job_id}, cause=e
            )

    def poll_job_status(
        self,
        job_id: str,
        poll_interval: int = 30,
        max_polls: int = 120,
        request_id: Optional[str] = None,
        upload_id: Optional[str] = None,
        project_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Poll job status until completion.

        Args:
            job_id: Job identifier
            poll_interval: Polling interval in seconds
            max_polls: Maximum number of polls
            request_id: Optional request ID for tracking
            upload_id: Optional upload ID for tracking
            project_id: Optional project ID for tracking

        Returns:
            Final job status

        Raises:
            DomainExtractionError: If polling fails or times out
        """
        # Log context for structured logging
        log_context: dict[str, Any] = {
            "service": "extraction_api_client",
            "operation": "poll_job_status",
            "job_id": job_id,
            "poll_interval": poll_interval,
            "max_polls": max_polls,
        }
        if request_id:
            log_context["request_id"] = request_id
        if upload_id:
            log_context["upload_id"] = upload_id
        if project_id:
            log_context["project_id"] = project_id

        self.logger.info("Starting job status polling", **log_context)

        for attempt in range(max_polls):
            try:
                status_data = self.check_job_status(job_id, request_id, upload_id, project_id)
                job_status = status_data.get("status", "").upper()

                poll_context = {**log_context, "attempt": attempt + 1, "job_status": job_status}

                if job_status in ["COMPLETED", "SUCCESS", "FINISHED"]:
                    self.logger.info("Job completed successfully", **poll_context)
                    return status_data
                elif job_status in ["FAILED", "ERROR", "CANCELLED"]:
                    self.logger.error("Job failed with terminal status", **poll_context)
                    raise DomainExtractionError(
                        message=f"Job failed with status: {job_status}",
                        context={"job_id": job_id, "status": job_status, "attempts": attempt + 1},
                    )
                else:
                    self.logger.debug("Job still processing, continuing to poll", **poll_context)
                    if attempt < max_polls - 1:
                        time.sleep(poll_interval)

            except DomainExtractionError:
                raise
            except Exception as e:
                error_context = {**log_context, "attempt": attempt + 1, "error_type": type(e).__name__}
                self.logger.error("Error during job polling", error_message=str(e), **error_context)
                if attempt == max_polls - 1:
                    raise DomainExtractionError(
                        message=f"Job polling failed after {max_polls} attempts",
                        context={"job_id": job_id, "attempts": max_polls},
                        cause=e,
                    )
                time.sleep(poll_interval)

        # Timeout reached
        self.logger.error("Job polling timeout reached", timeout_seconds=max_polls * poll_interval, **log_context)
        raise DomainExtractionError(
            message=f"Job polling timeout after {max_polls * poll_interval} seconds",
            context={"job_id": job_id, "max_polls": max_polls, "poll_interval": poll_interval},
        )
