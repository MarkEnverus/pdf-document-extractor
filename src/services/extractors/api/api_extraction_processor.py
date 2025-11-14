"""
API-based document extraction processor implementing the strategy pattern.

This processor handles document extraction via external API calls,
implementing the DocumentProcessingStrategy interface.
"""

import json
from typing import Dict, Any, List, Optional

from lib.models.mime_type import IngestionMimeType
from lib.logger import Logger
from lib.models.extraction_models import ExtractionType, PageExtractionResult, PageMetadata
from src.interfaces.document_processor import AbstractProcessingStrategy
from src.services.extraction_api_client import ExtractionApiClient
from src.interfaces.storage import StorageService
from src.services.status_tracker import StatusTracker
from src.services.storage_utils import (
    generate_page_text_s3_path,
    generate_results_json_s3_path,
    store_text_file_to_s3,
)

# DocumentInfoExtractor removed - using MimeTypeUtils directly if needed
from src.models.extraction_request import ExtractorRequest, ExtractionConfig, ConfigType
from src.models.processing_models import ProcessingResult, IngestionStepStatus

logger = Logger.get_logger(__name__)


class ApiExtractionProcessor(AbstractProcessingStrategy):
    """
    API-based document extraction processor.

    This strategy uses external API services for document extraction,
    suitable for:
    - Large batches of documents
    - Mixed file types
    - Documents requiring cloud-based AI processing
    - Scenarios where Docling is not optimal
    """

    def __init__(
        self, extraction_client: ExtractionApiClient, storage_service: StorageService, status_tracker: StatusTracker
    ):
        """
        Initialize ApiExtractionProcessor.

        Args:
            extraction_client: Client for calling extraction API
            storage_service: Storage service for storing results
            status_tracker: Service for updating processing status
        """
        super().__init__("api")
        self.extraction_client = extraction_client
        self.storage_service = storage_service
        self.status_tracker = status_tracker
        self.logger = logger

    async def process_documents(
        self,
        request_id: str,
        project_id: str,
        upload_id: str,
        document_urls: List[str],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Process documents using API extraction strategy.

        Args:
            request_id: Unique identifier for this processing request
            project_id: Project identifier (required for all logs)
            upload_id: Upload identifier - this is the key for tracking (required for all logs)
            document_urls: List of document URLs to process
            metadata: Optional metadata containing processing hints

        Returns:
            Dictionary containing API processing results
        """
        try:
            self.logger.info(
                "Starting API extraction processing",
                request_id=request_id,
                project_id=project_id,
                upload_id=upload_id,
                document_count=len(document_urls),
                strategy="api",
            )

            # Basic validation - ensure we have documents
            if not document_urls:
                return self._create_failure_result(
                    request_id=request_id,
                    document_urls=document_urls,
                    error_message="No documents provided for processing",
                    project_id=project_id,
                    upload_id=upload_id,
                )

            # Update status to processing
            await self.status_tracker.update_status(
                upload_id=upload_id,
                status=IngestionStepStatus.PROCESSING,
                project_id=project_id,
                request_id=request_id,
                message="Starting API extraction processing",
            )

            # Create extraction request
            extraction_request = self._create_extraction_request(document_urls, metadata)

            # Call extraction API
            extraction_result = self.extraction_client.extract(extraction_request)

            # Process results
            if extraction_result.success:
                return await self._handle_successful_extraction(
                    request_id=request_id,
                    document_urls=document_urls,
                    extraction_result=extraction_result,
                    project_id=project_id,
                    upload_id=upload_id,
                )
            else:
                return await self._handle_failed_extraction(
                    request_id=request_id,
                    document_urls=document_urls,
                    extraction_result=extraction_result,
                    project_id=project_id,
                    upload_id=upload_id,
                )

        except Exception as e:
            self.logger.error(
                "Unexpected error in API extraction processing",
                request_id=request_id,
                document_count=len(document_urls),
                error=str(e),
                exc_info=True,
            )

            # Update status to failed
            await self.status_tracker.update_status(
                upload_id=upload_id,
                status=IngestionStepStatus.FAILURE,
                project_id=project_id,
                request_id=request_id,
                message=f"API processing failed: {str(e)}",
            )

            return self._create_failure_result(
                request_id=request_id,
                document_urls=document_urls,
                error_message=f"Processing failed: {str(e)}",
                project_id=project_id,
                upload_id=upload_id,
            )

    def _create_extraction_request(
        self, document_urls: List[str], metadata: Optional[Dict[str, Any]] = None
    ) -> ExtractorRequest:
        """Create extraction API request with optimized configuration."""
        # Create default extraction configuration
        extraction_config = ExtractionConfig(
            config_type=ConfigType.BEDROCK,
            model_id="anthropic.claude-3-sonnet-20240229-v1:0",
            name=None,
            system_prompt=None,
            data_automation_project_arn=None,
            data_automation_profile_arn=None,
            output_s3_uri=None,
            blueprints=None,
            role_arn=None,
        )

        # Apply any custom configuration from metadata
        if metadata and "extraction_config" in metadata:
            custom_config = metadata["extraction_config"]
            if isinstance(custom_config, dict):
                for key, value in custom_config.items():
                    if hasattr(extraction_config, key):
                        setattr(extraction_config, key, value)

        return ExtractorRequest(
            document_urls=document_urls,
            extraction_configs=[extraction_config],
            extraction_mode=metadata.get("extraction_mode", "multi") if metadata else "multi",
            output_schema=metadata.get("output_schema") if metadata else None,
        )

    async def _handle_successful_extraction(
        self, request_id: str, document_urls: List[str], extraction_result: Any, project_id: str, upload_id: str
    ) -> Dict[str, Any]:
        """Handle successful API extraction results."""
        try:
            # Store extraction results to S3
            extraction_s3_path = self._store_extraction_results(
                extraction_result=extraction_result, project_id=project_id, upload_id=upload_id
            )

            # Update status to success
            await self.status_tracker.update_status(
                upload_id=upload_id,
                status=IngestionStepStatus.SUCCESS,
                project_id=project_id,
                s3_location=extraction_s3_path,
                request_id=request_id,
                message=f"Successfully processed {len(document_urls)} documents via API",
                metadata={
                    "extraction_result": extraction_result.data,
                    "extraction_type": ExtractionType.API.value,
                    "extractions": ["1/results.json"],  # Matches Docling structure for downstream consumers
                },
            )

            self.logger.info(
                "API extraction processing completed successfully",
                request_id=request_id,
                document_count=len(document_urls),
                extraction_s3_path=extraction_s3_path,
            )

            return self._create_standard_result_format(
                request_id=request_id,
                success=True,
                document_urls=document_urls,
                processing_data={
                    "extraction_s3_path": extraction_s3_path,
                    "extraction_result": extraction_result,
                    "processing_method": "api",
                },
            )

        except Exception as e:
            self.logger.error("Failed to handle successful extraction results", request_id=request_id, error=str(e))
            return self._create_failure_result(
                request_id=request_id,
                document_urls=document_urls,
                error_message=f"Failed to process extraction results: {str(e)}",
                project_id=project_id,
                upload_id=upload_id,
            )

    async def _handle_failed_extraction(
        self, request_id: str, document_urls: List[str], extraction_result: Any, project_id: str, upload_id: str
    ) -> Dict[str, Any]:
        """Handle failed API extraction results."""
        error_message = "Unknown extraction error"
        if extraction_result.error:
            error_message = extraction_result.error.error_message

        self.logger.error(
            "API extraction failed",
            request_id=request_id,
            error_message=error_message,
            document_count=len(document_urls),
        )

        # Update status to failed
        await self.status_tracker.update_status(
            upload_id=upload_id,
            status=IngestionStepStatus.FAILURE,
            project_id=project_id,
            request_id=request_id,
            message=f"API extraction failed: {error_message}",
        )

        return self._create_failure_result(
            request_id=request_id,
            document_urls=document_urls,
            error_message=error_message,
            project_id=project_id,
            upload_id=upload_id,
            extraction_result=extraction_result,
        )

    async def _handle_empty_documents(self, request_id: str, project_id: str, upload_id: str) -> Dict[str, Any]:
        """Handle the case of empty document list."""
        self.logger.info(
            "No documents to process via API", request_id=request_id, project_id=project_id, upload_id=upload_id
        )

        # Create empty extraction result for consistency
        empty_extraction_data = {
            "results": [],
            "total_documents": 0,
            "successful_documents": 0,
            "failed_documents": 0,
        }

        # Update status to success (empty processing is still successful)
        await self.status_tracker.update_status(
            upload_id=upload_id,
            status=IngestionStepStatus.SUCCESS,
            project_id=project_id,
            request_id=request_id,
            message="Successfully processed 0 documents via API",
            metadata={"extraction_result": empty_extraction_data},
        )

        return self._create_standard_result_format(
            request_id=request_id,
            success=True,
            document_urls=[],
            processing_data={"extraction_result": empty_extraction_data, "processing_method": "api"},
        )

    def _store_extraction_results(self, extraction_result: Any, project_id: str, upload_id: str) -> str:
        """Store extraction results to S3 using page-based structure and return S3 path."""
        from src.configs.settings import settings

        bucket = settings.S3_BUCKET_NAME
        page_number = 1  # API extraction treats entire document as single page

        # Extract text content from extraction result
        content = ""
        if hasattr(extraction_result, "data") and isinstance(extraction_result.data, dict):
            # Try to extract text content from various possible fields
            content = extraction_result.data.get("text", "")
            if not content:
                content = extraction_result.data.get("content", "")
            if not content:
                # Fallback: serialize the entire result as JSON
                self.logger.warning(
                    "Extraction result missing 'text' and 'content' fields; falling back to JSON serialization",
                    project_id=project_id,
                    upload_id=upload_id,
                    has_data_fields=list(extraction_result.data.keys()) if extraction_result.data else [],
                )
                content = json.dumps(extraction_result.data, default=str, indent=2)

        # Validate that we extracted meaningful content
        if not content or not content.strip():
            self.logger.warning(
                "No content extracted from API result",
                project_id=project_id,
                upload_id=upload_id,
                has_data=hasattr(extraction_result, "data"),
            )
            raise ValueError(
                f"API extraction returned no content for project {project_id}, upload {upload_id}. "
                "The document may be empty, corrupted, or in an unsupported format."
            )

        # Create page metadata
        word_count = len(content.split()) if content else 0
        page_metadata = PageMetadata(
            page_count=1,
            word_count=word_count,
            figure_count=0,
            table_count=0,
            extraction_config={"extraction_type": "api", "config_type": "bedrock"},
        )

        # Create page-based extraction result
        page_result = PageExtractionResult(
            page_number=page_number,
            document_url=f"s3://{bucket}/{project_id}/{upload_id}/",
            content=content,
            metadata=page_metadata,
            bounding_boxes=[],
            figures=[],
            tables=[],
            processing_time_seconds=None,
            extraction_type=ExtractionType.API,
        )

        # Store results.json
        results_key = generate_results_json_s3_path(project_id, upload_id, page_number)
        results_content = page_result.model_dump_json(indent=2).encode("utf-8")
        results_s3_path = self.storage_service.upload_content(
            results_content, bucket, results_key, project_id=project_id, upload_id=upload_id
        )

        self.logger.debug(
            "Stored page results to S3",
            s3_path=results_s3_path,
            page_number=page_number,
            project_id=project_id,
            upload_id=upload_id,
        )

        # Store page content as separate text file for easy downstream consumption
        # while also maintaining the full PageExtractionResult in results.json for
        # consistency with Docling output structure and existing pipeline consumers.
        text_key = generate_page_text_s3_path(project_id, upload_id, page_number)
        store_text_file_to_s3(
            storage_service=self.storage_service,
            logger=self.logger,
            text_content=content,
            bucket=bucket,
            s3_key=text_key,
            project_id=project_id,
            upload_id=upload_id,
            log_context={"page_number": page_number},
        )

        return results_s3_path

    def _create_failure_result(
        self,
        request_id: str,
        document_urls: List[str],
        error_message: str,
        project_id: str,
        upload_id: str,
        extraction_result: Any = None,
    ) -> Dict[str, Any]:
        """Create standardized failure result."""
        result_data = {"processing_method": "api", "project_id": project_id, "upload_id": upload_id}

        if extraction_result:
            result_data["extraction_result"] = extraction_result

        return self._create_standard_result_format(
            request_id=request_id,
            success=False,
            document_urls=document_urls,
            processing_data=result_data,
            error_message=error_message,
        )

    def health_check(self) -> Dict[str, str]:
        """
        Check the health of the API extraction processor.

        Returns:
            Dictionary with health status information
        """
        health = {"api_extraction_processor": "healthy", "strategy_name": self.strategy_name}

        # Check extraction client health
        try:
            if hasattr(self.extraction_client, "health_check"):
                client_health = self.extraction_client.health_check()
                health["extraction_client"] = "healthy" if client_health else "unhealthy"
            else:
                health["extraction_client"] = "available"
        except Exception:
            health["extraction_client"] = "unhealthy"

        # Check storage service health
        try:
            if hasattr(self.storage_service, "health_check"):
                storage_health = self.storage_service.health_check()
                health["storage_service"] = "healthy" if storage_health else "unhealthy"
            else:
                health["storage_service"] = "available"
        except Exception:
            health["storage_service"] = "unhealthy"

        # Check status tracker health
        try:
            status_health = self.status_tracker.health_check()
            health["status_tracker"] = "healthy" if status_health.get("status_tracker") == "healthy" else "unhealthy"
        except Exception:
            health["status_tracker"] = "unhealthy"

        return health

    def supports_file_type(self, file_type: str) -> bool:
        """
        Check if this strategy supports the given file type.

        API extraction generally supports most file types through cloud processing.

        Args:
            file_type: MIME type string to check

        Returns:
            True for most common file types
        """
        supported_types = {
            IngestionMimeType.PDF.value,
            IngestionMimeType.DOCX.value,
            IngestionMimeType.DOC.value,
            IngestionMimeType.PPTX.value,
            IngestionMimeType.PNG.value,
            IngestionMimeType.JPG.value,
            IngestionMimeType.XLSX.value,
            IngestionMimeType.XLS.value,
            IngestionMimeType.TXT.value,
            IngestionMimeType.RTF.value,
        }

        # API extraction is flexible and can handle most types
        return file_type in supported_types or file_type != "unknown"

    def get_processing_capabilities(self) -> Dict[str, Any]:
        """
        Get information about API processor capabilities.

        Returns:
            Dictionary describing processing capabilities
        """
        return {
            "strategy_name": self.strategy_name,
            "supported_file_types": [
                IngestionMimeType.PDF.value,
                IngestionMimeType.DOCX.value,
                IngestionMimeType.DOC.value,
                IngestionMimeType.PPTX.value,
                IngestionMimeType.PNG.value,
                IngestionMimeType.JPG.value,
                IngestionMimeType.XLSX.value,
                IngestionMimeType.XLS.value,
                IngestionMimeType.TXT.value,
                IngestionMimeType.RTF.value,
            ],
            "features": [
                "batch_processing",
                "cloud_ai_extraction",
                "mixed_file_types",
                "scalable_processing",
                "api_based_extraction",
            ],
            "limitations": ["requires_api_connectivity", "external_dependencies", "potential_latency"],
            "performance_characteristics": {
                "batch_size": "unlimited",
                "concurrency": "high",
                "processing_speed": "variable",
                "resource_usage": "low_local",
            },
            "optimal_use_cases": [
                "large_document_batches",
                "mixed_file_types",
                "cloud_based_processing",
                "scalable_workloads",
            ],
        }
