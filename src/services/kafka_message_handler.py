# Copyright 2025 by Enverus. All rights reserved.

"""
Message handler for processing Kafka document processing requests.
"""

import time
from typing import Any, Dict

from lib.logger import Logger

from src.interfaces.status_tracker import StatusTracker
from src.models.extraction_request import ConfigType, ExtractionConfig, ExtractorRequest
from src.models.processing_models import IncomingDocumentProcessingRequest
from src.models.status_models import IngestionStepStatus
from src.services.extraction_api_client import ExtractionApiClient
from src.services.processing_orchestrator import ProcessingOrchestrator

logger = Logger.get_logger(__name__)


class KafkaMessageHandler:
    """Handler for processing document processing requests from Kafka."""

    def __init__(
        self,
        status_tracker: StatusTracker,
        orchestrator: ProcessingOrchestrator,
        extraction_client: ExtractionApiClient,
    ):
        self.status_tracker = status_tracker
        self.orchestrator = orchestrator
        self.extraction_client = extraction_client

    async def handle_document_processing_request(
        self, request: IncomingDocumentProcessingRequest, metadata: dict[str, Any] | None = None
    ) -> None:
        """
        Handle incoming document processing request from Kafka.

        Args:
            request: Parsed document processing request with new schema
            metadata: Kafka message metadata (partition, offset, etc.)
        """
        try:
            logger.info(
                "Processing Kafka document request",
                request_id=request.request_id,
                project_id=request.project_id,
                upload_id=request.upload_id,
                partition=metadata.get("partition") if metadata else None,
                offset=metadata.get("offset") if metadata else None,
            )

            # Get the base S3 path where extracted content will be stored
            extraction_s3_base_path = request.get_extraction_s3_base_path()

            # Create initial extraction status record - this is the ONLY create operation
            status_update_success = await self.status_tracker.update_status(
                upload_id=request.upload_id,
                status=IngestionStepStatus.QUEUED,
                project_id=request.project_id,
                s3_location=extraction_s3_base_path,
                file_type=request.file_type,
                request_id=request.request_id,
                message="Queued for processing",
            )

            # Fail fast if status update failed (likely upload_id doesn't exist in database)
            if not status_update_success:
                logger.error(
                    "Failed to update status for upload. Skipping processing and moving to next file",
                    upload_id=request.upload_id,
                    request_id=request.request_id,
                )
                return  # Exit early, don't process this document

            logger.info("Starting document processing", request_id=request.request_id)

            # Default to docling processing (can be made configurable later)
            use_extraction_api = False  # Could be set from ENV variable or message field

            if use_extraction_api:
                # Use extraction API with polling
                extraction_config = ExtractionConfig(
                    config_type=ConfigType.BEDROCK,
                    output_s3_uri=extraction_s3_base_path,
                    name=None,
                    system_prompt=None,
                    model_id="anthropic.claude-3-sonnet-20240229-v1:0",
                    data_automation_project_arn=None,
                    data_automation_profile_arn=None,
                    blueprints=None,
                    role_arn=None,
                )

                extraction_request = ExtractorRequest(
                    document_urls=[request.location],
                    extraction_configs=[extraction_config],
                    extraction_mode="multi",
                    output_schema=None,
                )

                # Call extraction API directly
                extraction_result = self.extraction_client.extract(
                    extraction_request,
                    request_id=request.request_id,
                    upload_id=request.upload_id,
                    project_id=request.project_id,
                )

                # Use the new document processing pipeline (auto-selects strategy)
            logger.info("Using document processing pipeline", request_id=request.request_id)

            # Process through the orchestrator with explicit strategy (using "docling" as default)
            # User will specify the strategy they want, but for Kafka we default to "docling"
            pipeline_result = await self.orchestrator.process_documents(
                strategy="docling",
                request_id=request.request_id,
                project_id=request.project_id,
                upload_id=request.upload_id,
                document_urls=[request.location],
                source="kafka",
            )

            # Check if processing succeeded - if not, check if it's a hard failure
            processing_success = pipeline_result.get("success", False)
            is_hard_failure = pipeline_result.get("hard_failure", False)

            if not processing_success:
                if is_hard_failure:
                    # Hard failures (validation errors, corrupted files, etc.) should commit
                    # to move past poison pill messages
                    # TODO: Implement Dead Letter Queue (DLQ) for hard failures
                    # Hard failures should be sent to DLQ for manual review/reprocessing
                    logger.warning(
                        "🔴 HARD FAILURE - Kafka offset WILL BE COMMITTED, message will NOT retry.",
                        request_id=request.request_id,
                        project_id=request.project_id,
                        upload_id=request.upload_id,
                        partition=metadata.get("partition") if metadata else None,
                        offset=metadata.get("offset") if metadata else None,
                        location=request.location,
                        error_message=pipeline_result.get("error_message", "Unknown error"),
                    )
                    # Don't raise - allow commit to move past this poison pill message
                else:
                    # Soft failures (network issues, timeouts, etc.) should NOT commit
                    # to allow retries
                    error_msg = f"Document processing failed for {request.location}"
                    logger.error(
                        "🟡 SOFT FAILURE - Kafka offset will NOT be committed, message WILL RETRY. This is a transient error.",
                        request_id=request.request_id,
                        project_id=request.project_id,
                        upload_id=request.upload_id,
                        partition=metadata.get("partition") if metadata else None,
                        offset=metadata.get("offset") if metadata else None,
                        error_message=pipeline_result.get("error_message", "Unknown error"),
                    )
                    # Raise to prevent commit - message will be retried
                    raise RuntimeError(error_msg)

            # ProcessingOrchestrator already handled all status updates
            logger.info(
                "Document processing pipeline completed successfully",
                request_id=request.request_id,
                success=processing_success,
            )

        except Exception as e:
            logger.error(
                "Failed to process Kafka document request",
                request_id=request.request_id,
                project_id=request.project_id,
                upload_id=request.upload_id,
                partition=metadata.get("partition") if metadata else None,
                offset=metadata.get("offset") if metadata else None,
                error=str(e),
                error_type=type(e).__name__,
                exc_info=True,
            )
            # Re-raise to prevent Kafka commit - message will be retried
            # DocumentProcessingService should have already recorded failure status
            raise
