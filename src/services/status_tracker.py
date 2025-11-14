"""
Database-backed status tracking system using ingestion_status_service.

This module provides status tracking through the shared ingestion-status library.
"""

from typing import TYPE_CHECKING, Any, Dict, Optional
from uuid import UUID

from lib.models.ingestion_status import IngestionStatus, IngestionStepFileStatus
from lib.models.ingestion_status import IngestionStepStatus as LibIngestionStepStatus
from lib.logger import Logger
from lib.models.extraction_models import safe_uuid_conversion
from src.configs.settings import settings
from src.interfaces.status_tracker import AbstractStatusTracker
from src.models.status_models import IngestionStepStatus
from src.services.imported_services import ingestion_status_service

if TYPE_CHECKING:
    from idp_kafka import Kafka

logger = Logger.get_logger(__name__)


class StatusTracker(AbstractStatusTracker):
    """
    Database-backed status tracking using ingestion_status_service.

    This class provides status tracking through the shared ingestion-status library
    which stores status updates in PostgreSQL.

    Key Concepts:
    - upload_id: Primary key for status tracking (from Kafka/external system)
    - request_id: Internal correlation ID (for our processing pipeline)
    """

    def __init__(self) -> None:
        """
        Initialize StatusTracker.

        Kafka publishing is always enabled in production and messages are sent only on SUCCESS.
        In tests, Kafka can be disabled via STATUS_ENABLE_KAFKA_PUBLISHING=false for performance.
        """
        try:
            # Initialize Kafka service - messages sent only on SUCCESS
            self._kafka: Optional["Kafka"] = None
            self._kafka_initialized = False

            # Check if Kafka publishing is enabled (mainly for test performance)
            from src.configs.settings import settings

            if settings.STATUS_ENABLE_KAFKA_PUBLISHING:
                # Note: Kafka will be initialized async on first use
                self._kafka_initialized = False
            else:
                logger.info(
                    "Kafka publishing disabled via configuration",
                    kafka_enabled=False,
                    reason="STATUS_ENABLE_KAFKA_PUBLISHING=False",
                )

            logger.info(
                "StatusTracker initialized successfully",
                backend_type="database",
                kafka_enabled=settings.STATUS_ENABLE_KAFKA_PUBLISHING,
                service="status_tracker",
            )

        except Exception as e:
            logger.error(
                "StatusTracker initialization failed",
                error_type=type(e).__name__,
                error_message=str(e),
                service="status_tracker",
            )
            # Don't suppress exceptions in production - let them propagate
            raise

    async def _init_kafka(self) -> None:
        """Initialize Kafka service for publishing status updates."""
        try:
            from idp_kafka import Kafka

            kafka_config = settings.get_kafka_config()
            # Create producer-only async Kafka instance
            self._kafka = Kafka(kafka_config)
            await self._kafka.start_producer()
            self._kafka_initialized = True
            logger.info(
                "Kafka producer initialized for status publishing",
                kafka_producer="initialized",
                service="status_tracker",
            )
        except Exception as e:
            logger.error(
                "CRITICAL: Failed to initialize Kafka producer",
                error_type=type(e).__name__,
                error_message=str(e),
                service="status_tracker",
                criticality="high",
                exc_info=True,
            )
            # Fail-fast in production - Kafka is critical
            if settings.ENVIRONMENT_NAME != "test":
                logger.critical(
                    "Kafka initialization failed in production - service cannot continue",
                    environment=settings.ENVIRONMENT_NAME,
                    service="status_tracker",
                )
                raise RuntimeError(f"Critical failure: Kafka producer initialization failed: {e}") from e

            # In tests, allow graceful degradation
            self._kafka = None
            self._kafka_initialized = False

    async def _publish_extraction_status_to_kafka(
        self,
        upload_id: str,
        status: IngestionStepStatus,
        extraction_s3_location: str | None = None,
        request_id: str | None = None,
        file_type: str = "unknown",
        document_url: str | None = None,
        project_id: str = "default",
        metadata: Dict[str, Any] | None = None,
    ) -> None:
        """Publish pipeline event to Kafka (success only)."""
        # Initialize Kafka if needed
        if not self._kafka_initialized:
            await self._init_kafka()

        if not self._kafka:
            return

        try:
            from lib.models.mime_type import IngestionMimeType
            from lib.models.extraction_models import MimeTypeUtils, PipelineEvent

            # Determine proper MIME type using the helper function
            determined_mime_type_str = MimeTypeUtils.determine_mime_type(
                document_url or extraction_s3_location or "", file_type
            )

            # Convert MIME type string to enum
            determined_mime_type = IngestionMimeType(determined_mime_type_str)

            # Convert string IDs to UUIDs
            upload_uuid = safe_uuid_conversion(upload_id)
            project_uuid = safe_uuid_conversion(project_id)

            # Create PipelineEvent (only published on success)
            message = PipelineEvent(
                uuid=upload_uuid,
                project_id=project_uuid,
                upload_id=upload_uuid,
                location=extraction_s3_location or "",
                file_type=determined_mime_type,
                request_id=request_id,  # Optional, can be None
                metadata=metadata,  # Pass through metadata including extractions array
            )

            # Convert PipelineEvent to dict for publishing
            message_dict = message.model_dump(mode="json")

            # Publish to Kafka using async Kafka
            await self._kafka.produce(
                topic=settings.KAFKA_OUTPUT_TOPIC,
                message=message_dict,
                key=upload_id,  # Use upload_id string as partition key
            )

            logger.info(
                "Pipeline event published to Kafka",
                upload_id=upload_id,
                kafka_topic=settings.KAFKA_OUTPUT_TOPIC,
                status=status.value,
                service="status_tracker",
            )

        except Exception as e:
            logger.error(
                "Failed to publish pipeline event to Kafka",
                upload_id=upload_id,
                kafka_topic=settings.KAFKA_OUTPUT_TOPIC,
                error_type=type(e).__name__,
                error_message=str(e),
                service="status_tracker",
            )

    def health_check(self) -> Dict[str, str]:
        """
        Check health of status tracker components.

        Returns:
            Dictionary with health status of components
        """
        health = {
            "status_tracker": "healthy",
            "storage_backend": "database",
            "kafka_publishing": "enabled" if self._kafka else "disabled",
        }

        # Check database health through ingestion_status_service
        try:
            # The service itself being available means database is configured
            if ingestion_status_service:
                health["database"] = "healthy"
            else:
                health["database"] = "unavailable"
        except Exception as e:
            logger.error("Database health check failed", error=str(e), service="status_tracker")
            health["database"] = "unhealthy"

        # Check Kafka health (if enabled)
        if self._kafka:
            try:
                # Kafka is healthy if it's initialized and not None
                health["kafka"] = "healthy"
            except Exception as e:
                logger.error("Kafka health check failed", error=str(e), service="status_tracker")
                health["kafka"] = "unhealthy"
        else:
            health["kafka"] = "disabled"

        return health

    async def update_status(
        self,
        upload_id: str,  # Primary key for status tracking
        status: IngestionStepStatus,
        project_id: str = "default",
        s3_location: Optional[str] = None,
        file_type: str = "unknown",
        message: Optional[str] = None,
        request_id: Optional[str] = None,  # Internal correlation ID
        document_url: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Update status for document processing with database storage and Kafka publishing.

        Key Design:
        - upload_id: Primary key for status tracking (REQUIRED)
        - request_id: Internal correlation UUID for our pipeline (OPTIONAL)

        Args:
            upload_id: Primary key for status tracking (required for all operations)
            status: Processing status (PROCESSING, SUCCESS, FAILURE, etc.)
            project_id: Project identifier
            s3_location: S3 path to results/content (for SUCCESS status)
            file_type: MIME type of processed content
            message: Human-readable status message (for logging)
            request_id: Internal correlation UUID (for pipeline tracking)
            document_url: Original document URL (for MIME type detection)
            metadata: Additional structured data

        Returns:
            True if status update succeeded, False otherwise

        Example Usage:
            # Basic status update (from Kafka with upload_id)
            tracker.update_status("a2cc5ad8-a583-47f8-be1e-1e6273b06346", IngestionStepStatus.PROCESSING)

            # Internal processing (with both upload_id and request_id UUIDs)
            tracker.update_status(
                upload_id="a2cc5ad8-a583-47f8-be1e-1e6273b06346",
                status=IngestionStepStatus.SUCCESS,
                s3_location="s3://bucket/results.json",
                request_id="b3dd6be8-c774-4e9f-af2f-2f8394c7e847"  # Our internal UUID for correlation
            )
        """
        # Entry point logging - proves function is called
        logger.debug(
            "StatusTracker.update_status called",
            upload_id=upload_id,
            status=status.value,
            project_id=project_id,
            request_id=request_id,
            service="status_tracker",
        )

        # Track overall success - start optimistic
        db_update_succeeded = False
        kafka_publish_succeeded = False

        # Convert IDs upfront (needed for both DB and Kafka)
        upload_uuid = safe_uuid_conversion(upload_id)
        project_uuid = safe_uuid_conversion(project_id) if project_id and project_id != "default" else upload_uuid

        # Try database update first
        try:
            # Map local status enum to library status enum
            lib_status = LibIngestionStepStatus(status.value)

            # Build summary data
            summary = {}
            if s3_location:
                summary["extraction_s3_location"] = s3_location
            if message:
                summary["message"] = message
            if request_id:
                summary["request_id"] = request_id
            if metadata:
                summary.update(metadata)

            # Create IngestionStepFileStatus object
            file_status = IngestionStepFileStatus(
                project_id=project_uuid,
                file_upload_id=upload_uuid,
                step=IngestionStatus.EXTRACTION,
                status=lib_status,
                summary=summary if summary else None,
            )

            # Use upsert_file_status which handles create-or-update internally
            await ingestion_status_service.upsert_file_status(upload_uuid, file_status)
            db_update_succeeded = True

            logger.info(
                "Database status update completed successfully",
                upload_id=upload_id,
                request_id=request_id,
                status=status.value,
                project_id=project_id,
                has_s3_location=bool(s3_location),
                service="status_tracker",
            )

        except Exception as e:
            logger.error(
                "Database status update failed",
                upload_id=upload_id,
                request_id=request_id,
                status=status.value,
                error=str(e),
                error_type=type(e).__name__,
                error_repr=repr(e),
                error_module=type(e).__module__,
                service="status_tracker",
                exc_info=True,
            )

        # Publish to Kafka ONLY on SUCCESS (regardless of DB update status)
        if settings.STATUS_ENABLE_KAFKA_PUBLISHING and status == IngestionStepStatus.SUCCESS:
            try:
                logger.info(
                    "Publishing SUCCESS status to Kafka",
                    upload_id=upload_id,
                    project_id=project_id,
                    request_id=request_id,
                    kafka_topic=settings.KAFKA_OUTPUT_TOPIC,
                    db_update_succeeded=db_update_succeeded,
                    service="status_tracker",
                )
                await self._publish_extraction_status_to_kafka(
                    upload_id=upload_id,
                    status=status,
                    extraction_s3_location=s3_location,
                    request_id=request_id or upload_id,  # Prefer request_id, fallback to upload_id
                    file_type=file_type,
                    document_url=document_url,
                    project_id=project_id,
                    metadata=metadata,  # Pass through metadata including extractions array
                )
                kafka_publish_succeeded = True

            except Exception as e:
                logger.error(
                    "Kafka publish failed",
                    upload_id=upload_id,
                    request_id=request_id,
                    error=str(e),
                    error_type=type(e).__name__,
                    service="status_tracker",
                    exc_info=True,
                )
        else:
            logger.debug(
                "Skipping Kafka publishing for non-SUCCESS status or disabled config",
                upload_id=upload_id,
                status=status.value,
                kafka_enabled=settings.STATUS_ENABLE_KAFKA_PUBLISHING,
                service="status_tracker",
            )

        # Return True if either DB or Kafka succeeded (or if Kafka was skipped intentionally)
        # For non-SUCCESS statuses, only DB update matters
        if status != IngestionStepStatus.SUCCESS:
            return db_update_succeeded
        else:
            # For SUCCESS, we want at least Kafka to succeed (DB is nice-to-have)
            return kafka_publish_succeeded or db_update_succeeded
