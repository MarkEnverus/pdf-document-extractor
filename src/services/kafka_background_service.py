# Copyright 2025 by Enverus. All rights reserved.

"""
Background Kafka consumer service for the FastAPI application.

This service manages the Kafka consumer using the shared idp_kafka library (async Kafka).
"""

import asyncio
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Optional

from idp_kafka import Kafka
from lib.logger import Logger
from src.configs.settings import settings
from src.models.processing_models import IncomingDocumentProcessingRequest
from src.services.kafka_message_handler import KafkaMessageHandler
from src.utils.critical_failure_handler import CriticalFailureHandler

logger = Logger.get_logger(__name__)


class KafkaBackgroundService:
    """
    Background service to run Kafka consumer alongside FastAPI.

    This service uses async Kafka from the shared library with KafkaMessageHandler
    for message processing, ensuring complete consistency with FastAPI endpoints.

    Uses a queue + workers pattern:
    - Consume loop polls Kafka continuously (keeps consumer alive)
    - Worker tasks process messages with semaphore-controlled concurrency
    - Messages are committed after successful processing (at-least-once delivery)
    """

    def __init__(
        self,
        kafka: Kafka,
        message_handler: Any,
        message_queue: asyncio.Queue[tuple[str, dict[str, Any], dict[str, Any]]],
        processing_semaphore: asyncio.Semaphore,
    ):
        """
        Initialize Kafka background service.

        Args:
            kafka: Async Kafka instance
            message_handler: Async message handler function
            message_queue: Queue for buffering messages between consume and process (topic, message, metadata)
            processing_semaphore: Semaphore to limit concurrent processing
        """
        self._kafka = kafka
        self._message_handler = message_handler
        self._message_queue = message_queue
        self._processing_semaphore = processing_semaphore
        self._consumer_task: Optional[asyncio.Task[None]] = None
        self._worker_tasks: list[asyncio.Task[None]] = []
        self.running = False

    async def _consume_loop(self) -> None:
        """
        Consume loop - polls Kafka continuously and queues messages.

        This loop runs fast and keeps the consumer alive, preventing rebalances
        even when message processing takes a long time.
        """
        try:
            logger.info("Starting Kafka consume loop", service="kafka_background")

            async def queue_message(topic: str, message: dict[str, Any], metadata: dict[str, Any]) -> None:
                """Handler that queues messages for processing."""
                await self._message_queue.put((topic, message, metadata))
                logger.info(
                    "Message queued for processing",
                    topic=topic,
                    partition=metadata.get("partition"),
                    offset=metadata.get("offset"),
                    queue_size=self._message_queue.qsize(),
                    service="kafka_background",
                )

            await self._kafka.consume(queue_message)

        except Exception as e:
            logger.error(
                "Kafka consume loop failed",
                error=str(e),
                error_type=type(e).__name__,
                service="kafka_background",
                exc_info=True,
            )
            raise

    async def _worker_loop(self, worker_id: int) -> None:
        """
        Worker loop - processes messages from queue with semaphore-controlled concurrency.

        Args:
            worker_id: ID of this worker for logging
        """
        logger.info(f"Starting worker {worker_id}", service="kafka_background")

        try:
            while self.running:
                try:
                    # Get message from queue (with timeout to allow graceful shutdown)
                    topic, message, metadata = await asyncio.wait_for(self._message_queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue  # Check running flag and retry

                # Process with semaphore to limit concurrency
                async with self._processing_semaphore:
                    # Extract context for logging (before processing to have it for all logs)
                    upload_id = message.get("upload_id", "unknown")
                    project_id = message.get("project_id", "unknown")
                    partition = metadata.get("partition")
                    offset = metadata.get("offset")

                    try:
                        logger.debug(
                            f"Worker {worker_id} processing message",
                            topic=topic,
                            upload_id=upload_id,
                            project_id=project_id,
                            partition=partition,
                            offset=offset,
                            service="kafka_background",
                        )

                        await self._message_handler(topic, message, metadata)

                        # Commit offset after successful processing
                        try:
                            await self._kafka.commit()
                            logger.info(
                                "✅ Kafka offset committed successfully - message processing complete",
                                worker_id=worker_id,
                                upload_id=upload_id,
                                project_id=project_id,
                                topic=topic,
                                partition=partition,
                                offset=offset,
                                service="kafka_background",
                            )
                        except Exception as commit_error:
                            logger.error(
                                "❌ Kafka offset commit FAILED - offset will retry on next poll",
                                worker_id=worker_id,
                                upload_id=upload_id,
                                project_id=project_id,
                                topic=topic,
                                partition=partition,
                                offset=offset,
                                error=str(commit_error),
                                error_type=type(commit_error).__name__,
                            )
                            # Continue processing - commit failure shouldn't stop worker

                    except Exception as e:
                        logger.error(
                            f"Worker {worker_id} message processing failed - Kafka offset NOT committed, message will retry",
                            upload_id=upload_id,
                            project_id=project_id,
                            topic=topic,
                            partition=partition,
                            offset=offset,
                            error=str(e),
                            error_type=type(e).__name__,
                            service="kafka_background",
                            exc_info=True,
                        )
                        # Don't re-raise - log and continue to next message

                    finally:
                        self._message_queue.task_done()

        except Exception as e:
            logger.error(
                f"Worker {worker_id} loop failed",
                error=str(e),
                error_type=type(e).__name__,
                service="kafka_background",
                exc_info=True,
            )

        logger.info(f"Worker {worker_id} stopped", service="kafka_background")

    async def start(self) -> None:
        """
        Start the Kafka consumer background service.

        Starts both the consume loop (polls Kafka) and worker tasks (process messages).
        """
        if self.running:
            logger.warning("Kafka background service is already running", service="kafka_background")
            return

        logger.info(
            "Starting Kafka background consumer service",
            max_concurrent_processing=settings.MAX_CONCURRENT_PROCESSING,
            service="kafka_background",
        )

        try:
            # Start consumer
            await self._kafka.start_consumer(topics=[settings.KAFKA_INGEST_TOPIC])

            # Set running flag before starting tasks
            self.running = True

            # Start consume loop task (polls Kafka and queues messages)
            self._consumer_task = asyncio.create_task(self._consume_loop())

            # Start worker tasks (process messages with concurrency control)
            num_workers = settings.MAX_CONCURRENT_PROCESSING
            self._worker_tasks = [asyncio.create_task(self._worker_loop(worker_id=i)) for i in range(num_workers)]

            logger.info(
                "Kafka background service started successfully",
                topic=settings.KAFKA_INGEST_TOPIC,
                num_workers=num_workers,
                service="kafka_background",
            )

        except Exception as e:
            logger.error(
                "Failed to start Kafka background service",
                error=str(e),
                error_type=type(e).__name__,
                service="kafka_background",
            )
            await self.stop()

            # Check if this is a critical failure that should trigger shutdown
            if CriticalFailureHandler.is_critical_error(e) or settings.ENVIRONMENT_NAME != "test":
                await CriticalFailureHandler.handle_async_critical_failure(
                    e, "KafkaBackgroundService", "Service startup failed"
                )

    async def stop(self) -> None:
        """
        Stop the Kafka background service gracefully.

        Waits for in-flight messages to complete processing before shutting down.
        """
        if not self.running:
            return

        logger.info("Stopping Kafka background service", service="kafka_background")

        try:
            # Set running flag to false to signal workers to stop
            self.running = False

            # Cancel consumer task (stops polling new messages)
            if self._consumer_task:
                self._consumer_task.cancel()
                try:
                    await self._consumer_task
                except asyncio.CancelledError:
                    logger.info("Kafka consumer task cancelled", service="kafka_background")

            # Wait for queue to drain (with timeout)
            try:
                await asyncio.wait_for(self._message_queue.join(), timeout=30.0)
                logger.info("Message queue drained", service="kafka_background")
            except asyncio.TimeoutError:
                logger.warning(
                    "Timeout waiting for queue to drain",
                    queue_size=self._message_queue.qsize(),
                    service="kafka_background",
                )

            # Wait for all worker tasks to complete
            if self._worker_tasks:
                logger.info(
                    f"Waiting for {len(self._worker_tasks)} workers to complete",
                    service="kafka_background",
                )
                await asyncio.gather(*self._worker_tasks, return_exceptions=True)
                logger.info("All workers stopped", service="kafka_background")

            # Stop consumer and producer
            await self._kafka.stop_consumer()
            await self._kafka.stop_producer()

            logger.info("Kafka background service stopped", service="kafka_background")

        except Exception as e:
            logger.error(
                "Error stopping Kafka background service",
                error=str(e),
                service="kafka_background",
            )

    def is_healthy(self) -> bool:
        """Check if the background service is healthy."""
        return self.running and self._kafka is not None

    def get_status(self) -> dict[str, Any]:
        """Get status information about the background service."""
        return {
            "running": self.running,
            "consumer_initialized": self._kafka is not None,
            "topic": settings.KAFKA_INGEST_TOPIC if self.running else None,
        }


async def create_kafka_background_service() -> KafkaBackgroundService:
    """
    Create Kafka background service with dependency injection.

    Uses async Kafka with queue + workers pattern for concurrent processing.
    """
    from src.dependencies.providers import get_kafka_message_handler

    kafka_config = settings.get_kafka_config()
    message_handler_wrapper = get_kafka_message_handler()

    # Create message queue and semaphore for concurrency control
    message_queue: asyncio.Queue[tuple[str, dict[str, Any], dict[str, Any]]] = asyncio.Queue(maxsize=100)
    processing_semaphore = asyncio.Semaphore(settings.MAX_CONCURRENT_PROCESSING)

    # Create async Kafka instance for consumer
    kafka = Kafka(kafka_config)
    await kafka.start_producer()

    # Create async message handler
    async def async_message_handler(topic: str, message: dict[str, Any], metadata: dict[str, Any]) -> None:
        """Async message handler with manual commits (enable_auto_commit=False)."""
        try:
            # Parse and validate message
            request = IncomingDocumentProcessingRequest.model_validate(message)

            # Process message using async handler
            await message_handler_wrapper.handle_document_processing_request(request, metadata)

            logger.info(
                "Message processed successfully",
                upload_id=request.upload_id,
                project_id=request.project_id,
                partition=metadata.get("partition"),
                offset=metadata.get("offset"),
                service="kafka_background",
            )

        except Exception as e:
            # Log detailed error information including full message for debugging
            logger.error(
                "Message processing failed - offset will NOT be committed, message will retry",
                error=str(e),
                error_type=type(e).__name__,
                topic=topic,
                service="kafka_background",
            )
            # Re-raise to prevent commit - message will be retried
            raise

    return KafkaBackgroundService(kafka, async_message_handler, message_queue, processing_semaphore)


@asynccontextmanager
async def kafka_lifespan(app: Any) -> AsyncGenerator[None, None]:
    """
    FastAPI lifespan context manager to handle Kafka background service.

    Stores service instance in app.state for clean dependency injection.

    Usage:
        from src.services.kafka_background_service import kafka_lifespan
        app = FastAPI(lifespan=kafka_lifespan)
    """
    # Startup - create service with dependency injection
    logger.info("Application startup - initializing Kafka background service")
    service = await create_kafka_background_service()

    # Store in FastAPI app state
    app.state.kafka_background_service = service
    await service.start()

    yield

    # Shutdown
    logger.info("Application shutdown - stopping Kafka background service")
    if hasattr(app.state, "kafka_background_service"):
        await app.state.kafka_background_service.stop()


async def health_check(kafka_service: Optional[KafkaBackgroundService] = None) -> dict[str, Any]:
    """
    Health check for Kafka background service.

    Args:
        kafka_service: The KafkaBackgroundService instance to check

    Returns:
        Dictionary with health status information
    """
    if kafka_service is None:
        return {
            "kafka_background_service": {
                "status": "not_initialized",
                "details": {"running": False, "consumer_initialized": False},
            }
        }

    status = kafka_service.get_status()

    return {
        "kafka_background_service": {
            "status": "healthy" if kafka_service.is_healthy() else "unhealthy",
            "details": status,
        }
    }
