"""
Simple processing orchestrator - executes user-specified strategy, nothing more.
"""

import time
import threading
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

from lib.models.mime_type_utils import MimeTypeUtils

from src.interfaces.document_processor import DocumentProcessingStrategy
from lib.logger import Logger
from src.models.status_models import IngestionStepStatus
from src.services.status_tracker import StatusTracker

logger = Logger.get_logger(__name__)


@dataclass
class ActiveUploadTracking:
    """Track active upload processing for heartbeat updates."""

    upload_id: str
    project_id: str
    request_id: str
    strategy: str
    document_urls: List[str]
    start_time: float
    file_type: str
    current_doc_index: int = 0
    total_docs: int = 0


class ProcessingOrchestrator:
    """
    Simple document processing orchestrator.

    User says "docling" or "api", we execute it. That's it.
    """

    def __init__(
        self,
        docling_processor: DocumentProcessingStrategy,
        api_processor: DocumentProcessingStrategy,
        status_tracker: StatusTracker,
    ):
        """
        Initialize ProcessingOrchestrator.

        Args:
            docling_processor: Docling processing strategy
            api_processor: API processing strategy
            status_tracker: Status tracker for heartbeat updates
        """
        self.processors = {"docling": docling_processor, "api": api_processor}
        self.status_tracker = status_tracker
        self.logger = logger

        # Tracking structures for heartbeat updates
        self._active_uploads: Dict[str, ActiveUploadTracking] = {}
        self._tracking_lock = threading.Lock()
        self._heartbeat_running = True

        # Start background heartbeat worker thread
        self._start_heartbeat_worker()

    async def process_documents(
        self, strategy: str, request_id: str, project_id: str, upload_id: str, document_urls: List[str], source: str
    ) -> Dict[str, Any]:
        """
        Process documents using specified strategy.

        Args:
            strategy: "docling" or "api"
            request_id: Request identifier
            project_id: Project identifier
            upload_id: Upload identifier
            document_urls: Document URLs to process
            source: Optional metadata for processors

        Returns:
            Processing results

        Raises:
            ValueError: If strategy is invalid
        """
        if strategy not in self.processors:
            available = list(self.processors.keys())
            raise ValueError(f"Invalid strategy '{strategy}'. Available: {available}")

        self.logger.info(
            f"Processing {len(document_urls)} documents with {strategy} strategy",
            request_id=request_id,
            project_id=project_id,
            upload_id=upload_id,
        )

        # Prepare metadata (no longer need to embed IDs since they're explicit parameters)
        processing_metadata: Dict[str, Any] = {}
        processing_metadata["processing_strategy"] = strategy
        processing_metadata["source"] = source

        # Track processing start for heartbeat updates
        self._track_processing_start(
            upload_id=upload_id,
            project_id=project_id,
            request_id=request_id,
            strategy=strategy,
            document_urls=document_urls,
            processing_metadata=processing_metadata,
        )

        try:
            # Process documents using the selected strategy
            processor = self.processors[strategy]
            return await processor.process_documents(
                request_id, project_id, upload_id, document_urls, processing_metadata
            )
        finally:
            # Always stop tracking, even if processing fails
            self._track_processing_end(upload_id)

    def _start_heartbeat_worker(self) -> None:
        """Start background daemon thread for posting heartbeat updates."""
        heartbeat_thread = threading.Thread(
            target=self._heartbeat_worker, daemon=True, name="ProcessingOrchestratorHeartbeat"
        )
        heartbeat_thread.start()
        self.logger.info(
            "Started heartbeat worker thread",
            thread_name="ProcessingOrchestratorHeartbeat",
            heartbeat_interval_seconds=120,
        )

    def _heartbeat_worker(self) -> None:
        """Background thread that posts heartbeat updates every 2 minutes."""
        while self._heartbeat_running:
            try:
                # Sleep for 2 minutes (120 seconds)
                time.sleep(120)

                # Post heartbeat updates if there are active uploads
                with self._tracking_lock:
                    if self._active_uploads:
                        self._post_heartbeat_updates()

            except Exception as e:
                self.logger.error("Error in heartbeat worker thread", error=str(e), error_type=type(e).__name__)
                # Continue running even if one iteration fails
                continue

    def _post_heartbeat_updates(self) -> None:
        """
        Log heartbeat progress for monitoring (does NOT update status API).

        Logs queue summary and per-upload progress to INFO level for monitoring.
        Status API is only updated when status actually changes (QUEUED → PROCESSING → SUCCESS/FAILURE).

        Must be called with _tracking_lock held.
        """
        try:
            active_count = len(self._active_uploads)

            # Log queue summary for monitoring
            self.logger.info(
                f"Processing orchestrator heartbeat: {active_count} upload(s) actively processing",
                active_uploads=active_count,
                heartbeat_type="queue_summary",
            )

            # Post per-upload heartbeat for each active upload
            for upload_id, tracking in self._active_uploads.items():
                elapsed_time = self._get_elapsed_time_str(tracking.start_time)

                # Format heartbeat message
                message = (
                    f"Processing via {tracking.strategy} | {elapsed_time} | "
                    f"Doc {tracking.current_doc_index} of {tracking.total_docs} | "
                    f"Type: {tracking.file_type}"
                )

                # Log heartbeat for monitoring (don't update status API)
                self.logger.info(
                    f"Heartbeat: Been processing for {elapsed_time}",
                    upload_id=tracking.upload_id,
                    project_id=tracking.project_id,
                    request_id=tracking.request_id,
                    strategy=tracking.strategy,
                    elapsed_time=elapsed_time,
                    file_type=tracking.file_type,
                    progress=f"{tracking.current_doc_index}/{tracking.total_docs}",
                    details=message,
                )

        except Exception as e:
            self.logger.error("Failed to post heartbeat updates", error=str(e), error_type=type(e).__name__)

    def _track_processing_start(
        self,
        upload_id: str,
        project_id: str,
        request_id: str,
        strategy: str,
        document_urls: List[str],
        processing_metadata: Dict[str, Any],
    ) -> None:
        """Record when upload_id starts processing."""
        # Determine file type from first document URL
        provided_file_type = processing_metadata.get("file_type") if processing_metadata else None
        file_type = MimeTypeUtils.determine_mime_type(document_urls[0] if document_urls else "", provided_file_type)

        tracking = ActiveUploadTracking(
            upload_id=upload_id,
            project_id=project_id,
            request_id=request_id,
            strategy=strategy,
            document_urls=document_urls,
            start_time=time.time(),
            file_type=file_type,
            current_doc_index=1,  # Start at 1 for user-friendly display
            total_docs=len(document_urls),
        )

        with self._tracking_lock:
            self._active_uploads[upload_id] = tracking

        self.logger.info(
            "Started tracking upload processing",
            upload_id=upload_id,
            project_id=project_id,
            request_id=request_id,
            strategy=strategy,
            document_count=len(document_urls),
            file_type=file_type,
        )

    def _track_processing_end(self, upload_id: str) -> None:
        """Remove upload_id from tracking when processing completes."""
        with self._tracking_lock:
            if upload_id in self._active_uploads:
                tracking = self._active_uploads.pop(upload_id)
                elapsed_time = self._get_elapsed_time_str(tracking.start_time)

                self.logger.info(
                    "Stopped tracking upload processing",
                    upload_id=upload_id,
                    project_id=tracking.project_id,
                    request_id=tracking.request_id,
                    strategy=tracking.strategy,
                    elapsed_time=elapsed_time,
                )

    def _get_elapsed_time_str(self, start_time: float) -> str:
        """Format elapsed time as human-readable string (e.g., '5m 30s')."""
        elapsed_seconds = int(time.time() - start_time)
        minutes = elapsed_seconds // 60
        seconds = elapsed_seconds % 60

        if minutes > 0:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds}s"

    def shutdown(self) -> None:
        """Stop heartbeat worker thread gracefully."""
        self.logger.info("Shutting down ProcessingOrchestrator heartbeat worker")
        self._heartbeat_running = False

    def add_strategy(self, name: str, processor: DocumentProcessingStrategy) -> None:
        """Add a new processing strategy."""
        self.processors[name] = processor

    def get_available_strategies(self) -> List[str]:
        """Get list of available strategy names."""
        return list(self.processors.keys())

    def health_check(self) -> Dict[str, Any]:
        """
        Check health of processing orchestrator and all strategies.

        Returns:
            Dictionary with comprehensive health information
        """
        try:
            health: Dict[str, Any] = {
                "processing_orchestrator": "healthy",
                "available_strategies": list(self.processors.keys()),
            }

            # Check each strategy processor health
            strategy_health: Dict[str, Any] = {}
            all_strategies_healthy = True

            for strategy_name, processor in self.processors.items():
                try:
                    if hasattr(processor, "health_check"):
                        processor_health = processor.health_check()
                        strategy_health[strategy_name] = processor_health

                        # Check if this strategy is healthy
                        if not (
                            processor_health.get(f"{strategy_name}_strategy_processor") == "healthy"
                            or processor_health.get(f"{strategy_name}_extraction_processor") == "healthy"
                        ):
                            all_strategies_healthy = False
                    else:
                        strategy_health[strategy_name] = {"status": "no_health_check"}
                except Exception:
                    strategy_health[strategy_name] = {"status": "unhealthy"}
                    all_strategies_healthy = False

            health["strategies"] = strategy_health
            health["overall_health"] = "healthy" if all_strategies_healthy else "degraded"

            return health

        except Exception as e:
            self.logger.error("Orchestrator health check failed", error=str(e))
            return {"processing_orchestrator": "unhealthy", "error": str(e), "overall_health": "unhealthy"}
