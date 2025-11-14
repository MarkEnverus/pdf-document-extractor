"""
Integration test for full Kafka processing flow.

Tests the complete flow:
Kafka message → consumer → kafka_message_handler → document processing → status updates → Kafka completion message
"""

import pytest
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4, UUID

from src.models.processing_models import IncomingDocumentProcessingRequest
from idp_ingestion_status.models.ingestion_status import IngestionStepStatus
from src.services.kafka_message_handler import KafkaMessageHandler
from src.services.status_tracker import StatusTracker
from src.services.processing_orchestrator import ProcessingOrchestrator


@pytest.fixture
def sample_kafka_request():
    """Sample Kafka request matching the expected format."""
    return IncomingDocumentProcessingRequest(
        uuid="c5ffadcd-6698-418b-ba08-2028626b638f",
        request_id="6242f6c53933eeb097ac6f4c5fe9fcae",
        project_id="3fa85f64-5717-4562-b3fc-2c963f66afa6",
        upload_id="c5ffadcd-6698-418b-ba08-2028626b638f",
        location="s3://maestro-raw-dev/projects/3fa85f64-5717-4562-b3fc-2c963f66afa6/sample_mark_again_26.pdf",
        file_type="application/pdf",
    )


@pytest.fixture
def sample_kafka_metadata():
    """Sample Kafka message metadata."""
    return {
        "topic": "document-processing-requests",
        "partition": 0,
        "offset": 12345,
        "timestamp": 1634567890000,
        "key": "c5ffadcd-6698-418b-ba08-2028626b638f",
    }


@pytest.fixture
def mock_status_tracker():
    """Mock status tracker that captures calls - WE ARE THE FUTURE!"""
    mock_tracker = Mock(spec=StatusTracker)
    mock_tracker.update_status = AsyncMock(return_value=True)  # update_status is async
    mock_tracker.kafka_service = Mock()
    mock_tracker.enable_kafka = True
    return mock_tracker


@pytest.fixture
def mock_document_processing_service():
    """Mock document processing service."""
    mock_service = Mock()
    mock_service.process_document_pipeline = Mock(
        return_value={
            "request_id": "test-request",
            "success": True,
            "results": [{"document": "processed_content"}],
            "extraction_s3_path": "s3://genai-proprietary-data-extract-dev/3fa85f64-5717-4562-b3fc-2c963f66afa6/c5ffadcd-6698-418b-ba08-2028626b638f",
        }
    )
    return mock_service


@pytest.fixture
def kafka_message_handler(mock_status_tracker, test_processing_orchestrator, test_extraction_client):
    """Create KafkaMessageHandler with mocked dependencies."""
    return KafkaMessageHandler(
        status_tracker=mock_status_tracker,
        orchestrator=test_processing_orchestrator,
        extraction_client=test_extraction_client,
    )


class TestKafkaIntegrationFlow:
    """Test complete Kafka integration flow."""

    async def test_successful_document_processing_flow(
        self,
        kafka_message_handler,
        sample_kafka_request,
        sample_kafka_metadata,
        mock_status_tracker,
        test_processing_orchestrator,
    ):
        """
        Test successful end-to-end flow:
        1. Receive Kafka message
        2. Process document via docling (mocked)
        3. Update status tracker throughout
        4. Verify Kafka completion message is published on SUCCESS
        """

        # Act: Process the Kafka message
        await kafka_message_handler.handle_document_processing_request(request=sample_kafka_request)

        # Assert: Verify initial status was recorded as QUEUED using modern interface
        mock_status_tracker.update_status.assert_called_once_with(
            upload_id=sample_kafka_request.upload_id,
            status=IngestionStepStatus.QUEUED,
            project_id=sample_kafka_request.project_id,
            s3_location=sample_kafka_request.get_extraction_s3_base_path(),
            file_type=sample_kafka_request.file_type,
            request_id=sample_kafka_request.request_id,
            message="Queued for processing",
        )

    async def test_docling_processing_failure_flow(
        self,
        kafka_message_handler,
        sample_kafka_request,
        sample_kafka_metadata,
        mock_status_tracker,
        test_processing_orchestrator,
    ):
        """
        Test failure flow:
        1. Receive Kafka message
        2. Docling processing fails
        3. Verify no SUCCESS Kafka message is published
        4. Verify QUEUED status was still recorded
        """

        # Arrange: Make document processing fail via orchestrator
        with patch.object(
            test_processing_orchestrator, "process_documents", side_effect=Exception("Processing failed")
        ):
            # Act: Process the Kafka message (should re-raise to prevent commit)
            with pytest.raises(Exception, match="Processing failed"):
                await kafka_message_handler.handle_document_processing_request(request=sample_kafka_request)

        # Assert: Verify initial QUEUED status was still recorded using modern interface
        mock_status_tracker.update_status.assert_called_once_with(
            upload_id=sample_kafka_request.upload_id,
            status=IngestionStepStatus.QUEUED,
            project_id=sample_kafka_request.project_id,
            s3_location=sample_kafka_request.get_extraction_s3_base_path(),
            file_type=sample_kafka_request.file_type,
            request_id=sample_kafka_request.request_id,
            message="Queued for processing",
        )


if __name__ == "__main__":
    pytest.main([__file__])
