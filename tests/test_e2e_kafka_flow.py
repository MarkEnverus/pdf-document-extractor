"""
End-to-End Integration Tests for Complete Kafka Processing Flow.

These tests simulate the complete flow:
1. Incoming Kafka message → KafkaMessageHandler → ProcessingOrchestrator
2. Status updates via StatusTracker → Database + Kafka publishing
3. Document processing via orchestrator
4. Success/Failure flows with appropriate status updates and Kafka messages
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from uuid import uuid4, UUID
from datetime import datetime, timezone
from typing import Dict, Any

from src.services.kafka_message_handler import KafkaMessageHandler
from src.models.processing_models import IncomingDocumentProcessingRequest
from idp_pipeline_events import PipelineEvent
from idp_ingestion_status.models.ingestion_status import IngestionStepStatus
from src.models.status_models import IngestionStepStatus as LocalIngestionStepStatus


@pytest.fixture(autouse=True)
def mock_database_service():
    """Mock the database service for all tests in this module."""
    with patch('src.services.status_tracker.ingestion_status_service') as mock_status:
        mock_status.upsert_file_status = AsyncMock(return_value=None)
        yield mock_status


@pytest.fixture
def sample_incoming_request():
    """Sample incoming document processing request."""
    request_uuid = str(uuid4())
    project_id = str(uuid4())
    upload_id = str(uuid4())
    request_id = str(uuid4())

    return IncomingDocumentProcessingRequest(
        uuid=request_uuid,
        request_id=request_id,
        project_id=project_id,
        upload_id=upload_id,
        location="s3://maestro-raw-dev/projects/3fa85f64-5717-4562-b3fc-2c963f66afa6/sample_document.pdf",
        file_type="application/pdf",
    )


@pytest.fixture
def successful_processing_result():
    """Mock successful processing result from orchestrator."""
    return {
        "success": True,
        "extraction_s3_path": "s3://genai-proprietary-data-extract-dev/3fa85f64/test.json",
        "hard_failure": False,
    }


@pytest.fixture
def failed_processing_result():
    """Mock failed processing result (soft failure - should retry)."""
    return {
        "success": False,
        "error_message": "Network timeout connecting to Docling service",
        "hard_failure": False,
    }


@pytest.fixture
def hard_failure_processing_result():
    """Mock hard failure result (validation error - should not retry)."""
    return {
        "success": False,
        "error_message": "Invalid document format: File is corrupted",
        "hard_failure": True,
    }


class TestE2ESuccessfulFlow:
    """Test successful end-to-end processing flows."""

    @pytest.mark.asyncio
    async def test_successful_document_processing_flow(
        self, sample_incoming_request: IncomingDocumentProcessingRequest, successful_processing_result: Dict[str, Any]
    ) -> None:
        """Test successful end-to-end document processing."""
        # Arrange - Mock dependencies at service boundaries
        mock_status_tracker = Mock()
        mock_status_tracker.update_status = AsyncMock(return_value=True)

        mock_orchestrator = Mock()
        mock_orchestrator.process_documents = AsyncMock(return_value=successful_processing_result)

        mock_extraction_client = Mock()

        handler = KafkaMessageHandler(
            status_tracker=mock_status_tracker,
            orchestrator=mock_orchestrator,
            extraction_client=mock_extraction_client,
        )

        # Act
        await handler.handle_document_processing_request(sample_incoming_request)

        # Assert - Verify orchestrator was called
        mock_orchestrator.process_documents.assert_called_once()
        call_args = mock_orchestrator.process_documents.call_args[1]
        assert call_args["strategy"] == "docling"
        assert call_args["request_id"] == sample_incoming_request.request_id
        assert call_args["project_id"] == sample_incoming_request.project_id
        assert call_args["upload_id"] == sample_incoming_request.upload_id
        assert call_args["document_urls"] == [sample_incoming_request.location]
        assert call_args["source"] == "kafka"

        # Assert - Verify status updates were made
        # Handler calls update_status once for QUEUED
        # Orchestrator calls update_status for PROCESSING and SUCCESS (not tested here as it's mocked)
        assert mock_status_tracker.update_status.call_count >= 1

        # Verify QUEUED status
        first_call = mock_status_tracker.update_status.call_args_list[0][1]
        assert first_call["upload_id"] == sample_incoming_request.upload_id
        assert first_call["status"] == LocalIngestionStepStatus.QUEUED


class TestE2EFailureScenarios:
    """Test failure scenarios in end-to-end processing."""

    @pytest.mark.asyncio
    async def test_soft_failure_raises_exception(
        self, sample_incoming_request: IncomingDocumentProcessingRequest, failed_processing_result: Dict[str, Any]
    ) -> None:
        """Test that soft failures (network issues) raise exceptions to prevent Kafka commit."""
        # Arrange
        mock_status_tracker = Mock()
        mock_status_tracker.update_status = AsyncMock(return_value=True)

        mock_orchestrator = Mock()
        mock_orchestrator.process_documents = AsyncMock(return_value=failed_processing_result)

        mock_extraction_client = Mock()

        handler = KafkaMessageHandler(
            status_tracker=mock_status_tracker,
            orchestrator=mock_orchestrator,
            extraction_client=mock_extraction_client,
        )

        # Act & Assert - Expect exception to be raised (prevents Kafka commit)
        with pytest.raises(Exception) as exc_info:
            await handler.handle_document_processing_request(sample_incoming_request)

        # Verify error message
        assert "Document processing failed" in str(exc_info.value)

        # Verify orchestrator was called
        mock_orchestrator.process_documents.assert_called_once()

    @pytest.mark.asyncio
    async def test_hard_failure_does_not_raise(
        self, sample_incoming_request: IncomingDocumentProcessingRequest, hard_failure_processing_result: Dict[str, Any]
    ) -> None:
        """Test that hard failures (validation errors) do NOT raise to allow Kafka commit."""
        # Arrange
        mock_status_tracker = Mock()
        mock_status_tracker.update_status = AsyncMock(return_value=True)

        mock_orchestrator = Mock()
        mock_orchestrator.process_documents = AsyncMock(return_value=hard_failure_processing_result)

        mock_extraction_client = Mock()

        handler = KafkaMessageHandler(
            status_tracker=mock_status_tracker,
            orchestrator=mock_orchestrator,
            extraction_client=mock_extraction_client,
        )

        # Act - Should complete without raising (allows Kafka commit)
        await handler.handle_document_processing_request(sample_incoming_request)

        # Assert - Verify orchestrator was called
        mock_orchestrator.process_documents.assert_called_once()

        # Verify status tracker was called (QUEUED status)
        assert mock_status_tracker.update_status.call_count >= 1

    @pytest.mark.asyncio
    async def test_status_update_failure_exits_early(
        self, sample_incoming_request: IncomingDocumentProcessingRequest
    ) -> None:
        """Test that handler exits early if initial status update fails."""
        # Arrange
        mock_status_tracker = Mock()
        mock_status_tracker.update_status = AsyncMock(return_value=False)  # Status update fails

        mock_orchestrator = Mock()
        mock_orchestrator.process_documents = AsyncMock()

        mock_extraction_client = Mock()

        handler = KafkaMessageHandler(
            status_tracker=mock_status_tracker,
            orchestrator=mock_orchestrator,
            extraction_client=mock_extraction_client,
        )

        # Act
        await handler.handle_document_processing_request(sample_incoming_request)

        # Assert - Orchestrator should NOT be called if status update fails
        mock_orchestrator.process_documents.assert_not_called()

        # Status tracker should have been called once (QUEUED)
        mock_status_tracker.update_status.assert_called_once()


class TestKafkaMessageFormatValidation:
    """Test Kafka message format validation."""

    def test_kafka_message_format_validation(self):
        """
        Test that PipelineEvent format matches production requirements.

        This validates our outbound Kafka messages for successful processing.
        """
        # Test data matching exact production format
        production_format_data = {
            "uuid": "c5ffadcd-6698-418b-ba08-2028626b638f",
            "request_id": "c5ffadcd-6698-418b-ba08-2028626b638f",
            "project_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
            "upload_id": "c5ffadcd-6698-418b-ba08-2028626b638f",
            "location": "s3://genai-proprietary-data-extract-dev/3fa85f64-5717-4562-b3fc-2c963f66afa6/c5ffadcd-6698-418b-ba08-2028626b638f",
            "file_type": "application/pdf",
        }

        # Verify PipelineEvent can parse the production format
        message = PipelineEvent(**production_format_data)

        # Verify round-trip serialization maintains format
        serialized = message.model_dump(mode="json")

        # Check all required fields are present
        required_fields = {
            "uuid",
            "request_id",
            "project_id",
            "upload_id",
            "location",
            "file_type",
        }
        assert set(serialized.keys()) >= required_fields

        # Verify exact values are maintained
        assert serialized["request_id"] == production_format_data["request_id"]
        assert serialized["project_id"] == production_format_data["project_id"]
        assert serialized["upload_id"] == production_format_data["upload_id"]
        assert serialized["location"] == production_format_data["location"]
        assert serialized["file_type"] == production_format_data["file_type"]

        # Verify S3 location validation works
        with pytest.raises(ValueError):
            PipelineEvent(
                uuid="test",
                project_id="test",
                upload_id="test",
                location="/invalid/path",  # Invalid - not S3
                file_type="application/pdf",
            )


class TestStatusTrackerIntegration:
    """Test StatusTracker integration and failure handling."""

    @pytest.mark.asyncio
    async def test_status_tracker_database_updates(
        self, sample_incoming_request: IncomingDocumentProcessingRequest, successful_processing_result: Dict[str, Any], mock_database_service
    ) -> None:
        """Test that StatusTracker properly updates database via ingestion_status_service."""
        from src.services.status_tracker import StatusTracker

        # Arrange - Create real StatusTracker (with mocked database service via autouse fixture)
        status_tracker = StatusTracker()

        # Mock Kafka initialization to avoid actual Kafka connection
        with patch.object(status_tracker, '_kafka_initialized', False):
            with patch.object(status_tracker, '_kafka', None):
                # Create mocked orchestrator
                mock_orchestrator = Mock()
                mock_orchestrator.process_documents = AsyncMock(return_value=successful_processing_result)

                mock_extraction_client = Mock()

                handler = KafkaMessageHandler(
                    status_tracker=status_tracker,
                    orchestrator=mock_orchestrator,
                    extraction_client=mock_extraction_client,
                )

                # Act
                await handler.handle_document_processing_request(sample_incoming_request)

                # Assert - Verify database service was called
                # Should have upsert calls (QUEUED + other statuses handled by orchestrator)
                assert mock_database_service.upsert_file_status.call_count >= 1

                # Verify QUEUED status used upsert_file_status
                # Note: upsert_file_status signature is (upload_id, file_status), so file_status is second arg [1]
                upsert_call = mock_database_service.upsert_file_status.call_args_list[0][0][1]
                assert upsert_call.file_upload_id == UUID(sample_incoming_request.upload_id)
                assert upsert_call.status == IngestionStepStatus.QUEUED


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
