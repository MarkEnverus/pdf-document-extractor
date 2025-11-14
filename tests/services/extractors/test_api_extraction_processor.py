"""
Comprehensive tests for ApiExtractionProcessor covering API-based document extraction functionality.
"""

import json
import pytest
from unittest.mock import AsyncMock, Mock, patch

from src.models.extraction_request import ConfigType, ExtractorRequest
from src.services.extractors.api.api_extraction_processor import ApiExtractionProcessor
from src.models.processing_models import IngestionStepStatus
from idp_pipeline_events import ExtractionType


@pytest.fixture
def mock_extraction_client():
    """Mock extraction API client for testing."""
    mock = Mock()

    # Mock successful extraction result
    mock_result = Mock()
    mock_result.success = True
    mock_result.data = {
        "text": "Extracted document text content",
        "content": "Alternative content field",
        "metadata": {"pages": 5}
    }
    mock_result.error = None

    mock.extract = Mock(return_value=mock_result)
    mock.health_check = Mock(return_value=True)

    return mock


@pytest.fixture
def mock_storage_service():
    """Mock storage service for testing."""
    mock = Mock()
    mock.upload_content = Mock(return_value="s3://test-bucket/proj/upload/1/results.json")
    mock.health_check = Mock(return_value=True)
    return mock


@pytest.fixture
def mock_status_tracker():
    """Mock status tracker for testing."""
    mock = Mock()
    mock.update_status = AsyncMock()
    mock.health_check = Mock(return_value={"status_tracker": "healthy"})
    return mock


@pytest.fixture
def api_extraction_processor(mock_extraction_client, mock_storage_service, mock_status_tracker):
    """Create ApiExtractionProcessor instance for testing."""
    return ApiExtractionProcessor(
        extraction_client=mock_extraction_client,
        storage_service=mock_storage_service,
        status_tracker=mock_status_tracker
    )


class TestApiExtractionProcessorInit:
    """Test ApiExtractionProcessor initialization."""

    def test_init_success(self, api_extraction_processor, mock_extraction_client, mock_storage_service, mock_status_tracker):
        """Test successful initialization with all dependencies."""
        assert api_extraction_processor.extraction_client is mock_extraction_client
        assert api_extraction_processor.storage_service is mock_storage_service
        assert api_extraction_processor.status_tracker is mock_status_tracker
        assert api_extraction_processor.logger is not None
        assert api_extraction_processor.strategy_name == "api"


class TestProcessDocuments:
    """Test process_documents method."""

    @pytest.mark.asyncio
    async def test_process_documents_success(self, api_extraction_processor, mock_extraction_client, mock_status_tracker):
        """Test successful document processing via API."""
        document_urls = ["s3://bucket/doc1.pdf"]

        result = await api_extraction_processor.process_documents(
            request_id="req-123",
            project_id="proj-456",
            upload_id="upload-789",
            document_urls=document_urls,
            metadata=None
        )

        assert result["success"] is True
        assert "extraction_s3_path" in result  # processing_data is merged into result

        # Verify status updates were called
        assert mock_status_tracker.update_status.call_count == 2  # PROCESSING + SUCCESS

    @pytest.mark.asyncio
    async def test_process_documents_empty_list(self, api_extraction_processor):
        """Test processing with empty document list."""
        result = await api_extraction_processor.process_documents(
            request_id="req-123",
            project_id="proj-456",
            upload_id="upload-789",
            document_urls=[],
            metadata=None
        )

        assert result["success"] is False
        assert "No documents provided" in result["error_message"]

    @pytest.mark.asyncio
    async def test_process_documents_extraction_failure(self, api_extraction_processor, mock_extraction_client):
        """Test handling extraction API failure."""
        # Mock failed extraction
        mock_failed_result = Mock()
        mock_failed_result.success = False
        mock_failed_result.error = Mock(error_message="API extraction failed")
        mock_failed_result.data = None
        mock_extraction_client.extract.return_value = mock_failed_result

        result = await api_extraction_processor.process_documents(
            request_id="req-123",
            project_id="proj-456",
            upload_id="upload-789",
            document_urls=["s3://bucket/doc1.pdf"],
            metadata=None
        )

        assert result["success"] is False
        assert "API extraction failed" in result["error_message"]

    @pytest.mark.asyncio
    async def test_process_documents_unexpected_exception(self, api_extraction_processor, mock_extraction_client, mock_status_tracker):
        """Test handling unexpected exceptions during processing."""
        mock_extraction_client.extract.side_effect = Exception("Unexpected error")

        result = await api_extraction_processor.process_documents(
            request_id="req-123",
            project_id="proj-456",
            upload_id="upload-789",
            document_urls=["s3://bucket/doc1.pdf"],
            metadata=None
        )

        assert result["success"] is False
        assert "Processing failed" in result["error_message"]

        # Verify FAILURE status was set
        failure_calls = [call for call in mock_status_tracker.update_status.call_args_list
                         if call.kwargs.get("status") == IngestionStepStatus.FAILURE]
        assert len(failure_calls) > 0


class TestCreateExtractionRequest:
    """Test _create_extraction_request method."""

    def test_create_extraction_request_default(self, api_extraction_processor):
        """Test creating extraction request with default configuration."""
        document_urls = ["s3://bucket/doc1.pdf", "s3://bucket/doc2.pdf"]

        request = api_extraction_processor._create_extraction_request(document_urls, metadata=None)

        assert isinstance(request, ExtractorRequest)
        assert request.document_urls == document_urls
        assert request.extraction_mode == "multi"
        assert len(request.extraction_configs) == 1
        assert request.extraction_configs[0].config_type == ConfigType.BEDROCK
        assert request.extraction_configs[0].model_id == "anthropic.claude-3-sonnet-20240229-v1:0"

    def test_create_extraction_request_with_custom_config(self, api_extraction_processor):
        """Test creating extraction request with custom configuration from metadata."""
        document_urls = ["s3://bucket/doc1.pdf"]
        metadata = {
            "extraction_config": {
                "model_id": "custom-model-id",
                "name": "custom-config"
            },
            "extraction_mode": "single"
        }

        request = api_extraction_processor._create_extraction_request(document_urls, metadata)

        assert request.extraction_mode == "single"
        assert request.extraction_configs[0].model_id == "custom-model-id"
        assert request.extraction_configs[0].name == "custom-config"

    def test_create_extraction_request_with_output_schema(self, api_extraction_processor):
        """Test creating extraction request with output schema."""
        document_urls = ["s3://bucket/doc1.pdf"]
        metadata = {
            "output_schema": {"field1": "type1", "field2": "type2"}
        }

        request = api_extraction_processor._create_extraction_request(document_urls, metadata)

        assert request.output_schema == {"field1": "type1", "field2": "type2"}


class TestHandleSuccessfulExtraction:
    """Test _handle_successful_extraction method."""

    @pytest.mark.asyncio
    async def test_handle_successful_extraction(self, api_extraction_processor, mock_status_tracker):
        """Test handling successful extraction results."""
        mock_extraction_result = Mock()
        mock_extraction_result.success = True
        mock_extraction_result.data = {"text": "Extracted content"}

        result = await api_extraction_processor._handle_successful_extraction(
            request_id="req-123",
            document_urls=["s3://bucket/doc.pdf"],
            extraction_result=mock_extraction_result,
            project_id="proj-456",
            upload_id="upload-789"
        )

        assert result["success"] is True
        assert "extraction_s3_path" in result  # processing_data is merged into result

        # Verify SUCCESS status was set
        success_call = mock_status_tracker.update_status.call_args
        assert success_call.kwargs["status"] == IngestionStepStatus.SUCCESS
        assert success_call.kwargs["message"].startswith("Successfully processed")

    @pytest.mark.asyncio
    async def test_handle_successful_extraction_storage_failure(self, api_extraction_processor, mock_storage_service):
        """Test handling storage failure during successful extraction."""
        mock_storage_service.upload_content.side_effect = Exception("S3 upload failed")

        mock_extraction_result = Mock()
        mock_extraction_result.success = True
        mock_extraction_result.data = {"text": "Extracted content"}

        result = await api_extraction_processor._handle_successful_extraction(
            request_id="req-123",
            document_urls=["s3://bucket/doc.pdf"],
            extraction_result=mock_extraction_result,
            project_id="proj-456",
            upload_id="upload-789"
        )

        assert result["success"] is False
        assert "Failed to process extraction results" in result["error_message"]


class TestHandleFailedExtraction:
    """Test _handle_failed_extraction method."""

    @pytest.mark.asyncio
    async def test_handle_failed_extraction(self, api_extraction_processor, mock_status_tracker):
        """Test handling failed extraction results."""
        mock_extraction_result = Mock()
        mock_extraction_result.success = False
        mock_extraction_result.error = Mock(error_message="Document is corrupted")
        mock_extraction_result.data = None

        result = await api_extraction_processor._handle_failed_extraction(
            request_id="req-123",
            document_urls=["s3://bucket/doc.pdf"],
            extraction_result=mock_extraction_result,
            project_id="proj-456",
            upload_id="upload-789"
        )

        assert result["success"] is False
        assert "Document is corrupted" in result["error_message"]

        # Verify FAILURE status was set
        failure_call = mock_status_tracker.update_status.call_args
        assert failure_call.kwargs["status"] == IngestionStepStatus.FAILURE
        assert "API extraction failed" in failure_call.kwargs["message"]

    @pytest.mark.asyncio
    async def test_handle_failed_extraction_no_error_message(self, api_extraction_processor):
        """Test handling failed extraction with no error message."""
        mock_extraction_result = Mock()
        mock_extraction_result.success = False
        mock_extraction_result.error = None

        result = await api_extraction_processor._handle_failed_extraction(
            request_id="req-123",
            document_urls=["s3://bucket/doc.pdf"],
            extraction_result=mock_extraction_result,
            project_id="proj-456",
            upload_id="upload-789"
        )

        assert result["success"] is False
        assert "Unknown extraction error" in result["error_message"]


class TestHandleEmptyDocuments:
    """Test _handle_empty_documents method."""

    @pytest.mark.asyncio
    async def test_handle_empty_documents(self, api_extraction_processor, mock_status_tracker):
        """Test handling empty document list."""
        result = await api_extraction_processor._handle_empty_documents(
            request_id="req-123",
            project_id="proj-456",
            upload_id="upload-789"
        )

        assert result["success"] is True
        assert result["processing_method"] == "api"  # processing_data is merged into result

        # Verify SUCCESS status was set
        success_call = mock_status_tracker.update_status.call_args
        assert success_call.kwargs["status"] == IngestionStepStatus.SUCCESS
        assert "0 documents" in success_call.kwargs["message"]


class TestStoreExtractionResults:
    """Test _store_extraction_results method."""

    @patch("src.configs.settings.settings")
    def test_store_extraction_results_with_text_field(self, mock_settings, api_extraction_processor):
        """Test storing extraction results when 'text' field is present."""
        mock_settings.S3_BUCKET_NAME = "test-bucket"

        mock_extraction_result = Mock()
        mock_extraction_result.data = {"text": "Extracted document text"}

        result = api_extraction_processor._store_extraction_results(
            extraction_result=mock_extraction_result,
            project_id="proj-123",
            upload_id="upload-456"
        )

        assert result == "s3://test-bucket/proj/upload/1/results.json"

        # Verify both results.json and page_text.txt were stored
        assert api_extraction_processor.storage_service.upload_content.call_count == 2

    @patch("src.configs.settings.settings")
    def test_store_extraction_results_with_content_field(self, mock_settings, api_extraction_processor):
        """Test storing extraction results when 'content' field is present (fallback)."""
        mock_settings.S3_BUCKET_NAME = "test-bucket"

        mock_extraction_result = Mock()
        mock_extraction_result.data = {"content": "Alternative content field"}

        result = api_extraction_processor._store_extraction_results(
            extraction_result=mock_extraction_result,
            project_id="proj-123",
            upload_id="upload-456"
        )

        assert result == "s3://test-bucket/proj/upload/1/results.json"

    @patch("src.configs.settings.settings")
    def test_store_extraction_results_json_fallback(self, mock_settings, api_extraction_processor):
        """Test storing extraction results with JSON fallback when no text fields."""
        mock_settings.S3_BUCKET_NAME = "test-bucket"

        mock_extraction_result = Mock()
        mock_extraction_result.data = {"metadata": "some_data", "other_field": "value"}

        result = api_extraction_processor._store_extraction_results(
            extraction_result=mock_extraction_result,
            project_id="proj-123",
            upload_id="upload-456"
        )

        assert result == "s3://test-bucket/proj/upload/1/results.json"

        # Verify content was serialized as JSON
        call_args = api_extraction_processor.storage_service.upload_content.call_args_list
        # First call is results.json, second is page_text.txt
        text_content_call = call_args[1][1]["content"]  # Second call (page_text.txt)
        assert b"metadata" in text_content_call
        assert b"some_data" in text_content_call

    @patch("src.configs.settings.settings")
    def test_store_extraction_results_empty_content_raises_error(self, mock_settings, api_extraction_processor):
        """Test that no data attribute raises ValueError."""
        mock_settings.S3_BUCKET_NAME = "test-bucket"

        mock_extraction_result = Mock(spec=[])  # No 'data' attribute at all

        with pytest.raises(ValueError, match="API extraction returned no content"):
            api_extraction_processor._store_extraction_results(
                extraction_result=mock_extraction_result,
                project_id="proj-123",
                upload_id="upload-456"
            )

    @patch("src.configs.settings.settings")
    def test_store_extraction_results_whitespace_only_raises_error(self, mock_settings, api_extraction_processor):
        """Test that whitespace-only content raises ValueError."""
        mock_settings.S3_BUCKET_NAME = "test-bucket"

        mock_extraction_result = Mock()
        mock_extraction_result.data = {"text": "   \n\t  "}  # Whitespace only

        with pytest.raises(ValueError, match="API extraction returned no content"):
            api_extraction_processor._store_extraction_results(
                extraction_result=mock_extraction_result,
                project_id="proj-123",
                upload_id="upload-456"
            )

    @patch("src.configs.settings.settings")
    def test_store_extraction_results_creates_page_extraction_result(self, mock_settings, api_extraction_processor):
        """Test that PageExtractionResult is created with correct structure."""
        mock_settings.S3_BUCKET_NAME = "test-bucket"

        mock_extraction_result = Mock()
        mock_extraction_result.data = {"text": "Test content with multiple words"}

        api_extraction_processor._store_extraction_results(
            extraction_result=mock_extraction_result,
            project_id="proj-123",
            upload_id="upload-456"
        )

        # Verify results.json was created with PageExtractionResult structure
        results_call = api_extraction_processor.storage_service.upload_content.call_args_list[0]
        # The content parameter can be positional or keyword
        if "content" in results_call.kwargs:
            results_content = results_call.kwargs["content"]
        else:
            results_content = results_call.args[0]  # First positional argument

        # Decode and parse JSON to verify structure
        results_json = json.loads(results_content.decode("utf-8"))
        assert results_json["page_number"] == 1
        assert results_json["content"] == "Test content with multiple words"
        assert results_json["metadata"]["page_count"] == 1
        assert results_json["metadata"]["word_count"] == 5
        assert results_json["extraction_type"] == ExtractionType.API.value


class TestCreateFailureResult:
    """Test _create_failure_result method."""

    def test_create_failure_result_basic(self, api_extraction_processor):
        """Test creating basic failure result."""
        result = api_extraction_processor._create_failure_result(
            request_id="req-123",
            document_urls=["s3://bucket/doc.pdf"],
            error_message="Test error message",
            project_id="proj-456",
            upload_id="upload-789"
        )

        assert result["success"] is False
        assert result["error_message"] == "Test error message"
        # Note: processing_data is NOT nested on failure, only error_message is added

    def test_create_failure_result_with_extraction_result(self, api_extraction_processor):
        """Test creating failure result with extraction result."""
        mock_extraction_result = Mock()
        mock_extraction_result.error = Mock(error_message="Extraction failed")

        result = api_extraction_processor._create_failure_result(
            request_id="req-123",
            document_urls=["s3://bucket/doc.pdf"],
            error_message="Test error",
            project_id="proj-456",
            upload_id="upload-789",
            extraction_result=mock_extraction_result
        )

        assert result["success"] is False
        # Note: extraction_result is not added to the standard result format


class TestHealthCheck:
    """Test health_check method."""

    def test_health_check_all_healthy(self, api_extraction_processor):
        """Test health check when all services are healthy."""
        result = api_extraction_processor.health_check()

        assert result["api_extraction_processor"] == "healthy"
        assert result["strategy_name"] == "api"
        assert result["extraction_client"] == "healthy"
        assert result["storage_service"] == "healthy"
        assert result["status_tracker"] == "healthy"

    def test_health_check_extraction_client_unhealthy(self, api_extraction_processor, mock_extraction_client):
        """Test health check when extraction client is unhealthy."""
        mock_extraction_client.health_check.side_effect = Exception("Client error")

        result = api_extraction_processor.health_check()

        assert result["api_extraction_processor"] == "healthy"
        assert result["extraction_client"] == "unhealthy"

    def test_health_check_storage_service_unhealthy(self, api_extraction_processor, mock_storage_service):
        """Test health check when storage service is unhealthy."""
        mock_storage_service.health_check.side_effect = Exception("Storage error")

        result = api_extraction_processor.health_check()

        assert result["api_extraction_processor"] == "healthy"
        assert result["storage_service"] == "unhealthy"

    def test_health_check_status_tracker_unhealthy(self, api_extraction_processor, mock_status_tracker):
        """Test health check when status tracker is unhealthy."""
        mock_status_tracker.health_check.return_value = {"status_tracker": "unhealthy"}

        result = api_extraction_processor.health_check()

        assert result["api_extraction_processor"] == "healthy"
        assert result["status_tracker"] == "unhealthy"

    def test_health_check_no_health_check_method(self, api_extraction_processor, mock_extraction_client):
        """Test health check when service doesn't have health_check method."""
        del mock_extraction_client.health_check

        result = api_extraction_processor.health_check()

        assert result["api_extraction_processor"] == "healthy"
        assert result["extraction_client"] == "available"


class TestSupportsFileType:
    """Test supports_file_type method."""

    def test_supports_common_file_types(self, api_extraction_processor):
        """Test that common file types are supported."""
        from idp_file_management.models.mime_type import IngestionMimeType

        supported_types = [
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
        ]

        for file_type in supported_types:
            assert api_extraction_processor.supports_file_type(file_type) is True

    def test_supports_unknown_file_type(self, api_extraction_processor):
        """Test that unknown file type is not supported."""
        assert api_extraction_processor.supports_file_type("unknown") is False

    def test_supports_custom_file_types(self, api_extraction_processor):
        """Test that custom non-unknown file types are supported."""
        # API extraction is flexible and supports most types
        assert api_extraction_processor.supports_file_type("application/custom") is True


class TestGetProcessingCapabilities:
    """Test get_processing_capabilities method."""

    def test_get_processing_capabilities(self, api_extraction_processor):
        """Test getting processing capabilities."""
        from idp_file_management.models.mime_type import IngestionMimeType

        capabilities = api_extraction_processor.get_processing_capabilities()

        assert capabilities["strategy_name"] == "api"
        assert IngestionMimeType.PDF.value in capabilities["supported_file_types"]
        assert "batch_processing" in capabilities["features"]
        assert "cloud_ai_extraction" in capabilities["features"]
        assert "requires_api_connectivity" in capabilities["limitations"]
        assert capabilities["performance_characteristics"]["batch_size"] == "unlimited"
        assert "large_document_batches" in capabilities["optimal_use_cases"]
