"""
Pytest configuration and fixtures for the test suite.
"""

# Set environment variables BEFORE any imports to ensure they're available during module discovery
import os
os.environ["ENVIRONMENT_NAME"] = "test"
os.environ["STATUS_ENABLE_KAFKA_PUBLISHING"] = "false"

import pytest
from unittest.mock import Mock, patch

# Module-level mock for Kafka to prevent real Kafka initialization during test discovery
_sync_kafka_patcher = patch("idp_kafka.Kafka")
_mock_sync_kafka_class = _sync_kafka_patcher.start()

# Configure the mock to return a properly mocked instance
_mock_kafka_instance = Mock()
_mock_kafka_instance.initialize = Mock()
_mock_kafka_instance.produce = Mock()
_mock_kafka_instance.close = Mock()
_mock_kafka_instance.start_consumer = Mock()
_mock_kafka_instance.seek_back = Mock()
_mock_sync_kafka_class.return_value = _mock_kafka_instance
import json
from typing import Dict, Any, List
from src.models.docling_models import (
    DoclingConfig,
    DoclingProviderEnum,
    DoclingOutputFormatEnum,
)
from src.models.processing_models import RequestSourceEnum


@pytest.fixture
def mock_settings():
    """Mock settings for testing."""
    with patch("src.configs.settings.settings") as mock_settings:
        mock_settings.AWS_REGION = "us-east-1"
        mock_settings.EXTRACTION_ENDPOINT = "http://test-endpoint"
        mock_settings.KAFKA_INGEST_TOPIC = "test-ingest-topic"
        mock_settings.KAFKA_OUTPUT_TOPIC = "test-output-topic"
        mock_settings.DOCLING_DEFAULT_PROVIDER = "LOCAL"
        mock_settings.DOCLING_DEFAULT_OUTPUT_FORMAT = "MARKDOWN"
        mock_settings.DOCLING_ENABLE_OCR = True
        mock_settings.DOCLING_ENABLE_TABLE_STRUCTURE = True
        mock_settings.DOCLING_ENABLE_FIGURE_EXTRACTION = True
        mock_settings.DOCLING_MAX_TOKENS = 4000
        mock_settings.ENVIRONMENT_NAME = "test"
        mock_settings.STATUS_ENABLE_KAFKA_PUBLISHING = False
        mock_settings.mcp_secret = Mock(return_value={"token": "test-token"})
        yield mock_settings


@pytest.fixture
def sample_document_urls():
    """Sample S3 document URLs for testing."""
    return ["s3://test-bucket/document1.pdf", "s3://test-bucket/document2.pdf"]


@pytest.fixture
def sample_docling_config():
    """Sample DoclingConfig for testing."""
    return DoclingConfig(
        provider=DoclingProviderEnum.LOCAL,
        output_format=DoclingOutputFormatEnum.MARKDOWN,
        enable_ocr=True,
        enable_table_structure=True,
        enable_figure_extraction=True,
    )


@pytest.fixture
def mock_s3_service():
    """Mock S3Service for testing."""
    mock_service = Mock()
    mock_service.download_content = Mock(return_value=b"mock pdf content")
    mock_service.upload_content = Mock(return_value="s3://bucket/path/to/file.json")
    return mock_service


@pytest.fixture
def mock_sync_kafka():
    """Mock Kafka for testing."""
    mock_service = Mock()
    mock_service.initialize = Mock()
    mock_service.produce = Mock()
    mock_service.start_consumer = Mock()
    mock_service.close = Mock()
    return mock_service


@pytest.fixture
def mock_extraction_response():
    """Mock extraction API response."""
    return {
        "request_id": "test-request-id",
        "status": "completed",
        "results": [
            {
                "document_url": "s3://test-bucket/document1.pdf",
                "extracted_text": "Sample extracted text",
                "metadata": {"pages": 5},
            }
        ],
    }


@pytest.fixture
def mock_docling_result():
    """Mock Docling processing result."""
    return {
        "request_id": "test-request-id",
        "success": True,
        "total_documents": 2,
        "successful_documents": 2,
        "failed_documents": 0,
        "processing_time_seconds": 10.5,
        "results": [
            {
                "document_url": "s3://test-bucket/document1.pdf",
                "success": True,
                "content": "# Document 1\n\nSample markdown content",
                "metadata": {"page_count": 5, "word_count": 100},
                "processing_time_seconds": 5.2,
            },
            {
                "document_url": "s3://test-bucket/document2.pdf",
                "success": True,
                "content": "# Document 2\n\nAnother markdown content",
                "metadata": {"page_count": 3, "word_count": 75},
                "processing_time_seconds": 5.3,
            },
        ],
    }


@pytest.fixture
def mock_requests_client():
    """Mock requests client for testing."""
    mock_client = Mock()

    # Mock successful response
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "request_id": "test-request-id",
        "status": "completed",
        "results": [],
    }
    mock_response.text = json.dumps(mock_response.json.return_value)

    mock_client.post.return_value = mock_response

    return mock_client


# Removed mock_bedrock_client fixture - BEDROCK functionality removed


@pytest.fixture
def mock_docling_converter():
    """Mock Docling DocumentConverter for testing."""
    mock_converter = Mock()

    # Mock document result
    mock_doc = Mock()
    mock_doc.export_to_markdown.return_value = "# Sample Document\n\nContent here"
    mock_doc.export_to_text.return_value = "Sample Document\n\nContent here"
    mock_doc.export_to_html.return_value = "<h1>Sample Document</h1><p>Content here</p>"
    mock_doc.export_to_json.return_value = '{"title": "Sample Document", "content": "Content here"}'
    mock_doc.pages = [Mock(), Mock(), Mock()]  # 3 pages
    mock_doc.version = "1.0"
    mock_doc.input_format = "PDF"

    # Mock conversion result
    mock_result = Mock()
    mock_result.document = mock_doc

    mock_converter.convert.return_value = mock_result

    return mock_converter


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Set up test environment variables."""
    import os

    original_env = os.environ.copy()

    # Set test environment variables
    os.environ["ENVIRONMENT_NAME"] = "test"
    os.environ["STATUS_ENABLE_KAFKA_PUBLISHING"] = "false"

    yield

    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset any module-level state between tests."""
    # Clear any existing status_tracker module imports to force re-initialization
    import sys

    modules_to_remove = []
    for module_name in sys.modules:
        if "status_tracker" in module_name:
            modules_to_remove.append(module_name)

    for module_name in modules_to_remove:
        del sys.modules[module_name]

    # Clear lru_cache in providers for test isolation
    from src.dependencies import providers

    for attr_name in dir(providers):
        attr = getattr(providers, attr_name)
        try:
            attr.cache_clear()
        except AttributeError:
            # Attribute doesn't have cache_clear method, skip
            pass

    yield

    # Clean up after test
    modules_to_remove = []
    for module_name in sys.modules:
        if "status_tracker" in module_name:
            modules_to_remove.append(module_name)

    for module_name in modules_to_remove:
        if module_name in sys.modules:
            del sys.modules[module_name]


# Dependency Injection Test Fixtures
@pytest.fixture
def test_status_tracker():
    """Provide a StatusTracker instance for testing - Kafka always enabled."""
    from src.services.status_tracker import StatusTracker
    from unittest.mock import AsyncMock, Mock

    # Mock database service to prevent real database connections during tests
    with patch("src.services.status_tracker.ingestion_status_service") as mock_status_service, patch(
        "idp_kafka.Kafka"
    ) as mock_sync_kafka_class:
        mock_status_service.upsert_file_status = AsyncMock(return_value=None)

        # Mock Kafka instance to prevent actual Kafka connections
        mock_kafka_instance = Mock()
        mock_kafka_instance.initialize = Mock()
        mock_kafka_instance.produce = Mock()
        mock_kafka_instance.close = Mock()
        mock_sync_kafka_class.return_value = mock_kafka_instance

        tracker = StatusTracker()
        yield tracker


@pytest.fixture
def test_storage_service():
    """Provide a mock S3Service for testing."""
    from unittest.mock import Mock

    mock_storage = Mock()
    # Set up common mock methods that tests expect
    mock_storage.download_content = Mock(return_value=b"Mock PDF document content for testing")
    mock_storage.upload_content = Mock(return_value="s3://test-bucket/results/test-file.json")
    return mock_storage


@pytest.fixture
def test_document_processor(test_storage_service, test_logger):
    """Provide a DoclingStrategyProcessor for testing."""
    from unittest.mock import Mock, AsyncMock
    from src.services.extractors.docling.docling_strategy_processor import DoclingStrategyProcessor
    from src.services.asset_storage_service import AssetStorageService
    from src.services.processing_config_manager import ProcessingConfigManager

    # Create mock dependencies
    from src.models.docling_models import DoclingConfig, DoclingProviderEnum, DoclingOutputFormatEnum

    mock_status_tracker = Mock()
    mock_status_tracker.update_status = AsyncMock()  # Make update_status async-aware
    mock_status_tracker.health_check = Mock(return_value={"status_tracker": "healthy"})

    mock_config_manager = Mock(spec=ProcessingConfigManager)
    # Return a real DoclingConfig when create_optimized_config is called
    mock_config_manager.create_optimized_config.return_value = DoclingConfig(
        provider=DoclingProviderEnum.LOCAL,
        output_format=DoclingOutputFormatEnum.MARKDOWN,
    )

    mock_image_analysis = Mock()
    mock_asset_storage = Mock(spec=AssetStorageService)

    return DoclingStrategyProcessor(
        storage_service=test_storage_service,
        status_tracker=mock_status_tracker,
        config_manager=mock_config_manager,
        asset_storage=mock_asset_storage,
    )


@pytest.fixture
def test_processing_orchestrator(test_status_tracker):
    """Provide a ProcessingOrchestrator for testing."""
    from unittest.mock import Mock
    from src.services.processing_orchestrator import ProcessingOrchestrator
    from src.services.extractors.docling.docling_strategy_processor import DoclingStrategyProcessor
    from src.services.extractors.api.api_extraction_processor import ApiExtractionProcessor

    # Create mock strategy processors
    mock_docling_processor = Mock(spec=DoclingStrategyProcessor)
    mock_api_processor = Mock(spec=ApiExtractionProcessor)

    return ProcessingOrchestrator(
        docling_processor=mock_docling_processor, api_processor=mock_api_processor, status_tracker=test_status_tracker
    )


@pytest.fixture
def test_extraction_client():
    """Provide a mock ExtractionApiClient for testing."""
    from src.services.extraction_api_client import ExtractionApiClient

    return ExtractionApiClient(endpoint="http://test-endpoint", timeout=300)


@pytest.fixture
def test_logger():
    """Provide a logger instance for testing."""
    from idp_logger.logger import Logger

    return Logger.get_logger("test")


@pytest.fixture
def test_image_analysis_service():
    """Provide a mock ImageAnalysisService for testing."""
    from unittest.mock import Mock
    from datetime import datetime, timezone

    mock_service = Mock()

    # Default successful analysis result
    mock_service.analyze_image.return_value = {
        "model_used": "anthropic.claude-3-sonnet-20240229-v1:0",
        "analysis_summary": "This is a test image showing various graphical elements for testing purposes.",
        "analysis_timestamp": datetime.now(timezone.utc).isoformat(),
        "processing_time_ms": 1000,
        "analysis_confidence": 0.90,
    }

    mock_service.health_check.return_value = {
        "bedrock_image_analysis_service": "healthy",
        "bedrock_connectivity": "healthy",
    }

    return mock_service


# Duplicate test_processing_orchestrator fixture removed - using simpler mock version above


@pytest.fixture
def test_kafka_message_handler(test_status_tracker, test_processing_orchestrator, test_extraction_client):
    """Provide a KafkaMessageHandler for testing."""
    from src.services.kafka_message_handler import KafkaMessageHandler

    return KafkaMessageHandler(test_status_tracker, test_processing_orchestrator, test_extraction_client)


@pytest.fixture
def test_document_orchestrator(
    test_storage_service, test_document_processor, test_status_tracker, test_extraction_client, test_logger
):
    """Provide a DocumentOrchestrator for testing with all dependencies."""
    from src.services.document_orchestrator import DocumentOrchestrator  # type: ignore[import-untyped]

    return DocumentOrchestrator(
        storage_service=test_storage_service,
        document_processor=test_document_processor,
        status_tracker=test_status_tracker,
        extraction_client=test_extraction_client,
        logger_instance=test_logger,
    )


# Legacy fixtures for backward compatibility
@pytest.fixture
def status_tracker(test_status_tracker):
    """Legacy fixture name for StatusTracker - use test_status_tracker instead."""
    return test_status_tracker
