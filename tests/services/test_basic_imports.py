"""
Basic import and instantiation tests for services.
These tests verify that services can be imported and instantiated without errors.
"""

import pytest
from unittest.mock import Mock, patch


class TestServiceImports:
    """Test that all services can be imported and basic functionality works."""

    def test_docling_service_import(self, test_document_processor):
        """Test DoclingStrategyProcessor can be imported and instantiated."""
        from src.models.docling_models import DoclingConfig, DoclingProviderEnum, DoclingOutputFormatEnum

        # Test model creation
        config = DoclingConfig(provider=DoclingProviderEnum.LOCAL, output_format=DoclingOutputFormatEnum.MARKDOWN)
        assert config.provider == DoclingProviderEnum.LOCAL
        assert config.output_format == DoclingOutputFormatEnum.MARKDOWN

        # Test service instantiation via dependency injection
        service = test_document_processor
        assert service is not None

        # Test health check
        health = service.health_check()
        assert isinstance(health, dict)
        assert "docling_strategy_processor" in health

    def test_processing_orchestrator_import(self, test_processing_orchestrator):
        """Test ProcessingOrchestrator can be imported and instantiated with strategy pattern."""
        orchestrator = test_processing_orchestrator
        assert orchestrator is not None
        # Check for strategy-pattern methods
        assert hasattr(orchestrator, "process_documents")
        assert hasattr(orchestrator, "health_check")
        # Check that strategies are properly injected
        assert hasattr(orchestrator, "processors")
        assert "docling" in orchestrator.processors
        assert "api" in orchestrator.processors
        # Verify core orchestrator methods exist
        assert hasattr(orchestrator, "get_available_strategies")

    def test_s3_service_import(self):
        """Test S3Service can be imported and instantiated with dependency injection."""
        from src.services.s3_service import S3Service

        # Test with injected S3 client - no patching needed!
        mock_s3_client = Mock()
        service = S3Service(s3_client=mock_s3_client)

        assert service is not None
        assert hasattr(service, "upload_content")
        assert hasattr(service, "download_content")
        assert hasattr(service, "parse_s3_uri")
        assert service.s3_client == mock_s3_client

    async def test_status_tracker_import(self, test_status_tracker):
        """Test StatusTracker can be imported and basic functionality works."""
        from src.services.status_tracker import StatusTracker
        from src.models.processing_models import RequestSourceEnum
        from idp_ingestion_status.models.ingestion_status import IngestionStepStatus

        # Test that we can create a StatusTracker instance
        assert isinstance(test_status_tracker, StatusTracker)

        # Test basic functionality - new unified interface
        request_id = "test-import-123"

        # Test update_status method returns boolean (now async)
        result = await test_status_tracker.update_status(
            upload_id=request_id,
            status=IngestionStepStatus.PROCESSING,
            project_id="default",
            message="Test status update",
        )

        # New implementation returns boolean
        assert isinstance(result, bool)
        assert result is True  # Should succeed even without API configured

        # Test health check works
        health = test_status_tracker.health_check()
        assert isinstance(health, dict)
        assert "status_tracker" in health

    def test_processing_models_import(self):
        """Test processing models can be imported and created."""
        from src.models.processing_models import ProcessingStatus, RequestSourceEnum
        from idp_ingestion_status.models.ingestion_status import IngestionStepStatus

        # Test status enum values exist
        assert IngestionStepStatus.QUEUED
        assert IngestionStepStatus.PROCESSING
        assert IngestionStepStatus.SUCCESS
        assert IngestionStepStatus.FAILURE

        assert RequestSourceEnum.REST_API
        assert RequestSourceEnum.KAFKA

    def test_docling_models_import(self):
        """Test docling models can be imported and created."""
        from src.models.docling_models import (
            DoclingConfig,
            DoclingRequest,
            DoclingProviderEnum,
            DoclingOutputFormatEnum,
        )

        # Test enum values (BEDROCK_CLAUDE provider removed)
        assert DoclingProviderEnum.LOCAL

        assert DoclingOutputFormatEnum.MARKDOWN
        assert DoclingOutputFormatEnum.HTML
        assert DoclingOutputFormatEnum.TEXT
        assert DoclingOutputFormatEnum.JSON

        # Test model creation
        config = DoclingConfig(
            provider=DoclingProviderEnum.LOCAL,
            output_format=DoclingOutputFormatEnum.MARKDOWN,
            chunk_size=1000,
            overlap_size=100,
            extract_pages="all",
            single_page_mode=False,
        )
        assert config.provider == DoclingProviderEnum.LOCAL
        assert config.output_format == DoclingOutputFormatEnum.MARKDOWN

        request = DoclingRequest(
            request_id="test-123",
            project_id="test-project",
            upload_id="test-upload-678",
            document_urls=["s3://test/doc.pdf"],
            config=config,
        )
        assert request.request_id == "test-123"
        assert len(request.document_urls) == 1
