"""
Comprehensive tests for DoclingService covering local processing functionality.
"""

import pytest
import time
from unittest.mock import Mock, patch, mock_open

from src.services.extractors.docling.docling_strategy_processor import DoclingStrategyProcessor
from src.models.docling_models import (
    DoclingRequest,
    DoclingConfig,
    DoclingDocumentResult,
    DoclingProviderEnum,
    DoclingOutputFormatEnum,
)


class TestDoclingServiceComprehensive:
    """Comprehensive test cases for DoclingStrategyProcessor (local processing only)."""

    @pytest.fixture
    def service(self, mock_settings, test_document_processor):
        """Create DoclingService instance for testing."""
        return test_document_processor

    @pytest.fixture
    def sample_request(self):
        """Create sample DoclingRequest for testing."""
        config = DoclingConfig(
            provider=DoclingProviderEnum.LOCAL,
            output_format=DoclingOutputFormatEnum.MARKDOWN,
            enable_ocr=True,
            enable_table_structure=True,
            single_page_mode=False,
            extract_pages="all",
        )

        return DoclingRequest(
            request_id="test-request-123",
            project_id="test-project-456",
            upload_id="test-upload-678",
            document_urls=["s3://test-bucket/doc1.pdf", "s3://test-bucket/doc2.pdf"],
            config=config,
        )

    # Initialization Tests
    def test_init_success(self, service, mock_settings):
        """Test successful DoclingService initialization."""
        assert service.storage_service is not None
        assert service.status_tracker is not None
        assert service.config_manager is not None
        assert service.asset_storage is not None
        assert service.logger is not None

    # Main Processing Tests
    @pytest.mark.asyncio
    async def test_process_documents_local_success(self, service, sample_request):
        """Test successful local document processing."""
        from src.models.docling_models import DoclingBatchResult

        # Mock the batch result that would be returned
        mock_batch_result = DoclingBatchResult(
            request_id=sample_request.request_id,
            config_used=sample_request.config,
            success=True,
            total_documents=2,
            successful_documents=2,
            failed_documents=0,
            total_processing_time_seconds=4.3,
            results=[
                DoclingDocumentResult(
                    document_url="s3://test-bucket/doc1.pdf",
                    success=True,
                    content="# Document 1 Content",
                    metadata={"page_count": 5},
                    processing_time_seconds=2.5,
                ),
                DoclingDocumentResult(
                    document_url="s3://test-bucket/doc2.pdf",
                    success=True,
                    content="# Document 2 Content",
                    metadata={"page_count": 3},
                    processing_time_seconds=1.8,
                ),
            ],
        )

        # Mock the internal methods
        with patch.object(service, "_process_documents_batch", return_value=mock_batch_result), \
             patch.object(service, "_extract_and_store_assets", return_value={"figures": [], "tables": []}), \
             patch.object(service, "_store_docling_results", return_value={0: "s3://path/0/results.json", 1: "s3://path/1/results.json"}):

            result = await service.process_documents(
                request_id=sample_request.request_id,
                project_id=sample_request.project_id,
                upload_id=sample_request.upload_id,
                document_urls=sample_request.document_urls,
            )

            # Check standardized result format
            assert result["request_id"] == "test-request-123"
            assert result["success"] is True
            assert result["strategy"] == "docling"
            assert result["total_documents"] == 2
            # Processing data is merged into top level, not nested
            assert "base_s3_location" in result
            assert "processing_method" in result
            assert result["processing_method"] == "docling"

    @pytest.mark.asyncio
    async def test_process_documents_mixed_results(self, service, sample_request):
        """Test document processing with mixed success/failure results."""
        from src.models.docling_models import DoclingBatchResult

        # Partial success - some documents succeeded
        mock_batch_result = DoclingBatchResult(
            request_id=sample_request.request_id,
            config_used=sample_request.config,
            success=True,  # Overall success since at least one succeeded
            total_documents=2,
            successful_documents=1,
            failed_documents=1,
            total_processing_time_seconds=3.0,
            results=[
                DoclingDocumentResult(
                    document_url="s3://test-bucket/doc1.pdf",
                    success=True,
                    content="# Success",
                    processing_time_seconds=2.0
                ),
                DoclingDocumentResult(
                    document_url="s3://test-bucket/doc2.pdf",
                    success=False,
                    error_message="Processing failed",
                    processing_time_seconds=1.0,
                ),
            ],
        )

        with patch.object(service, "_process_documents_batch", return_value=mock_batch_result), \
             patch.object(service, "_extract_and_store_assets", return_value={"figures": [], "tables": []}), \
             patch.object(service, "_store_docling_results", return_value={0: "s3://path/0/results.json"}):

            result = await service.process_documents(
                request_id=sample_request.request_id,
                project_id=sample_request.project_id,
                upload_id=sample_request.upload_id,
                document_urls=sample_request.document_urls,
            )

            assert result["success"] is True  # Overall success if any document succeeds
            assert result["strategy"] == "docling"

    @pytest.mark.asyncio
    async def test_process_documents_all_failures(self, service, sample_request):
        """Test document processing where all documents fail."""
        from src.models.docling_models import DoclingBatchResult

        # All failed
        mock_batch_result = DoclingBatchResult(
            request_id=sample_request.request_id,
            config_used=sample_request.config,
            success=False,  # Overall failure since all failed
            total_documents=2,
            successful_documents=0,
            failed_documents=2,
            total_processing_time_seconds=2.0,
            results=[
                DoclingDocumentResult(
                    document_url="s3://test-bucket/doc1.pdf",
                    success=False,
                    error_message="Failed to process",
                    processing_time_seconds=1.0,
                ),
                DoclingDocumentResult(
                    document_url="s3://test-bucket/doc2.pdf",
                    success=False,
                    error_message="Failed to process",
                    processing_time_seconds=1.0,
                ),
            ],
        )

        with patch.object(service, "_process_documents_batch", return_value=mock_batch_result):
            result = await service.process_documents(
                request_id=sample_request.request_id,
                project_id=sample_request.project_id,
                upload_id=sample_request.upload_id,
                document_urls=sample_request.document_urls,
            )

            assert result["success"] is False  # No successful documents
            assert result["strategy"] == "docling"
            assert "error_message" in result or "processing_data" in result

    @pytest.mark.asyncio
    async def test_process_documents_exception_handling(self, service, sample_request):
        """Test document processing with exceptions."""
        # Mock an exception during processing
        with patch.object(service, "_create_docling_request", side_effect=Exception("Unexpected error")):
            result = await service.process_documents(
                request_id=sample_request.request_id,
                project_id=sample_request.project_id,
                upload_id=sample_request.upload_id,
                document_urls=sample_request.document_urls,
            )

            # Should return failure result, not raise exception
            assert result["success"] is False
            assert "error_message" in result
            assert "Unexpected error" in result["error_message"]

    # Local Processing Tests
    def test_process_document_local_success(self, service):
        """Test successful local document processing."""
        document_url = "s3://test-bucket/document.pdf"
        config = DoclingConfig(
            provider=DoclingProviderEnum.LOCAL,
            output_format=DoclingOutputFormatEnum.MARKDOWN,
            chunk_size=1000,
            overlap_size=100,
            extract_pages="all",
            single_page_mode=False,
        )

        mock_result = DoclingDocumentResult(
            document_url=document_url,
            success=True,
            content="# Content",
            metadata={"page_count": 3, "extraction_mode": "multipage"},
            bounding_box=[],
            processing_time_seconds=0.5,
            page_count=3,
            word_count=100,
        )

        with patch.object(
            service, "_download_document", return_value="/tmp/test_document.pdf"
        ) as mock_download, patch.object(
            service, "_docling_process_local", return_value=mock_result
        ) as mock_process, patch("os.path.exists", return_value=True), patch("os.unlink") as mock_unlink:
            result = service._process_document_local(
                document_url, config, request_id="test-request", project_id="test-project", upload_id="test-upload"
            )

            assert result.success is True
            assert result.document_url == document_url
            assert result.content == "# Content"
            assert result.metadata["page_count"] == 3
            assert result.processing_time_seconds > 0

            mock_download.assert_called_once_with(document_url, "test-request", "test-upload", "test-project")
            mock_process.assert_called_once()
            mock_unlink.assert_called_once_with("/tmp/test_document.pdf")

    def test_process_document_local_download_failure(self, service):
        """Test local processing with download failure."""
        document_url = "s3://test-bucket/document.pdf"
        config = DoclingConfig(
            provider=DoclingProviderEnum.LOCAL,
            output_format=DoclingOutputFormatEnum.MARKDOWN,
            chunk_size=1000,
            overlap_size=100,
            extract_pages="all",
            single_page_mode=False,
        )

        with patch.object(service, "_download_document", side_effect=Exception("Download failed")):
            result = service._process_document_local(
                document_url, config, "test-request", "test-upload", "test-project"
            )

            assert result.success is False
            assert result.document_url == document_url
            assert "Download failed" in result.error_message
            assert result.processing_time_seconds > 0

    def test_process_document_local_processing_failure(self, service):
        """Test local processing with docling processing failure."""
        document_url = "s3://test-bucket/document.pdf"
        config = DoclingConfig(
            provider=DoclingProviderEnum.LOCAL,
            output_format=DoclingOutputFormatEnum.MARKDOWN,
            chunk_size=1000,
            overlap_size=100,
            extract_pages="all",
            single_page_mode=False,
        )

        with patch.object(service, "_download_document", return_value="/tmp/test.pdf"), patch.object(
            service, "_docling_process_local", side_effect=Exception("Docling error")
        ), patch("os.path.exists", return_value=True), patch("os.unlink"):
            result = service._process_document_local(
                document_url, config, "test-request", "test-upload", "test-project"
            )

            assert result.success is False
            assert "Docling error" in result.error_message

    def test_docling_process_local_success_multipage(self, service):
        """Test successful local Docling processing in multipage mode."""
        file_path = "/tmp/test.pdf"
        document_url = "s3://test-bucket/test.pdf"
        config = DoclingConfig(
            provider=DoclingProviderEnum.LOCAL,
            output_format=DoclingOutputFormatEnum.MARKDOWN,
            enable_ocr=True,
            enable_table_structure=True,
            single_page_mode=False,
        )

        # Mock the docling objects
        mock_doc = Mock()
        mock_doc.export_to_markdown.return_value = "# Sample Document Content"
        mock_doc.pages = [Mock(), Mock(), Mock()]  # 3 pages
        mock_doc.version = "1.0"
        mock_doc.input_format = "PDF"
        mock_doc.texts = []  # Empty list for bounding box extraction
        mock_doc.pictures = []  # No figures
        mock_doc.tables = []  # No tables

        mock_result = Mock()
        mock_result.document = mock_doc

        mock_converter = Mock()
        mock_converter.convert.return_value = mock_result

        with patch("docling.document_converter.DocumentConverter", return_value=mock_converter):
            result = service._docling_process_local(
                file_path, document_url, config, "test_request_id", "test_upload_id", "test_project_id"
            )

            assert result.success is True
            assert result.document_url == document_url
            assert result.content == "# Sample Document Content"
            assert result.metadata["page_count"] == 3
            assert result.metadata["extraction_mode"] == "multipage"
            assert result.metadata["docling_version"] == "1.0"
            assert result.metadata["format_detected"] == "PDF"
            assert result.metadata["page_extraction_config"]["single_page_mode"] is False
            assert result.bounding_box == []
            assert result.processing_time_seconds > 0

            mock_converter.convert.assert_called_once_with(file_path)
            mock_doc.export_to_markdown.assert_called_once()

    def test_docling_process_local_success_single_page(self, service):
        """Test successful local Docling processing in single page mode."""
        file_path = "/tmp/test.pdf"
        document_url = "s3://test-bucket/test.pdf"
        config = DoclingConfig(
            provider=DoclingProviderEnum.LOCAL,
            output_format=DoclingOutputFormatEnum.MARKDOWN,
            chunk_size=1000,
            overlap_size=100,
            extract_pages="all",
            single_page_mode=True,
        )

        # Mock text elements with page numbers
        mock_text_elem1 = Mock()
        mock_text_elem1.text = "Text on page 1"
        mock_prov1 = Mock()
        mock_prov1.page_no = 1
        mock_text_elem1.prov = [mock_prov1]

        mock_text_elem2 = Mock()
        mock_text_elem2.text = "Text on page 2"
        mock_prov2 = Mock()
        mock_prov2.page_no = 2
        mock_text_elem2.prov = [mock_prov2]

        # Mock the docling objects
        mock_doc = Mock()
        mock_doc.pages = [Mock(), Mock()]  # 2 pages
        mock_doc.texts = [mock_text_elem1, mock_text_elem2]
        mock_doc.version = "1.0"
        mock_doc.input_format = "PDF"
        mock_doc.pictures = []  # No figures
        mock_doc.tables = []  # No tables

        mock_result = Mock()
        mock_result.document = mock_doc

        mock_converter = Mock()
        mock_converter.convert.return_value = mock_result

        with patch("docling.document_converter.DocumentConverter", return_value=mock_converter):
            result = service._docling_process_local(
                file_path, document_url, config, "test_request_id", "test_upload_id", "test_project_id"
            )

            assert result.success is True
            assert result.document_url == document_url
            assert "# Page 1" in result.content
            assert "Text on page 1" in result.content
            assert "# Page 2" in result.content
            assert "Text on page 2" in result.content
            assert "---" in result.content  # Page separator
            assert result.metadata["page_count"] == 2
            assert result.metadata["extraction_mode"] == "single_page"
            assert result.metadata["page_extraction_config"]["single_page_mode"] is True
            assert result.processing_time_seconds > 0

    def test_docling_process_local_different_formats(self, service):
        """Test local Docling processing with different output formats."""
        file_path = "/tmp/test.pdf"
        document_url = "s3://test-bucket/test.pdf"

        # Mock document with all format methods
        mock_doc = Mock()
        mock_doc.export_to_markdown.return_value = "# Markdown Content"
        mock_doc.export_to_text.return_value = "Text Content"
        mock_doc.export_to_html.return_value = "<h1>HTML Content</h1>"
        mock_doc.export_to_json.return_value = '{"content": "JSON Content"}'
        mock_doc.pages = [Mock()]
        mock_doc.texts = []  # Empty list for bounding box extraction
        mock_doc.version = "1.0"
        mock_doc.input_format = "PDF"
        mock_doc.pictures = []  # No figures
        mock_doc.tables = []  # No tables

        mock_result = Mock()
        mock_result.document = mock_doc

        mock_converter = Mock()
        mock_converter.convert.return_value = mock_result

        formats_and_expected = [
            (DoclingOutputFormatEnum.MARKDOWN, "# Markdown Content"),
            (DoclingOutputFormatEnum.TEXT, "Text Content"),
            (DoclingOutputFormatEnum.HTML, "<h1>HTML Content</h1>"),
            (DoclingOutputFormatEnum.JSON, '{"content": "JSON Content"}'),
        ]

        for output_format, expected_content in formats_and_expected:
            config = DoclingConfig(
                provider=DoclingProviderEnum.LOCAL,
                output_format=output_format,
                chunk_size=1000,
                overlap_size=100,
                extract_pages="all",
                single_page_mode=False,
            )

            with patch("docling.document_converter.DocumentConverter", return_value=mock_converter):
                result = service._docling_process_local(
                    file_path, document_url, config, "test_request_id", "test_upload_id", "test_project_id"
                )
                assert result.success is True
                assert result.content == expected_content
                assert result.metadata["extraction_mode"] == "multipage"
                assert result.bounding_box == []

    def test_docling_process_local_import_error(self, service):
        """Test local Docling processing with import error."""
        file_path = "/tmp/test.pdf"
        document_url = "s3://test-bucket/test.pdf"
        config = DoclingConfig(
            provider=DoclingProviderEnum.LOCAL,
            output_format=DoclingOutputFormatEnum.MARKDOWN,
            chunk_size=1000,
            overlap_size=100,
            extract_pages="all",
            single_page_mode=False,
        )

        with patch("docling.document_converter.DocumentConverter", side_effect=ImportError("Docling not installed")):
            with pytest.raises(Exception, match="Docling library is not available or not properly installed"):
                service._docling_process_local(
                    file_path, document_url, config, "test_request_id", "test_upload_id", "test_project_id"
                )

    def test_docling_process_local_processing_error(self, service):
        """Test local Docling processing with processing error."""
        file_path = "/tmp/test.pdf"
        document_url = "s3://test-bucket/test.pdf"
        config = DoclingConfig(
            provider=DoclingProviderEnum.LOCAL,
            output_format=DoclingOutputFormatEnum.MARKDOWN,
            chunk_size=1000,
            overlap_size=100,
            extract_pages="all",
            single_page_mode=False,
        )

        with patch("docling.document_converter.DocumentConverter", side_effect=Exception("Processing failed")):
            with pytest.raises(Exception, match="Processing failed"):
                service._docling_process_local(
                    file_path, document_url, config, "test_request_id", "test_upload_id", "test_project_id"
                )

    # Download Tests
    def test_download_document_success(self, service):
        """Test successful document download."""
        document_url = "s3://test-bucket/document.pdf"
        mock_content = b"mock pdf content"

        service.storage_service.download_content.return_value = mock_content

        with patch("tempfile.NamedTemporaryFile") as mock_temp:
            mock_file = Mock()
            mock_file.name = "/tmp/test_doc.pdf"
            mock_temp.return_value.__enter__.return_value = mock_file
            mock_temp.return_value.__exit__.return_value = None

            temp_path = service._download_document(
                document_url, request_id="test-request", upload_id="test-upload", project_id="test-project"
            )

            assert temp_path == "/tmp/test_doc.pdf"
            service.storage_service.download_content.assert_called_once_with(
                document_url, request_id="test-request", upload_id="test-upload", project_id="test-project"
            )
            mock_file.write.assert_called_once_with(mock_content)

    def test_download_document_s3_failure(self, service):
        """Test document download with S3 failure."""
        document_url = "s3://test-bucket/document.pdf"

        service.storage_service.download_content.side_effect = Exception("S3 access denied")

        with pytest.raises(Exception, match="Failed to download document"):
            service._download_document(document_url, "test-request", "test-upload", "test-project")

    # Health Check Tests
    def test_health_check_all_available(self, service):
        """Test health check with all components available."""
        health = service.health_check()

        # Verify basic health check structure (use actual keys returned)
        assert "docling_strategy_processor" in health
        assert health["docling_strategy_processor"] == "healthy"
        assert "local_docling" in health
        assert "storage_service" in health

    def test_health_check_components(self, service):
        """Test that health check reports all required components."""
        health = service.health_check()

        # Check that all expected keys are present (actual keys from implementation)
        expected_keys = ["docling_strategy_processor", "local_docling", "storage_service"]
        for key in expected_keys:
            assert key in health, f"Health check missing key: {key}"

        # Verify at least one component reports as healthy
        assert any("healthy" in str(v) or "available" in str(v) for v in health.values())

    # Edge Cases and Error Handling
    def test_process_empty_document_list(self, service):
        """Test that empty document list validation works correctly."""
        from pydantic import ValidationError

        config = DoclingConfig(
            provider=DoclingProviderEnum.LOCAL,
            output_format=DoclingOutputFormatEnum.MARKDOWN,
            chunk_size=1000,
            overlap_size=100,
            extract_pages="all",
            single_page_mode=False,
        )

        # Should raise validation error for empty document list
        with pytest.raises(ValidationError) as exc_info:
            DoclingRequest(
                request_id="empty-test",
                project_id="test-project",
                upload_id="test-upload",
                document_urls=[],
                config=config,
            )

        # Verify the validation error is about the document_urls field
        assert "document_urls" in str(exc_info.value)
        assert "at least 1 item" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_process_large_document_list(self, service):
        """Test processing with large document list."""
        from src.models.docling_models import DoclingBatchResult

        document_urls = [f"s3://test-bucket/doc{i}.pdf" for i in range(50)]
        config = DoclingConfig(
            provider=DoclingProviderEnum.LOCAL,
            output_format=DoclingOutputFormatEnum.MARKDOWN,
            chunk_size=1000,
            overlap_size=100,
            extract_pages="all",
            single_page_mode=False,
        )
        request = DoclingRequest(
            request_id="large-test",
            project_id="test-project",
            upload_id="test-upload",
            document_urls=document_urls,
            config=config,
        )

        # Mock all 50 documents as successful
        mock_results = [
            DoclingDocumentResult(
                document_url=f"s3://test-bucket/doc{i}.pdf",
                success=True,
                content=f"# Content {i}",
                processing_time_seconds=1.0,
            )
            for i in range(50)
        ]

        mock_batch_result = DoclingBatchResult(
            request_id=request.request_id,
            config_used=config,
            success=True,
            total_documents=50,
            successful_documents=50,
            failed_documents=0,
            total_processing_time_seconds=50.0,
            results=mock_results,
        )

        # Create S3 paths for 50 pages
        page_s3_paths = {i: f"s3://test-bucket/test-project/test-upload/{i}/results.json" for i in range(50)}

        with patch.object(service, "_process_documents_batch", return_value=mock_batch_result), \
             patch.object(service, "_extract_and_store_assets", return_value={"figures": [], "tables": []}), \
             patch.object(service, "_store_docling_results", return_value=page_s3_paths):

            result = await service.process_documents(
                request_id=request.request_id,
                project_id=request.project_id,
                upload_id=request.upload_id,
                document_urls=request.document_urls,
            )

            assert result["success"] is True
            assert result["total_documents"] == 50
            assert result["strategy"] == "docling"

    def test_config_validation(self, service):
        """Test DoclingConfig validation and defaults."""
        # Test default config
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
        assert config.enable_ocr is True
        assert config.enable_table_structure is True
        assert config.single_page_mode is False
        assert config.extract_pages == "all"

        # Test custom config
        custom_config = DoclingConfig(
            provider=DoclingProviderEnum.LOCAL,
            output_format=DoclingOutputFormatEnum.JSON,
            enable_ocr=False,
            single_page_mode=True,
            extract_pages="single",
            chunk_size=2000,
            overlap_size=200,
        )
        assert custom_config.provider == DoclingProviderEnum.LOCAL
        assert custom_config.output_format == DoclingOutputFormatEnum.JSON
        assert custom_config.enable_ocr is False
        assert custom_config.single_page_mode is True
        assert custom_config.extract_pages == "single"

    @pytest.mark.asyncio
    async def test_concurrent_document_processing(self, service, sample_request):
        """Test that documents can be processed concurrently."""
        from src.models.docling_models import DoclingBatchResult

        # Mock batch result with 2 successful documents
        mock_batch_result = DoclingBatchResult(
            request_id=sample_request.request_id,
            config_used=sample_request.config,
            success=True,
            total_documents=2,
            successful_documents=2,
            failed_documents=0,
            total_processing_time_seconds=2.0,
            results=[
                DoclingDocumentResult(
                    document_url="s3://test-bucket/doc1.pdf",
                    success=True,
                    content="# Processed doc1",
                    processing_time_seconds=1.0,
                ),
                DoclingDocumentResult(
                    document_url="s3://test-bucket/doc2.pdf",
                    success=True,
                    content="# Processed doc2",
                    processing_time_seconds=1.0,
                ),
            ],
        )

        with patch.object(service, "_process_documents_batch", return_value=mock_batch_result), \
             patch.object(service, "_extract_and_store_assets", return_value={"figures": [], "tables": []}), \
             patch.object(service, "_store_docling_results", return_value={0: "s3://path/0/results.json", 1: "s3://path/1/results.json"}):

            result = await service.process_documents(
                request_id=sample_request.request_id,
                project_id=sample_request.project_id,
                upload_id=sample_request.upload_id,
                document_urls=sample_request.document_urls,
            )

            # Both documents should have been processed successfully
            assert result["success"] is True
            assert result["total_documents"] == 2
            assert result["strategy"] == "docling"

    def test_extract_elements_with_bbox(self, service):
        """Test bounding box extraction from document elements."""
        # Mock document with text elements
        mock_bbox = Mock()
        mock_bbox.l = 10.0
        mock_bbox.t = 20.0
        mock_bbox.r = 100.0
        mock_bbox.b = 30.0
        mock_bbox.coord_origin.value = "TOP_LEFT"

        mock_prov = Mock()
        mock_prov.page_no = 1
        mock_prov.bbox = mock_bbox
        mock_prov.charspan = (0, 10)

        mock_text_item = Mock()
        mock_text_item.text = "Sample text"
        mock_text_item.label.value = "paragraph"
        mock_text_item.prov = [mock_prov]

        mock_doc = Mock()
        mock_doc.texts = [mock_text_item]
        mock_doc.origin.filename = "test.pdf"

        elements = service._extract_elements_with_bbox(mock_doc)

        assert len(elements) == 1
        element = elements[0]
        assert element["text"] == "Sample text"
        assert element["label"] == "paragraph"
        assert element["page_no"] == 1
        assert element["bbox"]["left"] == 10.0
        assert element["bbox"]["top"] == 20.0
        assert element["bbox"]["right"] == 100.0
        assert element["bbox"]["bottom"] == 30.0
        assert element["document_type"] == "document"

    def test_extract_elements_with_bbox_presentation(self, service):
        """Test bounding box extraction for presentation documents."""
        # Mock presentation document
        mock_bbox = Mock()
        mock_bbox.l = 10.0
        mock_bbox.t = 20.0
        mock_bbox.r = 100.0
        mock_bbox.b = 30.0
        mock_bbox.coord_origin.value = "TOP_LEFT"

        mock_prov = Mock()
        mock_prov.page_no = 2
        mock_prov.bbox = mock_bbox

        mock_text_item = Mock()
        mock_text_item.text = "Slide content"
        mock_text_item.label.value = "title"
        mock_text_item.prov = [mock_prov]

        mock_doc = Mock()
        mock_doc.texts = [mock_text_item]
        mock_doc.origin.filename = "presentation.pptx"

        elements = service._extract_elements_with_bbox(mock_doc)

        assert len(elements) == 1
        element = elements[0]
        assert element["text"] == "Slide content"
        assert element["slide_no"] == 2
        assert element["document_type"] == "presentation"
