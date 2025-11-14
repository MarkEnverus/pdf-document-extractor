"""Unit tests for centralized storage utilities."""

import pytest
from unittest.mock import Mock, MagicMock

from src.services.storage_utils import (
    generate_page_text_s3_path,
    generate_results_json_s3_path,
    store_text_file_to_s3,
)


class TestPathGeneration:
    """Test S3 path generation functions."""

    def test_generate_page_text_s3_path_format(self):
        """Test page_text.txt path generation follows correct format."""
        path = generate_page_text_s3_path("project-123", "upload-456", 1)
        assert path == "project-123/upload-456/1/page_text.txt"

    def test_generate_page_text_s3_path_multi_page(self):
        """Test page_text.txt path generation for multiple pages."""
        path1 = generate_page_text_s3_path("proj", "up", 1)
        path2 = generate_page_text_s3_path("proj", "up", 2)
        path3 = generate_page_text_s3_path("proj", "up", 10)

        assert path1 == "proj/up/1/page_text.txt"
        assert path2 == "proj/up/2/page_text.txt"
        assert path3 == "proj/up/10/page_text.txt"

    def test_generate_results_json_s3_path_format(self):
        """Test results.json path generation follows correct format."""
        path = generate_results_json_s3_path("project-123", "upload-456", 1)
        assert path == "project-123/upload-456/1/results.json"

    def test_generate_results_json_s3_path_multi_page(self):
        """Test results.json path generation for multiple pages."""
        path1 = generate_results_json_s3_path("proj", "up", 1)
        path2 = generate_results_json_s3_path("proj", "up", 5)
        path3 = generate_results_json_s3_path("proj", "up", 100)

        assert path1 == "proj/up/1/results.json"
        assert path2 == "proj/up/5/results.json"
        assert path3 == "proj/up/100/results.json"

    def test_path_consistency_between_functions(self):
        """Test that both functions use the same base path structure."""
        project_id = "test-project"
        upload_id = "test-upload"
        page_number = 5

        text_path = generate_page_text_s3_path(project_id, upload_id, page_number)
        json_path = generate_results_json_s3_path(project_id, upload_id, page_number)

        # Both should have the same directory structure
        text_dir = "/".join(text_path.split("/")[:-1])
        json_dir = "/".join(json_path.split("/")[:-1])

        assert text_dir == json_dir == f"{project_id}/{upload_id}/{page_number}"


class TestStoreTextFileToS3:
    """Test store_text_file_to_s3 utility function."""

    def test_encodes_text_to_utf8(self):
        """Test that text content is encoded to UTF-8 bytes."""
        mock_storage = Mock()
        mock_storage.upload_content.return_value = "s3://bucket/path/file.txt"
        mock_logger = Mock()

        text_content = "Hello, world! 你好"
        bucket = "test-bucket"
        s3_key = "test/path/file.txt"

        store_text_file_to_s3(
            storage_service=mock_storage,
            logger=mock_logger,
            text_content=text_content,
            bucket=bucket,
            s3_key=s3_key,
            project_id="proj-123",
            upload_id="upload-456",
        )

        # Verify upload_content was called with encoded bytes
        mock_storage.upload_content.assert_called_once()
        call_args = mock_storage.upload_content.call_args
        encoded_content = call_args[1]["content"]

        assert isinstance(encoded_content, bytes)
        assert encoded_content == text_content.encode("utf-8")

    def test_calls_storage_service_with_correct_parameters(self):
        """Test that storage service is called with all required parameters."""
        mock_storage = Mock()
        mock_storage.upload_content.return_value = "s3://bucket/key"
        mock_logger = Mock()

        store_text_file_to_s3(
            storage_service=mock_storage,
            logger=mock_logger,
            text_content="Test content",
            bucket="my-bucket",
            s3_key="my/key.txt",
            project_id="project-1",
            upload_id="upload-1",
        )

        mock_storage.upload_content.assert_called_once_with(
            content=b"Test content",
            bucket="my-bucket",
            key="my/key.txt",
            project_id="project-1",
            upload_id="upload-1",
        )

    def test_returns_s3_path(self):
        """Test that function returns the S3 path from storage service."""
        expected_path = "s3://test-bucket/test/path/file.txt"
        mock_storage = Mock()
        mock_storage.upload_content.return_value = expected_path
        mock_logger = Mock()

        result = store_text_file_to_s3(
            storage_service=mock_storage,
            logger=mock_logger,
            text_content="content",
            bucket="bucket",
            s3_key="key",
            project_id="proj",
            upload_id="upload",
        )

        assert result == expected_path

    def test_logs_with_default_context(self):
        """Test that function logs with standard context fields."""
        mock_storage = Mock()
        mock_storage.upload_content.return_value = "s3://bucket/key"
        mock_logger = Mock()

        text_content = "Test content for logging"

        store_text_file_to_s3(
            storage_service=mock_storage,
            logger=mock_logger,
            text_content=text_content,
            bucket="bucket",
            s3_key="key",
            project_id="proj-123",
            upload_id="upload-456",
        )

        # Verify logger.debug was called
        mock_logger.debug.assert_called_once()
        call_args = mock_logger.debug.call_args

        # Check log message
        assert call_args[0][0] == "Stored text file to S3"

        # Check log context
        log_context = call_args[1]
        assert log_context["s3_path"] == "s3://bucket/key"
        assert log_context["text_length"] == len(text_content)
        assert log_context["project_id"] == "proj-123"
        assert log_context["upload_id"] == "upload-456"

    def test_logs_with_additional_context(self):
        """Test that function merges additional log context."""
        mock_storage = Mock()
        mock_storage.upload_content.return_value = "s3://bucket/key"
        mock_logger = Mock()

        store_text_file_to_s3(
            storage_service=mock_storage,
            logger=mock_logger,
            text_content="content",
            bucket="bucket",
            s3_key="key",
            project_id="proj",
            upload_id="upload",
            log_context={"page_number": 5, "figure_id": "fig-123"},
        )

        # Verify additional context was included
        log_context = mock_logger.debug.call_args[1]
        assert log_context["page_number"] == 5
        assert log_context["figure_id"] == "fig-123"
        # Standard fields should also be present
        assert "s3_path" in log_context
        assert "text_length" in log_context

    def test_handles_empty_text_content(self):
        """Test that function handles empty string correctly."""
        mock_storage = Mock()
        mock_storage.upload_content.return_value = "s3://bucket/key"
        mock_logger = Mock()

        result = store_text_file_to_s3(
            storage_service=mock_storage,
            logger=mock_logger,
            text_content="",
            bucket="bucket",
            s3_key="key",
            project_id="proj",
            upload_id="upload",
        )

        # Should still upload (empty file is valid)
        mock_storage.upload_content.assert_called_once()
        assert result == "s3://bucket/key"

        # Log should show zero length
        log_context = mock_logger.debug.call_args[1]
        assert log_context["text_length"] == 0

    def test_handles_unicode_text_content(self):
        """Test that function handles Unicode characters correctly."""
        mock_storage = Mock()
        mock_storage.upload_content.return_value = "s3://bucket/key"
        mock_logger = Mock()

        unicode_text = "Hello 世界 🌍 Привет"

        store_text_file_to_s3(
            storage_service=mock_storage,
            logger=mock_logger,
            text_content=unicode_text,
            bucket="bucket",
            s3_key="key",
            project_id="proj",
            upload_id="upload",
        )

        # Verify content was encoded correctly
        call_args = mock_storage.upload_content.call_args
        encoded_content = call_args[1]["content"]

        assert encoded_content == unicode_text.encode("utf-8")

    def test_handles_multiline_text_content(self):
        """Test that function handles multi-line text correctly."""
        mock_storage = Mock()
        mock_storage.upload_content.return_value = "s3://bucket/key"
        mock_logger = Mock()

        multiline_text = """Line 1
Line 2
Line 3"""

        result = store_text_file_to_s3(
            storage_service=mock_storage,
            logger=mock_logger,
            text_content=multiline_text,
            bucket="bucket",
            s3_key="key",
            project_id="proj",
            upload_id="upload",
        )

        mock_storage.upload_content.assert_called_once()
        assert result == "s3://bucket/key"

        # Verify text length includes newlines
        log_context = mock_logger.debug.call_args[1]
        assert log_context["text_length"] == len(multiline_text)
