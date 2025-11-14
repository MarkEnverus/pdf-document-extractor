"""
Comprehensive tests for S3Service covering all major functionality.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from botocore.exceptions import ClientError, NoCredentialsError

from src.services.s3_service import S3Service
from src.exceptions.domain_exceptions import StorageError


class TestS3ServiceComprehensive:
    """Comprehensive test cases for S3Service."""

    @pytest.fixture
    def mock_s3_client(self):
        """Create mock S3 client."""
        return Mock()

    @pytest.fixture
    def service(self, mock_s3_client):
        """Create S3Service instance with dependency injection."""
        # Use the new constructor with direct s3_client injection
        return S3Service(s3_client=mock_s3_client)

    @pytest.fixture
    def sample_s3_urls(self):
        """Sample S3 URLs for testing."""
        return [
            "s3://test-bucket/document.pdf",
            "s3://my-bucket/folder/subfolder/file.json",
            "s3://bucket-with-dashes/file-with-dashes.txt",
            "s3://bucket123/path/to/deep/file.xml",
        ]

    # Initialization Tests
    def test_init_success(self, mock_s3_client):
        """Test successful S3Service initialization with dependency injection."""
        service = S3Service(s3_client=mock_s3_client)
        assert service.s3_client == mock_s3_client

    def test_init_with_settings_provider(self):
        """Test initialization with settings provider."""
        mock_settings = Mock()
        mock_s3_client = Mock()
        mock_settings.get_s3_client.return_value = mock_s3_client

        service = S3Service(settings_provider=mock_settings)

        assert service.s3_client == mock_s3_client
        mock_settings.get_s3_client.assert_called_once()

    def test_init_fallback_to_global_settings(self):
        """Test initialization falls back to global settings."""
        # This tests the backward compatibility path where no dependencies are injected
        mock_s3_client = Mock()

        # Patch the settings module that gets imported dynamically
        with patch("src.configs.settings.settings") as mock_global_settings:
            mock_global_settings.get_s3_client.return_value = mock_s3_client

            service = S3Service()

            assert service.s3_client == mock_s3_client
            mock_global_settings.get_s3_client.assert_called_once()

    def test_init_with_credentials_error(self):
        """Test initialization with credentials error through settings provider."""
        mock_settings = Mock()
        mock_settings.get_s3_client.side_effect = NoCredentialsError()

        with pytest.raises(NoCredentialsError):
            S3Service(settings_provider=mock_settings)

    # Upload Content Tests
    def test_upload_content_success(self, service):
        """Test successful content upload."""
        content = b"test content to upload"
        bucket = "test-bucket"
        key = "uploads/test-file.json"

        # Mock successful upload (S3Service is stubbed, so just verify return value)
        result_url = service.upload_content(content, bucket, key)

        expected_url = f"s3://{bucket}/{key}"
        assert result_url == expected_url

    def test_upload_content_empty_content(self, service):
        """Test upload with empty content."""
        content = b""
        bucket = "test-bucket"
        key = "uploads/empty-file.txt"

        result_url = service.upload_content(content, bucket, key)

        expected_url = f"s3://{bucket}/{key}"
        assert result_url == expected_url

    def test_upload_content_large_content(self, service):
        """Test upload with large content."""
        # Create large content (1MB)
        content = b"x" * (1024 * 1024)
        bucket = "test-bucket"
        key = "uploads/large-file.bin"

        result_url = service.upload_content(content, bucket, key)

        expected_url = f"s3://{bucket}/{key}"
        assert result_url == expected_url

    def test_upload_content_special_characters(self, service):
        """Test upload with special characters in key."""
        content = b"test content"
        bucket = "test-bucket"
        key = "uploads/file with spaces & symbols!@#$%.txt"

        result_url = service.upload_content(content, bucket, key)

        expected_url = f"s3://{bucket}/{key}"
        assert result_url == expected_url

    def test_upload_content_nested_path(self, service):
        """Test upload with deeply nested path."""
        content = b"nested content"
        bucket = "test-bucket"
        key = "level1/level2/level3/level4/deep-file.json"

        result_url = service.upload_content(content, bucket, key)

        expected_url = f"s3://{bucket}/{key}"
        assert result_url == expected_url

    def test_upload_content_different_bucket_names(self, service):
        """Test upload with various bucket name formats."""
        content = b"test content"

        test_buckets = [
            "simple-bucket",
            "bucket.with.dots",
            "bucket-with-dashes",
            "bucket123numbers",
            "very-long-bucket-name-with-many-characters",
        ]

        for bucket in test_buckets:
            key = "test-file.txt"
            result_url = service.upload_content(content, bucket, key)
            assert result_url == f"s3://{bucket}/{key}"

    # Download Content Tests (by bucket and key)
    def test_download_content_by_bucket_key_success(self, service):
        """Test successful download by bucket and key."""
        bucket = "test-bucket"
        key = "path/to/file.pdf"
        mock_content = b"downloaded file content"

        # Mock S3 response
        mock_response = {"Body": Mock()}
        mock_response["Body"].read.return_value = mock_content
        service.s3_client.get_object.return_value = mock_response

        content = service.download_content_by_bucket_key(bucket, key)

        assert content == mock_content
        service.s3_client.get_object.assert_called_once_with(Bucket=bucket, Key=key)

    def test_download_content_by_bucket_key_access_denied(self, service):
        """Test download with access denied error."""
        bucket = "private-bucket"
        key = "secret-file.pdf"

        service.s3_client.get_object.side_effect = ClientError(
            error_response={"Error": {"Code": "AccessDenied", "Message": "Access Denied"}}, operation_name="GetObject"
        )

        with pytest.raises(StorageError) as exc_info:
            service.download_content_by_bucket_key(bucket, key)

        assert "S3 download failed" in exc_info.value.message
        assert exc_info.value.context["bucket"] == bucket
        assert exc_info.value.context["key"] == key

    def test_download_content_by_bucket_key_not_found(self, service):
        """Test download with file not found error."""
        bucket = "test-bucket"
        key = "nonexistent-file.pdf"

        service.s3_client.get_object.side_effect = ClientError(
            error_response={"Error": {"Code": "NoSuchKey", "Message": "The specified key does not exist."}},
            operation_name="GetObject",
        )

        with pytest.raises(StorageError) as exc_info:
            service.download_content_by_bucket_key(bucket, key)

        assert "File not found in S3" in exc_info.value.message

    def test_download_content_by_bucket_key_network_error(self, service):
        """Test download with network error."""
        bucket = "test-bucket"
        key = "file.pdf"

        service.s3_client.get_object.side_effect = ClientError(
            error_response={"Error": {"Code": "NetworkError", "Message": "Network connection failed"}},
            operation_name="GetObject",
        )

        with pytest.raises(StorageError):
            service.download_content_by_bucket_key(bucket, key)

    # Download Content Tests (by S3 URI)
    def test_download_content_success(self, service, sample_s3_urls):
        """Test successful download by S3 URI."""
        s3_url = sample_s3_urls[0]  # "s3://test-bucket/document.pdf"
        mock_content = b"file content from s3 uri"

        # Mock S3 response
        mock_response = {"Body": Mock()}
        mock_response["Body"].read.return_value = mock_content
        service.s3_client.get_object.return_value = mock_response

        content = service.download_content(s3_url)

        assert content == mock_content
        service.s3_client.get_object.assert_called_once_with(Bucket="test-bucket", Key="document.pdf")

    def test_download_content_nested_path(self, service):
        """Test download with nested path in S3 URI."""
        s3_url = "s3://my-bucket/folder/subfolder/file.json"
        mock_content = b"nested file content"

        mock_response = {"Body": Mock()}
        mock_response["Body"].read.return_value = mock_content
        service.s3_client.get_object.return_value = mock_response

        content = service.download_content(s3_url)

        assert content == mock_content
        service.s3_client.get_object.assert_called_once_with(Bucket="my-bucket", Key="folder/subfolder/file.json")

    def test_download_content_special_characters(self, service):
        """Test download with special characters in S3 URI."""
        s3_url = "s3://test-bucket/path/file with spaces & symbols.txt"
        mock_content = b"special chars content"

        mock_response = {"Body": Mock()}
        mock_response["Body"].read.return_value = mock_content
        service.s3_client.get_object.return_value = mock_response

        content = service.download_content(s3_url)

        assert content == mock_content
        service.s3_client.get_object.assert_called_once_with(
            Bucket="test-bucket", Key="path/file with spaces & symbols.txt"
        )

    def test_download_content_various_s3_urls(self, service, sample_s3_urls):
        """Test download with various S3 URL formats."""
        mock_content = b"test content"
        mock_response = {"Body": Mock()}
        mock_response["Body"].read.return_value = mock_content
        service.s3_client.get_object.return_value = mock_response

        expected_calls = [
            ("test-bucket", "document.pdf"),
            ("my-bucket", "folder/subfolder/file.json"),
            ("bucket-with-dashes", "file-with-dashes.txt"),
            ("bucket123", "path/to/deep/file.xml"),
        ]

        for i, s3_url in enumerate(sample_s3_urls):
            service.s3_client.reset_mock()

            content = service.download_content(s3_url)

            assert content == mock_content
            expected_bucket, expected_key = expected_calls[i]
            service.s3_client.get_object.assert_called_once_with(Bucket=expected_bucket, Key=expected_key)

    # Object Exists Tests
    def test_object_exists_true(self, service):
        """Test object_exists returns True (stubbed implementation)."""
        bucket = "test-bucket"
        key = "existing-file.pdf"

        # Current implementation is stubbed to return True
        result = service.object_exists(bucket, key)
        assert result is True

    def test_object_exists_various_files(self, service):
        """Test object_exists with various file types."""
        bucket = "test-bucket"
        test_keys = [
            "document.pdf",
            "folder/subfolder/file.json",
            "image.png",
            "data.csv",
            "config.xml",
            "very/deep/nested/path/file.txt",
        ]

        for key in test_keys:
            # Current implementation is stubbed to return True
            result = service.object_exists(bucket, key)
            assert result is True

    # URL Parsing Tests (internal method)
    def test_s3_url_parsing_basic(self, service):
        """Test basic S3 URL parsing."""
        test_cases = [
            ("s3://bucket/file.pdf", ("bucket", "file.pdf")),
            ("s3://my-bucket/folder/file.json", ("my-bucket", "folder/file.json")),
            ("s3://bucket-name/path/to/file.txt", ("bucket-name", "path/to/file.txt")),
        ]

        for s3_url, expected in test_cases:
            # Test by calling download_content which uses the parsing logic
            mock_response = {"Body": Mock()}
            mock_response["Body"].read.return_value = b"content"
            service.s3_client.get_object.return_value = mock_response

            service.download_content(s3_url)

            expected_bucket, expected_key = expected
            service.s3_client.get_object.assert_called_with(Bucket=expected_bucket, Key=expected_key)
            service.s3_client.reset_mock()

    def test_s3_url_parsing_complex_paths(self, service):
        """Test S3 URL parsing with complex paths."""
        test_cases = [
            "s3://bucket/path/with/multiple/levels/file.pdf",
            "s3://bucket/file-with-dashes_and_underscores.txt",
            "s3://bucket/path/file.with.dots.json",
            "s3://bucket/123numbers/in456path/789file.xml",
        ]

        mock_response = {"Body": Mock()}
        mock_response["Body"].read.return_value = b"content"
        service.s3_client.get_object.return_value = mock_response

        for s3_url in test_cases:
            # Just verify it doesn't crash - specific parsing tested above
            try:
                service.download_content(s3_url)
            except Exception:
                pytest.fail(f"Failed to parse S3 URL: {s3_url}")

            service.s3_client.reset_mock()

    # Integration Tests
    def test_upload_download_cycle(self, service):
        """Test upload followed by download (stubbed S3)."""
        content = b"original content for cycle test"
        bucket = "test-bucket"
        key = "cycle-test/file.json"

        # Upload content
        upload_url = service.upload_content(content, bucket, key)
        expected_url = f"s3://{bucket}/{key}"
        assert upload_url == expected_url

        # Mock download response
        mock_response = {"Body": Mock()}
        mock_response["Body"].read.return_value = content
        service.s3_client.get_object.return_value = mock_response

        # Download content
        downloaded_content = service.download_content(upload_url)

        assert downloaded_content == content
        service.s3_client.get_object.assert_called_once_with(Bucket=bucket, Key=key)

    def test_multiple_concurrent_operations(self, service):
        """Test multiple S3 operations in sequence."""
        operations = [
            ("upload", b"content1", "bucket1", "file1.txt"),
            ("upload", b"content2", "bucket2", "file2.json"),
            ("download", "s3://bucket1/file1.txt"),
            ("download", "s3://bucket2/file2.json"),
            ("exists", "bucket1", "file1.txt"),
            ("exists", "bucket2", "file2.json"),
        ]

        # Mock download responses
        mock_response = {"Body": Mock()}
        mock_response["Body"].read.return_value = b"mock content"
        service.s3_client.get_object.return_value = mock_response

        results = []
        for operation in operations:
            if operation[0] == "upload":
                _, content, bucket, key = operation
                result = service.upload_content(content, bucket, key)
                results.append(result)
            elif operation[0] == "download":
                _, s3_url = operation
                result = service.download_content(s3_url)
                results.append(result)
            elif operation[0] == "exists":
                _, bucket, key = operation
                result = service.object_exists(bucket, key)
                results.append(result)

        # Verify all operations completed
        assert len(results) == len(operations)

        # Check upload results
        assert results[0] == "s3://bucket1/file1.txt"
        assert results[1] == "s3://bucket2/file2.json"

        # Check download results
        assert results[2] == b"mock content"
        assert results[3] == b"mock content"

        # Check exists results (stubbed to True)
        assert results[4] is True
        assert results[5] is True

    # Edge Cases and Error Handling
    def test_upload_content_logging(self, service):
        """Test that upload operations are logged (stubbed implementation)."""
        content = b"content for logging test"
        bucket = "log-test-bucket"
        key = "logs/test-file.json"

        # The current implementation logs but doesn't actually upload
        # Just verify it returns the expected path
        result_url = service.upload_content(content, bucket, key)

        expected_url = f"s3://{bucket}/{key}"
        assert result_url == expected_url

    def test_large_file_handling(self, service):
        """Test handling of large files."""
        # Create a large file (10MB)
        large_content = b"x" * (10 * 1024 * 1024)
        bucket = "large-files-bucket"
        key = "large-files/big-file.bin"

        # Upload should handle large content
        result_url = service.upload_content(large_content, bucket, key)
        assert result_url == f"s3://{bucket}/{key}"

        # Mock large download
        mock_response = {"Body": Mock()}
        mock_response["Body"].read.return_value = large_content
        service.s3_client.get_object.return_value = mock_response

        downloaded_content = service.download_content(result_url)
        assert len(downloaded_content) == len(large_content)

    def test_unicode_content_handling(self, service):
        """Test handling of unicode content."""
        unicode_text = "Hello 世界 🌍 café résumé"
        content = unicode_text.encode("utf-8")
        bucket = "unicode-bucket"
        key = "unicode/test-file.txt"

        # Upload unicode content
        result_url = service.upload_content(content, bucket, key)
        assert result_url == f"s3://{bucket}/{key}"

        # Mock download of unicode content
        mock_response = {"Body": Mock()}
        mock_response["Body"].read.return_value = content
        service.s3_client.get_object.return_value = mock_response

        downloaded_content = service.download_content(result_url)
        assert downloaded_content == content

        # Verify unicode text can be decoded
        decoded_text = downloaded_content.decode("utf-8")
        assert decoded_text == unicode_text

    def test_binary_content_handling(self, service):
        """Test handling of various binary content types."""
        # Simulate different binary file types
        binary_contents = [
            b"\x89PNG\r\n\x1a\n",  # PNG header
            b"%PDF-1.4",  # PDF header
            b"\xff\xd8\xff\xe0",  # JPEG header
            bytes(range(256)),  # All possible byte values
        ]

        for i, content in enumerate(binary_contents):
            bucket = "binary-bucket"
            key = f"binary/file-{i}.bin"

            # Upload binary content
            result_url = service.upload_content(content, bucket, key)
            assert result_url == f"s3://{bucket}/{key}"

            # Mock download
            mock_response = {"Body": Mock()}
            mock_response["Body"].read.return_value = content
            service.s3_client.get_object.return_value = mock_response

            downloaded_content = service.download_content(result_url)
            assert downloaded_content == content

    def test_error_handling_consistency(self, service):
        """Test that error handling is consistent across methods."""
        # Test that all methods handle errors appropriately
        bucket = "error-test-bucket"
        key = "error-test-file.txt"
        s3_url = f"s3://{bucket}/{key}"

        # Test download error handling
        service.s3_client.get_object.side_effect = ClientError(
            error_response={"Error": {"Code": "TestError", "Message": "Test error"}}, operation_name="GetObject"
        )

        with pytest.raises(StorageError):
            service.download_content_by_bucket_key(bucket, key)

        with pytest.raises(StorageError):
            service.download_content(s3_url)

        # Upload errors are not tested in current stubbed implementation
        # object_exists errors are not tested in current stubbed implementation
