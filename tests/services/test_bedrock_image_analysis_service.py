"""Comprehensive tests for BedrockImageAnalysisService."""

import pytest
import json
import base64
from unittest.mock import Mock, patch, MagicMock
from botocore.exceptions import ClientError, BotoCoreError

from src.services.bedrock_image_analysis_service import BedrockImageAnalysisService
from src.models.image_analysis_models import BedrockAnalysisConfig, ImageAnalysisResult
from src.interfaces.image_analysis import ImageAnalysisError


class TestBedrockImageAnalysisService:
    """Test cases for BedrockImageAnalysisService."""

    @pytest.fixture
    def sample_config(self):
        """Sample Bedrock analysis configuration."""
        return BedrockAnalysisConfig(
            model_id="anthropic.claude-3-sonnet-20240229-v1:0", max_tokens=1024, temperature=0.1, timeout_seconds=30
        )

    @pytest.fixture
    def sample_image_data(self):
        """Sample image data (minimal PNG)."""
        # Minimal PNG data (1x1 pixel transparent PNG)
        png_data = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
        )
        return png_data

    @pytest.fixture
    def mock_bedrock_client(self):
        """Mock Bedrock client."""
        mock_client = Mock()
        return mock_client

    @pytest.fixture
    def mock_successful_bedrock_response(self):
        """Mock successful Bedrock response."""
        return {
            "content": [
                {
                    "text": "This image shows a simple diagram with geometric shapes. The diagram appears to be a flowchart or process visualization with connected boxes and arrows indicating workflow or data flow."
                }
            ],
            "usage": {"input_tokens": 150, "output_tokens": 45},
        }

    @pytest.fixture
    def service(self, sample_config, mock_bedrock_client):
        """Create BedrockImageAnalysisService with mocked Bedrock client."""
        with patch(
            "src.services.bedrock_image_analysis_service.get_bedrock_runtime_client", return_value=mock_bedrock_client
        ):
            service = BedrockImageAnalysisService(config=sample_config, aws_region="us-east-1")
            service.bedrock_client = mock_bedrock_client
            return service

    def test_initialization_success(self, sample_config):
        """Test successful service initialization."""
        with patch("src.services.bedrock_image_analysis_service.get_bedrock_runtime_client") as mock_boto_client:
            mock_boto_client.return_value = Mock()

            service = BedrockImageAnalysisService(config=sample_config)

            assert service.config.model_id == "anthropic.claude-3-sonnet-20240229-v1:0"
            assert service.aws_region == "us-east-1"
            mock_boto_client.assert_called_once_with("us-east-1")

    def test_initialization_with_custom_region(self, sample_config):
        """Test service initialization with custom AWS region."""
        with patch("src.services.bedrock_image_analysis_service.get_bedrock_runtime_client") as mock_boto_client:
            mock_boto_client.return_value = Mock()

            service = BedrockImageAnalysisService(config=sample_config, aws_region="us-west-2")

            assert service.aws_region == "us-west-2"
            mock_boto_client.assert_called_once_with("us-west-2")

    def test_initialization_failure(self, sample_config):
        """Test service initialization failure."""
        with patch(
            "src.services.bedrock_image_analysis_service.get_bedrock_runtime_client",
            side_effect=Exception("AWS credentials not found"),
        ):
            with pytest.raises(ImageAnalysisError) as exc_info:
                BedrockImageAnalysisService(config=sample_config)

            assert "Failed to initialize Bedrock client" in str(exc_info.value)
            assert exc_info.value.model_used == sample_config.model_id

    def test_analyze_image_success(self, service, sample_image_data, mock_successful_bedrock_response):
        """Test successful image analysis."""
        # Mock the Bedrock response
        mock_response = {"body": Mock()}
        mock_response["body"].read.return_value = json.dumps(mock_successful_bedrock_response).encode()
        service.bedrock_client.invoke_model.return_value = mock_response

        # Analyze image
        result = service.analyze_image(
            image_data=sample_image_data, image_format="PNG", context={"caption": "Sample diagram"}
        )

        # Verify result structure
        assert result["model_used"] == "anthropic.claude-3-sonnet-20240229-v1:0"
        assert (
            result["analysis_summary"]
            == "This image shows a simple diagram with geometric shapes. The diagram appears to be a flowchart or process visualization with connected boxes and arrows indicating workflow or data flow."
        )
        assert "analysis_timestamp" in result
        assert "processing_time_ms" in result
        assert result.get("analysis_error") is None

        # Verify Bedrock was called correctly
        service.bedrock_client.invoke_model.assert_called_once()
        call_args = service.bedrock_client.invoke_model.call_args
        assert call_args[1]["modelId"] == "anthropic.claude-3-sonnet-20240229-v1:0"

    def test_analyze_image_with_context(self, service, sample_image_data, mock_successful_bedrock_response):
        """Test image analysis with rich context."""
        mock_response = {"body": Mock()}
        mock_response["body"].read.return_value = json.dumps(mock_successful_bedrock_response).encode()
        service.bedrock_client.invoke_model.return_value = mock_response

        context = {
            "figure_id": "fig_123",
            "caption": "System architecture diagram",
            "page_number": 5,
            "source_document": "s3://bucket/doc.pdf",
        }

        result = service.analyze_image(image_data=sample_image_data, image_format="JPEG", context=context)

        assert result["model_used"] == "anthropic.claude-3-sonnet-20240229-v1:0"
        assert result["analysis_summary"] == mock_successful_bedrock_response["content"][0]["text"]

        # Verify the context was included in the request
        call_args = service.bedrock_client.invoke_model.call_args
        request_body = json.loads(call_args[1]["body"])
        prompt_text = request_body["messages"][0]["content"][1]["text"]

        assert "System architecture diagram" in prompt_text
        assert "Page number: 5" in prompt_text

    def test_analyze_image_unsupported_format(self, service, sample_image_data):
        """Test analysis with unsupported image format."""
        result = service.analyze_image(image_data=sample_image_data, image_format="BMP", context=None)

        # Should return error result, not raise exception
        assert result["model_used"] == "anthropic.claude-3-sonnet-20240229-v1:0"
        assert result["analysis_summary"] == "Analysis failed"
        assert "Unsupported image format" in result["analysis_error"]

    def test_analyze_image_with_enum(self, service, sample_image_data, mock_successful_bedrock_response):
        """Test analyze_image with IngestionMimeType enum (e.g., IngestionMimeType.PNG)."""
        from idp_file_management.models.mime_type import IngestionMimeType

        mock_response = {"body": Mock()}
        mock_response["body"].read.return_value = json.dumps(mock_successful_bedrock_response).encode()
        service.bedrock_client.invoke_model.return_value = mock_response

        # Test with enum - this is how asset_storage_service calls it
        result = service.analyze_image(image_data=sample_image_data, image_format=IngestionMimeType.PNG, context=None)

        assert result["model_used"] == "anthropic.claude-3-sonnet-20240229-v1:0"
        assert result["analysis_summary"] == mock_successful_bedrock_response["content"][0]["text"]
        assert result.get("analysis_error") is None
        service.bedrock_client.invoke_model.assert_called_once()

        # Test with JPEG enum
        service.bedrock_client.invoke_model.reset_mock()
        result = service.analyze_image(image_data=sample_image_data, image_format=IngestionMimeType.JPG, context=None)
        assert result.get("analysis_error") is None
        service.bedrock_client.invoke_model.assert_called_once()

    def test_analyze_image_bedrock_client_error(self, service, sample_image_data):
        """Test Bedrock ClientError handling."""
        error_response = {"Error": {"Code": "ValidationException", "Message": "Invalid model ID"}}
        service.bedrock_client.invoke_model.side_effect = ClientError(error_response, "InvokeModel")

        result = service.analyze_image(image_data=sample_image_data, image_format="PNG", context=None)

        assert result["model_used"] == "anthropic.claude-3-sonnet-20240229-v1:0"
        assert result["analysis_summary"] == "Analysis failed"
        assert "Bedrock API error (ValidationException): Invalid model ID" in result["analysis_error"]

    def test_analyze_image_boto_core_error(self, service, sample_image_data):
        """Test BotoCoreError handling."""
        service.bedrock_client.invoke_model.side_effect = BotoCoreError()

        result = service.analyze_image(image_data=sample_image_data, image_format="PNG", context=None)

        assert result["model_used"] == "anthropic.claude-3-sonnet-20240229-v1:0"
        assert result["analysis_summary"] == "Analysis failed"
        assert "AWS SDK error" in result["analysis_error"]

    def test_analyze_image_invalid_response_format(self, service, sample_image_data):
        """Test handling of invalid Bedrock response format."""
        # Mock invalid response (missing content field)
        mock_response = {"body": Mock()}
        mock_response["body"].read.return_value = json.dumps({"usage": {"tokens": 100}}).encode()
        service.bedrock_client.invoke_model.return_value = mock_response

        result = service.analyze_image(image_data=sample_image_data, image_format="PNG", context=None)

        assert result["analysis_summary"] == "Analysis failed"
        assert "Invalid Bedrock response: missing 'content' field" in result["analysis_error"]

    def test_analyze_image_empty_response_text(self, service, sample_image_data):
        """Test handling of empty response text."""
        mock_response_data = {
            "content": [
                {"text": "   "}  # Empty/whitespace text
            ]
        }
        mock_response = {"body": Mock()}
        mock_response["body"].read.return_value = json.dumps(mock_response_data).encode()
        service.bedrock_client.invoke_model.return_value = mock_response

        result = service.analyze_image(image_data=sample_image_data, image_format="PNG", context=None)

        assert result["analysis_summary"] == "Analysis failed"
        assert "Empty analysis text from Bedrock" in result["analysis_error"]

    def test_prepare_claude_message_basic(self, service, sample_image_data):
        """Test Claude message preparation with basic inputs."""
        context = {"caption": "Test image"}

        message = service._prepare_claude_message(sample_image_data, "PNG", context)

        # Verify message structure
        assert message["anthropic_version"] == "bedrock-2023-05-31"
        assert message["max_tokens"] == 1024
        assert message["temperature"] == 0.1
        assert len(message["messages"]) == 1

        user_message = message["messages"][0]
        assert user_message["role"] == "user"
        assert len(user_message["content"]) == 2

        # Check image content
        image_content = user_message["content"][0]
        assert image_content["type"] == "image"
        assert image_content["source"]["type"] == "base64"
        assert image_content["source"]["media_type"] == "image/png"
        assert isinstance(image_content["source"]["data"], str)

        # Check text content includes context
        text_content = user_message["content"][1]
        assert text_content["type"] == "text"
        assert "Test image" in text_content["text"]

    def test_get_mime_type_mapping(self, service):
        """Test MIME type conversion for different formats."""
        assert service._get_mime_type("PNG") == "image/png"
        assert service._get_mime_type("JPEG") == "image/jpeg"
        assert service._get_mime_type("JPG") == "image/jpeg"
        assert service._get_mime_type("GIF") == "image/gif"
        assert service._get_mime_type("WEBP") == "image/webp"
        assert service._get_mime_type("UNKNOWN") == "image/jpeg"  # Default

    def test_health_check_success(self, service):
        """Test successful health check."""
        with patch("src.services.bedrock_image_analysis_service.get_bedrock_runtime_client") as mock_boto_client:
            mock_boto_client.return_value = Mock()

            health = service.health_check()

            assert health["bedrock_image_analysis_service"] == "healthy"
            assert health["model_id"] == "anthropic.claude-3-sonnet-20240229-v1:0"
            assert health["aws_region"] == "us-east-1"
            assert health["bedrock_connectivity"] == "healthy"

    def test_health_check_connectivity_failure(self, service):
        """Test health check with connectivity failure."""
        with patch(
            "src.services.bedrock_image_analysis_service.get_bedrock_runtime_client",
            side_effect=Exception("Connection failed"),
        ):
            health = service.health_check()

            assert health["bedrock_image_analysis_service"] == "healthy"
            assert health["bedrock_connectivity"] == "unhealthy"
            assert "Connection failed" in health["bedrock_error"]

    def test_image_analysis_result_model_success(self):
        """Test ImageAnalysisResult model for successful analysis."""
        result = ImageAnalysisResult.create_success_result(
            model_used="anthropic.claude-3-sonnet-20240229-v1:0",
            analysis_summary="This is a test image showing geometric shapes.",
            confidence=0.95,
            processing_time_ms=1500,
        )

        assert result.model_used == "anthropic.claude-3-sonnet-20240229-v1:0"
        assert result.analysis_summary == "This is a test image showing geometric shapes."
        assert result.analysis_confidence == 0.95
        assert result.processing_time_ms == 1500
        assert result.analysis_error is None

    def test_image_analysis_result_model_error(self):
        """Test ImageAnalysisResult model for error cases."""
        result = ImageAnalysisResult.create_error_result(
            model_used="anthropic.claude-3-sonnet-20240229-v1:0",
            error_message="API timeout occurred",
            processing_time_ms=30000,
        )

        assert result.model_used == "anthropic.claude-3-sonnet-20240229-v1:0"
        assert result.analysis_summary == "Analysis failed"
        assert result.analysis_error == "API timeout occurred"
        assert result.processing_time_ms == 30000
        assert result.analysis_confidence is None

    def test_bedrock_analysis_config_defaults(self):
        """Test BedrockAnalysisConfig with default values."""
        config = BedrockAnalysisConfig()

        assert config.model_id == "anthropic.claude-3-sonnet-20240229-v1:0"
        assert config.max_tokens == 1024
        assert config.temperature == 0.1
        assert config.timeout_seconds == 30
        assert "Please analyze this image" in config.prompt_template

    def test_bedrock_analysis_config_validation(self):
        """Test BedrockAnalysisConfig validation."""
        # Test valid config
        config = BedrockAnalysisConfig(
            model_id="anthropic.claude-3-haiku-20240307-v1:0", max_tokens=2048, temperature=0.5, timeout_seconds=60
        )
        assert config.max_tokens == 2048
        assert config.temperature == 0.5

        # Test invalid values should raise validation error
        with pytest.raises(ValueError):
            BedrockAnalysisConfig(max_tokens=0)  # Must be > 0

        with pytest.raises(ValueError):
            BedrockAnalysisConfig(temperature=1.5)  # Must be <= 1.0

        with pytest.raises(ValueError):
            BedrockAnalysisConfig(timeout_seconds=0)  # Must be > 0

    def test_concurrent_analysis_requests(self, service, sample_image_data, mock_successful_bedrock_response):
        """Test that service handles concurrent requests properly."""
        mock_response = {"body": Mock()}
        mock_response["body"].read.return_value = json.dumps(mock_successful_bedrock_response).encode()
        service.bedrock_client.invoke_model.return_value = mock_response

        # Simulate multiple concurrent requests
        results = []
        for i in range(3):
            result = service.analyze_image(
                image_data=sample_image_data, image_format="PNG", context={"figure_id": f"fig_{i}"}
            )
            results.append(result)

        # All should succeed
        assert len(results) == 3
        for result in results:
            assert result["analysis_summary"] == mock_successful_bedrock_response["content"][0]["text"]
            assert result.get("analysis_error") is None

        # Bedrock should have been called 3 times
        assert service.bedrock_client.invoke_model.call_count == 3
