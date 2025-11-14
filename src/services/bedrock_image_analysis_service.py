"""Bedrock image analysis service for LLM-based image processing."""

import json
import base64
import time
from typing import Dict, Any, Optional

from botocore.exceptions import ClientError, BotoCoreError
from lib.aws import get_bedrock_runtime_client

from lib.logger import Logger
from src.interfaces.image_analysis import AbstractImageAnalysisService, ImageAnalysisError
from src.models.image_analysis_models import ImageAnalysisResult, BedrockAnalysisConfig

logger = Logger.get_logger(__name__)


class BedrockImageAnalysisService(AbstractImageAnalysisService):
    """
    Bedrock-based image analysis service using Claude vision models.

    This service analyzes images using AWS Bedrock's Claude models to generate
    detailed descriptions and analysis for document processing workflows.
    """

    def __init__(self, config: BedrockAnalysisConfig, aws_region: Optional[str] = None):
        """
        Initialize BedrockImageAnalysisService.

        Args:
            config: Configuration for Bedrock analysis
            aws_region: AWS region for Bedrock client (defaults to us-east-1)
        """
        self.config = config
        self.aws_region = aws_region or "us-east-1"
        self.logger = logger

        try:
            self.bedrock_client = get_bedrock_runtime_client(self.aws_region)
            self.logger.info(
                "Bedrock image analysis service initialized", model_id=self.config.model_id, region=self.aws_region
            )
        except Exception as e:
            self.logger.error("Failed to initialize Bedrock client", error=str(e), region=self.aws_region)
            raise ImageAnalysisError(
                f"Failed to initialize Bedrock client: {str(e)}", model_used=self.config.model_id, original_error=e
            )

    def analyze_image(
        self, image_data: bytes, image_format: str, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Analyze an image using Bedrock Claude and return description/analysis.

        Args:
            image_data: Raw image bytes
            image_format: Image format (PNG, JPEG, etc.)
            context: Optional context about the image (caption, source document, etc.)

        Returns:
            Dictionary containing analysis results

        Raises:
            ImageAnalysisError: If analysis fails critically
        """
        start_time = time.time()

        try:
            self.logger.info(
                "Starting Bedrock image analysis",
                model_id=self.config.model_id,
                image_format=image_format,
                image_size_bytes=len(image_data),
            )

            # Validate image format
            # Handle IngestionMimeType enum objects (e.g., IngestionMimeType.PNG)
            if hasattr(image_format, "value"):
                image_format = image_format.value  # Extract "image/png" from enum

            # Bedrock Claude supports: image/png, image/jpeg, image/gif, image/webp
            # Derive from IngestionMimeType enum for supported formats, plus additional Bedrock formats
            from lib.models.mime_type import IngestionMimeType

            # Get image formats from enum (PNG, JPG)
            supported_formats = {
                "PNG",  # From IngestionMimeType.PNG
                "JPEG",  # From IngestionMimeType.JPG
                "JPG",  # Alias for JPEG
                "GIF",  # Bedrock supports but not in enum (optional format)
                "WEBP",  # Bedrock supports but not in enum (optional format)
            }

            format_upper = image_format.upper()

            # Handle MIME types (e.g., "image/png" -> "PNG")
            if "/" in format_upper:
                format_upper = format_upper.split("/")[1]  # "IMAGE/PNG" -> "PNG"

            if format_upper not in supported_formats:
                processing_time_ms = int((time.time() - start_time) * 1000)
                result = ImageAnalysisResult.create_error_result(
                    model_used=self.config.model_id,
                    error_message=f"Unsupported image format: {image_format}. Supported: {supported_formats}",
                    processing_time_ms=processing_time_ms,
                )
                return result.to_dict()

            # Prepare the message for Claude
            message = self._prepare_claude_message(image_data, image_format, context)

            # Call Bedrock
            response = self._call_bedrock(message)

            # Parse response
            analysis_summary = self._parse_bedrock_response(response)

            processing_time_ms = int((time.time() - start_time) * 1000)

            # Create successful result
            result = ImageAnalysisResult.create_success_result(
                model_used=self.config.model_id,
                analysis_summary=analysis_summary,
                processing_time_ms=processing_time_ms,
            )

            self.logger.info(
                "Bedrock image analysis completed successfully",
                model_id=self.config.model_id,
                processing_time_ms=processing_time_ms,
                summary_length=len(analysis_summary),
            )

            return result.to_dict()

        except ImageAnalysisError as e:
            # Convert ImageAnalysisError to error result
            processing_time_ms = int((time.time() - start_time) * 1000)
            result = ImageAnalysisResult.create_error_result(
                model_used=self.config.model_id, error_message=str(e), processing_time_ms=processing_time_ms
            )
            return result.to_dict()
        except Exception as e:
            processing_time_ms = int((time.time() - start_time) * 1000)

            self.logger.error(
                "Bedrock image analysis failed",
                model_id=self.config.model_id,
                error=str(e),
                processing_time_ms=processing_time_ms,
            )

            # Create error result
            result = ImageAnalysisResult.create_error_result(
                model_used=self.config.model_id, error_message=str(e), processing_time_ms=processing_time_ms
            )

            return result.to_dict()

    def _prepare_claude_message(
        self, image_data: bytes, image_format: str, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Prepare the message structure for Claude vision model."""
        try:
            # Encode image to base64
            image_base64 = base64.b64encode(image_data).decode("utf-8")

            # Determine MIME type
            mime_type = self._get_mime_type(image_format)

            # Build prompt with context if available
            prompt = self.config.prompt_template
            if context:
                if "caption" in context and context["caption"]:
                    prompt += f"\n\nImage caption: {context['caption']}"
                if "source_document" in context and context["source_document"]:
                    prompt += f"\nSource document: {context['source_document']}"
                if "page_number" in context and context["page_number"]:
                    prompt += f"\nPage number: {context['page_number']}"

            # Prepare message for Claude
            message = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": self.config.max_tokens,
                "temperature": self.config.temperature,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {"type": "base64", "media_type": mime_type, "data": image_base64},
                            },
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
            }

            return message

        except Exception as e:
            raise ImageAnalysisError(
                f"Failed to prepare Claude message: {str(e)}", model_used=self.config.model_id, original_error=e
            )

    def _call_bedrock(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Call Bedrock with the prepared message."""
        try:
            response = self.bedrock_client.invoke_model(
                modelId=self.config.model_id,
                body=json.dumps(message),
                accept="application/json",
                contentType="application/json",
            )

            # Parse response body
            response_body: Dict[str, Any] = json.loads(response["body"].read())
            return response_body

        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            error_message = e.response["Error"]["Message"]
            raise ImageAnalysisError(
                f"Bedrock API error ({error_code}): {error_message}", model_used=self.config.model_id, original_error=e
            )
        except BotoCoreError as e:
            raise ImageAnalysisError(f"AWS SDK error: {str(e)}", model_used=self.config.model_id, original_error=e)
        except Exception as e:
            raise ImageAnalysisError(
                f"Unexpected error calling Bedrock: {str(e)}", model_used=self.config.model_id, original_error=e
            )

    def _parse_bedrock_response(self, response: Dict[str, Any]) -> str:
        """Parse the Bedrock response to extract analysis text."""
        try:
            # Claude response format: {"content": [{"text": "..."}], "usage": {...}}
            if "content" not in response:
                raise ImageAnalysisError(
                    "Invalid Bedrock response: missing 'content' field", model_used=self.config.model_id
                )

            content = response["content"]
            if not isinstance(content, list) or not content:
                raise ImageAnalysisError(
                    "Invalid Bedrock response: 'content' is not a non-empty list", model_used=self.config.model_id
                )

            # Extract text from first content block
            first_content = content[0]
            if "text" not in first_content:
                raise ImageAnalysisError(
                    "Invalid Bedrock response: missing 'text' in content block", model_used=self.config.model_id
                )

            analysis_text: str = first_content["text"].strip()
            if not analysis_text:
                raise ImageAnalysisError("Empty analysis text from Bedrock", model_used=self.config.model_id)

            return analysis_text

        except ImageAnalysisError:
            raise
        except Exception as e:
            raise ImageAnalysisError(
                f"Failed to parse Bedrock response: {str(e)}", model_used=self.config.model_id, original_error=e
            )

    def _get_mime_type(self, image_format: str) -> str:
        """Convert image format to MIME type."""
        format_upper = image_format.upper()
        mime_types = {
            "PNG": "image/png",
            "JPEG": "image/jpeg",
            "JPG": "image/jpeg",
            "GIF": "image/gif",
            "WEBP": "image/webp",
        }
        return mime_types.get(format_upper, "image/jpeg")

    def health_check(self) -> Dict[str, str]:
        """
        Check health of Bedrock image analysis service.

        Returns:
            Dictionary with health status information
        """
        health = {
            "bedrock_image_analysis_service": "healthy",
            "model_id": self.config.model_id,
            "aws_region": self.aws_region,
        }

        try:
            # Test Bedrock connectivity with a simple call
            # Note: We don't actually invoke a model here to avoid costs
            # Instead, we just verify the client can be created
            test_client = get_bedrock_runtime_client(self.aws_region)
            health["bedrock_connectivity"] = "healthy"
        except Exception as e:
            self.logger.warning("Bedrock connectivity check failed", error=str(e))
            health["bedrock_connectivity"] = "unhealthy"
            health["bedrock_error"] = str(e)

        return health
