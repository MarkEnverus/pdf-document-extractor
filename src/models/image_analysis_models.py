"""Models for image analysis results from LLM services."""

from datetime import datetime, timezone
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field, ConfigDict


class ImageAnalysisResult(BaseModel):
    """Result of LLM-based image analysis."""

    model_used: str = Field(..., description="LLM model used for analysis")
    analysis_summary: str = Field(..., description="LLM-generated description/analysis of the image")
    analysis_timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc), description="Timestamp when analysis was performed"
    )
    analysis_confidence: Optional[float] = Field(
        None, description="Confidence score from the model (0.0-1.0)", ge=0.0, le=1.0
    )
    analysis_error: Optional[str] = Field(None, description="Error message if analysis failed or had issues")
    processing_time_ms: Optional[int] = Field(None, description="Time taken for analysis in milliseconds")

    model_config = ConfigDict(json_encoders={datetime: lambda dt: dt.isoformat()})

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for storage."""
        return self.model_dump(mode="json")

    @classmethod
    def create_error_result(
        cls, model_used: str, error_message: str, processing_time_ms: Optional[int] = None
    ) -> "ImageAnalysisResult":
        """
        Create an ImageAnalysisResult for error cases.

        Args:
            model_used: Model that was attempted
            error_message: Error description
            processing_time_ms: Time taken before error occurred

        Returns:
            ImageAnalysisResult with error information
        """
        return cls(
            model_used=model_used,
            analysis_summary="Analysis failed",
            analysis_confidence=None,
            analysis_error=error_message,
            processing_time_ms=processing_time_ms,
        )

    @classmethod
    def create_success_result(
        cls,
        model_used: str,
        analysis_summary: str,
        confidence: Optional[float] = None,
        processing_time_ms: Optional[int] = None,
    ) -> "ImageAnalysisResult":
        """
        Create an ImageAnalysisResult for successful analysis.

        Args:
            model_used: Model that performed the analysis
            analysis_summary: LLM-generated description
            confidence: Optional confidence score
            processing_time_ms: Time taken for analysis

        Returns:
            ImageAnalysisResult with analysis information
        """
        return cls(
            model_used=model_used,
            analysis_summary=analysis_summary,
            analysis_confidence=confidence,
            analysis_error=None,
            processing_time_ms=processing_time_ms,
        )


class BedrockAnalysisConfig(BaseModel):
    """Configuration for Bedrock image analysis."""

    model_id: str = Field(
        default="anthropic.claude-3-sonnet-20240229-v1:0", description="Bedrock model ID to use for analysis"
    )
    max_tokens: int = Field(default=1024, description="Maximum tokens for analysis response", gt=0, le=4096)
    temperature: float = Field(default=0.1, description="Temperature for model sampling", ge=0.0, le=1.0)
    timeout_seconds: int = Field(default=30, description="Timeout for analysis request in seconds", gt=0, le=300)
    prompt_template: str = Field(
        default=(
            "Please analyze this image and provide a detailed description. "
            "Focus on the main content, any text visible, charts or diagrams, "
            "and the overall purpose or meaning of the image. "
            "Be concise but thorough in your analysis."
        ),
        description="Prompt template to use for analysis",
    )

    model_config = ConfigDict(json_encoders={datetime: lambda dt: dt.isoformat()})
