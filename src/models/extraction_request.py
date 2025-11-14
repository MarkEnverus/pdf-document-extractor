# Copyright 2025 by Enverus. All rights reserved.

from enum import Enum
from typing import Optional, Dict, List, Literal, Any
from pydantic import BaseModel, Field, field_validator
from uuid import UUID

"""Extraction Request Models"""


class LLMConfig(BaseModel):
    prompt: str = Field(..., description="The ID of the model to be used for extraction.")
    output_schema: Dict[str, Any] = Field(
        ...,
        description="Schema defining the structure of the extracted data, must be valid JSON. Required only for multimodal extraction.",
    )


class ExtractionRequest(BaseModel):
    document_urls: List[str] = Field(..., min_length=1, description="List of URLs of the documents to be extracted.")
    llm_config: LLMConfig | None = Field(
        None,
        description="Configuration for document extraction. Must be a valid JSON object.",
    )


"""Those models are used to define the structure of the extraction-service job request and configuration."""


class ConfigType(str, Enum):
    BEDROCK = "bedrock"
    MULTIMODAL = "multimodal"


class ExtractionConfig(BaseModel):
    config_type: ConfigType = Field(..., description="Type of extraction configuration (bedrock or multimodal)")

    name: Optional[str] = Field(None, description="Optional name for the extraction configuration")

    system_prompt: Optional[str] = Field(None, description="Optional system prompt to guide the extraction process")

    model_id: Optional[str] = Field(
        None,
        description="Optional model ID for the extraction backend (required for multimodal config_type)",
    )

    data_automation_project_arn: Optional[str] = Field(
        None,
        description="ARN of the Bedrock Data Automation project (required for bedrock config_type)",
    )

    data_automation_profile_arn: Optional[str] = Field(
        None,
        description="ARN of the Bedrock Data Automation profile (required for bedrock config_type)",
    )

    output_s3_uri: Optional[str] = Field(None, description="S3 URI for output data (required for bedrock config_type)")

    blueprints: Optional[List[Dict[str, str]]] = Field(None, description="Optional list of blueprint configurations")

    role_arn: Optional[str] = Field(None, description="ARN of the IAM role to assume for cross-account access")

    @field_validator("model_id")
    @classmethod
    def validate_model_id(cls, v: Optional[str], info: Any) -> Optional[str]:
        config_type = info.data.get("config_type")

        if config_type == ConfigType.MULTIMODAL and not v:
            raise ValueError("model_id is required when config_type is 'multimodal'")

        return v

    @field_validator("output_s3_uri")
    @classmethod
    def validate_s3_uri(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and not v.startswith("s3://"):
            raise ValueError("output_s3_uri must be a valid S3 URI starting with 's3://'")

        return v

    @field_validator("role_arn")
    @classmethod
    def validate_role_arn(cls, v: Optional[str]) -> Optional[str]:
        """
        Validate that role_arn, if provided, is a valid AWS ARN.
        """
        if v is not None:
            if not isinstance(v, str) or not v.startswith("arn:aws:iam::"):
                raise ValueError("role_arn must be a valid AWS IAM ARN starting with 'arn:aws:iam::'")
            # Basic ARN format check: arn:aws:iam::<account-id>:role/<role-name>
            parts = v.split(":")
            if len(parts) < 6 or not parts[5].startswith("role/"):
                raise ValueError("role_arn must be a valid IAM role ARN (arn:aws:iam::<account-id>:role/<role-name>)")
        return v


ExtractionMode = Literal["single", "multi"]


class ExtractorRequest(BaseModel):
    document_urls: List[str] = Field(..., min_length=1, description="List of URLs of the documents to be extracted.")

    extraction_configs: List[ExtractionConfig] = Field(
        ...,
        min_length=1,
        description="List of configurations for document extraction. Each document will be processed with each configuration.",
    )

    extraction_mode: ExtractionMode = Field(
        "multi",
        description="The mode of extraction to be used. Options are 'single' for document extraction using only one configuration or 'multi' for processing multiple files with multiple configurations.",
    )

    output_schema: Dict[str, Any] | None = Field(
        None,
        description="Schema defining the structure of the extracted data, must be valid JSON. Required only for multimodal extraction.",
    )


class JobStatusRequest(BaseModel):
    job_id: UUID = Field(..., description="The UUID of the job to check status for.")
