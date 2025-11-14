"""AWS configuration for local development support."""

import os
from pydantic_settings import BaseSettings


class AWSConfig(BaseSettings):
    """AWS configuration for S3 endpoint override (local development support)."""

    S3_ENDPOINT_URL: str | None = None
    S3_ACCESS_KEY_ID: str | None = None
    S3_SECRET_ACCESS_KEY: str | None = None

    class Config:
        env_prefix = ""


# Global configuration instance
aws_config = AWSConfig()
