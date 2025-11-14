"""AWS helper for boto3 operations with dependency injection support."""

import json
import os
from functools import lru_cache
from typing import Any

import boto3

from lib.aws.config import aws_config
from lib.logger import Logger

logger = Logger.get_logger(__name__)

# Global instance for singleton pattern
_aws_helper: "AWSHelper | None" = None


class AWSHelper:
    """Simplified AWS helper for boto3 operations."""

    def __init__(self) -> None:
        self.aws_region_name = os.getenv("AWS_REGION", "us-east-1")
        self.environment_name = os.getenv("ENVIRONMENT_NAME", "local")
        self._session = self._initialize_session()
        logger.info(
            f"AWSHelper initialized for environment: {self.environment_name}, region: {self.aws_region_name}"
        )

    def _initialize_session(self) -> boto3.Session:
        """Initialize boto3 session based on environment."""
        if self.environment_name == "local":
            # Local: use AWS profile for development
            profile = os.getenv("AWS_PROFILE_NAME", "default")
            logger.info(f"Local environment: using AWS profile '{profile}'")
            try:
                return boto3.Session(
                    profile_name=profile, region_name=self.aws_region_name
                )
            except Exception as e:
                logger.warning(
                    f"Failed to create session with profile '{profile}': {e}. Using default credentials."
                )
                return boto3.Session(region_name=self.aws_region_name)
        else:
            # Cloud: use IAM role credentials
            logger.info("Cloud environment: using IAM role credentials")
            return boto3.Session(region_name=self.aws_region_name)

    @lru_cache(maxsize=256)
    def get_secret(self, secret_name: str) -> dict[str, Any]:
        """
        Fetch secret from AWS Secrets Manager.

        Args:
            secret_name: Name of the secret in Secrets Manager

        Returns:
            Dictionary with secret key-value pairs
        """
        if self.environment_name in ["local", "test"]:
            logger.warning(
                f"Local/test environment: returning empty dict for secret '{secret_name}'"
            )
            return {}

        try:
            client = self._session.client("secretsmanager")
            response = client.get_secret_value(SecretId=secret_name)
            return json.loads(response["SecretString"])
        except Exception as e:
            logger.error(f"Failed to fetch secret '{secret_name}': {e}")
            raise

    def get_session(self) -> boto3.Session:
        """Get boto3 session."""
        return self._session

    @lru_cache(maxsize=1)
    def get_aws_account_number(self) -> str:
        """
        Get AWS account ID using STS.

        Returns:
            AWS account number as string, or "000000000000" if unavailable
        """
        try:
            sts = self._session.client("sts")
            account_id = sts.get_caller_identity()["Account"]
            logger.debug(f"AWS account number: {account_id}")
            return account_id
        except Exception as e:
            logger.warning(f"Failed to get AWS account number: {e}")
            return "000000000000"


def get_aws_client() -> AWSHelper:
    """
    Get or create the global AWSHelper instance.

    Returns:
        Singleton AWSHelper instance
    """
    global _aws_helper
    if _aws_helper is None:
        _aws_helper = AWSHelper()
    return _aws_helper


def get_s3_client() -> Any:
    """
    Get S3 client with MinIO support for local development.

    Returns:
        boto3 S3 client configured for local (MinIO) or cloud (AWS S3)
    """
    # Check for local development endpoint (MinIO)
    s3_endpoint = aws_config.S3_ENDPOINT_URL

    if s3_endpoint:
        # Local dev: use MinIO or localstack
        logger.info(f"Using S3 endpoint override: {s3_endpoint}")
        access_key = aws_config.S3_ACCESS_KEY_ID or "minioadmin"
        secret_key = aws_config.S3_SECRET_ACCESS_KEY or "minioadmin"

        session = boto3.Session(
            aws_access_key_id=access_key, aws_secret_access_key=secret_key
        )
        return session.client("s3", endpoint_url=s3_endpoint)

    # Cloud: use standard session with IAM role or AWS credentials
    helper = get_aws_client()
    session = helper.get_session()
    return session.client("s3", region_name=helper.aws_region_name)


def get_bedrock_runtime_client(region: str = "us-east-1") -> Any:
    """
    Get Bedrock Runtime client for model invocation.

    Args:
        region: AWS region for Bedrock service (default: us-east-1)

    Returns:
        boto3 Bedrock Runtime client
    """
    helper = get_aws_client()
    session = helper.get_session()
    logger.debug(f"Creating Bedrock Runtime client for region: {region}")
    return session.client("bedrock-runtime", region_name=region)


def get_secret(secret_name: str) -> dict[str, Any]:
    """
    Convenience function to get secret from AWS Secrets Manager.

    Args:
        secret_name: Name of the secret

    Returns:
        Dictionary with secret key-value pairs
    """
    helper = get_aws_client()
    return helper.get_secret(secret_name)
