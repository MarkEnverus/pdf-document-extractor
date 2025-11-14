import asyncio
import threading
from typing import Dict, Optional, Any
from botocore.exceptions import ClientError

from src.interfaces.storage import AbstractStorageService
from src.exceptions.domain_exceptions import StorageError
from lib.logger import Logger

logger = Logger.get_logger(__name__)


class S3Service(AbstractStorageService):
    def __init__(self, s3_client: Optional[Any] = None, settings_provider: Optional[Any] = None) -> None:
        """
        Initialize S3Service with dependency injection support.

        Args:
            s3_client: Optional S3 client for dependency injection
            settings_provider: Optional settings provider for dependency injection
        """
        if s3_client is not None:
            self.s3_client = s3_client
        elif settings_provider is not None:
            self.s3_client = settings_provider.get_s3_client()
        else:
            # Fallback to global settings for backward compatibility
            from src.configs.settings import settings

            if settings is None:
                raise StorageError(
                    message="Settings not initialized and no s3_client provided",
                    context={"service": "S3Service", "action": "initialization"},
                )
            self.s3_client = settings.get_s3_client()

        # Thread-local storage for thread-safe concurrent operations
        self._thread_local = threading.local()
        self._settings_provider = settings_provider

    def _get_thread_safe_client(self) -> Any:
        """
        Get a thread-local S3 client for safe concurrent operations.
        Each thread in asyncio thread pool gets its own client instance.
        """
        try:
            return self._thread_local.s3_client
        except AttributeError:
            # Create new client for this thread
            if self._settings_provider is not None:
                self._thread_local.s3_client = self._settings_provider.get_s3_client()
            else:
                from src.configs.settings import settings

                self._thread_local.s3_client = settings.get_s3_client()
            return self._thread_local.s3_client

    def upload_content(
        self,
        content: bytes,
        bucket: str,
        key: str,
        request_id: Optional[str] = None,
        upload_id: Optional[str] = None,
        project_id: Optional[str] = None,
    ) -> str:
        """
        Upload content to S3 and return the S3 path.

        Args:
            content: The bytes content to upload
            bucket: S3 bucket name
            key: S3 object key
            request_id: Optional request ID for tracking
            upload_id: Optional upload ID for tracking
            project_id: Optional project ID for tracking

        Returns:
            S3 path in format s3://bucket/key
        """
        try:
            s3_path = f"s3://{bucket}/{key}"

            # Log context for structured logging
            log_context: dict[str, Any] = {
                "service": "s3_service",
                "operation": "upload_content",
                "bucket": bucket,
                "key": key,
                "content_size": len(content),
                "s3_path": s3_path,
            }
            if request_id:
                log_context["request_id"] = request_id
            if upload_id:
                log_context["upload_id"] = upload_id
            if project_id:
                log_context["project_id"] = project_id

            logger.debug("Uploading content to S3", **log_context)

            # Uncomment when ready to use actual S3:
            self.s3_client.put_object(Bucket=bucket, Key=key, Body=content)

            logger.info("S3 upload completed successfully", **log_context)
            return s3_path
        except Exception as e:
            error_context: dict[str, Any] = {
                "service": "s3_service",
                "operation": "upload_content",
                "bucket": bucket,
                "key": key,
                "content_size": len(content),
                "error": str(e),
                "error_type": type(e).__name__,
            }
            if request_id:
                error_context["request_id"] = request_id
            if upload_id:
                error_context["upload_id"] = upload_id
            if project_id:
                error_context["project_id"] = project_id

            logger.error("S3 upload failed", **error_context)
            raise StorageError(
                message=f"S3 upload failed: {str(e)}",
                context={"bucket": bucket, "key": key, "content_size": len(content)},
                cause=e,
            )

    def download_content_by_bucket_key(
        self,
        bucket: str,
        key: str,
        request_id: Optional[str] = None,
        upload_id: Optional[str] = None,
        project_id: Optional[str] = None,
    ) -> bytes:
        """
        Download content from S3.

        Args:
            bucket: S3 bucket name
            key: S3 object key
            request_id: Optional request ID for tracking
            upload_id: Optional upload ID for tracking
            project_id: Optional project ID for tracking

        Returns:
            The content as bytes
        """
        try:
            # Log context for structured logging
            log_context: dict[str, Any] = {
                "service": "s3_service",
                "operation": "download_content",
                "bucket": bucket,
                "key": key,
                "s3_path": f"s3://{bucket}/{key}",
            }
            if request_id:
                log_context["request_id"] = request_id
            if upload_id:
                log_context["upload_id"] = upload_id
            if project_id:
                log_context["project_id"] = project_id

            logger.debug("Downloading content from S3", **log_context)

            response = self.s3_client.get_object(Bucket=bucket, Key=key)
            content = response["Body"].read()
            content_bytes = bytes(content) if not isinstance(content, bytes) else content

            logger.info("S3 download completed successfully", content_size=len(content_bytes), **log_context)
            return content_bytes
        except ClientError as e:
            error_context: dict[str, Any] = {
                "service": "s3_service",
                "operation": "download_content",
                "bucket": bucket,
                "key": key,
                "error": str(e),
                "error_type": type(e).__name__,
            }
            if request_id:
                error_context["request_id"] = request_id
            if upload_id:
                error_context["upload_id"] = upload_id
            if project_id:
                error_context["project_id"] = project_id

            # Check if this is a NoSuchKey error (file doesn't exist)
            error_code = e.response.get("Error", {}).get("Code", "")
            if error_code == "NoSuchKey":
                # Log as warning since missing files may be expected (deleted/moved)
                logger.warning(
                    "S3 file not found (may have been deleted or moved)",
                    **error_context,
                )
                raise StorageError(
                    message=f"File not found in S3: s3://{bucket}/{key}",
                    context={"bucket": bucket, "key": key},
                    cause=e,
                )
            else:
                # Log as error for actual S3 failures
                logger.error("S3 download failed", **error_context)
                raise StorageError(
                    message=f"S3 download failed: {str(e)}",
                    context={"bucket": bucket, "key": key},
                    cause=e,
                )

    def download_content(
        self,
        s3_uri: str,
        request_id: Optional[str] = None,
        upload_id: Optional[str] = None,
        project_id: Optional[str] = None,
    ) -> bytes:
        """
        Download content from S3.

        Args:
            s3_uri: S3 URI in format s3://bucket/key

        Returns:
            The content as bytes
        """
        bucket, key = self.parse_s3_uri(s3_uri)
        return self.download_content_by_bucket_key(bucket, key, request_id, upload_id, project_id)

    def parse_s3_uri(self, s3_uri: str) -> tuple[str, str]:
        """
        Parse S3 URI into bucket and key components.

        Args:
            s3_uri: S3 URI to parse

        Returns:
            Tuple of (bucket, key)

        Raises:
            ValueError: If URI is invalid
        """
        if not s3_uri.startswith("s3://"):
            raise ValueError(f"Invalid S3 URI format: {s3_uri}")

        try:
            parts = s3_uri[5:].split("/", 1)
            if len(parts) < 2:
                raise ValueError(f"Invalid S3 URI format: {s3_uri}")
            return parts[0], parts[1]
        except Exception as e:
            raise ValueError(f"Failed to parse S3 URI: {s3_uri}") from e

    def object_exists(self, bucket: str, key: str) -> bool:
        """
        Check if an object exists in S3.

        Args:
            bucket: S3 bucket name
            key: S3 object key

        Returns:
            True if object exists, False otherwise
        """
        try:
            self.s3_client.head_object(Bucket=bucket, Key=key)
            return True
        except ClientError:
            return False

    async def upload_content_async(
        self,
        content: bytes,
        bucket: str,
        key: str,
        request_id: Optional[str] = None,
        upload_id: Optional[str] = None,
        project_id: Optional[str] = None,
    ) -> str:
        """
        Async upload content to S3 and return the S3 path.

        Uses asyncio.to_thread to prevent blocking Kafka heartbeats during upload.

        Args:
            content: The bytes content to upload
            bucket: S3 bucket name
            key: S3 object key
            request_id: Optional request ID for tracking
            upload_id: Optional upload ID for tracking
            project_id: Optional project ID for tracking

        Returns:
            S3 path in format s3://bucket/key
        """
        try:
            s3_path = f"s3://{bucket}/{key}"

            # Log context for structured logging
            log_context: dict[str, Any] = {
                "service": "s3_service",
                "operation": "upload_content_async",
                "bucket": bucket,
                "key": key,
                "content_size": len(content),
                "s3_path": s3_path,
            }
            if request_id:
                log_context["request_id"] = request_id
            if upload_id:
                log_context["upload_id"] = upload_id
            if project_id:
                log_context["project_id"] = project_id

            logger.debug("Async uploading content to S3", **log_context)

            # Use thread-safe client in thread pool to avoid blocking event loop
            def _upload_with_thread_safe_client() -> None:
                client = self._get_thread_safe_client()
                client.put_object(Bucket=bucket, Key=key, Body=content)

            await asyncio.to_thread(_upload_with_thread_safe_client)

            logger.info("Async S3 upload completed successfully", **log_context)
            return s3_path
        except Exception as e:
            error_context: dict[str, Any] = {
                "service": "s3_service",
                "operation": "upload_content_async",
                "bucket": bucket,
                "key": key,
                "content_size": len(content),
                "error": str(e),
                "error_type": type(e).__name__,
            }
            if request_id:
                error_context["request_id"] = request_id
            if upload_id:
                error_context["upload_id"] = upload_id
            if project_id:
                error_context["project_id"] = project_id

            logger.error("Async S3 upload failed", **error_context)
            raise StorageError(
                message=f"Async S3 upload failed: {str(e)}",
                context={"bucket": bucket, "key": key, "content_size": len(content)},
                cause=e,
            )

    async def download_content_async(
        self,
        s3_uri: str,
        request_id: Optional[str] = None,
        upload_id: Optional[str] = None,
        project_id: Optional[str] = None,
    ) -> bytes:
        """
        Async download content from S3.

        Uses asyncio.to_thread to prevent blocking Kafka heartbeats during download.

        Args:
            s3_uri: S3 URI in format s3://bucket/key
            request_id: Optional request ID for tracking
            upload_id: Optional upload ID for tracking
            project_id: Optional project ID for tracking

        Returns:
            The content as bytes
        """
        try:
            bucket, key = self.parse_s3_uri(s3_uri)

            # Log context for structured logging
            log_context: dict[str, Any] = {
                "service": "s3_service",
                "operation": "download_content_async",
                "bucket": bucket,
                "key": key,
                "s3_path": s3_uri,
            }
            if request_id:
                log_context["request_id"] = request_id
            if upload_id:
                log_context["upload_id"] = upload_id
            if project_id:
                log_context["project_id"] = project_id

            logger.debug("Async downloading content from S3", **log_context)

            # Use thread-safe client in thread pool to avoid blocking event loop
            def _download_with_thread_safe_client() -> bytes:
                client = self._get_thread_safe_client()
                response = client.get_object(Bucket=bucket, Key=key)
                content = response["Body"].read()
                return bytes(content) if not isinstance(content, bytes) else content

            content_bytes = await asyncio.to_thread(_download_with_thread_safe_client)

            logger.info("Async S3 download completed successfully", content_size=len(content_bytes), **log_context)
            return content_bytes
        except ClientError as e:
            error_context: dict[str, Any] = {
                "service": "s3_service",
                "operation": "download_content_async",
                "s3_uri": s3_uri,
                "error": str(e),
                "error_type": type(e).__name__,
            }
            if request_id:
                error_context["request_id"] = request_id
            if upload_id:
                error_context["upload_id"] = upload_id
            if project_id:
                error_context["project_id"] = project_id

            # Check if this is a NoSuchKey error (file doesn't exist)
            error_code = e.response.get("Error", {}).get("Code", "")
            if error_code == "NoSuchKey":
                logger.warning(
                    "Async S3 file not found (may have been deleted or moved)",
                    **error_context,
                )
                raise StorageError(
                    message=f"File not found in S3: {s3_uri}",
                    context={"s3_uri": s3_uri},
                    cause=e,
                )
            else:
                logger.error("Async S3 download failed", **error_context)
                raise StorageError(
                    message=f"Async S3 download failed: {str(e)}",
                    context={"s3_uri": s3_uri},
                    cause=e,
                )

    def health_check(self, bucket_name: Optional[str] = None) -> Dict[str, str]:
        """
        Check S3 service health by testing connection and access.

        Args:
            bucket_name: Optional bucket name to test. If not provided,
                        uses a default bucket or skips bucket-specific tests.

        Returns:
            Dictionary with S3 health status details
        """
        health = {"s3_service": "healthy", "connection": "unknown", "bucket_access": "unknown"}

        if bucket_name is None:
            # If no bucket specified, we can only check if S3 client is configured
            health["connection"] = "healthy" if self.s3_client else "unhealthy"
            health["bucket_access"] = "not_tested"
            return health

        try:
            # Test S3 connection by listing objects (limited to 1)
            response = self.s3_client.list_objects_v2(Bucket=bucket_name, MaxKeys=1)
            health["connection"] = "healthy"
            health["bucket_access"] = "healthy"
            logger.info(f"S3 health check passed for bucket: {bucket_name}")

        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code == "NoSuchBucket":
                health["connection"] = "healthy"
                health["bucket_access"] = "bucket_not_found"
                health["s3_service"] = "degraded"
            elif error_code in ["AccessDenied", "Forbidden"]:
                health["connection"] = "healthy"
                health["bucket_access"] = "access_denied"
                health["s3_service"] = "degraded"
            else:
                health["connection"] = "unhealthy"
                health["bucket_access"] = "failed"
                health["s3_service"] = "unhealthy"
            logger.error(f"S3 health check failed: {str(e)}")

        except Exception as e:
            health["connection"] = "unhealthy"
            health["bucket_access"] = "failed"
            health["s3_service"] = "unhealthy"
            logger.error(f"S3 health check failed with unexpected error: {str(e)}")

        return health
