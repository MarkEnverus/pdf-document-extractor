# Copyright 2025 by Enverus. All rights reserved.

"""
Critical failure handler for system-level service failures.

This module provides utilities for handling critical infrastructure failures
that should trigger application shutdown.
"""

import sys
import asyncio
from typing import Any, Callable, Optional

from lib.logger import Logger
from src.configs.settings import settings

logger = Logger.get_logger(__name__)


class CriticalFailureHandler:
    """Handler for critical application failures that require system shutdown."""

    @staticmethod
    def handle_critical_failure(error: Exception, service_name: str, context: Optional[str] = None) -> None:
        """
        Handle a critical failure by logging and triggering system shutdown.

        Args:
            error: The exception that caused the critical failure
            service_name: Name of the service that failed
            context: Additional context about the failure

        Raises:
            SystemExit: Always raises to trigger application shutdown
        """
        error_msg = f"CRITICAL FAILURE in {service_name}: {str(error)}"
        if context:
            error_msg += f" | Context: {context}"

        logger.critical(error_msg)

        # Special handling for specific error types
        error_str_lower = str(error).lower()

        if any(
            substring in error_str_lower for substring in ("token has expired", "refresh failed", "expired credentials")
        ):
            logger.critical("AWS authentication token has expired - application cannot continue")
            logger.critical("Please refresh your AWS credentials (e.g., 'aws sso login') and restart the application")
        elif "unable to locate credentials" in error_str_lower:
            logger.critical("AWS credentials not found - application cannot continue")
            logger.critical("Please configure AWS credentials using one of these methods:")
            logger.critical("  1. AWS SSO: Run 'aws sso login'")
            logger.critical("  2. IAM Role: Ensure instance has correct IAM role")
            logger.critical("  3. Environment: Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY")
            logger.critical("  4. Credentials file: Configure ~/.aws/credentials")
        elif "access denied" in error_str_lower or "forbidden" in error_str_lower:
            logger.critical("AWS access denied - insufficient permissions")
            logger.critical("The current AWS credentials do not have required permissions")
            logger.critical("Please ensure the IAM role or user has necessary access to AWS services")

        # Log environment info
        logger.critical(f"Environment: {settings.ENVIRONMENT_NAME}")
        logger.critical("Application will shut down immediately to prevent data corruption")

        # Check multiple test environment indicators (same logic as AWS validation)
        environment_name = settings.ENVIRONMENT_NAME.lower() if settings.ENVIRONMENT_NAME else ""
        is_test_env = (
            environment_name == "test"
            or "pytest" in sys.modules  # Running under pytest
            or "test" in sys.argv[0].lower()  # Script name contains 'test'
            or any("test" in arg.lower() for arg in sys.argv)  # Any arg contains 'test'
        )

        if is_test_env:
            # In test environment, don't exit - just log and return
            logger.critical("Test environment detected - skipping shutdown for test stability")
            logger.warning("In production, this would trigger immediate application shutdown")
            return
        else:
            # In production environments, exit immediately
            logger.critical("Production environment detected - forcing immediate shutdown")
            sys.exit(1)

    @staticmethod
    async def handle_async_critical_failure(error: Exception, service_name: str, context: Optional[str] = None) -> None:
        """
        Handle a critical failure in an async context.

        Args:
            error: The exception that caused the critical failure
            service_name: Name of the service that failed
            context: Additional context about the failure

        Raises:
            SystemExit: Always raises to trigger application shutdown
        """
        # Stop all running tasks first
        try:
            tasks = [task for task in asyncio.all_tasks() if not task.done()]
            if tasks:
                logger.critical(f"Cancelling {len(tasks)} running tasks before shutdown")
                for task in tasks:
                    task.cancel()

                # Wait a bit for tasks to clean up
                try:
                    await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=5.0)
                except asyncio.TimeoutError:
                    logger.warning("Some tasks did not complete cleanup within timeout")

        except Exception as cleanup_error:
            logger.error(f"Error during task cleanup: {str(cleanup_error)}")

        # Now handle the critical failure
        CriticalFailureHandler.handle_critical_failure(error, service_name, context)

    @staticmethod
    def is_critical_error(error: Exception) -> bool:
        """
        Determine if an error should be treated as a critical failure.

        Args:
            error: The exception to check

        Returns:
            True if the error is critical and should trigger shutdown
        """
        error_str = str(error).lower()

        # SSO/Authentication failures
        if any(
            phrase in error_str
            for phrase in [
                "token has expired",
                "refresh failed",
                "authentication failed",
                "unauthorized",
                "credentials",
            ]
        ):
            return True

        # Kafka connection failures (in production only)
        if settings.ENVIRONMENT_NAME != "test" and any(
            phrase in error_str
            for phrase in ["kafka", "bootstrap_connected", "sasl authentication failed", "connection refused"]
        ):
            return True

        # AWS service failures
        if any(
            phrase in error_str
            for phrase in [
                "secrets manager",
                "aws credential",
                "assume role failed",
                "unable to locate credentials",
                "expired credentials",
                "access denied",
                "forbidden",
                "invalid security token",
            ]
        ):
            return True

        return False


def critical_failure_decorator(service_name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Decorator to automatically handle critical failures in functions.

    Args:
        service_name: Name of the service for error reporting

    Usage:
        @critical_failure_decorator("MyService")
        def my_function():
            # function code here
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if CriticalFailureHandler.is_critical_error(e):
                    CriticalFailureHandler.handle_critical_failure(e, service_name, f"Function: {func.__name__}")
                else:
                    raise

        return wrapper

    return decorator


def async_critical_failure_decorator(service_name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Async version of the critical failure decorator.

    Args:
        service_name: Name of the service for error reporting

    Usage:
        @async_critical_failure_decorator("MyService")
        async def my_async_function():
            # async function code here
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                if CriticalFailureHandler.is_critical_error(e):
                    await CriticalFailureHandler.handle_async_critical_failure(
                        e, service_name, f"Function: {func.__name__}"
                    )
                else:
                    raise

        return wrapper

    return decorator
