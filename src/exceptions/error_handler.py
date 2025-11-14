import os
import traceback
from collections import defaultdict
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, TypedDict, Union

from fastapi import status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from pydantic import ValidationError as PydanticValidationError

from lib.logger import Logger

logger = Logger.get_logger(__name__)


def build_http_fastapi_error_response(
    status_code: int = status.HTTP_400_BAD_REQUEST,
    error_code: str = "VALIDATION_ERROR",
    message: str = "Validation failed",
    details: str | dict[str, Any] = "",
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error_code": error_code, "message": message, "details": details},
    )


class ErrorSeverity(Enum):
    """Enum to represent severity levels for exceptions"""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class BaseError(Exception):
    """Base exception for all application errors

    This serves as the foundation for our exception hierarchy.
    All custom exceptions should inherit from this class.
    """

    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        error_code: str = "GENERIC_ERROR",
    ):
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.severity = severity
        self.http_status_code = status_code
        self.error_code = error_code
        self.traceback = traceback.format_exc()

    def __str__(self) -> str:
        return f"{self.error_code}: {self.message}"

    def log(self) -> None:
        """Log the exception with appropriate severity level"""
        log_message = f"{self.error_code}: {self.message}"

        if self.details:
            log_message += f" - Details: {self.details}"

        if self.severity == ErrorSeverity.DEBUG:
            logger.debug(log_message)
        elif self.severity == ErrorSeverity.INFO:
            logger.info(log_message)
        elif self.severity == ErrorSeverity.WARNING:
            logger.warning(log_message)
        elif self.severity == ErrorSeverity.CRITICAL:
            logger.critical(log_message)
        else:  # Default to ERROR
            logger.error(log_message)

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for response formatting"""
        return {
            "error_code": self.error_code,
            "message": self.message,
            "details": self.details,
            "severity": self.severity.value,
        }


# ========== Configuration Errors ==========


class ConfigurationError(BaseError):
    """Base exception for configuration related errors"""

    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
    ):
        super().__init__(
            message=message,
            details=details,
            severity=severity,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            error_code="CONFIGURATION_ERROR",
        )


# ========== Request/Response Errors ==========


class RequestError(BaseError):
    """Base exception for all request-related errors"""

    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        severity: ErrorSeverity = ErrorSeverity.WARNING,
    ):
        super().__init__(
            message=message,
            details=details,
            severity=severity,
            status_code=status.HTTP_400_BAD_REQUEST,
            error_code="REQUEST_ERROR",
        )


class BadRequestError(RequestError):
    """Exception raised when the request is malformed or invalid"""

    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        severity: ErrorSeverity = ErrorSeverity.WARNING,
    ):
        super().__init__(message=message, details=details, severity=severity)
        self.error_code = "BAD_REQUEST_ERROR"


class InternalServerError(BaseError):
    """Exception raised for unexpected internal server errors"""

    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        severity: ErrorSeverity = ErrorSeverity.CRITICAL,
    ):
        super().__init__(
            message=message,
            details=details,
            severity=severity,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            error_code="INTERNAL_SERVER_ERROR",
        )


# ========== Quota Errors ==========


class QuotaError(BaseError):
    """Base exception for all quota-related errors"""

    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        severity: ErrorSeverity = ErrorSeverity.WARNING,
    ):
        super().__init__(
            message=message,
            details=details,
            severity=severity,
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            error_code="QUOTA_ERROR",
        )


class PromptQuotaExceededError(QuotaError):
    """Exception raised when prompt quota is exceeded"""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, details)
        self.error_code = "PROMPT_QUOTA_EXCEEDED"


class ConversationPromptQuotaExceededError(QuotaError):
    """Exception raised when conversation prompt quota is exceeded"""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, details)
        self.error_code = "CONVERSATION_PROMPT_QUOTA_EXCEEDED"


# ========== Retry Errors ==========


class RetryError(BaseError):
    """Base exception for retry-related errors"""

    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        severity: ErrorSeverity = ErrorSeverity.WARNING,
        retry_count: int = 0,
        max_retries: int = 0,
    ):
        super().__init__(
            message=message,
            details=details,
            severity=severity,
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            error_code="RETRY_ERROR",
        )
        self.retry_count = retry_count
        self.max_retries = max_retries
        if details is None:
            self.details = {}
        self.details.update({"retry_count": retry_count, "max_retries": max_retries})


class RetryException(RetryError):
    """Exception raised for operations that should be retried"""

    pass


# ========== Search Errors ==========


class SearchError(BaseError):
    """Base exception for search-related errors"""

    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        severity: ErrorSeverity = ErrorSeverity.WARNING,
    ):
        super().__init__(
            message=message,
            details=details,
            severity=severity,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            error_code="SEARCH_ERROR",
        )


# ========== Tool Errors ==========


class ToolError(BaseError):
    """Base exception for all tool-related errors"""

    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        tool_name: Optional[str] = None,
    ):
        super().__init__(
            message=message,
            details=details,
            severity=severity,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            error_code="TOOL_ERROR",
        )
        self.tool_name = tool_name
        if tool_name:
            if details is None:
                self.details = {}
            self.details["tool_name"] = tool_name


class ToolExecutionError(ToolError):
    """Exception raised when tool execution fails"""

    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        tool_name: Optional[str] = None,
    ):
        super().__init__(message, details, tool_name=tool_name)
        self.error_code = "TOOL_EXECUTION_ERROR"


class ToolTimeoutError(ToolExecutionError):
    """Exception raised when tool execution times out"""

    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        tool_name: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
    ):
        super().__init__(message, details, tool_name)
        self.error_code = "TOOL_TIMEOUT_ERROR"
        self.timeout_seconds = timeout_seconds
        if timeout_seconds:
            if details is None:
                self.details = {}
            self.details["timeout_seconds"] = timeout_seconds


class ToolValidationError(ToolError):
    """Exception raised when tool input validation fails"""

    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        tool_name: Optional[str] = None,
    ):
        super().__init__(
            message=message,
            details=details,
            severity=ErrorSeverity.WARNING,
            tool_name=tool_name,
        )
        self.error_code = "TOOL_VALIDATION_ERROR"
        self.status_code = status.HTTP_400_BAD_REQUEST


class ToolRetryError(ToolError):
    """Exception raised when max retries for a tool are exceeded"""

    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        tool_name: Optional[str] = None,
        retry_count: int = 0,
        max_retries: int = 0,
    ):
        super().__init__(message, details, tool_name=tool_name)
        self.error_code = "TOOL_RETRY_ERROR"
        self.retry_count = retry_count
        self.max_retries = max_retries
        if details is None:
            self.details = {}
        self.details.update({"retry_count": retry_count, "max_retries": max_retries})


# ========== Agent Errors ==========


class AgentError(BaseError):
    """Base exception for all agent-related errors"""

    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        agent_name: Optional[str] = None,
    ):
        super().__init__(
            message=message,
            details=details,
            severity=severity,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            error_code="AGENT_ERROR",
        )
        self.agent_name = agent_name
        if agent_name:
            if details is None:
                self.details = {}
            self.details["agent_name"] = agent_name


class AgentConfigError(AgentError):
    """Exception raised for agent configuration issues"""

    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        agent_name: Optional[str] = None,
    ):
        super().__init__(message, details, agent_name=agent_name)
        self.error_code = "AGENT_CONFIG_ERROR"


class AgentExecutionError(AgentError):
    """Exception raised for agent execution issues"""

    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        agent_name: Optional[str] = None,
    ):
        super().__init__(message, details, agent_name=agent_name)
        self.error_code = "AGENT_EXECUTION_ERROR"


class AgentTimeoutError(AgentError):
    """Exception raised when agent execution times out"""

    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        agent_name: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
    ):
        super().__init__(message, details, agent_name=agent_name)
        self.error_code = "AGENT_TIMEOUT_ERROR"
        self.timeout_seconds = timeout_seconds
        if timeout_seconds:
            if details is None:
                self.details = {}
            self.details["timeout_seconds"] = timeout_seconds


class AgentValidationError(AgentError):
    """Exception raised when agent input validation fails"""

    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        agent_name: Optional[str] = None,
    ):
        super().__init__(
            message=message,
            details=details,
            severity=ErrorSeverity.WARNING,
            agent_name=agent_name,
        )
        self.error_code = "AGENT_VALIDATION_ERROR"
        self.status_code = status.HTTP_400_BAD_REQUEST


# ========== Workflow Errors ==========


class WorkflowError(BaseError):
    """Base exception for all workflow-related errors"""

    def __init__(
        self,
        message: str,
        node_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        workflow_name: Optional[str] = None,
    ):
        super().__init__(
            message=message,
            details=details,
            severity=severity,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            error_code="WORKFLOW_ERROR",
        )
        self.node_id = node_id
        self.workflow_name = workflow_name

        if details is None:
            self.details = {}

        if node_id:
            self.details["node_id"] = node_id

        if workflow_name:
            self.details["workflow_name"] = workflow_name


class NodeExecutionError(WorkflowError):
    """Exception raised during node execution"""

    def __init__(
        self,
        message: str,
        node_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        workflow_name: Optional[str] = None,
    ):
        super().__init__(
            message=message,
            node_id=node_id,
            details=details,
            workflow_name=workflow_name,
        )
        self.error_code = "NODE_EXECUTION_ERROR"


class WorkflowSetupError(WorkflowError):
    """Exception raised during workflow setup"""

    def __init__(
        self,
        message: str,
        node_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        workflow_name: Optional[str] = None,
    ):
        super().__init__(
            message=message,
            node_id=node_id,
            details=details,
            workflow_name=workflow_name,
        )
        self.error_code = "WORKFLOW_SETUP_ERROR"


Loc = Tuple[Union[int, str], ...]


# ========== Validation Errors ==========


class CustomValidationError(BaseError):
    """Base exception for all custom validation errors"""

    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        severity: ErrorSeverity = ErrorSeverity.WARNING,
    ):
        super().__init__(
            message=message,
            details=details,
            severity=severity,
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            error_code="VALIDATION_ERROR",
        )


class _ErrorDictRequired(TypedDict):
    loc: Loc
    msg: str
    type: str


class ErrorDict(_ErrorDictRequired, total=False):
    ctx: Dict[str, Any]


class MultipleValidationErrors(CustomValidationError):
    """Exception for handling multiple validation errors at once"""

    def __init__(
        self,
        errors: List[ErrorDict],
        message: str = "Multiple validation errors occurred",
        severity: ErrorSeverity = ErrorSeverity.WARNING,
    ):
        super().__init__(message=message, severity=severity)
        self.errors = errors
        self.details = {"errors": errors}
        self.error_code = "MULTIPLE_VALIDATION_ERRORS"


# ========== Helper Functions ==========


def get_errors(errors: list[Any] = []) -> dict[str, list[str]]:
    """Format validation errors into a more readable structure"""
    reformatted_message: dict[str, list[str]] = defaultdict(list)
    for pydantic_error in errors:
        loc, msg = pydantic_error["loc"], pydantic_error["msg"]
        filtered_loc = loc[1:] if loc[0] in ("body", "query", "path") else loc
        filtered_loc = [f if isinstance(f, str) else "body" if isinstance(f, int) else str(f) for f in filtered_loc]
        field_string = ".".join(filtered_loc)
        if field_string == "":
            field_string = "body"
        reformatted_message[field_string].append(msg)
    return reformatted_message


def raise_validation_error(loc: str = "body", msg: str = "Wrong value") -> None:
    """Raise a FastAPI validation error with the given location and message"""
    raise RequestValidationError(
        [
            {
                "loc": ["path", loc],
                "msg": "Value error, " + msg,
            }
        ]
    )


def should_retry_exception(exception: Exception) -> bool:
    """Determine whether an exception should be retried

    Args:
        exception: The exception to check

    Returns:
        True if the exception should be retried, False otherwise
    """
    # Check if it's a RetryError with explicit information
    if isinstance(exception, RetryError):
        if exception.retry_count < exception.max_retries:
            logger.info(
                f"Retrying due to RetryError: {exception}, retry {exception.retry_count}/{exception.max_retries}"
            )
            return True
        else:
            logger.info(f"Not retrying due to max retries exceeded: {exception}")
            return False

    # List of exception types that should be retried
    retryable_exceptions = (
        ConnectionError,  # Network issues
        TimeoutError,  # Timeouts
        ToolRetryError,  # Tool retry errors
        RetryException,  # Custom retry exception
        CustomValidationError,  # Custom validation errors (might be temporary)
    )

    # Check if exception is of a retryable type
    if isinstance(exception, retryable_exceptions):
        logger.info(f"Retrying due to exception type: {type(exception).__name__}")
        return True

    # List of error message patterns that suggest retrying
    retryable_messages = [
        "rate limit",
        "too many requests",
        "429",  # Rate limiting
        "timeout",
        "timed out",  # Timeouts
        "connection error",
        "connection refused",  # Connection issues
        "service unavailable",
        "503",  # Service availability
        "temporary failure",
        "try again",  # Temporary failures
        "overloaded",
        "under heavy load",  # Load issues
    ]

    # Check if exception message contains retryable phrases
    error_message = str(exception).lower()
    for msg in retryable_messages:
        if msg in error_message:
            logger.info(f"Retrying due to message pattern '{msg}' in: {error_message}")
            return True

    logger.info(f"Not retrying exception: {exception}")
    return False
