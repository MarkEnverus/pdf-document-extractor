"""Domain-specific exceptions for business logic."""

from typing import Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class DomainError(Exception):
    """Base domain exception."""

    message: str
    context: Optional[Dict[str, Any]] = None
    cause: Optional[Exception] = None

    def __str__(self) -> str:
        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            return f"{self.message} (context: {context_str})"
        return self.message


@dataclass
class StorageError(DomainError):
    """Storage operation failed."""

    pass


@dataclass
class ProcessingError(DomainError):
    """Document processing failed."""

    pass


@dataclass
class ExtractionError(DomainError):
    """Extraction API error."""

    status_code: Optional[int] = None
    response_body: Optional[str] = None


@dataclass
class StatusError(DomainError):
    """Status tracking error."""

    pass


@dataclass
class DomainValidationError(DomainError):
    """Input validation error."""

    field_name: Optional[str] = None
    invalid_value: Optional[Any] = None


@dataclass
class ConfigurationError(DomainError):
    """Configuration error."""

    config_key: Optional[str] = None
