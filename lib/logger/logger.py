"""Simplified logger for PDF document extractor."""

import logging
import os
from enum import Enum
from typing import Any

from pydantic_settings import BaseSettings


class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LoggerSettings(BaseSettings):
    level: LogLevel = LogLevel.INFO
    format_str: str = "%(asctime)s - [%(levelname)s] - %(name)s - %(funcName)s - %(message)s"

    class Config:
        env_prefix = "LOG_"


ExcInfoType = (
    bool
    | tuple[type[BaseException], BaseException, Any]
    | tuple[None, None, None]
    | BaseException
    | None
)


class Logger:
    _instances: dict[str, "Logger"] = {}
    _root_configured: bool = False

    @property
    def logger(self) -> logging.Logger:
        return self._log_instance

    @classmethod
    def get_logger(cls, name: str) -> "Logger":
        if not cls._root_configured:
            cls._configure_root_logging()
            cls._root_configured = True

        if name not in cls._instances:
            cls._instances[name] = cls(name)
        return cls._instances[name]

    @classmethod
    def _configure_root_logging(cls) -> None:
        """Configure root logger once"""
        settings = LoggerSettings()

        formatter = logging.Formatter(
            fmt=settings.format_str, datefmt="%Y-%m-%d %H:%M:%S"
        )

        root_logger = logging.getLogger()
        root_logger.handlers.clear()
        root_logger.setLevel(settings.level.value)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    def __init__(self, name: str):
        self.name = name
        self._log_instance = logging.getLogger(name)
        self.settings = LoggerSettings()
        self._log_instance.setLevel(self.settings.level.value)

    def debug(
        self,
        message: str,
        *,
        exc_info: ExcInfoType = None,
        extra: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self._log_instance.debug(message, exc_info=exc_info, extra=extra or kwargs)

    def info(
        self,
        message: str,
        *,
        exc_info: ExcInfoType = None,
        extra: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self._log_instance.info(message, exc_info=exc_info, extra=extra or kwargs)

    def warning(
        self,
        message: str,
        *,
        exc_info: ExcInfoType = None,
        extra: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self._log_instance.warning(message, exc_info=exc_info, extra=extra or kwargs)

    def error(
        self,
        message: str,
        *,
        exc_info: ExcInfoType = None,
        extra: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self._log_instance.error(message, exc_info=exc_info, extra=extra or kwargs)

    def critical(
        self,
        message: str,
        *,
        exc_info: ExcInfoType = None,
        extra: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self._log_instance.critical(message, exc_info=exc_info, extra=extra or kwargs)
