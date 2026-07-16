"""Structured logging setup for production."""

import logging
import logging.handlers
from pathlib import Path
from typing import Optional
import json
from datetime import datetime

from dashboard.config import get_config


class JsonFormatter(logging.Formatter):
    """JSON logging formatter for structured logs."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_obj = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)

        if hasattr(record, "extra_data"):
            log_obj["extra"] = record.extra_data

        return json.dumps(log_obj, default=str)


class _LoggerManager:
    """Centralized logger management."""

    _instance: Optional["_LoggerManager"] = None
    _loggers: dict = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.config = get_config()
        self._setup_root_logger()
        self._initialized = True

    def _setup_root_logger(self) -> None:
        """Setup root logger with file and console handlers."""
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, self.config.logging.level))

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(self.config.logging.format)
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)

        # File handler with rotation
        if self.config.logging.file:
            self.config.logging.file.parent.mkdir(parents=True, exist_ok=True)

            file_handler = logging.handlers.RotatingFileHandler(
                self.config.logging.file,
                maxBytes=self.config.logging.max_bytes,
                backupCount=self.config.logging.backup_count,
                encoding="utf-8",
            )
            file_handler.setLevel(logging.DEBUG)
            json_formatter = JsonFormatter()
            file_handler.setFormatter(json_formatter)
            root_logger.addHandler(file_handler)

    def get_logger(self, name: str) -> logging.Logger:
        """Get or create a logger by name."""
        if name not in self._loggers:
            self._loggers[name] = logging.getLogger(name)
        return self._loggers[name]


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance."""
    manager = _LoggerManager()
    return manager.get_logger(name)


def log_audit(event: str, requester: str, resource: str, status: str, **kwargs):
    """Log an audit event."""
    logger = get_logger("audit")
    extra_data = {
        "event": event,
        "requester": requester,
        "resource": resource,
        "status": status,
        **kwargs,
    }
    logger.info(f"{event}: {resource} by {requester}", extra=extra_data)


# Convenient module-level logger
_default_logger = get_logger(__name__)
