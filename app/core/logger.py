"""
Centralized logging configuration.

Provides a single `get_logger` factory used across the entire application
so log formatting, rotation, and destinations stay consistent.
"""
import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from app.core.config import get_settings

_LOG_FORMAT = (
    "%(asctime)s | %(levelname)-8s | %(name)s | "
    "req_id=%(request_id)s | %(message)s"
)
_CONFIGURED = False


class _RequestIdFilter(logging.Filter):
    """Injects a default request_id when none is bound to the log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "request_id"):
            record.request_id = "-"
        return True


def _configure_root_logging() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return

    settings = get_settings()
    log_dir = Path(settings.LOG_DIR)
    log_dir.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(_LOG_FORMAT)
    request_filter = _RequestIdFilter()

    root_logger = logging.getLogger("tts_app")
    root_logger.setLevel(logging.DEBUG if settings.DEBUG else logging.INFO)
    root_logger.propagate = False

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.addFilter(request_filter)
    root_logger.addHandler(console_handler)

    # Rotating file handler: 10 MB per file, keep 5 backups
    file_handler = RotatingFileHandler(
        filename=str(log_dir / "tts_app.log"),
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    file_handler.addFilter(request_filter)
    root_logger.addHandler(file_handler)

    # Dedicated error log
    error_handler = RotatingFileHandler(
        filename=str(log_dir / "tts_errors.log"),
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    error_handler.setFormatter(formatter)
    error_handler.addFilter(request_filter)
    error_handler.setLevel(logging.ERROR)
    root_logger.addHandler(error_handler)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced child logger under the `tts_app` root logger."""
    _configure_root_logging()
    return logging.getLogger(f"tts_app.{name}")