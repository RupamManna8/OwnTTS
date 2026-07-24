"""
Application-specific exception hierarchy.

Every exception carries an HTTP status code and a machine-readable error
code so the API layer can translate them into consistent JSON error
responses without inspecting exception types elsewhere.
"""
from typing import Any, Dict, Optional


class TTSBaseException(Exception):
    """Base class for all application exceptions."""

    status_code: int = 500
    error_code: str = "internal_error"

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "error": self.error_code,
            "message": self.message,
            "details": self.details,
        }


class ValidationError(TTSBaseException):
    """Raised when request input fails validation rules."""

    status_code = 422
    error_code = "validation_error"


class TextTooLongError(ValidationError):
    error_code = "text_too_long"


class EmptyTextError(ValidationError):
    error_code = "empty_text"


class UnsupportedVoiceError(ValidationError):
    error_code = "unsupported_voice"


class UnsupportedFormatError(ValidationError):
    error_code = "unsupported_format"


class InvalidSpeedError(ValidationError):
    error_code = "invalid_speed"


class ModelNotLoadedError(TTSBaseException):
    """Raised when inference is attempted before the model finished loading."""

    status_code = 503
    error_code = "model_not_loaded"


class ModelLoadError(TTSBaseException):
    """Raised when the Kokoro model fails to load from disk."""

    status_code = 500
    error_code = "model_load_error"


class SynthesisError(TTSBaseException):
    """Raised when audio generation fails for a valid request."""

    status_code = 500
    error_code = "synthesis_error"


class AudioProcessingError(TTSBaseException):
    """Raised when post-processing (format conversion, effects) fails."""

    status_code = 500
    error_code = "audio_processing_error"


class CacheError(TTSBaseException):
    """Raised on cache read/write failures. Never fatal to a request."""

    status_code = 500
    error_code = "cache_error"


class RateLimitExceededError(TTSBaseException):
    status_code = 429
    error_code = "rate_limit_exceeded"