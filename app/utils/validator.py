"""
Request validation helpers.

Kept separate from the Pydantic schema layer because some rules depend on
runtime state (e.g. which voices are actually installed) that cannot be
expressed as static field constraints.
"""
from typing import Iterable

from app.core.config import Settings
from app.core.exception import (
    EmptyTextError,
    InvalidSpeedError,
    TextTooLongError,
    UnsupportedFormatError,
    UnsupportedVoiceError,
)


def validate_text(text: str, settings: Settings) -> str:
    stripped = text.strip()
    if len(stripped) < settings.MIN_TEXT_LENGTH:
        raise EmptyTextError("Text must not be empty.")
    if len(stripped) > settings.MAX_TEXT_LENGTH:
        raise TextTooLongError(
            f"Text exceeds maximum length of {settings.MAX_TEXT_LENGTH} characters.",
            details={"length": len(stripped), "max_length": settings.MAX_TEXT_LENGTH},
        )
    return stripped


def validate_voice(voice: str, available_voices: Iterable[str]) -> str:
    available = set(available_voices)
    if voice not in available:
        raise UnsupportedVoiceError(
            f"Voice '{voice}' is not available.",
            details={"requested_voice": voice, "available_voices": sorted(available)},
        )
    return voice


def validate_format(fmt: str, settings: Settings) -> str:
    normalized = fmt.lower().strip()
    if normalized not in settings.ALLOWED_FORMATS:
        raise UnsupportedFormatError(
            f"Format '{fmt}' is not supported.",
            details={"requested_format": fmt, "allowed_formats": settings.ALLOWED_FORMATS},
        )
    return normalized


def validate_speed(speed: float, settings: Settings) -> float:
    if not (settings.MIN_SPEED <= speed <= settings.MAX_SPEED):
        raise InvalidSpeedError(
            f"Speed must be between {settings.MIN_SPEED} and {settings.MAX_SPEED}.",
            details={"requested_speed": speed, "min": settings.MIN_SPEED, "max": settings.MAX_SPEED},
        )
    return speed