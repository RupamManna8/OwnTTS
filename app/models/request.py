"""Pydantic request schemas."""
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

AudioFormat = Literal["wav", "mp3", "pcm"]


class SpeechRequest(BaseModel):
    """Payload for POST /v1/audio/speech."""

    text: str = Field(..., description="Text to synthesize", min_length=1)
    voice: str = Field(default="af_heart", description="Voice identifier")
    speed: float = Field(default=1.0, ge=0.25, le=4.0, description="Playback speed multiplier")
    format: AudioFormat = Field(default="wav", description="Output audio format")
    lang: Optional[str] = Field(default=None, description="Override language code")
    normalize_volume: bool = Field(default=False, description="Normalize output loudness")
    trim_silence: bool = Field(default=False, description="Trim leading/trailing silence")

    @field_validator("text")
    @classmethod
    def _strip_text(cls, value: str) -> str:
        return value.strip()

    @field_validator("voice")
    @classmethod
    def _normalize_voice(cls, value: str) -> str:
        return value.strip()

    model_config = {
        "json_schema_extra": {
            "example": {
                "text": "Hello World",
                "voice": "af_heart",
                "speed": 1.0,
                "format": "wav",
            }
        }
    }


class VoicePreviewRequest(BaseModel):
    """Payload for POST /v1/audio/preview."""

    voice: str = Field(..., description="Voice identifier to preview")
    format: AudioFormat = Field(default="wav")