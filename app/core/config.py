"""
Application configuration.

All settings are loaded from environment variables (via .env) using
pydantic-settings. This is the single source of truth for configuration
and must never be duplicated elsewhere in the codebase.
"""
from functools import lru_cache
from pathlib import Path
from typing import List

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Strongly typed application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Server ---
    HOST: str = Field(default="0.0.0.0", description="Bind host")
    PORT: int = Field(default=8000, description="Bind port")
    WORKERS: int = Field(default=1, description="Uvicorn worker count")
    APP_NAME: str = Field(default="Kokoro TTS API")
    APP_VERSION: str = Field(default="1.0.0")
    ENVIRONMENT: str = Field(default="production")
    DEBUG: bool = Field(default=False)
    PROFILE_ENABLED: bool = Field(default=True, description="Enable latency and profiling statistics")
    PROFILE_DEEP: bool = Field(default=False, description="Enable deep request cProfiling")

    # --- Model ---
    MODEL_PATH: str = Field(
        default=r"D:\OwnTTs\models\kokoro-v1.0.onnx",
        description="Path to the Kokoro ONNX model file",
    )
    VOICES_PATH: str = Field(
        default=r"D:\OwnTTs\models\voices-v1.0.bin",
        description="Path to the Kokoro voices bundle",
    )
    DEFAULT_VOICE: str = Field(default="am_adam")
    DEFAULT_SPEED: float = Field(default=1.0)
    DEFAULT_LANG: str = Field(default="en-us")

    # --- Cache ---
    CACHE_ENABLED: bool = Field(default=True)
    CACHE_DIR: str = Field(default="./cache")
    CACHE_SIZE: int = Field(default=256, description="Max cached items (LRU)")
    CACHE_TTL_SECONDS: int = Field(default=300, description="Keep cache entries for 5 minutes")

    # --- Output ---
    OUTPUT_DIR: str = Field(default="./outputs")
    LOG_DIR: str = Field(default="./logs")

    # --- Validation limits ---
    MAX_TEXT_LENGTH: int = Field(default=5000)
    MIN_TEXT_LENGTH: int = Field(default=1)
    ALLOWED_FORMATS: List[str] = Field(default_factory=lambda: ["wav", "mp3", "pcm"])
    MIN_SPEED: float = Field(default=0.5)
    MAX_SPEED: float = Field(default=2.0)

    # --- Concurrency / performance ---
    MAX_CONCURRENT_REQUESTS: int = Field(default=50)
    INFERENCE_THREADS: int = Field(default=4)

    # --- Security / misc ---
    CORS_ORIGINS: List[str] = Field(default_factory=lambda: ["*"])
    ENABLE_RATE_LIMIT: bool = Field(default=True)
    RATE_LIMIT_PER_MINUTE: int = Field(default=60)
    REQUEST_ID_HEADER: str = Field(default="X-Request-ID")

    @field_validator("CACHE_DIR", "OUTPUT_DIR", "LOG_DIR", mode="after")
    @classmethod
    def _ensure_dir_exists(cls, value: str) -> str:
        path = Path(value)
        path.mkdir(parents=True, exist_ok=True)
        return str(path)

    @property
    def cache_path(self) -> Path:
        return Path(self.CACHE_DIR)

    @property
    def output_path(self) -> Path:
        return Path(self.OUTPUT_DIR)

    @property
    def log_path(self) -> Path:
        return Path(self.LOG_DIR)


@lru_cache
def get_settings() -> Settings:
    """Return a cached, process-wide Settings singleton."""
    return Settings()