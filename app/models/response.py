"""Pydantic response schemas."""
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class ServiceInfoResponse(BaseModel):
    name: str
    version: str
    description: str
    docs_url: str
    status: str


class HealthResponse(BaseModel):
    status: str = Field(description="'ok' or 'degraded'")
    model_loaded: bool
    uptime_seconds: float
    cache_enabled: bool
    cached_items: int
    active_requests: int


class VoiceInfo(BaseModel):
    name: str
    gender: str
    language: str
    language_code: str
    sample_rate: int
    quality: Optional[str] = None


class VoicesResponse(BaseModel):
    total: int
    voices_by_language: Dict[str, List[VoiceInfo]]


class ErrorResponse(BaseModel):
    error: str
    message: str
    details: Dict = Field(default_factory=dict)


class AudioDurationResponse(BaseModel):
    text_length: int
    estimated_duration_seconds: float
    voice: str
    speed: float


class CacheStatsResponse(BaseModel):
    enabled: bool
    total_entries: int
    max_entries: int
    total_size_bytes: int
    hits: int
    misses: int
    hit_rate: float