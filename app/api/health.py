"""Service info and health-check endpoints."""
import time

from fastapi import APIRouter, Depends

from app.core.config import Settings, get_settings
from app.models.response import HealthResponse, ServiceInfoResponse
from app.services.cache import AudioCache
from app.services.tts_service import KokoroTTSService, get_tts_service

router = APIRouter(tags=["health"])

_SERVICE_START_TIME = time.time()
_ACTIVE_REQUESTS = {"count": 0}


def get_active_requests_counter() -> dict:
    return _ACTIVE_REQUESTS


@router.get("/", response_model=ServiceInfoResponse)
async def get_service_info(settings: Settings = Depends(get_settings)) -> ServiceInfoResponse:
    """Basic service information."""
    return ServiceInfoResponse(
        name=settings.APP_NAME,
        version=settings.APP_VERSION,
        description="Local, offline, CPU-optimized text-to-speech API powered by Kokoro-82M.",
        docs_url="/docs",
        status="running",
    )


@router.get("/health", response_model=HealthResponse)
async def health_check(
    settings: Settings = Depends(get_settings),
) -> HealthResponse:
    """Liveness/readiness probe."""
    try:
        service = await KokoroTTSService.get_instance()
        model_loaded = service.is_ready()
    except Exception:  # noqa: BLE001
        model_loaded = False

    from app.api.routes import get_cache_dependency  # local import avoids circular import
    cache: AudioCache = await get_cache_dependency()

    return HealthResponse(
        status="ok" if model_loaded else "degraded",
        model_loaded=model_loaded,
        uptime_seconds=round(time.time() - _SERVICE_START_TIME, 2),
        cache_enabled=settings.CACHE_ENABLED,
        cached_items=cache.size(),
        active_requests=_ACTIVE_REQUESTS["count"],
    )