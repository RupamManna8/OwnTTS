import sys
import time
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

APP_IMPORT_START_TIME = time.perf_counter()

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse

from app.api import health, routes
from app.core.config import get_settings
from app.core.exception import TTSBaseException
from app.core.logger import get_logger
from app.core.middleware import RateLimitMiddleware, RequestIDMiddleware, ProfilingMiddleware
from app.services.tts_service import KokoroTTSService

logger = get_logger("main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Preload the Kokoro model once at startup so the first real request
    doesn't pay the cold-start cost."""
    settings = get_settings()
    logger.info("Starting %s v%s (%s)", settings.APP_NAME, settings.APP_VERSION, settings.ENVIRONMENT)

    init_start = time.perf_counter()
    service = await KokoroTTSService.get_instance(settings)
    try:
        await service.initialize()
        
        # Extract individual loading metrics from service
        model_load_time = service.model_load_duration_seconds()
        voice_load_time = service.voice_load_duration_seconds()
        
        init_duration = time.perf_counter() - init_start
        startup_duration = time.perf_counter() - APP_IMPORT_START_TIME
        
        if settings.PROFILE_ENABLED:
            logger.info("=== STARTUP PERFORMANCE REPORT ===")
            logger.info("FastAPI startup time : %.4f s", startup_duration)
            logger.info("Kokoro model load    : %.4f s", model_load_time or 0.0)
            logger.info("Voice loading time   : %.4f s", voice_load_time or 0.0)
            logger.info("Total initialization : %.4f s", init_duration)
            logger.info("==================================")
            
    except Exception as exc:  # noqa: BLE001
        # Do not crash the process: expose a degraded /health instead so
        # orchestrators can surface the real error and retry deploys.
        logger.error("Model failed to preload at startup: %s", exc)

    yield

    logger.info("Shutting down %s", settings.APP_NAME)


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description=(
            "Local, offline, unlimited, CPU-optimized text-to-speech API "
            "built on Kokoro-82M and ONNX Runtime."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # --- Middleware (order matters: outermost added last) ---
    app.add_middleware(GZipMiddleware, minimum_size=512)
    app.add_middleware(CORSMiddleware, allow_origins=settings.CORS_ORIGINS, allow_credentials=True,
                        allow_methods=["*"], allow_headers=["*"])
    app.add_middleware(RateLimitMiddleware, settings=settings)
    app.add_middleware(RequestIDMiddleware, header_name=settings.REQUEST_ID_HEADER)
    app.add_middleware(ProfilingMiddleware, settings=settings)

    # --- Routers ---
    app.include_router(health.router)
    app.include_router(routes.router)

    # --- Exception handlers ---
    @app.exception_handler(TTSBaseException)
    async def handle_tts_exception(request: Request, exc: TTSBaseException) -> JSONResponse:
        request_id = getattr(request.state, "request_id", "-")
        logger.error(
            "Handled exception %s: %s (request_id=%s)",
            exc.error_code, exc.message, request_id,
        )
        return JSONResponse(status_code=exc.status_code, content=exc.to_dict())

    @app.exception_handler(Exception)
    async def handle_unexpected_exception(request: Request, exc: Exception) -> JSONResponse:
        request_id = getattr(request.state, "request_id", "-")
        logger.exception("Unhandled exception (request_id=%s): %s", request_id, exc)
        return JSONResponse(
            status_code=500,
            content={
                "error": "internal_error",
                "message": "An unexpected error occurred.",
                "details": {},
            },
        )

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        workers=settings.WORKERS,
        reload=settings.DEBUG,
    )