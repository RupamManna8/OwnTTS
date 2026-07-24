"""Request ID tagging and a lightweight in-memory rate limiter."""
import time
from collections import defaultdict, deque
from typing import Deque, Dict

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from app.core.config import Settings
from app.core.logger import get_logger
from app.utils.helpers import new_request_id

logger = get_logger("middleware")


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Attaches a unique request ID to each request/response for traceability."""

    def __init__(self, app: ASGIApp, header_name: str = "X-Request-ID") -> None:
        super().__init__(app)
        self._header_name = header_name

    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get(self._header_name) or new_request_id()
        request.state.request_id = request_id

        start = time.perf_counter()
        response: Response = await call_next(request)
        elapsed_ms = round((time.perf_counter() - start) * 1000, 2)

        response.headers[self._header_name] = request_id
        response.headers["X-Process-Time-Ms"] = str(elapsed_ms)
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Fixed-window per-client-IP rate limiter, in-memory (no external deps).

    Suitable for a single-process deployment. For multi-worker/multi-node
    deployments, back this with Redis instead.
    """

    def __init__(self, app: ASGIApp, settings: Settings) -> None:
        super().__init__(app)
        self._enabled = settings.ENABLE_RATE_LIMIT
        self._limit = settings.RATE_LIMIT_PER_MINUTE
        self._window_seconds = 60.0
        self._hits: Dict[str, Deque[float]] = defaultdict(deque)

    async def dispatch(self, request: Request, call_next):
        if not self._enabled or request.url.path in {"/health", "/", "/docs", "/openapi.json"}:
            return await call_next(request)

        client_ip = request.client.host if request.client else "unknown"
        now = time.time()
        window = self._hits[client_ip]

        while window and now - window[0] > self._window_seconds:
            window.popleft()

        if len(window) >= self._limit:
            logger.warning("Rate limit exceeded for %s", client_ip)
            from fastapi.responses import JSONResponse

            return JSONResponse(
                status_code=429,
                content={
                    "error": "rate_limit_exceeded",
                    "message": f"Rate limit of {self._limit} requests/minute exceeded.",
                    "details": {"retry_after_seconds": self._window_seconds},
                },
                headers={"Retry-After": str(int(self._window_seconds))},
            )

        window.append(now)
        return await call_next(request)


class ProfilingMiddleware(BaseHTTPMiddleware):
    """Measures latency, memory, CPU, and generates performance reports for requests."""

    def __init__(self, app: ASGIApp, settings: Settings) -> None:
        super().__init__(app)
        self._settings = settings

    async def dispatch(self, request: Request, call_next):
        if not self._settings.PROFILE_ENABLED:
            return await call_next(request)

        from app.core.profiler import (
            profile_context,
            get_memory_usage,
            get_peak_memory,
            get_cpu_stats,
            print_performance_reports,
        )

        # Profile context initialization
        ctx = {
            "start_time": time.perf_counter(),
            "timings": {
                "Validation": 0.0,
                "Voice Validation": 0.0,
                "Cache Lookup": 0.0,
                "Inference": 0.0,
                "Trim Silence": 0.0,
                "Normalize Volume": 0.0,
                "Encoding": 0.0,
                "Cache Save": 0.0,
                "Response Build": 0.0,
                "Streaming": 0.0,
                "HTTP Total": 0.0,
            },
            "metadata": {
                "characters": 0,
                "words": 0,
                "est_duration": 0.0,
                "samples": 0,
                "sample_rate": 0,
                "duration": 0.0,
                "output_bytes": 0,
                "total_bytes_streamed": 0,
            },
            "memory": {
                "before": get_memory_usage(),
                "after_inference": 0.0,
                "after_encoding": 0.0,
                "after_response": 0.0,
                "peak": get_peak_memory(),
            },
            "cpu": {
                "pct": 0.0,
                "logical_cores": 0,
                "thread_count": 0,
                "process_cpu_time": 0.0,
            },
        }
        token = profile_context.set(ctx)

        try:
            response: Response = await call_next(request)
            
            # Record total response building & latency
            http_total_ms = (time.perf_counter() - ctx["start_time"]) * 1000
            ctx["timings"]["HTTP Total"] = http_total_ms
            
            # RAM usage after response
            ctx["memory"]["after_response"] = get_memory_usage()
            ctx["memory"]["peak"] = max(ctx["memory"]["peak"], get_peak_memory())
            
            # CPU telemetry
            cpu_pct, logical_cores, thread_count, cpu_time = get_cpu_stats()
            ctx["cpu"]["pct"] = cpu_pct
            ctx["cpu"]["logical_cores"] = logical_cores
            ctx["cpu"]["thread_count"] = thread_count
            ctx["cpu"]["process_cpu_time"] = cpu_time
            
            # Print performance reports only for TTS generation requests
            if request.url.path in {"/v1/audio/speech", "/v1/audio/preview"}:
                print_performance_reports(ctx)
                
            return response
        finally:
            profile_context.reset(token)