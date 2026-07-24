"""Performance profiling engine and metrics collection helpers."""
import contextvars
import time
from functools import wraps
import inspect
from typing import Dict, Any, Optional
import psutil
from app.core.logger import get_logger

logger = get_logger("profiler")

# Thread/async-safe context variable to store request-scoped measurements
profile_context: contextvars.ContextVar[Optional[Dict[str, Any]]] = contextvars.ContextVar("profile_context", default=None)

def get_memory_usage() -> float:
    """Returns current RSS memory usage in MB."""
    try:
        import os
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)
    except Exception:
        return 0.0

def get_peak_memory() -> float:
    """Returns peak memory usage (peak working set on Windows, RSS fallback on other OS) in MB."""
    try:
        import os
        process = psutil.Process(os.getpid())
        info = process.memory_info()
        if hasattr(info, 'peak_wset'):
            return info.peak_wset / (1024 * 1024)
        return info.rss / (1024 * 1024)
    except Exception:
        return 0.0

def get_cpu_stats():
    """Returns process CPU usage percentage, logical cores, thread count, and CPU time (sec)."""
    try:
        import os
        process = psutil.Process(os.getpid())
        cpu_pct = process.cpu_percent(interval=None)
        logical_cores = psutil.cpu_count(logical=True)
        thread_count = process.num_threads()
        cpu_times = process.cpu_times()
        cpu_time = cpu_times.user + cpu_times.system
        return cpu_pct, logical_cores, thread_count, cpu_time
    except Exception:
        return 0.0, 0, 0, 0.0

def profile_step(name: str):
    """Decorator to profile a function step. Logs and updates request context."""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            from app.core.config import get_settings
            settings = get_settings()
            if not settings.PROFILE_ENABLED:
                return await func(*args, **kwargs)
            
            start = time.perf_counter()
            res = await func(*args, **kwargs)
            elapsed_ms = (time.perf_counter() - start) * 1000
            
            ctx = profile_context.get()
            if ctx:
                ctx["timings"][name] = elapsed_ms
            
            logger.info("%s duration: %.2f ms", name, elapsed_ms)
            return res

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            from app.core.config import get_settings
            settings = get_settings()
            if not settings.PROFILE_ENABLED:
                return func(*args, **kwargs)
            
            start = time.perf_counter()
            res = func(*args, **kwargs)
            elapsed_ms = (time.perf_counter() - start) * 1000
            
            ctx = profile_context.get()
            if ctx:
                ctx["timings"][name] = elapsed_ms
            
            logger.info("%s duration: %.2f ms", name, elapsed_ms)
            return res

        if inspect.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    return decorator

def print_performance_reports(ctx: Dict[str, Any]) -> None:
    """Print Test 2 (Request Timeline) and Test 14 (Performance Report) exactly as specified."""
    timings = ctx.get("timings", {})
    metadata = ctx.get("metadata", {})
    memory = ctx.get("memory", {})
    cpu = ctx.get("cpu", {})
    
    # Extract timings or default to 0.0
    validation = timings.get("Validation", 0.0)
    voice_validation = timings.get("Voice Validation", 0.0)
    cache_lookup = timings.get("Cache Lookup", 0.0)
    inference = timings.get("Inference", 0.0)
    trim_silence = timings.get("Trim Silence", 0.0)
    normalize_volume = timings.get("Normalize Volume", 0.0)
    encoding = timings.get("Encoding", 0.0)
    cache_save = timings.get("Cache Save", 0.0)
    response_build = timings.get("Response Build", 0.0)
    streaming = timings.get("Streaming", 0.0)
    http_total = timings.get("HTTP Total", 0.0)
    
    post_processing = trim_silence + normalize_volume
    
    # Print Test 2 Timeline
    timeline_str = (
        "\n==================================================\n"
        "Request Timeline\n\n"
        f"Validation        : {validation:.2f} ms\n"
        f"Voice Validation  : {voice_validation:.2f} ms\n"
        f"Cache Lookup      : {cache_lookup:.2f} ms\n"
        f"Inference         : {inference:.2f} ms\n"
        f"Trim Silence      : {trim_silence:.2f} ms\n"
        f"Normalize Volume  : {normalize_volume:.2f} ms\n"
        f"Encoding          : {encoding:.2f} ms\n"
        f"Cache Save        : {cache_save:.2f} ms\n"
        f"Response Build    : {response_build:.2f} ms\n\n"
        f"TOTAL             : {http_total:.2f} ms\n"
        "=================================================="
    )
    print(timeline_str)
    
    # Print Test 14 Performance Report
    report_str = (
        "\n================ PERFORMANCE REPORT ================\n\n"
        f"Characters          : {metadata.get('characters', 0)}\n"
        f"Words               : {metadata.get('words', 0)}\n"
        f"Inference Time      : {inference:.2f} ms\n"
        f"Encoding Time       : {encoding:.2f} ms\n"
        f"Cache Lookup        : {cache_lookup:.2f} ms\n"
        f"Cache Save          : {cache_save:.2f} ms\n"
        f"Post Processing     : {post_processing:.2f} ms\n"
        f"Streaming           : {streaming:.2f} ms\n"
        f"HTTP Total          : {http_total:.2f} ms\n\n"
        f"CPU Usage           : {cpu.get('pct', 0.0):.2f} %\n"
        f"RAM Usage           : {memory.get('after_response', 0.0):.2f} MB\n"
        f"Peak RAM            : {memory.get('peak', 0.0):.2f} MB\n\n"
        f"Audio Duration      : {metadata.get('duration', 0.0):.3f} s\n"
        f"Output Size         : {metadata.get('output_bytes', 0)} bytes\n\n"
        "===================================================="
    )
    print(report_str)

    # TEST 12: Long Text Scaling Log
    if inference > 0:
        inf_seconds = inference / 1000.0
        chars = metadata.get('characters', 0)
        words = metadata.get('words', 0)
        samples = metadata.get('samples', 0)
        duration = metadata.get('duration', 0.0)
        
        chars_per_sec = chars / inf_seconds if inf_seconds > 0 else 0.0
        words_per_sec = words / inf_seconds if inf_seconds > 0 else 0.0
        samples_per_sec = samples / inf_seconds if inf_seconds > 0 else 0.0
        
        scale_str = (
            f"Scaling Details -> Input Length: {chars} chars | "
            f"Inference Time: {inference:.2f} ms | "
            f"Output Duration: {duration:.3f} s | "
            f"Speed: {chars_per_sec:.2f} chars/sec, {words_per_sec:.2f} words/sec, {samples_per_sec:.2f} samples/sec"
        )
        logger.info(scale_str)
