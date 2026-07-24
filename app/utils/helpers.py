"""Small stateless helper functions shared across services."""
import hashlib
import time
import uuid
from typing import Optional


def make_cache_key(text: str, voice: str, speed: float, fmt: str, lang: Optional[str] = None) -> str:
    """Deterministically hash (text + voice + speed + format + lang) -> cache key."""
    payload = f"{text}|{voice}|{speed:.2f}|{fmt}|{lang or ''}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def new_request_id() -> str:
    return uuid.uuid4().hex[:16]


def estimate_duration_seconds(text: str, speed: float = 1.0, words_per_minute: int = 155) -> float:
    """Rough duration estimate based on average speaking rate, adjusted for speed."""
    word_count = max(len(text.split()), 1)
    base_minutes = word_count / words_per_minute
    return round((base_minutes * 60.0) / max(speed, 0.01), 2)


class Timer:
    """Lightweight context manager for measuring elapsed wall-clock time."""

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        self.elapsed_seconds = 0.0
        return self

    def __exit__(self, *_exc_info) -> None:
        self.elapsed_seconds = round(time.perf_counter() - self._start, 4)


def bytes_to_human(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024.0:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} TB"