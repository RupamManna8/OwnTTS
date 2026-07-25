"""
Audio result cache.

Caches generated audio bytes on disk, keyed by a hash of
(text + voice + speed + format). Uses an in-memory OrderedDict as an LRU
index over the on-disk files so lookups are O(1) and eviction is cheap,
while surviving process restarts because the actual bytes live in
`cache/`.
"""
import asyncio
import json
import time
from collections import OrderedDict
from pathlib import Path
from typing import Optional

from app.core.config import Settings
from app.core.exception import CacheError
from app.core.logger import get_logger

logger = get_logger("cache")


class AudioCache:
    """Thread-safe (asyncio-safe) LRU cache for generated audio bytes."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._cache_dir = settings.cache_path
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self._cache_dir / "index.json"
        self._lock = asyncio.Lock()
        self._index: "OrderedDict[str, dict]" = OrderedDict()
        self._hits = 0
        self._misses = 0
        self._load_index()

    def _load_index(self) -> None:
        start_time = time.perf_counter()
        if self._index_path.exists():
            try:
                raw = json.loads(self._index_path.read_text(encoding="utf-8"))
                ttl = self._settings.CACHE_TTL_SECONDS
                now = time.time()
                valid_entries = {}
                expired_count = 0
                for k, v in raw.items():
                    created_at = v.get("created_at", 0)
                    fmt = v.get("format", "wav")
                    if ttl > 0 and (now - created_at) > ttl:
                        self._file_path(k, fmt).unlink(missing_ok=True)
                        expired_count += 1
                    else:
                        valid_entries[k] = v

                if expired_count > 0:
                    logger.info("Evicted %d expired cache entries during startup", expired_count)

                # Preserve insertion/recency order as stored.
                self._index = OrderedDict(
                    (k, v) for k, v in sorted(valid_entries.items(), key=lambda kv: kv[1].get("last_used", 0))
                )
                if expired_count > 0:
                    self._persist_index()
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Failed to load cache index, starting fresh: %s", exc)
                self._index = OrderedDict()
        if self._settings.PROFILE_ENABLED:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            logger.info("Index deserialization time: %.2f ms", elapsed_ms)

    def _persist_index(self) -> None:
        start_time = time.perf_counter()
        try:
            # Strip out "data" field (raw bytes cache) before serializing index
            serializable_index = {
                k: {ki: vi for ki, vi in v.items() if ki != "data"}
                for k, v in self._index.items()
            }
            self._index_path.write_text(json.dumps(serializable_index), encoding="utf-8")
        except OSError as exc:
            raise CacheError(f"Failed to persist cache index: {exc}") from exc
        if self._settings.PROFILE_ENABLED:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            logger.info("Index serialization time: %.2f ms", elapsed_ms)

    def _file_path(self, key: str, fmt: str) -> Path:
        return self._cache_dir / f"{key}.{fmt}"

    async def get(self, key: str, fmt: str) -> Optional[bytes]:
        if not self._settings.CACHE_ENABLED:
            return None

        start_time = time.perf_counter()
        hit = False
        data = None
        try:
            async with self._lock:
                entry = self._index.get(key)
                if entry is None:
                    self._misses += 1
                    return None

                # Check TTL expiration (both memory/disk entries)
                ttl = self._settings.CACHE_TTL_SECONDS
                if ttl > 0 and (time.time() - entry.get("created_at", 0)) > ttl:
                    self._index.pop(key, None)
                    self._file_path(key, fmt).unlink(missing_ok=True)
                    self._misses += 1
                    return None

                # Return memory-cached bytes instantly (L1 hit)
                if "data" in entry:
                    self._index.move_to_end(key)
                    entry["last_used"] = time.time()
                    self._hits += 1
                    hit = True
                    data = entry["data"]
                    return data

                path = self._file_path(key, fmt)
                if not path.exists():
                    # Index/disk drifted apart; self-heal.
                    self._index.pop(key, None)
                    self._misses += 1
                    return None

                self._index.move_to_end(key)
                entry["last_used"] = time.time()
                self._hits += 1
                hit = True

            data = path.read_bytes()
            # Store in in-memory cache for subsequent hits
            async with self._lock:
                if key in self._index:
                    self._index[key]["data"] = data
            return data
        except OSError as exc:
            raise CacheError(f"Failed to read cached audio: {exc}") from exc
        finally:
            if self._settings.PROFILE_ENABLED:
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                status = "HIT" if hit and data is not None else "MISS"
                print(f"Cache {status}\nLookup time: {elapsed_ms:.2f} ms")
                from app.core.profiler import profile_context
                ctx = profile_context.get()
                if ctx:
                    ctx["timings"]["Cache Lookup"] = elapsed_ms
                    ctx["metadata"]["cache_status"] = status

    async def set(self, key: str, fmt: str, data: bytes) -> None:
        if not self._settings.CACHE_ENABLED:
            return

        start_time = time.perf_counter()
        try:
            path = self._file_path(key, fmt)
            try:
                path.write_bytes(data)
            except OSError as exc:
                raise CacheError(f"Failed to write cached audio: {exc}") from exc

            async with self._lock:
                now = time.time()
                
                # Proactively evict any expired items from the cache index
                ttl = self._settings.CACHE_TTL_SECONDS
                if ttl > 0:
                    expired_keys = [
                        k for k, v in self._index.items()
                        if (now - v.get("created_at", 0)) > ttl
                    ]
                    for k in expired_keys:
                        entry = self._index.pop(k, None)
                        if entry:
                            self._file_path(k, entry.get("format", "wav")).unlink(missing_ok=True)
                            logger.debug("Evicted expired cache entry %s", k)

                self._index[key] = {
                    "format": fmt,
                    "created_at": now,
                    "last_used": now,
                    "size": len(data),
                    "data": data  # Cache the bytes in memory!
                }
                self._index.move_to_end(key)
                await self._evict_if_needed()
                self._persist_index()
        finally:
            if self._settings.PROFILE_ENABLED:
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                print(f"Store time: {elapsed_ms:.2f} ms")
                from app.core.profiler import profile_context
                ctx = profile_context.get()
                if ctx:
                    ctx["timings"]["Cache Save"] = elapsed_ms

    async def _evict_if_needed(self) -> None:
        max_size = self._settings.CACHE_SIZE
        while len(self._index) > max_size:
            oldest_key, oldest_entry = next(iter(self._index.items()))
            self._index.pop(oldest_key, None)
            self._file_path(oldest_key, oldest_entry.get("format", "wav")).unlink(missing_ok=True)
            logger.debug("Evicted cache entry %s (LRU)", oldest_key)

    def stats(self) -> dict:
        total_size = sum(entry.get("size", 0) for entry in self._index.values())
        total_lookups = self._hits + self._misses
        hit_rate = (self._hits / total_lookups) if total_lookups else 0.0
        return {
            "enabled": self._settings.CACHE_ENABLED,
            "total_entries": len(self._index),
            "max_entries": self._settings.CACHE_SIZE,
            "total_size_bytes": total_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(hit_rate, 4),
        }

    def size(self) -> int:
        return len(self._index)