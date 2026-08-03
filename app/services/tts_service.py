"""
Text-to-speech inference service.

Design notes
------------
`BaseTTSService` defines the contract every backend must satisfy. Adding a
new engine (Piper, XTTS, MeloTTS, ...) means writing one new subclass and
wiring it up in `get_tts_service()` -- nothing else in the app needs to
change (routes, caching, validation, and audio post-processing are all
engine-agnostic).

`KokoroTTSService` wraps the `kokoro-onnx` package (ONNX Runtime port of
Kokoro-82M). The model is loaded exactly once per process (singleton) and
guarded by an asyncio.Lock so concurrent requests during startup can't
trigger duplicate loads. Inference itself is CPU-bound, so it is executed
in a worker thread via `asyncio.to_thread` to avoid blocking the event
loop and to let FastAPI serve many requests concurrently.
"""
import asyncio
import os
import re
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from app.core.config import Settings, get_settings
from app.core.exception import ModelLoadError, ModelNotLoadedError, SynthesisError
from app.core.logger import get_logger
from app.services.voice_manager import VoiceManager

logger = get_logger("tts_service")

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_MAX_CHUNK_CHARS = 240


class BaseTTSService(ABC):
    """Contract that every TTS engine backend must implement."""

    @abstractmethod
    async def initialize(self) -> None:
        """Load model weights. Must be idempotent and safe to await twice."""

    @abstractmethod
    async def synthesize(
        self, text: str, voice: str, speed: float, lang: Optional[str] = None
    ) -> Tuple[np.ndarray, int]:
        """Generate audio. Returns (float32 mono samples, sample_rate)."""

    @abstractmethod
    def is_ready(self) -> bool:
        """Whether the model has finished loading."""

    @abstractmethod
    def voice_manager(self) -> VoiceManager:
        """Return the VoiceManager populated with this engine's voices."""


class KokoroTTSService(BaseTTSService):
    """Singleton Kokoro-82M ONNX TTS backend."""

    _instance: Optional["KokoroTTSService"] = None
    _instance_lock = asyncio.Lock()

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model = None  # kokoro_onnx.Kokoro instance
        self._voice_manager = VoiceManager()
        self._init_lock = asyncio.Lock()
        self._ready = False
        self._load_started_at: Optional[float] = None
        self._load_duration_seconds: Optional[float] = None
        self._model_load_duration: Optional[float] = None
        self._voice_load_duration: Optional[float] = None

    def model_load_duration_seconds(self) -> Optional[float]:
        return self._model_load_duration

    def voice_load_duration_seconds(self) -> Optional[float]:
        return self._voice_load_duration

    @staticmethod
    def _split_text_into_chunks(text: str) -> List[str]:
        """Split text into sentence-sized chunks for parallel synthesis."""
        normalized = " ".join(text.split())
        if not normalized:
            return []

        sentence_chunks = [chunk.strip() for chunk in _SENTENCE_SPLIT_RE.split(normalized) if chunk.strip()]
        if not sentence_chunks:
            sentence_chunks = [normalized]

        chunks: List[str] = []
        for sentence in sentence_chunks:
            if len(sentence) <= _MAX_CHUNK_CHARS:
                chunks.append(sentence)
                continue

            words = sentence.split()
            current_words: List[str] = []
            current_length = 0
            for word in words:
                extra = len(word) if not current_words else len(word) + 1
                if current_words and current_length + extra > _MAX_CHUNK_CHARS:
                    chunks.append(" ".join(current_words))
                    current_words = [word]
                    current_length = len(word)
                else:
                    current_words.append(word)
                    current_length += extra

            if current_words:
                chunks.append(" ".join(current_words))

        return chunks

    async def _synthesize_chunk(self, text: str, voice: str, speed: float, lang: Optional[str]) -> Tuple[np.ndarray, int]:
        if not self._ready or self._model is None:
            raise ModelNotLoadedError("Model is still loading. Please retry shortly.")

        effective_lang = lang or self._settings.DEFAULT_LANG
        return await asyncio.to_thread(
            self._model.create,
            text,
            voice=voice,
            speed=speed,
            lang=effective_lang,
        )

    @classmethod
    async def get_instance(cls, settings: Optional[Settings] = None) -> "KokoroTTSService":
        """Thread/async-safe accessor for the process-wide singleton."""
        if cls._instance is not None:
            return cls._instance

        async with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls(settings or get_settings())
            return cls._instance

    async def initialize(self) -> None:
        if self._ready:
            return

        async with self._init_lock:
            if self._ready:  # re-check after acquiring the lock
                return

            self._load_started_at = time.perf_counter()
            model_path = Path(self._settings.MODEL_PATH)
            voices_path = Path(self._settings.VOICES_PATH)

            if not model_path.exists():
                raise ModelLoadError(
                    f"Kokoro model file not found at '{model_path}'. "
                    "Download kokoro-v1.0.onnx and set MODEL_PATH in .env."
                )
            if not voices_path.exists():
                raise ModelLoadError(
                    f"Kokoro voices file not found at '{voices_path}'. "
                    "Download voices-v1.0.bin and set VOICES_PATH in .env."
                )

            model_load_start = time.perf_counter()
            try:
                # Imported lazily so the rest of the app can be imported/tested
                # without the (relatively heavy) onnxruntime dependency present.
                from kokoro_onnx import Kokoro
                import onnxruntime as rt

                # Configure ONNX Runtime SessionOptions for CPU
                opts = rt.SessionOptions()
                # Explicitly enable CPU memory arena and memory pattern (critical for performance)
                opts.enable_cpu_mem_arena = True
                opts.enable_mem_pattern = True
                opts.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
                opts.execution_mode = rt.ExecutionMode.ORT_SEQUENTIAL
                
                # Do NOT set intra_op_num_threads or inter_op_num_threads to allow OpenMP dynamic scheduling.

                sess = rt.InferenceSession(str(model_path), sess_options=opts, providers=["CPUExecutionProvider"])

                # 1. Apply Tokenizer.phonemize LRU cache monkeypatch
                from functools import lru_cache
                from kokoro_onnx.tokenizer import Tokenizer
                
                original_phonemize = Tokenizer.phonemize
                
                @lru_cache(maxsize=1024)
                def cached_phonemize(self_tok, text, lang="en-us", norm=True):
                    return original_phonemize(self_tok, text, lang, norm)
                    
                Tokenizer.phonemize = cached_phonemize

                # 2. Apply Kokoro._create_audio copy reduction monkeypatch
                from kokoro_onnx.config import MAX_PHONEME_LENGTH, SAMPLE_RATE
                
                def optimized_create_audio(self_kokoro, phonemes: str, voice: np.ndarray, speed: float) -> tuple[np.ndarray, int]:
                    phonemes = phonemes[:MAX_PHONEME_LENGTH]
                    
                    tokens_list = self_kokoro.tokenizer.tokenize(phonemes)
                    token_count = len(tokens_list)
                    
                    # Direct NumPy array construction (avoids Python lists/lists within wrapper)
                    tokens_arr = np.zeros((1, token_count + 2), dtype=np.int64)
                    tokens_arr[0, 1:-1] = tokens_list
                    
                    part_voice = voice[token_count]
                    
                    input_names = [i.name for i in self_kokoro.sess.get_inputs()]
                    if "input_ids" in input_names:
                        inputs = {
                            "input_ids": tokens_arr,
                            "style": part_voice,
                            "speed": np.array([speed], dtype=np.int32),
                        }
                    else:
                        inputs = {
                            "tokens": tokens_arr,
                            "style": part_voice,
                            "speed": np.array([speed], dtype=np.float32),
                        }
                        
                    audio = self_kokoro.sess.run(None, inputs)[0]
                    return audio, SAMPLE_RATE
                    
                Kokoro._create_audio = optimized_create_audio

                # Create the Kokoro instance from session
                self._model = await asyncio.to_thread(
                    Kokoro.from_session, sess, str(voices_path)
                )
            except ImportError as exc:
                raise ModelLoadError(
                    "kokoro-onnx package is not installed. Run: pip install kokoro-onnx"
                ) from exc
            except Exception as exc:  # noqa: BLE001
                raise ModelLoadError(f"Failed to load Kokoro model: {exc}") from exc
            self._model_load_duration = time.perf_counter() - model_load_start

            voice_load_start = time.perf_counter()
            voice_names = self._discover_voice_names()
            self._voice_manager.load_from_names(voice_names)
            self._voice_load_duration = time.perf_counter() - voice_load_start

            self._ready = True
            self._load_duration_seconds = round(time.perf_counter() - self._load_started_at, 3)
            logger.info(
                "Kokoro model loaded in %.3fs with %d voices",
                self._load_duration_seconds,
                len(voice_names),
            )

    def _discover_voice_names(self) -> List[str]:
        """Best-effort extraction of the bundled voice list from kokoro-onnx."""
        try:
            voices_attr = getattr(self._model, "voices", None)
            if voices_attr is not None:
                return sorted(list(voices_attr.keys()) if hasattr(voices_attr, "keys") else list(voices_attr))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not introspect voices from model, using fallback list: %s", exc)

        # Fallback: the well-known Kokoro-82M v1.0 voice set.
        return [
            "af_heart", "af_bella", "af_nicole", "af_sarah", "af_sky",
            "am_adam", "am_michael",
            "bf_emma", "bf_isabella",
            "bm_george", "bm_lewis",
        ]

    async def synthesize(
        self, text: str, voice: str, speed: float, lang: Optional[str] = None
    ) -> Tuple[np.ndarray, int]:
        total_start = time.perf_counter()

        # Measure waiting for initialization lock
        lock_start = time.perf_counter()
        async with self._init_lock:
            pass
        lock_wait_ms = (time.perf_counter() - lock_start) * 1000

        if not self._ready or self._model is None:
            raise ModelNotLoadedError("Model is still loading. Please retry shortly.")

        create_start = time.perf_counter()
        try:
            chunks = self._split_text_into_chunks(text)
            if not chunks:
                raise SynthesisError("Synthesis received empty text after normalization.")

            if len(chunks) == 1:
                samples, sample_rate = await self._synthesize_chunk(chunks[0], voice, speed, lang)
            else:
                max_workers = min(len(chunks), max(1, os.cpu_count() or 1))
                semaphore = asyncio.Semaphore(max_workers)

                async def synthesize_with_limit(chunk: str) -> Tuple[np.ndarray, int]:
                    async with semaphore:
                        return await self._synthesize_chunk(chunk, voice, speed, lang)

                chunk_results = await asyncio.gather(*(synthesize_with_limit(chunk) for chunk in chunks))

                sample_rate = int(chunk_results[0][1])
                merged_samples: List[np.ndarray] = []
                for chunk_samples, chunk_sample_rate in chunk_results:
                    if int(chunk_sample_rate) != sample_rate:
                        raise SynthesisError("Parallel synthesis returned mismatched sample rates.")
                    merged_samples.append(np.asarray(chunk_samples, dtype=np.float32))

                samples = np.concatenate(merged_samples) if len(merged_samples) > 1 else merged_samples[0]

            create_ms = (time.perf_counter() - create_start) * 1000
            total_synth_ms = (time.perf_counter() - total_start) * 1000

            if self._settings.PROFILE_ENABLED:
                print(
                    f"\nInitialization Wait : {lock_wait_ms:.2f} ms\n"
                    f"create()            : {create_ms:.2f} ms\n"
                    f"Total synthesize()  : {total_synth_ms:.2f} ms\n"
                )

                # Record timings in request-scoped profile context
                from app.core.profiler import profile_context
                ctx = profile_context.get()
                if ctx:
                    ctx["timings"]["Inference"] = total_synth_ms
                    ctx["metadata"]["samples"] = len(samples)
                    ctx["metadata"]["sample_rate"] = int(sample_rate)

            return np.asarray(samples, dtype=np.float32), int(sample_rate)
        except Exception as exc:  # noqa: BLE001
            raise SynthesisError(f"Audio generation failed: {exc}") from exc

    def is_ready(self) -> bool:
        return self._ready

    def load_duration_seconds(self) -> Optional[float]:
        return self._load_duration_seconds

    def voice_manager(self) -> VoiceManager:
        return self._voice_manager


async def get_tts_service() -> KokoroTTSService:
    """FastAPI dependency: returns the (already-initializing) singleton service."""
    service = await KokoroTTSService.get_instance()
    if not service.is_ready():
        await service.initialize()
    return service