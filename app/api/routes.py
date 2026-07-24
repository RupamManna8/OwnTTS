import asyncio
import time
from typing import Optional

from fastapi import APIRouter, Depends, Request
from fastapi.responses import Response, StreamingResponse

from app.api.health import get_active_requests_counter
from app.core.config import Settings, get_settings
from app.core.exception import SynthesisError
from app.core.logger import get_logger
from app.models.request import SpeechRequest, VoicePreviewRequest
from app.models.response import AudioDurationResponse, CacheStatsResponse, VoicesResponse, VoiceInfo
from app.services.audio_processor import AudioProcessor
from app.services.cache import AudioCache
from app.services.tts_service import KokoroTTSService, get_tts_service
from app.utils.helpers import estimate_duration_seconds, make_cache_key
from app.utils.validator import validate_format, validate_speed, validate_text, validate_voice

logger = get_logger("routes")
router = APIRouter(tags=["tts"])

_cache_instance: Optional[AudioCache] = None
_cache_lock = asyncio.Lock()


async def get_cache_dependency() -> AudioCache:
    """FastAPI dependency returning the process-wide AudioCache singleton."""
    global _cache_instance
    if _cache_instance is not None:
        return _cache_instance
    async with _cache_lock:
        if _cache_instance is None:
            _cache_instance = AudioCache(get_settings())
        return _cache_instance


@router.get("/voices", response_model=VoicesResponse)
async def list_voices(
    service: KokoroTTSService = Depends(get_tts_service),
) -> VoicesResponse:
    """List all available Kokoro voices grouped by language."""
    vm = service.voice_manager()
    grouped = vm.grouped_by_language()
    response_grouped = {
        language: [
            VoiceInfo(
                name=v.name,
                gender=v.gender,
                language=v.language,
                language_code=v.language_code,
                sample_rate=v.sample_rate,
            )
            for v in voices
        ]
        for language, voices in grouped.items()
    }
    return VoicesResponse(total=len(vm.all_voices()), voices_by_language=response_grouped)


@router.post("/v1/audio/duration", response_model=AudioDurationResponse)
async def estimate_audio_duration(
    payload: SpeechRequest,
    settings: Settings = Depends(get_settings),
) -> AudioDurationResponse:
    """Estimate spoken duration for a given text/speed without generating audio."""
    text = validate_text(payload.text, settings)
    speed = validate_speed(payload.speed, settings)
    return AudioDurationResponse(
        text_length=len(text),
        estimated_duration_seconds=estimate_duration_seconds(text, speed),
        voice=payload.voice,
        speed=speed,
    )


@router.get("/v1/cache/stats", response_model=CacheStatsResponse)
async def cache_stats(cache: AudioCache = Depends(get_cache_dependency)) -> CacheStatsResponse:
    """Expose cache hit/miss statistics for observability."""
    return CacheStatsResponse(**cache.stats())


async def _generate_audio_bytes(
    payload: SpeechRequest,
    service: KokoroTTSService,
    cache: AudioCache,
    settings: Settings,
    bypass_cache: bool = False,
) -> tuple[bytes, str, bool]:
    """Shared synthesis pipeline used by both /speech and /preview.

    Returns (audio_bytes, mime_type, cache_hit).
    """
    vm = service.voice_manager()

    from app.core.profiler import profile_context, get_memory_usage
    ctx = profile_context.get()

    val_start = time.perf_counter()
    text = validate_text(payload.text, settings)
    fmt = validate_format(payload.format, settings)
    speed = validate_speed(payload.speed, settings)
    validation_ms = (time.perf_counter() - val_start) * 1000

    voice_val_start = time.perf_counter()
    voice = validate_voice(payload.voice, vm.voice_names())
    voice_validation_ms = (time.perf_counter() - voice_val_start) * 1000

    if ctx:
        ctx["timings"]["Validation"] = validation_ms
        ctx["timings"]["Voice Validation"] = voice_validation_ms
        ctx["metadata"]["characters"] = len(text)
        ctx["metadata"]["words"] = len(text.split())
        ctx["metadata"]["est_duration"] = estimate_duration_seconds(text, speed)

    cache_key = make_cache_key(text, voice, speed, fmt, payload.lang)

    if bypass_cache:
        cached = None
    else:
        cached = await cache.get(cache_key, fmt)

    if cached is not None:
        logger.info("Cache hit voice=%s format=%s chars=%d", voice, fmt, len(text))
        if ctx:
            ctx["metadata"]["output_bytes"] = len(cached)
            ctx["metadata"]["duration"] = estimate_duration_seconds(text, speed)
        return cached, AudioProcessor.mime_type_for(fmt), True

    samples, sample_rate = await service.synthesize(text, voice, speed, payload.lang)

    if ctx:
        ctx["memory"]["after_inference"] = get_memory_usage()

    if payload.trim_silence:
        samples = AudioProcessor.trim_silence(samples)
    if payload.normalize_volume:
        samples = AudioProcessor.normalize_volume(samples)

    if len(samples) == 0:
        raise SynthesisError("Synthesis produced no audio samples.")

    audio_bytes, mime_type = AudioProcessor.encode(samples, sample_rate, fmt)

    if ctx:
        ctx["memory"]["after_encoding"] = get_memory_usage()
        ctx["metadata"]["output_bytes"] = len(audio_bytes)
        ctx["metadata"]["duration"] = AudioProcessor.duration_seconds(samples, sample_rate)

    await cache.set(cache_key, fmt, audio_bytes)

    logger.info(
        "Synthesized voice=%s format=%s chars=%d duration=%.2fs",
        voice, fmt, len(text), AudioProcessor.duration_seconds(samples, sample_rate),
    )
    return audio_bytes, mime_type, False


@router.post("/v1/audio/speech")
async def generate_speech(
    request: Request,
    payload: SpeechRequest,
    service: KokoroTTSService = Depends(get_tts_service),
    cache: AudioCache = Depends(get_cache_dependency),
    settings: Settings = Depends(get_settings),
):
    """Generate speech audio from text. Streams the result; nothing is written
    to a temporary file on disk."""
    counter = get_active_requests_counter()
    counter["count"] += 1
    
    bypass_cache = request.headers.get("x-bypass-cache") == "true"
    
    from app.core.profiler import profile_context
    ctx = profile_context.get()
    
    import cProfile
    import os
    from datetime import datetime
    
    try:
        if settings.PROFILE_ENABLED and settings.PROFILE_DEEP:
            os.makedirs("profiles", exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            prof_path = f"profiles/profile_{timestamp}.prof"
            
            pr = cProfile.Profile()
            pr.enable()
            try:
                audio_bytes, mime_type, cache_hit = await _generate_audio_bytes(
                    payload, service, cache, settings, bypass_cache=bypass_cache
                )
            finally:
                pr.disable()
                pr.dump_stats(prof_path)
        else:
            audio_bytes, mime_type, cache_hit = await _generate_audio_bytes(
                payload, service, cache, settings, bypass_cache=bypass_cache
            )
    finally:
        counter["count"] -= 1

    resp_build_start = time.perf_counter()
    headers = {
        "X-Cache-Status": "HIT" if cache_hit else "MISS",
        "Content-Disposition": f'inline; filename="speech.{payload.format}"',
    }

    if settings.PROFILE_ENABLED and ctx:
        headers["X-Inference-Time-Ms"] = f"{ctx['timings'].get('Inference', 0.0):.2f}"
        headers["X-Encoding-Time-Ms"] = f"{ctx['timings'].get('Encoding', 0.0):.2f}"
        headers["X-Validation-Time-Ms"] = f"{ctx['timings'].get('Validation', 0.0):.2f}"

    async def _iter_bytes():
        first_byte_start = time.perf_counter()
        yield audio_bytes
        first_byte_ready_ms = (time.perf_counter() - first_byte_start) * 1000
        
        if settings.PROFILE_ENABLED and ctx:
            ctx["timings"]["Streaming"] = first_byte_ready_ms
            ctx["metadata"]["total_bytes_streamed"] = len(audio_bytes)

    response = StreamingResponse(_iter_bytes(), media_type=mime_type, headers=headers)
    
    resp_build_ms = (time.perf_counter() - resp_build_start) * 1000
    if ctx:
        ctx["timings"]["Response Build"] = resp_build_ms
        
    return response


@router.post("/v1/audio/preview")
async def preview_voice(
    payload: VoicePreviewRequest,
    service: KokoroTTSService = Depends(get_tts_service),
    cache: AudioCache = Depends(get_cache_dependency),
    settings: Settings = Depends(get_settings),
):
    """Generate a short sample clip for a given voice."""
    resp_build_start = time.perf_counter()
    sample_text = "Hello! This is a preview of my voice."
    speech_payload = SpeechRequest(text=sample_text, voice=payload.voice, speed=1.0, format=payload.format)
    audio_bytes, mime_type, _ = await _generate_audio_bytes(speech_payload, service, cache, settings)
    
    response = Response(content=audio_bytes, media_type=mime_type)
    
    resp_build_ms = (time.perf_counter() - resp_build_start) * 1000
    from app.core.profiler import profile_context
    ctx = profile_context.get()
    if ctx:
        ctx["timings"]["Response Build"] = resp_build_ms
    return response


@router.post("/benchmark")
async def run_benchmark(
    request: Request,
    settings: Settings = Depends(get_settings),
):
    """Automatically run benchmarks across multiple text lengths and report metrics."""
    benchmark_cases = [
        {"words": 1, "text": "Hello."},
        {"words": 5, "text": "Hello, how are you today?"},
        {"words": 20, "text": "This is a benchmark test to measure the performance and latency of the text to speech engine using kokoro."},
        {"words": 50, "text": "Kokoro is a lightweight text to speech model that runs extremely fast on cpu architectures and this benchmark will evaluate how well it scales with longer text sequences by measuring inference encoding and total request times over various character counts to help identify bottleneck patterns."},
        {"words": 100, "text": "Text to speech technology has advanced significantly allowing for natural sounding voices to be generated locally on consumer hardware without the need for cloud APIs or internet connections which preserves user privacy and reduces latency. This project uses the kokoro model running on onnx runtime which is optimized for cpu execution and this test evaluates the throughput of the system under different lengths of text input to verify scaling characteristics and CPU efficiency. We expect to see linear scaling in inference time with respect to the character count although setup overhead may dominate for very short words."},
        {"words": 250, "text": "Artificial intelligence and machine learning have revolutionized the field of speech synthesis over the past decade leading to the development of highly realistic neural text to speech models that can run efficiently on edge devices. Kokoro is one such model built to provide high quality audio generation while keeping the compute footprint minimal. By deploying it using onnx runtime we leverage hardware acceleration and optimizations that are native to the host machine. In this benchmark we are measuring the response times across a wide range of input lengths to understand where the system spends most of its time. The process involves multiple steps including input text cleaning phonemization feature extraction model inference audio post processing such as silence trimming and volume normalization and finally audio container encoding into formats like WAV MP3 or PCM. For each of these steps we track the exact duration to identify latency spikes and compute metrics like characters per second and words per second. This data is critical for performance engineering and optimization. By analyzing the scaling factor we can determine if there are non linear complexities in the phonemization or ONNX execution phase. Generally speech synthesis scales linearly with the number of generated audio frames which is proportional to the number of input phonemes. This benchmark automates the execution of these test cases and returns a structured performance report for analysis."}
    ]

    results = []
    
    from fastapi.testclient import TestClient
    from app.main import app
    
    with TestClient(app) as client:
        for case in benchmark_cases:
            payload = {
                "text": case["text"],
                "voice": settings.DEFAULT_VOICE,
                "speed": 1.0,
                "format": "wav",
                "normalize_volume": False,
                "trim_silence": False
            }
            
            http_start = time.perf_counter()
            response = client.post(
                "/v1/audio/speech",
                json=payload,
                headers={"X-Bypass-Cache": "true"}
            )
            http_duration_ms = (time.perf_counter() - http_start) * 1000
            
            if response.status_code != 200:
                raise Exception(f"Benchmark failed on case {case['words']} words: {response.text}")
                
            inference_ms = float(response.headers.get("X-Inference-Time-Ms", "0.0"))
            encoding_ms = float(response.headers.get("X-Encoding-Time-Ms", "0.0"))
            
            inf_seconds = inference_ms / 1000.0
            char_count = len(case["text"])
            word_count = case["words"]
            
            chars_per_sec = char_count / inf_seconds if inf_seconds > 0 else 0.0
            words_per_sec = word_count / inf_seconds if inf_seconds > 0 else 0.0
            
            results.append({
                "word_count": word_count,
                "character_count": char_count,
                "inference_time_ms": round(inference_ms, 2),
                "encoding_time_ms": round(encoding_ms, 2),
                "http_time_ms": round(http_duration_ms, 2),
                "characters_per_sec": round(chars_per_sec, 2),
                "words_per_sec": round(words_per_sec, 2)
            })

    return {"benchmark_results": results}