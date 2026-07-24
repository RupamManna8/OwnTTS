"""
Audio post-processing: format encoding, speed adjustment, volume
normalization and silence trimming.

All operations work on in-memory numpy arrays / byte buffers -- nothing
is ever written to a temporary file, per the streaming requirement.
"""
import io
import subprocess
import time
from typing import Tuple

import numpy as np
import soundfile as sf

from app.core.exception import AudioProcessingError
from app.core.logger import get_logger

logger = get_logger("audio_processor")

_SILENCE_THRESHOLD = 0.01  # amplitude below this is considered silence
_MIME_TYPES = {
    "wav": "audio/wav",
    "mp3": "audio/mpeg",
    "pcm": "audio/L16",
}


class AudioProcessor:
    """Stateless audio transformation utilities operating on raw PCM samples."""

    @staticmethod
    def mime_type_for(fmt: str) -> str:
        return _MIME_TYPES.get(fmt, "application/octet-stream")

    @staticmethod
    def change_speed(samples: np.ndarray, sample_rate: int, speed: float) -> np.ndarray:
        """Time-stretch audio by resampling (changes pitch slightly, like a
        tape speed change). Kokoro also accepts a native `speed` parameter;
        this is used as a safety-net for any additional client-side tuning."""
        if speed == 1.0:
            return samples

        try:
            original_length = len(samples)
            new_length = int(original_length / speed)
            if new_length <= 0:
                raise AudioProcessingError("Resulting audio length is zero after speed change.")

            original_indices = np.arange(original_length)
            new_indices = np.linspace(0, original_length - 1, new_length)
            resampled = np.interp(new_indices, original_indices, samples)
            return resampled.astype(samples.dtype)
        except Exception as exc:  # noqa: BLE001
            raise AudioProcessingError(f"Failed to change audio speed: {exc}") from exc

    @staticmethod
    def normalize_volume(samples: np.ndarray, target_peak: float = 0.95) -> np.ndarray:
        start_time = time.perf_counter()
        try:
            peak = np.max(np.abs(samples))
            if peak == 0:
                res = samples
            else:
                gain = target_peak / peak
                res = np.clip(samples * gain, -1.0, 1.0).astype(samples.dtype)
            
            from app.core.config import get_settings
            settings = get_settings()
            if settings.PROFILE_ENABLED:
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                print(f"normalize_volume()  : {elapsed_ms:.2f} ms")
                from app.core.profiler import profile_context
                ctx = profile_context.get()
                if ctx:
                    ctx["timings"]["Normalize Volume"] = elapsed_ms
            return res
        except Exception as exc:  # noqa: BLE001
            raise AudioProcessingError(f"Failed to normalize volume: {exc}") from exc

    @staticmethod
    def trim_silence(samples: np.ndarray, threshold: float = _SILENCE_THRESHOLD) -> np.ndarray:
        start_time = time.perf_counter()
        try:
            mask = np.abs(samples) > threshold
            if not np.any(mask):
                res = samples
            else:
                start = int(np.argmax(mask))
                end = int(len(mask) - np.argmax(mask[::-1]))
                res = samples[start:end]
            
            from app.core.config import get_settings
            settings = get_settings()
            if settings.PROFILE_ENABLED:
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                print(f"trim_silence()      : {elapsed_ms:.2f} ms")
                from app.core.profiler import profile_context
                ctx = profile_context.get()
                if ctx:
                    ctx["timings"]["Trim Silence"] = elapsed_ms
            return res
        except Exception as exc:  # noqa: BLE001
            raise AudioProcessingError(f"Failed to trim silence: {exc}") from exc

    @staticmethod
    def to_wav_bytes(samples: np.ndarray, sample_rate: int) -> bytes:
        start_time = time.perf_counter()
        buffer = io.BytesIO()
        try:
            sf.write(buffer, samples, sample_rate, format="WAV", subtype="PCM_16")
            res_bytes = buffer.getvalue()
            
            from app.core.config import get_settings
            settings = get_settings()
            if settings.PROFILE_ENABLED:
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                print(f"to_wav_bytes()      : {elapsed_ms:.2f} ms")
                print(f"WAV encoding        : {elapsed_ms:.2f} ms")
                from app.core.profiler import profile_context
                ctx = profile_context.get()
                if ctx:
                    ctx["timings"]["Encoding"] = elapsed_ms
            return res_bytes
        except Exception as exc:  # noqa: BLE001
            raise AudioProcessingError(f"Failed to encode WAV: {exc}") from exc

    @staticmethod
    def to_pcm_bytes(samples: np.ndarray) -> bytes:
        start_time = time.perf_counter()
        try:
            clipped = np.clip(samples, -1.0, 1.0)
            int16_samples = (clipped * 32767.0).astype(np.int16)
            res_bytes = int16_samples.tobytes()
            
            from app.core.config import get_settings
            settings = get_settings()
            if settings.PROFILE_ENABLED:
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                print(f"to_pcm_bytes()      : {elapsed_ms:.2f} ms")
                print(f"PCM encoding        : {elapsed_ms:.2f} ms")
                from app.core.profiler import profile_context
                ctx = profile_context.get()
                if ctx:
                    ctx["timings"]["Encoding"] = elapsed_ms
            return res_bytes
        except Exception as exc:  # noqa: BLE001
            raise AudioProcessingError(f"Failed to encode PCM: {exc}") from exc

    @staticmethod
    def to_mp3_bytes(samples: np.ndarray, sample_rate: int, bitrate: str = "192k") -> bytes:
        """Encode via ffmpeg (piped, no temp files). Requires ffmpeg on PATH."""
        start_time = time.perf_counter()
        wav_bytes = AudioProcessor.to_wav_bytes(samples, sample_rate)
        try:
            process = subprocess.run(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel", "error",
                    "-f", "wav",
                    "-i", "pipe:0",
                    "-b:a", bitrate,
                    "-f", "mp3",
                    "pipe:1",
                ],
                input=wav_bytes,
                capture_output=True,
                check=True,
                timeout=30,
            )
            res_bytes = process.stdout
            
            from app.core.config import get_settings
            settings = get_settings()
            if settings.PROFILE_ENABLED:
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                print(f"to_mp3_bytes()      : {elapsed_ms:.2f} ms")
                print(f"MP3 encoding        : {elapsed_ms:.2f} ms")
                from app.core.profiler import profile_context
                ctx = profile_context.get()
                if ctx:
                    ctx["timings"]["Encoding"] = elapsed_ms
            return res_bytes
        except FileNotFoundError as exc:
            raise AudioProcessingError(
                "ffmpeg binary not found. Install ffmpeg to enable MP3 output."
            ) from exc
        except subprocess.CalledProcessError as exc:
            raise AudioProcessingError(
                f"ffmpeg failed to encode MP3: {exc.stderr.decode(errors='ignore')}"
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise AudioProcessingError("ffmpeg timed out while encoding MP3.") from exc

    @classmethod
    def encode(cls, samples: np.ndarray, sample_rate: int, fmt: str) -> Tuple[bytes, str]:
        """Encode raw float32 samples into the requested container format.

        Returns (audio_bytes, mime_type).
        """
        fmt = fmt.lower()
        if fmt == "wav":
            return cls.to_wav_bytes(samples, sample_rate), cls.mime_type_for(fmt)
        if fmt == "pcm":
            return cls.to_pcm_bytes(samples), cls.mime_type_for(fmt)
        if fmt == "mp3":
            return cls.to_mp3_bytes(samples, sample_rate), cls.mime_type_for(fmt)
        raise AudioProcessingError(f"Unsupported audio format: {fmt}")

    @staticmethod
    def duration_seconds(samples: np.ndarray, sample_rate: int) -> float:
        return round(len(samples) / float(sample_rate), 3)