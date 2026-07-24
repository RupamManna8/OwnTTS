"""
Voice discovery and metadata management for Kokoro voices.

Kokoro voice identifiers encode gender and language via a naming
convention, e.g.:
    af_heart  -> a=American English, f=female, "heart" style
    bm_george -> b=British English,  m=male,  "george" style

This module parses that convention and exposes a clean lookup API so the
rest of the app never has to know about the raw naming scheme.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional

from app.core.exception import UnsupportedVoiceError
from app.core.logger import get_logger

logger = get_logger("voice_manager")

_LANG_CODE_MAP: Dict[str, str] = {
    "a": "en-us",
    "b": "en-gb",
    "j": "ja",
    "z": "zh",
    "e": "es",
    "f": "fr",
    "h": "hi",
    "i": "it",
    "p": "pt-br",
}

_LANG_NAME_MAP: Dict[str, str] = {
    "en-us": "English (US)",
    "en-gb": "English (UK)",
    "ja": "Japanese",
    "zh": "Chinese",
    "es": "Spanish",
    "fr": "French",
    "hi": "Hindi",
    "it": "Italian",
    "pt-br": "Portuguese (Brazil)",
}

DEFAULT_SAMPLE_RATE = 24000


@dataclass(frozen=True)
class VoiceInfo:
    name: str
    gender: str
    language: str
    language_code: str
    sample_rate: int = DEFAULT_SAMPLE_RATE


class VoiceManager:
    """Discovers and serves metadata for installed Kokoro voices."""

    def __init__(self) -> None:
        self._voices: Dict[str, VoiceInfo] = {}

    def load_from_names(self, voice_names: List[str]) -> None:
        """Populate the registry from the list of voice names bundled in the model."""
        voices: Dict[str, VoiceInfo] = {}
        for raw_name in voice_names:
            info = self._parse_voice_name(raw_name)
            voices[raw_name] = info
        self._voices = voices
        logger.info("Loaded %d voices across %d languages", len(voices), len(self.languages()))

    @staticmethod
    def _parse_voice_name(raw_name: str) -> VoiceInfo:
        prefix = raw_name.split("_", 1)[0] if "_" in raw_name else raw_name[:2]
        lang_char = prefix[0] if prefix else "a"
        gender_char = prefix[1] if len(prefix) > 1 else "f"

        language_code = _LANG_CODE_MAP.get(lang_char, "en-us")
        language = _LANG_NAME_MAP.get(language_code, "Unknown")
        gender = "female" if gender_char == "f" else "male" if gender_char == "m" else "unknown"

        return VoiceInfo(
            name=raw_name,
            gender=gender,
            language=language,
            language_code=language_code,
            sample_rate=DEFAULT_SAMPLE_RATE,
        )

    def all_voices(self) -> List[VoiceInfo]:
        return list(self._voices.values())

    def voice_names(self) -> List[str]:
        return list(self._voices.keys())

    def languages(self) -> List[str]:
        return sorted({v.language for v in self._voices.values()})

    def get(self, name: str) -> VoiceInfo:
        info = self._voices.get(name)
        if info is None:
            raise UnsupportedVoiceError(
                f"Voice '{name}' is not available.",
                details={"available_voices": sorted(self._voices.keys())},
            )
        return info

    def exists(self, name: str) -> bool:
        return name in self._voices

    def grouped_by_language(self) -> Dict[str, List[VoiceInfo]]:
        grouped: Dict[str, List[VoiceInfo]] = {}
        for voice in self._voices.values():
            grouped.setdefault(voice.language, []).append(voice)
        for voices in grouped.values():
            voices.sort(key=lambda v: v.name)
        return grouped

    def is_loaded(self) -> bool:
        return len(self._voices) > 0