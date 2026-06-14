# Copyright (c) 2026 Sunghwan Mun. Licensed under the Apache License, Version 2.0.
"""TTS Engine Integration for K-Moshi Stage 2.

This module provides TTS engine integrations:
- OpenAudio S1 Mini: High-quality voice cloning TTS
- Supertonic-2: Fast ONNX-based Korean TTS

Usage:
    from data_preparation_stage2.tts import SupertonicTTS, TTSEngine

    # Direct Supertonic usage
    tts = SupertonicTTS(model_dir="/models/supertonic-2")
    audio = tts.synthesize("안녕하세요")

    # Unified engine interface
    engine = TTSEngine.from_config(config)
    audio = engine.synthesize(text, speaker="moshi")
"""

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .supertonic_wrapper import SupertonicTTS
    from .openaudio_wrapper import OpenAudioTTS
    from .tts_engine import TTSEngine

__all__ = [
    "SupertonicTTS",
    "OpenAudioTTS",
    "TTSEngine",
]
