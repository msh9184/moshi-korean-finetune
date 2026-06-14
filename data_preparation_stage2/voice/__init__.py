# Copyright (c) 2026 Sunghwan Mun. Licensed under the Apache License, Version 2.0.
"""Voice Management Module for K-Moshi Stage 2.

This module handles voice preparation and management:
- Reference audio validation and preparation
- Voice profile management (Moshi + multiple users)
- Audio quality checks (SNR, duration, clipping)

Usage:
    from data_preparation_stage2.voice import (
        validate_reference_audio,
        prepare_reference_audio,
        VoiceManager,
    )

    # Validate reference audio
    result = validate_reference_audio("moshi_reference.wav")

    # Prepare for voice cloning
    prepare_reference_audio("raw.wav", "processed.wav")
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .reference_preparation import (
        validate_reference_audio,
        prepare_reference_audio,
        ReferenceAudioConfig,
    )
    from .voice_manager import VoiceManager, VoiceProfile

__all__ = [
    "validate_reference_audio",
    "prepare_reference_audio",
    "ReferenceAudioConfig",
    "VoiceManager",
    "VoiceProfile",
]
