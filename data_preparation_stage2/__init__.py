# Copyright (c) 2026 Sunghwan Mun. Licensed under the Apache License, Version 2.0.
# K-Moshi: Korean Full-Duplex Spoken Dialogue Model
"""
Data Preparation Stage 2: Synthetic Dialogue Generation

This module provides tools for generating synthetic Korean dialogue data
using external TTS systems for K-Moshi training.

Key Components:
- corpus: Text corpus readers and processors
- identity: K-Moshi identity Q&A system
- timing: Full-duplex dialogue timing control
- tts: TTS engine integrations (OpenAudio S1 Mini, Supertonic-2)
- voice: Voice management and selection
- synthesis: Dialogue synthesis engine
- quality: Quality assurance (WER filtering, audio validation)
- orchestrators: Pipeline orchestration

Usage:
    from data_preparation_stage2 import SynthesisPipeline

    pipeline = SynthesisPipeline.from_yaml("configs/default.yaml")
    pipeline.run()
"""

__version__ = "0.1.0"
__author__ = "K-Moshi Development Team"

from pathlib import Path

# Package root directory
PACKAGE_ROOT = Path(__file__).parent

# Default paths
DEFAULT_CONFIG_PATH = PACKAGE_ROOT / "configs" / "default.yaml"
DEFAULT_IDENTITY_DATA = PACKAGE_ROOT / "identity" / "data"
DEFAULT_VOICE_SAMPLES = PACKAGE_ROOT / "voice" / "samples"
