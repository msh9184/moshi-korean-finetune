# Copyright (c) 2026 Sunghwan Mun. Licensed under the Apache License, Version 2.0.
"""Corpus Processing Module for K-Moshi Stage 2.

This module handles text corpus loading, processing, and mixing:
- Korean corpus readers (AI Hub, 모두의말뭉치)
- English corpus integration (DailyDialog, EmpatheticDialogues)
- Spoken style conversion (written → spoken)
- Language mixing for bilingual capability

Usage:
    from data_preparation_stage2.corpus import (
        EnglishMixer,
        RuleBasedConverter,
        CorpusLoader,
    )

    # Mix English into Korean corpus
    mixer = EnglishMixer(english_ratio=0.1)
    mixed = mixer.mix_corpus(korean_dialogues, english_dialogues)

    # Convert written to spoken style
    converter = RuleBasedConverter()
    spoken = converter.convert("무엇을 도와드릴까요?")
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .english_mixer import EnglishMixer
    from .spoken_converter import RuleBasedConverter, LocalLLMConverter
    from .corpus_loader import CorpusLoader

__all__ = [
    "EnglishMixer",
    "RuleBasedConverter",
    "LocalLLMConverter",
    "CorpusLoader",
]
