# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
K-Moshi Speaker Conditioning Modules

This package provides speaker conditioning capabilities for K-Moshi,
enabling zero-shot speaker adaptation for the Korean Moshi model.

Modules:
    - speaker_encoder: Speaker embedding extraction
        - ECAPATDNNSpeakerEncoder: SpeechBrain ECAPA-TDNN (192-dim)
        - W2vBERT2SpeakerEncoder: W2v-BERT 2.0 SV (256-dim, SOTA)
        - DummySpeakerEncoder: Testing without dependencies
    - speaker_conditioner: Integration with Temporal Transformer via sum_condition
    - audio_prompt: VALL-E style audio/text prompting

Speaker Encoder Options:
    1. ECAPA-TDNN (Default):
       - 192-dimensional embedding
       - SpeechBrain pre-trained on VoxCeleb
       - Requires: pip install speechbrain

    2. W2v-BERT 2.0 SV (Recommended for best quality):
       - 256-dimensional embedding
       - 0.14% EER on VoxCeleb1-O (SOTA)
       - Requires: pip install transformers
       - Model: https://huggingface.co/zl389/w2v-bert-2.0_SV/blob/main/model_lmft_0.14.pth
       - Paper: https://arxiv.org/abs/2510.04213

Audio Prompting Modes:
    1. speaker_embedding (Default): Global speaker embedding only
    2. audio_only: VALL-E style audio codes prepended as prompt
    3. audio_text: Both audio codes and text tokens as prompt (PersonaPlex style)
"""

from .speaker_encoder import (
    BaseSpeakerEncoder,
    ECAPATDNNSpeakerEncoder,
    W2vBERT2SpeakerEncoder,
    DummySpeakerEncoder,
    SpeakerEncoderConfig,
    create_speaker_encoder,
    get_default_speaker_encoder,
)
from .speaker_conditioner import (
    SpeakerConditioner,
    SpeakerConditionerConfig,
    SpeakerConditioningModule,
    ReferenceSampler,
)
from .audio_prompt import (
    AudioPromptConfig,
    AudioPromptSample,
    AudioPromptSampler,
    AudioPromptEncoder,
    AudioPromptModule,
    create_audio_prompt_module,
    get_default_audio_prompt_config,
)

__all__ = [
    # Speaker Encoders
    "BaseSpeakerEncoder",
    "ECAPATDNNSpeakerEncoder",
    "W2vBERT2SpeakerEncoder",
    "DummySpeakerEncoder",
    "SpeakerEncoderConfig",
    "create_speaker_encoder",
    "get_default_speaker_encoder",
    # Speaker Conditioner
    "SpeakerConditioner",
    "SpeakerConditionerConfig",
    "SpeakerConditioningModule",
    "ReferenceSampler",
    # Audio Prompting (VALL-E style)
    "AudioPromptConfig",
    "AudioPromptSample",
    "AudioPromptSampler",
    "AudioPromptEncoder",
    "AudioPromptModule",
    "create_audio_prompt_module",
    "get_default_audio_prompt_config",
]
