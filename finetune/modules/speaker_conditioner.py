# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
Speaker Conditioner Module for K-Moshi Zero-Shot Speaker Conditioning

This module provides the bridge between speaker encoder outputs and
the Temporal Transformer's sum_condition mechanism.

Architecture Overview:
    Speaker Embedding (D_spk) → Projection → Scale → sum_condition (D_model)

    Where:
        D_spk = 192 (ECAPA-TDNN output)
        D_model = 4096 (Temporal Transformer hidden dim)

Key Components:
    - Learnable projection layer: Linear(D_spk, D_model)
    - Learnable scale parameter: Initialized to 0.1 for stable training
    - Optional LayerNorm for embedding normalization

Integration Point:
    The output is added to the Temporal Transformer input via sum_condition:
        input_ = text_emb[MOSHI] + Σaudio_emb[MOSHI] + speaker_condition
                                                       ↑ SpeakerConditioner output

Usage:
    config = SpeakerConditionerConfig(
        input_dim=192,    # Speaker encoder output
        output_dim=4096,  # Temporal TF hidden dim
        initial_scale=0.1,
        use_layernorm=True,
    )
    conditioner = SpeakerConditioner(config)

    # During forward pass
    spk_emb = speaker_encoder(reference_audio)  # [B, 192]
    spk_cond = conditioner(spk_emb)             # [B, 1, 4096]

    # In LMModel.forward_text()
    input_ = text_emb + audio_emb + spk_cond.to(input_)
"""

from dataclasses import dataclass
from typing import Optional
import logging
import math

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class SpeakerConditionerConfig:
    """Configuration for Speaker Conditioner.

    Attributes:
        input_dim: Input dimension from speaker encoder (192 for ECAPA-TDNN)
        output_dim: Output dimension matching Temporal TF hidden dim (4096)
        initial_scale: Initial value for learnable scale parameter
        use_layernorm: Whether to apply LayerNorm to projected embedding
        dropout: Dropout probability after projection
        learnable_scale: Whether scale parameter is learnable
        scale_mode: How to apply scale ("multiply" or "gated")
    """
    input_dim: int = 192
    output_dim: int = 4096
    initial_scale: float = 0.1
    use_layernorm: bool = True
    dropout: float = 0.0
    learnable_scale: bool = True
    scale_mode: str = "multiply"  # "multiply" or "gated"


class SpeakerConditioner(nn.Module):
    """Speaker Conditioner for Temporal Transformer integration.

    This module transforms speaker embeddings from the speaker encoder
    into the format expected by the Temporal Transformer's sum_condition.

    The transformation consists of:
        1. Linear projection: D_spk → D_model
        2. Optional LayerNorm
        3. Optional Dropout
        4. Learnable scaling

    The learnable scale is initialized small (0.1) to ensure stable training
    at the beginning, allowing the model to gradually learn how much speaker
    information to incorporate.

    Shape Flow:
        Input:  [B, D_spk]     e.g., [B, 192]
        Output: [B, 1, D_model] e.g., [B, 1, 4096]

    The output dimension includes an extra time dimension (1) to broadcast
    across all time steps when added to the Temporal TF input.
    """

    def __init__(self, config: SpeakerConditionerConfig):
        super().__init__()
        self.config = config

        # Projection layer: D_spk → D_model
        self.projection = nn.Linear(config.input_dim, config.output_dim)

        # Optional LayerNorm
        self.layernorm = nn.LayerNorm(config.output_dim) if config.use_layernorm else None

        # Optional Dropout
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else None

        # Learnable scale parameter
        if config.scale_mode == "gated":
            # Gated scaling with sigmoid
            self.scale_gate = nn.Linear(config.output_dim, config.output_dim)
            self.scale = None
        else:
            # Simple scalar multiplication
            self.scale = nn.Parameter(
                torch.tensor(config.initial_scale),
                requires_grad=config.learnable_scale,
            )
            self.scale_gate = None

        # Initialize weights
        self._init_weights()

        logger.info(
            f"SpeakerConditioner initialized: {config.input_dim} → {config.output_dim}, "
            f"scale={config.initial_scale}, mode={config.scale_mode}"
        )

    def _init_weights(self) -> None:
        """Initialize projection weights for stable training."""
        # Xavier uniform initialization for projection
        nn.init.xavier_uniform_(self.projection.weight)
        if self.projection.bias is not None:
            nn.init.zeros_(self.projection.bias)

        # Gated scale initialization
        if self.scale_gate is not None:
            # Initialize gate to output ~0.1 after sigmoid
            # sigmoid(-2.2) ≈ 0.1
            nn.init.zeros_(self.scale_gate.weight)
            nn.init.constant_(self.scale_gate.bias, -2.2)

    def forward(self, speaker_embedding: torch.Tensor) -> torch.Tensor:
        """Transform speaker embedding to sum_condition format.

        Args:
            speaker_embedding: Speaker embedding [B, D_spk]

        Returns:
            Conditioned embedding [B, 1, D_model] ready for sum_condition
        """
        # Ensure input is 2D
        if speaker_embedding.dim() == 1:
            speaker_embedding = speaker_embedding.unsqueeze(0)

        # Cast to projection layer's dtype (handles float32 → bfloat16 conversion)
        # This is necessary because speaker_encoder runs outside FSDP (float32)
        # while SpeakerConditioner is inside FSDP (bfloat16)
        target_dtype = self.projection.weight.dtype
        if speaker_embedding.dtype != target_dtype:
            speaker_embedding = speaker_embedding.to(dtype=target_dtype)

        # Project to model dimension
        x = self.projection(speaker_embedding)  # [B, D_model]

        # Apply LayerNorm if configured
        if self.layernorm is not None:
            x = self.layernorm(x)

        # Apply Dropout if configured
        if self.dropout is not None:
            x = self.dropout(x)

        # Apply scaling
        if self.config.scale_mode == "gated" and self.scale_gate is not None:
            # Gated scaling: element-wise scaling via sigmoid
            gate = torch.sigmoid(self.scale_gate(x))
            x = x * gate
        elif self.scale is not None:
            # Simple scalar multiplication
            x = x * self.scale

        # Add time dimension for broadcasting: [B, D_model] → [B, 1, D_model]
        x = x.unsqueeze(1)

        return x

    @property
    def scale_value(self) -> float:
        """Return current scale value for logging."""
        if self.scale is not None:
            return self.scale.item()
        return float("nan")  # Gated mode doesn't have single scale

    def get_stats(self) -> dict:
        """Return statistics for monitoring."""
        stats = {
            "speaker_conditioner/scale": self.scale_value,
        }

        if self.layernorm is not None:
            stats["speaker_conditioner/layernorm_weight_mean"] = self.layernorm.weight.mean().item()

        stats["speaker_conditioner/projection_weight_norm"] = self.projection.weight.norm().item()

        return stats


class SpeakerConditioningModule(nn.Module):
    """Complete speaker conditioning module combining encoder and conditioner.

    This is a convenience wrapper that combines:
        - Speaker Encoder (e.g., ECAPA-TDNN)
        - Speaker Conditioner (projection + scaling)

    It provides a single interface for extracting speaker-conditioned embeddings
    from raw reference audio.

    Usage:
        module = SpeakerConditioningModule(encoder_config, conditioner_config)
        sum_condition = module(reference_audio)  # [B, 1, D_model]
    """

    def __init__(
        self,
        encoder_config: "SpeakerEncoderConfig",  # Forward reference
        conditioner_config: SpeakerConditionerConfig,
    ):
        super().__init__()

        # Import here to avoid circular dependency
        from .speaker_encoder import create_speaker_encoder

        self.encoder = create_speaker_encoder(encoder_config)
        self.conditioner = SpeakerConditioner(conditioner_config)

        # Validate dimensions
        if encoder_config.output_dim != conditioner_config.input_dim:
            logger.warning(
                f"Dimension mismatch: encoder output ({encoder_config.output_dim}) "
                f"!= conditioner input ({conditioner_config.input_dim})"
            )

    def forward(
        self,
        reference_audio: torch.Tensor,
        audio_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Extract speaker-conditioned embedding from reference audio.

        Args:
            reference_audio: Reference audio [B, T] at encoder's sample rate
            audio_lengths: Optional actual lengths [B] for padded batches

        Returns:
            sum_condition tensor [B, 1, D_model]
        """
        # Extract speaker embedding
        speaker_embedding = self.encoder(reference_audio, audio_lengths)  # [B, D_spk]

        # Transform to sum_condition format
        sum_condition = self.conditioner(speaker_embedding)  # [B, 1, D_model]

        return sum_condition

    def get_stats(self) -> dict:
        """Return combined statistics."""
        return self.conditioner.get_stats()


class ReferenceSampler:
    """Utility class for sampling reference audio during training.

    During training, we need to sample a random segment from the MOSHI
    audio stream to use as speaker reference. This class handles:
        - Random segment selection
        - Minimum/maximum duration constraints
        - Avoiding overlap with current training segment

    Reference: ZERO_SHOT_SPEAKER_CONDITIONING_SPECIFICATION.md Section 5

    Usage:
        sampler = ReferenceSampler(
            min_duration_sec=3.0,
            max_duration_sec=10.0,
            sample_rate=24000,  # Mimi's rate
            target_sample_rate=16000,  # Speaker encoder's rate
        )
        ref_audio, ref_text = sampler.sample_from_moshi_stream(
            moshi_audio, moshi_text, current_start, current_end
        )
    """

    def __init__(
        self,
        min_duration_sec: float = 3.0,
        max_duration_sec: float = 10.0,
        sample_rate: int = 24000,
        target_sample_rate: int = 16000,
    ):
        self.min_duration_sec = min_duration_sec
        self.max_duration_sec = max_duration_sec
        self.sample_rate = sample_rate
        self.target_sample_rate = target_sample_rate

        self.min_samples = int(min_duration_sec * sample_rate)
        self.max_samples = int(max_duration_sec * sample_rate)

    def sample_from_moshi_stream(
        self,
        moshi_audio: torch.Tensor,
        moshi_text_tokens: Optional[torch.Tensor] = None,
        exclude_start: Optional[int] = None,
        exclude_end: Optional[int] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Sample a random reference segment from MOSHI audio stream.

        Args:
            moshi_audio: Full MOSHI audio stream [T] or [C, T]
            moshi_text_tokens: Optional corresponding text tokens [T]
            exclude_start: Start index of current training segment to avoid
            exclude_end: End index of current training segment to avoid

        Returns:
            Tuple of (reference_audio, reference_text_tokens)
            reference_audio is resampled to target_sample_rate
        """
        import torchaudio.functional as F

        # Handle channel dimension
        if moshi_audio.dim() == 2:
            moshi_audio = moshi_audio[0]  # Use first channel

        total_samples = moshi_audio.shape[-1]

        # Determine valid sampling regions
        valid_regions = self._get_valid_regions(
            total_samples, exclude_start, exclude_end
        )

        if not valid_regions:
            # Fallback: use entire audio
            logger.warning("No valid regions for reference sampling, using full audio")
            valid_regions = [(0, total_samples)]

        # Sample duration
        duration_samples = torch.randint(
            self.min_samples,
            min(self.max_samples, total_samples) + 1,
            (1,),
        ).item()

        # Sample start position from valid regions
        start_pos = self._sample_from_regions(valid_regions, duration_samples)

        # Extract reference segment
        end_pos = min(start_pos + duration_samples, total_samples)
        ref_audio = moshi_audio[start_pos:end_pos]

        # Resample to target rate if needed
        if self.sample_rate != self.target_sample_rate:
            ref_audio = F.resample(
                ref_audio.unsqueeze(0),
                self.sample_rate,
                self.target_sample_rate,
            ).squeeze(0)

        # Extract corresponding text tokens if available
        ref_text = None
        if moshi_text_tokens is not None:
            # Convert audio samples to token frames (12.5Hz = 80ms per frame)
            frame_rate = 12.5
            start_frame = int(start_pos / self.sample_rate * frame_rate)
            end_frame = int(end_pos / self.sample_rate * frame_rate)
            ref_text = moshi_text_tokens[start_frame:end_frame]

        return ref_audio, ref_text

    def _get_valid_regions(
        self,
        total_samples: int,
        exclude_start: Optional[int],
        exclude_end: Optional[int],
    ) -> list[tuple[int, int]]:
        """Get valid sampling regions avoiding excluded segment."""
        if exclude_start is None or exclude_end is None:
            return [(0, total_samples)]

        regions = []

        # Region before excluded segment
        if exclude_start > self.min_samples:
            regions.append((0, exclude_start))

        # Region after excluded segment
        if total_samples - exclude_end > self.min_samples:
            regions.append((exclude_end, total_samples))

        return regions

    def _sample_from_regions(
        self,
        regions: list[tuple[int, int]],
        duration: int,
    ) -> int:
        """Sample a start position from valid regions."""
        # Calculate total valid length
        valid_lengths = [
            max(0, end - start - duration)
            for start, end in regions
        ]
        total_valid = sum(valid_lengths)

        if total_valid <= 0:
            # Fallback to first region
            return regions[0][0]

        # Sample position
        pos = torch.randint(0, total_valid + 1, (1,)).item()

        # Find which region this falls into
        cumsum = 0
        for (start, end), length in zip(regions, valid_lengths):
            if cumsum + length >= pos:
                return start + (pos - cumsum)
            cumsum += length

        return regions[-1][0]
