# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
Unit tests for K-Moshi Speaker Conditioning modules.

Tests cover:
    - SpeakerEncoderConfig validation
    - DummySpeakerEncoder functionality
    - SpeakerConditioner projection and scaling
    - SpeakerConditioningModule integration
    - ReferenceSampler segment extraction

Run with: pytest tests/test_speaker_conditioning.py -v
"""

import pytest
import torch
import torch.nn as nn


class TestSpeakerEncoderConfig:
    """Tests for SpeakerEncoderConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        from finetune.modules.speaker_encoder import SpeakerEncoderConfig

        config = SpeakerEncoderConfig()
        assert config.encoder_type == "ecapa_tdnn"
        assert config.pretrained_path == "speechbrain/spkrec-ecapa-voxceleb"
        assert config.freeze is True
        assert config.output_dim == 192
        assert config.sample_rate == 16000
        assert config.normalize_embedding is True

    def test_custom_config(self):
        """Test custom configuration."""
        from finetune.modules.speaker_encoder import SpeakerEncoderConfig

        config = SpeakerEncoderConfig(
            encoder_type="ecapa_tdnn",
            freeze=False,
            output_dim=256,
            normalize_embedding=False,
        )
        assert config.freeze is False
        assert config.output_dim == 256
        assert config.normalize_embedding is False


class TestDummySpeakerEncoder:
    """Tests for DummySpeakerEncoder (testing without SpeechBrain)."""

    def test_forward_single_sample(self):
        """Test forward pass with single sample."""
        from finetune.modules.speaker_encoder import DummySpeakerEncoder, SpeakerEncoderConfig

        config = SpeakerEncoderConfig(output_dim=192)
        encoder = DummySpeakerEncoder(config)

        # Single audio sample
        audio = torch.randn(16000)  # 1 second at 16kHz
        embedding = encoder(audio)

        assert embedding.shape == (1, 192)
        assert embedding.dtype == audio.dtype

    def test_forward_batch(self):
        """Test forward pass with batch."""
        from finetune.modules.speaker_encoder import DummySpeakerEncoder, SpeakerEncoderConfig

        config = SpeakerEncoderConfig(output_dim=192)
        encoder = DummySpeakerEncoder(config)

        # Batch of audio samples
        batch_size = 4
        audio = torch.randn(batch_size, 16000)
        embedding = encoder(audio)

        assert embedding.shape == (batch_size, 192)

    def test_normalization(self):
        """Test embedding L2 normalization."""
        from finetune.modules.speaker_encoder import DummySpeakerEncoder, SpeakerEncoderConfig

        # With normalization
        config_norm = SpeakerEncoderConfig(output_dim=192, normalize_embedding=True)
        encoder_norm = DummySpeakerEncoder(config_norm)

        audio = torch.randn(2, 16000)
        embedding = encoder_norm(audio)

        # Check L2 norm is approximately 1
        norms = torch.norm(embedding, p=2, dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_freeze_unfreeze(self):
        """Test freeze and unfreeze functionality."""
        from finetune.modules.speaker_encoder import DummySpeakerEncoder, SpeakerEncoderConfig

        config = SpeakerEncoderConfig(output_dim=192)
        encoder = DummySpeakerEncoder(config)

        # Initially not frozen
        assert all(p.requires_grad for p in encoder.parameters())

        # Freeze
        encoder.freeze()
        assert all(not p.requires_grad for p in encoder.parameters())

        # Unfreeze
        encoder.unfreeze()
        assert all(p.requires_grad for p in encoder.parameters())


class TestSpeakerConditioner:
    """Tests for SpeakerConditioner projection layer."""

    def test_basic_projection(self):
        """Test basic projection from speaker embedding to model dimension."""
        from finetune.modules.speaker_conditioner import SpeakerConditioner, SpeakerConditionerConfig

        config = SpeakerConditionerConfig(
            input_dim=192,
            output_dim=4096,
            initial_scale=0.1,
        )
        conditioner = SpeakerConditioner(config)

        # Speaker embedding
        batch_size = 2
        speaker_emb = torch.randn(batch_size, 192)

        # Forward
        output = conditioner(speaker_emb)

        # Check output shape: [B, 1, D_model]
        assert output.shape == (batch_size, 1, 4096)

    def test_scale_parameter(self):
        """Test learnable scale parameter."""
        from finetune.modules.speaker_conditioner import SpeakerConditioner, SpeakerConditionerConfig

        config = SpeakerConditionerConfig(
            input_dim=192,
            output_dim=4096,
            initial_scale=0.5,
            learnable_scale=True,
        )
        conditioner = SpeakerConditioner(config)

        # Check initial scale value
        assert conditioner.scale_value == pytest.approx(0.5, abs=1e-5)

        # Scale should be a parameter
        assert conditioner.scale.requires_grad is True

    def test_non_learnable_scale(self):
        """Test non-learnable scale parameter."""
        from finetune.modules.speaker_conditioner import SpeakerConditioner, SpeakerConditionerConfig

        config = SpeakerConditionerConfig(
            input_dim=192,
            output_dim=4096,
            initial_scale=0.1,
            learnable_scale=False,
        )
        conditioner = SpeakerConditioner(config)

        # Scale should not require gradients
        assert conditioner.scale.requires_grad is False

    def test_with_layernorm(self):
        """Test conditioner with LayerNorm."""
        from finetune.modules.speaker_conditioner import SpeakerConditioner, SpeakerConditionerConfig

        config = SpeakerConditionerConfig(
            input_dim=192,
            output_dim=4096,
            use_layernorm=True,
        )
        conditioner = SpeakerConditioner(config)

        assert conditioner.layernorm is not None

        speaker_emb = torch.randn(2, 192)
        output = conditioner(speaker_emb)

        assert output.shape == (2, 1, 4096)

    def test_without_layernorm(self):
        """Test conditioner without LayerNorm."""
        from finetune.modules.speaker_conditioner import SpeakerConditioner, SpeakerConditionerConfig

        config = SpeakerConditionerConfig(
            input_dim=192,
            output_dim=4096,
            use_layernorm=False,
        )
        conditioner = SpeakerConditioner(config)

        assert conditioner.layernorm is None

    def test_gated_scale_mode(self):
        """Test gated scaling mode."""
        from finetune.modules.speaker_conditioner import SpeakerConditioner, SpeakerConditionerConfig

        config = SpeakerConditionerConfig(
            input_dim=192,
            output_dim=4096,
            scale_mode="gated",
        )
        conditioner = SpeakerConditioner(config)

        assert conditioner.scale_gate is not None
        assert conditioner.scale is None

        speaker_emb = torch.randn(2, 192)
        output = conditioner(speaker_emb)

        assert output.shape == (2, 1, 4096)

    def test_get_stats(self):
        """Test statistics retrieval."""
        from finetune.modules.speaker_conditioner import SpeakerConditioner, SpeakerConditionerConfig

        config = SpeakerConditionerConfig(
            input_dim=192,
            output_dim=4096,
            initial_scale=0.1,
            use_layernorm=True,
        )
        conditioner = SpeakerConditioner(config)

        stats = conditioner.get_stats()

        assert "speaker_conditioner/scale" in stats
        assert "speaker_conditioner/projection_weight_norm" in stats
        assert "speaker_conditioner/layernorm_weight_mean" in stats

    def test_gradient_flow(self):
        """Test that gradients flow correctly through the conditioner."""
        from finetune.modules.speaker_conditioner import SpeakerConditioner, SpeakerConditionerConfig

        config = SpeakerConditionerConfig(
            input_dim=192,
            output_dim=4096,
            learnable_scale=True,
        )
        conditioner = SpeakerConditioner(config)

        speaker_emb = torch.randn(2, 192, requires_grad=True)
        output = conditioner(speaker_emb)

        # Compute loss and backward
        loss = output.sum()
        loss.backward()

        # Check gradients
        assert speaker_emb.grad is not None
        assert conditioner.projection.weight.grad is not None
        assert conditioner.scale.grad is not None


class TestReferenceSampler:
    """Tests for ReferenceSampler utility class."""

    def test_basic_sampling(self):
        """Test basic reference sampling."""
        from finetune.modules.speaker_conditioner import ReferenceSampler

        sampler = ReferenceSampler(
            min_duration_sec=1.0,
            max_duration_sec=3.0,
            sample_rate=24000,
            target_sample_rate=16000,
        )

        # Simulate 10 seconds of MOSHI audio
        moshi_audio = torch.randn(10 * 24000)

        ref_audio, ref_text = sampler.sample_from_moshi_stream(moshi_audio)

        # Check resampled audio length
        min_samples = int(1.0 * 16000)
        max_samples = int(3.0 * 16000)
        assert min_samples <= ref_audio.shape[-1] <= max_samples

    def test_sampling_with_exclusion(self):
        """Test sampling with excluded region."""
        from finetune.modules.speaker_conditioner import ReferenceSampler

        sampler = ReferenceSampler(
            min_duration_sec=1.0,
            max_duration_sec=2.0,
            sample_rate=24000,
            target_sample_rate=24000,  # No resampling for easier testing
        )

        # 10 seconds of audio
        moshi_audio = torch.randn(10 * 24000)

        # Exclude middle 4 seconds (3s to 7s)
        exclude_start = 3 * 24000
        exclude_end = 7 * 24000

        ref_audio, _ = sampler.sample_from_moshi_stream(
            moshi_audio,
            exclude_start=exclude_start,
            exclude_end=exclude_end,
        )

        # Should have valid output
        assert ref_audio.shape[-1] > 0

    def test_with_text_tokens(self):
        """Test sampling with corresponding text tokens."""
        from finetune.modules.speaker_conditioner import ReferenceSampler

        sampler = ReferenceSampler(
            min_duration_sec=2.0,
            max_duration_sec=4.0,
            sample_rate=24000,
            target_sample_rate=16000,
        )

        # 10 seconds of audio → 125 frames (12.5Hz)
        moshi_audio = torch.randn(10 * 24000)
        moshi_text = torch.randint(0, 32000, (125,))

        ref_audio, ref_text = sampler.sample_from_moshi_stream(
            moshi_audio, moshi_text_tokens=moshi_text
        )

        # Both should be returned
        assert ref_audio is not None
        assert ref_text is not None
        assert ref_text.shape[0] > 0


class TestSpeakerConditioningIntegration:
    """Integration tests for complete speaker conditioning flow."""

    def test_full_pipeline(self):
        """Test complete pipeline: audio → encoder → conditioner → sum_condition."""
        from finetune.modules.speaker_encoder import DummySpeakerEncoder, SpeakerEncoderConfig
        from finetune.modules.speaker_conditioner import SpeakerConditioner, SpeakerConditionerConfig

        # Setup encoder
        encoder_config = SpeakerEncoderConfig(output_dim=192)
        encoder = DummySpeakerEncoder(encoder_config)

        # Setup conditioner
        conditioner_config = SpeakerConditionerConfig(
            input_dim=192,
            output_dim=4096,
            initial_scale=0.1,
        )
        conditioner = SpeakerConditioner(conditioner_config)

        # Input audio
        batch_size = 2
        audio = torch.randn(batch_size, 16000)

        # Full pipeline
        speaker_emb = encoder(audio)  # [B, 192]
        sum_condition = conditioner(speaker_emb)  # [B, 1, 4096]

        # Verify shapes
        assert speaker_emb.shape == (batch_size, 192)
        assert sum_condition.shape == (batch_size, 1, 4096)

        # Simulate adding to transformer input
        seq_len = 100
        transformer_input = torch.randn(batch_size, seq_len, 4096)
        conditioned_input = transformer_input + sum_condition.to(transformer_input)

        assert conditioned_input.shape == (batch_size, seq_len, 4096)

    def test_broadcasting(self):
        """Test that sum_condition broadcasts correctly across time."""
        from finetune.modules.speaker_conditioner import SpeakerConditioner, SpeakerConditionerConfig

        conditioner_config = SpeakerConditionerConfig(
            input_dim=192,
            output_dim=4096,
        )
        conditioner = SpeakerConditioner(conditioner_config)

        batch_size = 2
        seq_len = 100

        speaker_emb = torch.randn(batch_size, 192)
        sum_condition = conditioner(speaker_emb)  # [B, 1, 4096]

        # Create transformer input
        transformer_input = torch.zeros(batch_size, seq_len, 4096)

        # Add sum_condition (should broadcast)
        result = transformer_input + sum_condition

        # Check that all time steps got the same speaker condition
        for t in range(seq_len):
            assert torch.allclose(result[:, t, :], result[:, 0, :])


class TestSpeakerConditioningArgs:
    """Tests for configuration args in finetune/args.py."""

    def test_default_args(self):
        """Test default SpeakerConditioningArgs."""
        from finetune.args import SpeakerConditioningArgs

        args = SpeakerConditioningArgs()

        assert args.enabled is False
        assert args.method == "encoder"
        assert args.encoder.encoder_type == "ecapa_tdnn"
        assert args.conditioner.output_dim == 4096
        assert args.reference_sampler.min_duration_sec == 3.0

    def test_invalid_method(self):
        """Test invalid method raises error."""
        from finetune.args import SpeakerConditioningArgs

        with pytest.raises(ValueError, match="speaker.method must be one of"):
            SpeakerConditioningArgs(method="invalid")

    def test_nested_config(self):
        """Test nested configuration."""
        from finetune.args import (
            SpeakerConditioningArgs,
            SpeakerEncoderArgs,
            SpeakerConditionerArgs,
        )

        args = SpeakerConditioningArgs(
            enabled=True,
            encoder=SpeakerEncoderArgs(freeze=False, output_dim=256),
            conditioner=SpeakerConditionerArgs(initial_scale=0.2),
        )

        assert args.enabled is True
        assert args.encoder.freeze is False
        assert args.encoder.output_dim == 256
        assert args.conditioner.initial_scale == 0.2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
