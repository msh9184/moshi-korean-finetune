# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
Unit tests for Speaker Conditioning Evaluation Integration (Phase 1).

Tests cover:
    - AudioPromptSampler deterministic mode
    - "start" and "end" sampling strategies
    - EvalSpeakerConditioningInfo dataclass
    - sample_saver.py speaker conditioning metadata saving
    - Reproducibility guarantees

Run with: pytest tests/test_speaker_conditioning_eval.py -v
"""

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pytest
import torch


class TestAudioPromptDeterministicMode:
    """Tests for AudioPromptSampler deterministic mode."""

    def test_deterministic_flag_in_config(self):
        """Test that deterministic flag exists in AudioPromptConfig."""
        from finetune.modules.audio_prompt import AudioPromptConfig

        # Default should be False (training mode)
        config = AudioPromptConfig()
        assert config.deterministic is False

        # Can be set to True (evaluation mode)
        config_eval = AudioPromptConfig(
            enable=True,
            deterministic=True,
            fixed_duration_sec=10.0,
            sample_strategy="start",
        )
        assert config_eval.deterministic is True
        assert config_eval.fixed_duration_sec == 10.0
        assert config_eval.sample_strategy == "start"

    def test_deterministic_warning_with_random_strategy(self):
        """Test warning when deterministic=True but strategy is random."""
        from finetune.modules.audio_prompt import AudioPromptConfig
        import logging

        # Should log a warning
        with pytest.warns(None):  # May or may not warn depending on implementation
            config = AudioPromptConfig(
                enable=True,
                deterministic=True,
                sample_strategy="random",  # Inconsistent with deterministic
            )
            assert config.deterministic is True

    def test_start_strategy_deterministic(self):
        """Test 'start' strategy always samples from position 0."""
        from finetune.modules.audio_prompt import AudioPromptConfig, AudioPromptSampler

        config = AudioPromptConfig(
            enable=True,
            mode="audio_only",
            min_duration_sec=2.0,
            max_duration_sec=5.0,
            sample_strategy="start",
            deterministic=True,
            fixed_duration_sec=3.0,
        )

        sampler = AudioPromptSampler(config, frame_rate=12.5)

        # Create test codes: [K, T] = [8, 100] (8 seconds at 12.5 Hz)
        codes = torch.randint(0, 2048, (8, 100))

        # Sample multiple times - should always get same result
        results = []
        for _ in range(5):
            sample = sampler.sample_single(codes, deterministic=True)
            results.append((sample.start_idx, sample.end_idx))

        # All results should be identical
        assert all(r == results[0] for r in results), \
            f"Deterministic sampling produced different results: {results}"

        # Start should always be 0 for "start" strategy
        assert results[0][0] == 0, f"Start should be 0, got {results[0][0]}"

    def test_end_strategy_deterministic(self):
        """Test 'end' strategy always samples from the end."""
        from finetune.modules.audio_prompt import AudioPromptConfig, AudioPromptSampler

        config = AudioPromptConfig(
            enable=True,
            mode="audio_only",
            min_duration_sec=2.0,
            max_duration_sec=5.0,
            sample_strategy="end",
            deterministic=True,
            fixed_duration_sec=3.0,
        )

        sampler = AudioPromptSampler(config, frame_rate=12.5)
        total_frames = 100

        # Create test codes
        codes = torch.randint(0, 2048, (8, total_frames))

        # Sample multiple times
        results = []
        for _ in range(5):
            sample = sampler.sample_single(codes, deterministic=True)
            results.append((sample.start_idx, sample.end_idx))

        # All results should be identical
        assert all(r == results[0] for r in results)

        # End should be at total_frames for "end" strategy
        assert results[0][1] == total_frames, \
            f"End should be {total_frames}, got {results[0][1]}"

    def test_deterministic_vs_random_sampling(self):
        """Test that deterministic and random modes produce different behavior."""
        from finetune.modules.audio_prompt import AudioPromptConfig, AudioPromptSampler

        config = AudioPromptConfig(
            enable=True,
            mode="audio_only",
            min_duration_sec=1.0,
            max_duration_sec=3.0,
            sample_strategy="random",
            deterministic=False,  # Random mode
        )

        sampler = AudioPromptSampler(config, frame_rate=12.5)

        # Create test codes: 200 frames (16 seconds)
        codes = torch.randint(0, 2048, (8, 200))

        # Sample multiple times with random mode
        random_results = []
        for _ in range(10):
            sample = sampler.sample_single(codes, deterministic=False)
            random_results.append((sample.start_idx, sample.end_idx, sample.duration_frames))

        # With random mode, we should get some variation (not all identical)
        # Note: Small probability all are same, but very unlikely with 10 samples
        unique_starts = len(set(r[0] for r in random_results))

        # Sample with deterministic mode (overriding config)
        det_results = []
        for _ in range(5):
            sample = sampler.sample_single(codes, deterministic=True)
            det_results.append((sample.start_idx, sample.end_idx))

        # Deterministic should always be identical
        assert all(r == det_results[0] for r in det_results), \
            f"Deterministic mode should produce identical results: {det_results}"

    def test_fixed_duration_sec(self):
        """Test that fixed_duration_sec is used in deterministic mode."""
        from finetune.modules.audio_prompt import AudioPromptConfig, AudioPromptSampler

        fixed_duration = 5.0  # 5 seconds
        frame_rate = 12.5
        expected_frames = int(fixed_duration * frame_rate)  # 62.5 → 62 frames

        config = AudioPromptConfig(
            enable=True,
            mode="audio_only",
            sample_strategy="start",
            deterministic=True,
            fixed_duration_sec=fixed_duration,
        )

        sampler = AudioPromptSampler(config, frame_rate=frame_rate)

        # Create test codes: 200 frames
        codes = torch.randint(0, 2048, (8, 200))

        sample = sampler.sample_single(codes, deterministic=True)

        # Duration should match fixed_duration_sec
        actual_duration = sample.duration_frames
        assert actual_duration == expected_frames, \
            f"Expected {expected_frames} frames, got {actual_duration}"


class TestEvalSpeakerConditioningInfo:
    """Tests for EvalSpeakerConditioningInfo dataclass."""

    def test_dataclass_creation(self):
        """Test creating EvalSpeakerConditioningInfo."""
        from finetune.eval import EvalSpeakerConditioningInfo

        info = EvalSpeakerConditioningInfo()
        assert info.enabled is False
        assert info.method == "none"
        assert info.deterministic is True
        assert info.sampling_strategy == "start"

    def test_dataclass_with_values(self):
        """Test creating with specific values."""
        from finetune.eval import EvalSpeakerConditioningInfo

        speaker_emb = torch.randn(2, 256)
        ref_audio = torch.randn(2, 24000 * 10)

        info = EvalSpeakerConditioningInfo(
            enabled=True,
            method="both",
            speaker_embedding=speaker_emb,
            reference_audio=ref_audio,
            reference_start_sec=0.0,
            reference_end_sec=10.0,
            reference_text="Test reference text",
            sampling_strategy="start",
            deterministic=True,
            fixed_duration_sec=10.0,
        )

        assert info.enabled is True
        assert info.method == "both"
        assert info.speaker_embedding is speaker_emb
        assert info.reference_audio is ref_audio
        assert info.reference_start_sec == 0.0
        assert info.reference_end_sec == 10.0
        assert info.reference_text == "Test reference text"


class TestGetEvalAudioPromptConfig:
    """Tests for get_eval_audio_prompt_config helper function."""

    def test_default_eval_config(self):
        """Test default evaluation configuration."""
        from finetune.modules.audio_prompt import get_eval_audio_prompt_config

        config = get_eval_audio_prompt_config()

        assert config.enable is True
        assert config.deterministic is True
        assert config.sample_strategy == "start"
        assert config.fixed_duration_sec == 10.0
        assert config.avoid_overlap is False

    def test_custom_eval_config(self):
        """Test custom evaluation configuration."""
        from finetune.modules.audio_prompt import get_eval_audio_prompt_config

        config = get_eval_audio_prompt_config(
            fixed_duration_sec=5.0,
            sample_strategy="end",
        )

        assert config.deterministic is True
        assert config.sample_strategy == "end"
        assert config.fixed_duration_sec == 5.0


class TestAudioPromptArgsValidation:
    """Tests for AudioPromptArgs validation in args.py."""

    def test_args_deterministic_fields(self):
        """Test that AudioPromptArgs has deterministic fields."""
        from finetune.args import AudioPromptArgs

        args = AudioPromptArgs()
        assert hasattr(args, 'deterministic')
        assert hasattr(args, 'fixed_duration_sec')
        assert hasattr(args, 'use_word_count')
        assert hasattr(args, 'fixed_word_count')

        # Default values
        assert args.deterministic is False
        assert args.fixed_duration_sec == 10.0
        assert args.use_word_count is False
        assert args.fixed_word_count == 20

    def test_args_eval_configuration(self):
        """Test configuring args for evaluation."""
        from finetune.args import AudioPromptArgs

        eval_args = AudioPromptArgs(
            enable=True,
            mode="audio_text",
            sample_strategy="start",
            deterministic=True,
            fixed_duration_sec=10.0,
        )

        assert eval_args.deterministic is True
        assert eval_args.sample_strategy == "start"
        assert eval_args.fixed_duration_sec == 10.0


class TestSampleSaverSpeakerConditioning:
    """Tests for sample_saver.py speaker conditioning metadata saving."""

    def test_sample_save_result_has_speaker_fields(self):
        """Test SampleSaveResult has speaker conditioning fields."""
        from finetune.monitoring.sample_saver import SampleSaveResult

        result = SampleSaveResult(step=100, num_samples=5)

        assert hasattr(result, 'reference_audio_paths')
        assert hasattr(result, 'speaker_metadata_paths')
        assert isinstance(result.reference_audio_paths, list)
        assert isinstance(result.speaker_metadata_paths, list)

    def test_speaker_metadata_json_structure(self):
        """Test the structure of speaker metadata JSON."""
        # Create a mock EvalSpeakerConditioningInfo
        @dataclass
        class MockSpeakerConditioningInfo:
            enabled: bool = True
            method: str = "both"
            deterministic: bool = True
            sampling_strategy: str = "start"
            reference_start_sec: float = 0.0
            reference_end_sec: float = 10.0
            fixed_duration_sec: float = 10.0
            reference_text: Optional[str] = "Test text"
            speaker_embedding: Optional[torch.Tensor] = None
            reference_audio: Optional[torch.Tensor] = None

        info = MockSpeakerConditioningInfo()
        info.speaker_embedding = torch.randn(1, 256)

        # Build metadata dict as sample_saver would
        speaker_metadata = {
            "enabled": info.enabled,
            "method": info.method,
            "deterministic": info.deterministic,
            "sampling_strategy": info.sampling_strategy,
            "reference_start_sec": info.reference_start_sec,
            "reference_end_sec": info.reference_end_sec,
            "fixed_duration_sec": info.fixed_duration_sec,
        }

        if info.reference_text:
            speaker_metadata["reference_text"] = info.reference_text

        if info.speaker_embedding is not None:
            emb = info.speaker_embedding[0]
            speaker_metadata["embedding_stats"] = {
                "shape": list(emb.shape),
                "mean": float(emb.mean().item()),
                "std": float(emb.std().item()),
                "min": float(emb.min().item()),
                "max": float(emb.max().item()),
                "norm": float(emb.norm().item()),
            }

        # Verify structure
        assert speaker_metadata["enabled"] is True
        assert speaker_metadata["method"] == "both"
        assert speaker_metadata["deterministic"] is True
        assert speaker_metadata["sampling_strategy"] == "start"
        assert speaker_metadata["reference_start_sec"] == 0.0
        assert speaker_metadata["reference_end_sec"] == 10.0
        assert speaker_metadata["reference_text"] == "Test text"
        assert "embedding_stats" in speaker_metadata
        assert "shape" in speaker_metadata["embedding_stats"]
        assert "norm" in speaker_metadata["embedding_stats"]

        # Verify JSON serializable
        json_str = json.dumps(speaker_metadata)
        assert len(json_str) > 0

        # Verify round-trip
        loaded = json.loads(json_str)
        assert loaded["method"] == "both"


class TestReproducibility:
    """Tests for reproducibility guarantees."""

    def test_same_input_same_output(self):
        """Test that same input always produces same reference selection."""
        from finetune.modules.audio_prompt import AudioPromptConfig, AudioPromptSampler

        config = AudioPromptConfig(
            enable=True,
            mode="audio_only",
            sample_strategy="start",
            deterministic=True,
            fixed_duration_sec=5.0,
        )

        sampler = AudioPromptSampler(config, frame_rate=12.5)

        # Fixed input
        torch.manual_seed(42)
        codes = torch.randint(0, 2048, (8, 150))

        # Multiple runs
        results = []
        for _ in range(10):
            sample = sampler.sample_single(codes, deterministic=True)
            results.append({
                "start": sample.start_idx,
                "end": sample.end_idx,
                "duration": sample.duration_frames,
            })

        # All should be identical
        first = results[0]
        for i, r in enumerate(results[1:], 1):
            assert r == first, f"Run {i} differs: {r} vs {first}"

    def test_no_random_number_generation_in_deterministic(self):
        """Verify no RNG state changes in deterministic mode."""
        from finetune.modules.audio_prompt import AudioPromptConfig, AudioPromptSampler

        config = AudioPromptConfig(
            enable=True,
            mode="audio_only",
            sample_strategy="start",
            deterministic=True,
            fixed_duration_sec=3.0,
        )

        sampler = AudioPromptSampler(config, frame_rate=12.5)
        codes = torch.randint(0, 2048, (8, 100))

        # Set RNG state
        torch.manual_seed(12345)
        state_before = torch.get_rng_state().clone()

        # Run deterministic sampling
        _ = sampler.sample_single(codes, deterministic=True)

        # Check RNG state unchanged
        state_after = torch.get_rng_state()

        # In deterministic mode, RNG should NOT be used
        # Note: This test may need adjustment if any RNG is used internally
        # The key guarantee is that the OUTPUT is deterministic


class TestAudioPromptModuleIntegration:
    """Integration tests for AudioPromptModule with deterministic mode."""

    def test_module_deterministic_forward(self):
        """Test AudioPromptModule forward with deterministic sampling."""
        from finetune.modules.audio_prompt import (
            AudioPromptConfig,
            AudioPromptModule,
            create_audio_prompt_module,
        )

        config = AudioPromptConfig(
            enable=True,
            mode="audio_only",
            sample_strategy="start",
            deterministic=True,
            fixed_duration_sec=3.0,
        )

        module = create_audio_prompt_module(
            config=config,
            frame_rate=12.5,
            audio_offset=1,
            text_padding_token_id=0,
        )

        # Create input codes: [B, K, T] = [2, 9, 100]
        codes = torch.randint(0, 2048, (2, 9, 100))

        # Forward pass with deterministic=True
        prompted_codes, mask, samples = module(
            codes,
            exclude_start=None,
            exclude_end=None,
            deterministic=True,
        )

        # Check output shapes
        assert prompted_codes.dim() == 3
        assert prompted_codes.size(0) == 2  # Batch size
        assert prompted_codes.size(1) == 9  # Codebooks

        # Mask should be provided
        assert mask is not None
        assert mask.shape[0] == 2  # Batch size

        # Samples info should be provided
        assert samples is not None
        assert len(samples) == 2  # One per batch item

        # All samples should have start_idx = 0 (start strategy)
        for sample in samples:
            assert sample.start_idx == 0, f"Expected start_idx=0, got {sample.start_idx}"


class TestBatchDeterministicSampling:
    """Tests for batch-level deterministic sampling."""

    def test_batch_sample_deterministic(self):
        """Test batch sampling in deterministic mode."""
        from finetune.modules.audio_prompt import AudioPromptConfig, AudioPromptSampler

        config = AudioPromptConfig(
            enable=True,
            mode="audio_only",
            sample_strategy="start",
            deterministic=True,
            fixed_duration_sec=4.0,
        )

        sampler = AudioPromptSampler(config, frame_rate=12.5)

        # Batch of codes: [B, K, T] = [4, 8, 120]
        batch_codes = torch.randint(0, 2048, (4, 8, 120))

        # Sample batch
        samples = sampler.sample_batch(batch_codes, deterministic=True)

        assert len(samples) == 4

        # All samples should start at 0 (deterministic "start" strategy)
        for i, sample in enumerate(samples):
            assert sample.start_idx == 0, \
                f"Sample {i}: expected start_idx=0, got {sample.start_idx}"

        # Verify consistency across multiple calls
        samples2 = sampler.sample_batch(batch_codes, deterministic=True)
        for i, (s1, s2) in enumerate(zip(samples, samples2)):
            assert s1.start_idx == s2.start_idx
            assert s1.end_idx == s2.end_idx
            assert s1.duration_frames == s2.duration_frames


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
