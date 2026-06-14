# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
PersonaPlex Style Audio/Text Prompting Module for K-Moshi

This module implements PersonaPlex style audio and text prompting for zero-shot
speaker adaptation during training. It samples reference audio AND corresponding
text from the Moshi stream and prepends them as prompts to the main sequence.

Unlike VALL-E (audio-only), PersonaPlex always includes BOTH audio codes and
text tokens in the prompt, providing richer speaker conditioning through both
acoustic and linguistic patterns.

Architecture Overview:
    ┌──────────────────────────────────────────────────────────────────┐
    │                    PersonaPlex Style Prompting                   │
    ├──────────────────────────────────────────────────────────────────┤
    │                                                                  │
    │  Reference Audio+Text (10-15s)    Main Sequence (Training Target)│
    │  ─────────────────────────────    ──────────────────────────────│
    │                                                                  │
    │  ┌─────────────────────────────┐  ┌────────────────────────────┐│
    │  │ Text:  [안녕] [PAD] [하세요] │→│ Text:  [오늘] [날씨] [...] ││
    │  │ Audio: [A0-7] [A0-7] [A0-7] │→│ Audio: [A0-7] [A0-7] [...] ││
    │  └─────────────────────────────┘  └────────────────────────────┘│
    │        (prompt_mask=True)              (prompt_mask=False)       │
    │           ↓                                                      │
    │  ┌─────────────────────────────┐                                │
    │  │ Speaker Encoder → Embedding │ (Global Speaker Condition)     │
    │  └─────────────────────────────┘                                │
    │                                                                  │
    └──────────────────────────────────────────────────────────────────┘

Key Features:
    1. Random reference segment sampling from Moshi stream
    2. BOTH audio codes AND text tokens in prompt (PersonaPlex style)
    3. Configurable prompt duration (10-15 seconds recommended)
    4. Synchronized audio-text alignment preservation
    5. Integration with speaker encoder (method="both" for best results)

References:
    - NVIDIA PersonaPlex: Reference audio/text conditioning approach
    - K-Moshi Zero-Shot Speaker Conditioning Specification
"""

from dataclasses import dataclass, field
from typing import Optional, Literal, Tuple, List
import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class AudioPromptConfig:
    """Configuration for PersonaPlex style audio/text prompting.

    PersonaPlex style ALWAYS includes both audio codes AND text tokens in the
    prompt. This provides richer speaker conditioning compared to audio-only
    approaches like VALL-E.

    Attributes:
        enable: Enable audio prompting (default: False for backward compatibility)
        mode: Prompting mode
            - "audio_text": Use both audio codes and text tokens (PersonaPlex style)
            - "speaker_embedding": Only global speaker embedding (no prompt prepending)
        min_duration_sec: Minimum prompt duration in seconds (10-15s recommended)
        max_duration_sec: Maximum prompt duration in seconds
        sample_strategy: How to sample reference segment
            - "random": Random segment from Moshi stream
            - "start": First N seconds
            - "end": Last N seconds
            - "voiced": Prefer voiced segments (requires VAD)
        include_special_tokens: Include BOS/EOS markers around prompt
        prompt_position: Where to place the prompt
            - "prefix": Before main sequence (PersonaPlex style)
            - "interleaved": Interleaved with main sequence (future)
        audio_sample_rate: Audio sample rate for processing (24000 for Mimi)
        text_frame_rate: Text token frame rate (12.5Hz for Moshi)
        avoid_overlap: Avoid sampling from the same segment being trained
    """
    enable: bool = False
    mode: Literal["audio_text", "speaker_embedding"] = "speaker_embedding"

    min_duration_sec: float = 3.0
    max_duration_sec: float = 10.0

    sample_strategy: Literal["random", "start", "end", "voiced"] = "random"
    include_special_tokens: bool = True
    prompt_position: Literal["prefix", "interleaved"] = "prefix"

    audio_sample_rate: int = 24000  # Mimi's native rate
    text_frame_rate: float = 12.5   # Moshi's text token rate

    avoid_overlap: bool = True

    # Special token IDs (will be populated from model)
    prompt_start_token_id: int = -1  # Placeholder, set from model
    prompt_end_token_id: int = -1    # Placeholder, set from model

    # =========================================================================
    # DETERMINISTIC MODE FOR EVALUATION/INFERENCE
    # =========================================================================
    # When deterministic=True:
    #   - NO torch.randint() calls
    #   - Fixed duration (fixed_duration_sec) instead of random [min, max]
    #   - Fixed position based on sample_strategy ("start" recommended)
    #
    # This ensures reproducible evaluation and inference results.
    # The same input will ALWAYS produce the same reference selection.
    # =========================================================================
    deterministic: bool = False
    fixed_duration_sec: float = 10.0  # Used when deterministic=True

    # =========================================================================
    # WORD-COUNT BASED SELECTION (alternative to duration-based)
    # =========================================================================
    # When use_word_count=True:
    # - min_words/max_words define the valid range of non-padding text tokens
    # - Segment is selected to contain approximately min_words~max_words tokens
    # - This ensures meaningful speech content in reference audio
    # =========================================================================
    use_word_count: bool = False
    min_words: int = 5     # Minimum non-padding text tokens (approx. 5+ words)
    max_words: int = 30    # Maximum non-padding text tokens (approx. 30 words)
    fixed_word_count: int = 20  # Fixed word count when deterministic=True

    # Text padding token IDs to exclude from word count
    text_padding_token_ids: Tuple[int, ...] = (0, 3, 32000)  # PAD, EOS, END_OF_TEXT

    def __post_init__(self) -> None:
        """Validate configuration (PersonaPlex-only)."""
        valid_modes = ("audio_text", "speaker_embedding")
        if self.mode not in valid_modes:
            raise ValueError(
                f"AudioPromptConfig.mode must be one of {valid_modes}, got '{self.mode}'. "
                "Note: 'audio_only' (VALL-E style) is not supported. "
                "Use 'audio_text' for PersonaPlex style prompting (Audio + Text)."
            )

        # Validation for deterministic mode
        if self.deterministic:
            if self.sample_strategy == "random":
                logger.warning(
                    "AudioPromptConfig: deterministic=True with sample_strategy='random'. "
                    "Consider using 'start' for fully deterministic behavior."
                )
            if self.fixed_duration_sec <= 0:
                raise ValueError(
                    f"fixed_duration_sec must be positive when deterministic=True, "
                    f"got {self.fixed_duration_sec}"
                )


@dataclass
class AudioPromptSample:
    """A sampled audio/text prompt for training.

    This data structure holds a reference segment sampled from the Moshi stream,
    including both audio codes and corresponding text tokens (if available).

    Supports both 9-codebook (Monologue) and 17-codebook (Full-Duplex) modes:
        - Monologue: audio_codes [8, T] for Moshi only
        - Full-Duplex: audio_codes [8, T] for Moshi, user_audio_codes [8, T] for User

    Attributes:
        audio_codes: Moshi audio codes [8, T_prompt] for the reference segment
        text_tokens: Corresponding text tokens [T_prompt] (if mode="audio_text")
        user_audio_codes: User audio codes [8, T_prompt] (if 17-codebook mode)
        duration_sec: Actual duration of the prompt in seconds
        start_idx: Start index in original stream (for debugging/logging)
        end_idx: End index in original stream
        speaker_embedding: Optional pre-computed speaker embedding [D_spk]
        num_codebooks: Number of codebooks in source (9 or 17)
    """
    audio_codes: torch.Tensor  # [8, T_prompt] - 8 moshi audio codebooks
    text_tokens: Optional[torch.Tensor] = None  # [T_prompt]
    user_audio_codes: Optional[torch.Tensor] = None  # [8, T_prompt] - 8 user audio codebooks (17-codebook mode)
    duration_sec: float = 0.0
    start_idx: int = 0
    end_idx: int = 0
    speaker_embedding: Optional[torch.Tensor] = None  # [D_spk]
    num_codebooks: int = 9  # Source codebook count (9 or 17)


class AudioPromptSampler:
    """Sampler for extracting reference audio/text prompts from Moshi stream.

    This class handles the core logic of sampling reference segments from
    the Moshi training data stream, ensuring proper alignment between audio
    codes and text tokens.

    The sampler operates on the code-level (post Mimi encoding) rather than
    raw audio, which is more efficient during training.

    Supports both 9-codebook (Monologue) and 17-codebook (Full-Duplex) modes:
        - 9 codebooks: [1 text + 8 moshi audio]
        - 17 codebooks: [1 text + 8 moshi audio + 8 user audio]

    Usage:
        sampler = AudioPromptSampler(config)

        # During training dataloader
        codes = batch["codes"]  # [B, 9/17, T] - 1 text + 8 audio (+ 8 user audio)
        prompt_samples = sampler.sample_batch(
            codes,
            current_start=0,  # Start of training segment
            current_end=T,    # End of training segment
        )

        # Get prompted codes for training
        prompted_codes = sampler.apply_prompts(codes, prompt_samples)
    """

    def __init__(self, config: AudioPromptConfig):
        self.config = config
        self.frame_rate = config.text_frame_rate  # 12.5Hz

        # Calculate frame counts
        self.min_frames = int(config.min_duration_sec * self.frame_rate)
        self.max_frames = int(config.max_duration_sec * self.frame_rate)

        # Deterministic mode: fixed duration
        self.fixed_frames = int(config.fixed_duration_sec * self.frame_rate)

        # Word count settings
        self.use_word_count = config.use_word_count
        self.min_words = config.min_words
        self.max_words = config.max_words
        self.fixed_word_count = config.fixed_word_count
        self.text_padding_token_ids = set(config.text_padding_token_ids)

        mode_str = "DETERMINISTIC" if config.deterministic else "random"
        word_str = f", word_count={config.min_words}-{config.max_words}" if config.use_word_count else ""
        logger.info(
            f"AudioPromptSampler initialized: mode={config.mode}, "
            f"sampling={mode_str}, strategy={config.sample_strategy}, "
            f"duration={config.min_duration_sec}-{config.max_duration_sec}s "
            f"(fixed={config.fixed_duration_sec}s if deterministic){word_str}, "
            f"frames={self.min_frames}-{self.max_frames}"
        )

    def sample_single(
        self,
        codes: torch.Tensor,
        exclude_start: Optional[int] = None,
        exclude_end: Optional[int] = None,
        deterministic: Optional[bool] = None,
    ) -> AudioPromptSample:
        """Sample a single reference segment from codes.

        Args:
            codes: Full sequence codes [9, T] - 1 text + 8 audio codebooks
            exclude_start: Start frame to avoid (current training segment)
            exclude_end: End frame to avoid
            deterministic: Override config.deterministic for this call.
                          If None, uses config.deterministic.
                          If True, uses fixed duration and position (for eval/inference).

        Returns:
            AudioPromptSample containing the sampled reference segment

        Deterministic Mode Behavior:
            When deterministic=True:
            - "start" strategy: Always selects [0, fixed_duration_sec]
            - "end" strategy: Always selects [total - fixed_duration_sec, total]
            - "random" strategy: Same as "start" (with warning)
            - No torch.randint() calls - completely reproducible
        """
        assert codes.dim() == 2, f"Expected [K, T] where K=9 or 17, got {codes.shape}"
        num_codebooks, total_frames = codes.shape
        assert num_codebooks in (9, 17), f"Expected 9 or 17 codebooks, got {num_codebooks}"

        # Determine if we're in deterministic mode
        use_deterministic = deterministic if deterministic is not None else self.config.deterministic
        strategy = self.config.sample_strategy

        # =================================================================
        # WORD-COUNT BASED SELECTION (if enabled)
        # =================================================================
        # When use_word_count=True, select segment based on number of
        # non-padding text tokens instead of duration. This ensures
        # meaningful speech content in reference audio.
        # =================================================================
        if self.use_word_count:
            return self.sample_single_word_count(
                codes,
                exclude_start=exclude_start,
                exclude_end=exclude_end,
                deterministic=use_deterministic,
            )

        if use_deterministic:
            # =====================================================================
            # DETERMINISTIC SAMPLING (for eval/inference)
            # =====================================================================
            # No random number generation - same input always produces same output
            # =====================================================================

            # Calculate fixed duration in frames
            duration_frames = min(self.fixed_frames, total_frames)

            if strategy == "start" or strategy == "random":
                # "start": Always from position 0
                # "random" in deterministic mode: treated as "start" with warning
                if strategy == "random":
                    logger.debug(
                        "Deterministic mode with 'random' strategy - using 'start' behavior"
                    )
                start_frame = 0
                end_frame = min(duration_frames, total_frames)

            elif strategy == "end":
                # "end": Always from the end of the sequence
                end_frame = total_frames
                start_frame = max(0, total_frames - duration_frames)

            elif strategy == "voiced":
                # "voiced" in deterministic mode: use first voiced region
                # For now, fallback to "start" behavior
                logger.debug(
                    "Deterministic mode with 'voiced' strategy - using 'start' behavior "
                    "(VAD not implemented for deterministic mode)"
                )
                start_frame = 0
                end_frame = min(duration_frames, total_frames)

            else:
                # Fallback to "start"
                start_frame = 0
                end_frame = min(duration_frames, total_frames)

        else:
            # =====================================================================
            # RANDOM SAMPLING (for training)
            # =====================================================================
            # Original behavior with random duration and position
            # =====================================================================

            # Determine valid sampling regions (avoiding current training segment)
            valid_regions = self._get_valid_regions(
                total_frames, exclude_start, exclude_end
            )

            if not valid_regions:
                logger.warning("No valid regions for prompt sampling, using full sequence")
                valid_regions = [(0, total_frames)]

            # Calculate max possible frames from valid regions
            max_possible_frames = min(
                self.max_frames,
                max(end - start for start, end in valid_regions)
            )

            # Sample duration randomly
            if max_possible_frames < self.min_frames:
                # Sequence too short, use what we have
                duration_frames = max_possible_frames
            else:
                duration_frames = torch.randint(
                    self.min_frames,
                    max_possible_frames + 1,
                    (1,),
                ).item()

            # Sample start position based on strategy
            if strategy == "start":
                # Prefer start of first valid region
                start_frame = valid_regions[0][0]
            elif strategy == "end":
                # Prefer end of last valid region
                last_region = valid_regions[-1]
                start_frame = max(last_region[0], last_region[1] - duration_frames)
            elif strategy == "voiced":
                # TODO: Implement VAD-based voiced segment detection
                # For now, use random sampling
                start_frame = self._sample_from_regions(valid_regions, duration_frames)
            else:  # "random"
                start_frame = self._sample_from_regions(valid_regions, duration_frames)

            end_frame = min(start_frame + duration_frames, total_frames)

        # Extract audio codes and text tokens
        # Support both 9-codebook (Monologue) and 17-codebook (Full-Duplex) modes
        text_tokens = codes[0, start_frame:end_frame]    # [T_prompt] - index 0
        audio_codes = codes[1:9, start_frame:end_frame]  # [8, T_prompt] - indices 1-8 (Moshi audio)

        # Extract user audio codes if 17-codebook mode (Full-Duplex)
        user_audio_codes = None
        if num_codebooks == 17:
            user_audio_codes = codes[9:17, start_frame:end_frame]  # [8, T_prompt] - indices 9-16

        duration_sec = (end_frame - start_frame) / self.frame_rate

        # PersonaPlex style: Always include BOTH audio codes AND text tokens
        # This provides richer speaker conditioning than audio-only approaches
        return AudioPromptSample(
            audio_codes=audio_codes,
            text_tokens=text_tokens,  # Always include text (PersonaPlex style)
            user_audio_codes=user_audio_codes,  # User audio for 17-codebook mode
            duration_sec=duration_sec,
            start_idx=start_frame,
            end_idx=end_frame,
            num_codebooks=num_codebooks,
        )

    def sample_batch(
        self,
        codes: torch.Tensor,
        exclude_start: Optional[int] = None,
        exclude_end: Optional[int] = None,
        deterministic: Optional[bool] = None,
    ) -> List[AudioPromptSample]:
        """Sample reference segments for a batch of sequences.

        Args:
            codes: Batch of codes [B, 9, T]
            exclude_start: Start frame to avoid (same for all batch items)
            exclude_end: End frame to avoid
            deterministic: Override config.deterministic for this batch.
                          If None, uses config.deterministic.
                          If True, uses fixed duration and position (for eval/inference).

        Returns:
            List of AudioPromptSample, one per batch item

        Note:
            In deterministic mode, ALL samples in the batch will use the same
            start position and duration (based on their individual sequence lengths).
            This ensures reproducibility across evaluation runs.
        """
        batch_size = codes.shape[0]
        samples = []

        for b in range(batch_size):
            sample = self.sample_single(
                codes[b],
                exclude_start=exclude_start,
                exclude_end=exclude_end,
                deterministic=deterministic,
            )
            samples.append(sample)

        return samples

    def apply_prompts(
        self,
        codes: torch.Tensor,
        prompts: List[AudioPromptSample],
        pad_token_id: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply prompts to codes by prepending.

        This creates a new codes tensor with prompts prepended, along with
        a mask indicating which positions are prompts (should not compute loss).

        Supports both 9-codebook (Monologue) and 17-codebook (Full-Duplex) modes:
            - 9 codebooks: [1 text + 8 moshi audio]
            - 17 codebooks: [1 text + 8 moshi audio + 8 user audio]

        Args:
            codes: Original codes [B, 9/17, T]
            prompts: List of AudioPromptSample for each batch item
            pad_token_id: Token ID for padding shorter prompts

        Returns:
            Tuple of:
                - prompted_codes: [B, 9/17, T_prompt + T]
                - prompt_mask: [B, T_prompt + T] - True for prompt positions
        """
        if not self.config.enable or self.config.mode == "speaker_embedding":
            # No audio prompting, return original codes
            return codes, torch.zeros(codes.shape[0], codes.shape[2], dtype=torch.bool, device=codes.device)

        batch_size, num_codebooks, seq_len = codes.shape

        # Find max prompt length
        max_prompt_len = max(p.audio_codes.shape[1] for p in prompts)

        # Create output tensors (preserving num_codebooks from input)
        total_len = max_prompt_len + seq_len
        prompted_codes = torch.zeros(
            batch_size, num_codebooks, total_len,
            dtype=codes.dtype, device=codes.device
        )
        prompt_mask = torch.zeros(
            batch_size, total_len,
            dtype=torch.bool, device=codes.device
        )

        for b, prompt in enumerate(prompts):
            prompt_len = prompt.audio_codes.shape[1]

            # Fill prompt region (PersonaPlex style: both text AND audio)
            # Text tokens go to index 0 (Moshi text stream)
            prompted_codes[b, 0, :prompt_len] = prompt.text_tokens
            # Moshi audio codes go to indices 1-8 (8 Mimi codebooks)
            prompted_codes[b, 1:9, :prompt_len] = prompt.audio_codes

            # User audio codes go to indices 9-16 if 17-codebook mode
            if num_codebooks == 17 and prompt.user_audio_codes is not None:
                prompted_codes[b, 9:17, :prompt_len] = prompt.user_audio_codes

            # Fill main sequence
            prompted_codes[b, :, max_prompt_len:] = codes[b]

            # Mark prompt positions (excluded from loss computation)
            prompt_mask[b, :max_prompt_len] = True

        return prompted_codes, prompt_mask

    def _get_valid_regions(
        self,
        total_frames: int,
        exclude_start: Optional[int],
        exclude_end: Optional[int],
    ) -> List[Tuple[int, int]]:
        """Get valid sampling regions avoiding excluded segment."""
        if not self.config.avoid_overlap or exclude_start is None or exclude_end is None:
            return [(0, total_frames)]

        regions = []

        # Region before excluded segment
        if exclude_start > self.min_frames:
            regions.append((0, exclude_start))

        # Region after excluded segment
        if total_frames - exclude_end > self.min_frames:
            regions.append((exclude_end, total_frames))

        return regions

    def _sample_from_regions(
        self,
        regions: List[Tuple[int, int]],
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

    def _count_valid_tokens(self, text_tokens: torch.Tensor) -> int:
        """Count non-padding text tokens (proxy for word count).

        Args:
            text_tokens: Text token tensor [T]

        Returns:
            Number of tokens that are not padding tokens
        """
        count = 0
        for token in text_tokens.tolist():
            if int(token) not in self.text_padding_token_ids:
                count += 1
        return count

    def _find_word_count_segment(
        self,
        text_tokens: torch.Tensor,
        target_words: int,
        deterministic: bool = False,
    ) -> Tuple[int, int]:
        """Find a segment with approximately target_words non-padding tokens.

        This method scans the text tokens to find a contiguous segment
        containing approximately the target number of non-padding tokens.

        Args:
            text_tokens: Text token tensor [T]
            target_words: Target number of non-padding tokens
            deterministic: If True, always return from the start

        Returns:
            Tuple of (start_idx, end_idx) for the segment
        """
        total_frames = text_tokens.shape[0]
        token_list = text_tokens.tolist()

        # Find all valid (non-padding) token positions
        valid_positions = []
        for i, token in enumerate(token_list):
            if int(token) not in self.text_padding_token_ids:
                valid_positions.append(i)

        if len(valid_positions) == 0:
            # No valid tokens - return full sequence
            logger.warning("No valid text tokens found, using full sequence")
            return 0, total_frames

        if len(valid_positions) <= target_words:
            # Not enough tokens - return full sequence
            return 0, total_frames

        # Find start position for target word count
        if deterministic:
            # Always start from the first valid token
            start_word_idx = 0
        else:
            # Random start position
            max_start = len(valid_positions) - target_words
            start_word_idx = torch.randint(0, max_start + 1, (1,)).item()

        end_word_idx = min(start_word_idx + target_words, len(valid_positions))

        # Convert word indices to frame indices
        start_frame = valid_positions[start_word_idx]
        end_frame = valid_positions[end_word_idx - 1] + 1  # +1 for exclusive end

        # Extend slightly to include surrounding context (optional)
        # This helps include natural word boundaries
        context_frames = 2  # ~160ms context
        start_frame = max(0, start_frame - context_frames)
        end_frame = min(total_frames, end_frame + context_frames)

        return start_frame, end_frame

    def sample_single_word_count(
        self,
        codes: torch.Tensor,
        exclude_start: Optional[int] = None,
        exclude_end: Optional[int] = None,
        deterministic: bool = False,
    ) -> AudioPromptSample:
        """Sample a reference segment based on word count.

        This method selects a segment containing min_words~max_words
        non-padding text tokens, ensuring meaningful speech content.

        Args:
            codes: Full sequence codes [K, T] where K=9 or 17
            exclude_start: Start frame to avoid
            exclude_end: End frame to avoid
            deterministic: If True, use fixed word count from start

        Returns:
            AudioPromptSample containing the sampled reference segment
        """
        num_codebooks, total_frames = codes.shape
        text_tokens = codes[0]  # Text stream at index 0

        # Determine target word count
        if deterministic:
            target_words = self.fixed_word_count
        else:
            target_words = torch.randint(
                self.min_words,
                self.max_words + 1,
                (1,),
            ).item()

        # Find segment with target word count
        start_frame, end_frame = self._find_word_count_segment(
            text_tokens,
            target_words,
            deterministic=deterministic,
        )

        # Avoid overlap with excluded region if specified
        if exclude_start is not None and exclude_end is not None:
            if start_frame < exclude_end and end_frame > exclude_start:
                # Overlap detected - try to find alternative
                if exclude_start > target_words:
                    # Use region before excluded
                    start_frame, end_frame = self._find_word_count_segment(
                        text_tokens[:exclude_start],
                        target_words,
                        deterministic=deterministic,
                    )
                elif total_frames - exclude_end > target_words:
                    # Use region after excluded
                    sub_start, sub_end = self._find_word_count_segment(
                        text_tokens[exclude_end:],
                        target_words,
                        deterministic=deterministic,
                    )
                    start_frame = exclude_end + sub_start
                    end_frame = exclude_end + sub_end

        # Extract segments
        selected_text = codes[0, start_frame:end_frame]
        audio_codes = codes[1:9, start_frame:end_frame]
        user_audio_codes = None
        if num_codebooks == 17:
            user_audio_codes = codes[9:17, start_frame:end_frame]

        duration_sec = (end_frame - start_frame) / self.frame_rate
        actual_word_count = self._count_valid_tokens(selected_text)

        logger.debug(
            f"Word-count sampling: target={target_words}, actual={actual_word_count}, "
            f"frames={start_frame}-{end_frame}, duration={duration_sec:.2f}s"
        )

        return AudioPromptSample(
            audio_codes=audio_codes,
            text_tokens=selected_text,
            user_audio_codes=user_audio_codes,
            duration_sec=duration_sec,
            start_idx=start_frame,
            end_idx=end_frame,
            num_codebooks=num_codebooks,
        )


class AudioPromptEncoder(nn.Module):
    """Encoder for audio prompt representations.

    This module encodes the audio prompt into a representation that can be
    used to condition the main sequence generation. It's used when we want
    learnable prompt encoding beyond just concatenation.

    Architecture Options:
        1. Simple: Just concatenate prompt codes (no learnable params)
        2. Attention: Cross-attention from main sequence to prompt
        3. Summary: Pool prompt into fixed-size representation

    For initial implementation, we use the simple concatenation approach
    (VALL-E style), which is handled by AudioPromptSampler.apply_prompts().
    This module is for future advanced encoding strategies.
    """

    def __init__(
        self,
        config: AudioPromptConfig,
        hidden_dim: int = 4096,
        num_audio_codebooks: int = 8,
        audio_vocab_size: int = 2048,
        text_vocab_size: int = 32000,
    ):
        super().__init__()
        self.config = config
        self.hidden_dim = hidden_dim

        # Only create learnable components for advanced modes
        if config.mode == "audio_text":
            # Separate embeddings for prompt context (optional)
            self.prompt_audio_embed = nn.Embedding(audio_vocab_size, hidden_dim)
            self.prompt_text_embed = nn.Embedding(text_vocab_size, hidden_dim)

            # Cross-attention for prompt conditioning (future)
            self.cross_attention = None  # Placeholder

        logger.info(f"AudioPromptEncoder initialized: mode={config.mode}")

    def forward(
        self,
        prompt_sample: AudioPromptSample,
    ) -> Optional[torch.Tensor]:
        """Encode a prompt sample.

        Currently returns None as we use simple concatenation.
        Future: Return encoded prompt representation.

        Args:
            prompt_sample: The sampled audio/text prompt

        Returns:
            Encoded prompt representation or None for concatenation mode
        """
        # For now, we use simple concatenation in AudioPromptSampler
        # Future: Implement cross-attention or summary encoding
        return None


class AudioPromptModule(nn.Module):
    """Complete audio prompting module combining sampler and encoder.

    This is the main interface for audio prompting during training. It handles:
        1. Sampling reference segments from training data
        2. Encoding prompts (if using advanced encoding)
        3. Applying prompts to create training sequences

    Integration with K-Moshi Training:
        ```python
        # In train.py or wrapped_model.py
        audio_prompt_module = AudioPromptModule(config)

        # During batch processing
        codes = batch["codes"]  # [B, 9, T]
        prompted_codes, prompt_mask = audio_prompt_module(
            codes,
            exclude_start=0,
            exclude_end=seq_len,
        )

        # Use prompted_codes for model forward
        # Use prompt_mask to exclude prompt from loss computation
        ```

    Usage:
        config = AudioPromptConfig(
            enable=True,
            mode="audio_text",
            min_duration_sec=3.0,
            max_duration_sec=10.0,
        )
        module = AudioPromptModule(config)

        prompted_codes, prompt_mask = module(codes)
    """

    def __init__(
        self,
        config: AudioPromptConfig,
        hidden_dim: int = 4096,
    ):
        super().__init__()
        self.config = config

        self.sampler = AudioPromptSampler(config)

        if config.mode != "speaker_embedding":
            self.encoder = AudioPromptEncoder(
                config,
                hidden_dim=hidden_dim,
            )
        else:
            self.encoder = None

        logger.info(
            f"AudioPromptModule initialized: enable={config.enable}, mode={config.mode}"
        )

    def forward(
        self,
        codes: torch.Tensor,
        exclude_start: Optional[int] = None,
        exclude_end: Optional[int] = None,
        pad_token_id: int = 0,
        deterministic: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[AudioPromptSample]]:
        """Apply audio prompting to a batch of codes.

        Args:
            codes: Input codes [B, 9, T] - 1 text + 8 audio codebooks
            exclude_start: Start frame to exclude from sampling
            exclude_end: End frame to exclude from sampling
            pad_token_id: Token ID for padding
            deterministic: Override config.deterministic for this call.
                          If None, uses config.deterministic.
                          If True, uses fixed duration and position (for eval/inference).

        Returns:
            Tuple of:
                - prompted_codes: [B, 9, T_prompt + T] with prompts prepended
                - prompt_mask: [B, T_prompt + T] - True for prompt positions
                - prompt_samples: List of AudioPromptSample for additional processing

        Usage for Training vs Evaluation:
            # Training: random sampling
            prompted_codes, mask, samples = module(codes, deterministic=False)

            # Evaluation/Inference: deterministic sampling
            prompted_codes, mask, samples = module(codes, deterministic=True)
        """
        if not self.config.enable:
            # Audio prompting disabled, return original
            batch_size, _, seq_len = codes.shape
            return codes, torch.zeros(batch_size, seq_len, dtype=torch.bool, device=codes.device), []

        # Sample prompts (with deterministic override if specified)
        prompt_samples = self.sampler.sample_batch(
            codes,
            exclude_start=exclude_start,
            exclude_end=exclude_end,
            deterministic=deterministic,
        )

        # Apply prompts (concatenation)
        prompted_codes, prompt_mask = self.sampler.apply_prompts(
            codes, prompt_samples, pad_token_id
        )

        return prompted_codes, prompt_mask, prompt_samples

    def get_prompt_raw_audio(
        self,
        raw_audio: torch.Tensor,
        prompt_sample: AudioPromptSample,
        audio_sample_rate: int = 24000,
    ) -> torch.Tensor:
        """Extract raw audio corresponding to a prompt sample.

        This is useful when we need the raw audio for speaker encoder
        in addition to the audio codes for prompting.

        Args:
            raw_audio: Full raw audio [T_audio] at audio_sample_rate
            prompt_sample: The prompt sample with frame indices
            audio_sample_rate: Sample rate of raw_audio

        Returns:
            Raw audio segment [T_prompt_audio] corresponding to the prompt
        """
        samples_per_frame = audio_sample_rate / self.sampler.frame_rate  # 24000 / 12.5 = 1920

        start_sample = int(prompt_sample.start_idx * samples_per_frame)
        end_sample = int(prompt_sample.end_idx * samples_per_frame)

        return raw_audio[start_sample:end_sample]


# =============================================================================
# Convenience Functions
# =============================================================================

def create_audio_prompt_module(
    config: Optional[AudioPromptConfig] = None,
    *,
    # Legacy individual parameters (for backward compatibility)
    enable: bool = False,
    mode: str = "speaker_embedding",
    min_duration_sec: float = 10.0,
    max_duration_sec: float = 15.0,
    hidden_dim: int = 4096,
    deterministic: bool = False,
    sample_strategy: str = "random",
    fixed_duration_sec: float = 10.0,
    # Additional parameters (ignored but accepted for compatibility)
    frame_rate: Optional[float] = None,
    audio_offset: Optional[int] = None,
    text_padding_token_id: Optional[int] = None,
) -> AudioPromptModule:
    """Factory function to create AudioPromptModule.

    Supports two calling conventions:
        1. With config object: create_audio_prompt_module(config=my_config)
        2. With individual params: create_audio_prompt_module(enable=True, mode="audio_text", ...)

    Args:
        config: AudioPromptConfig object (takes precedence if provided)
        enable: Enable audio prompting (used if config is None)
        mode: Prompting mode ("audio_text" for PersonaPlex, "speaker_embedding" for global only)
        min_duration_sec: Minimum prompt duration (10-15s recommended)
        max_duration_sec: Maximum prompt duration
        hidden_dim: Model hidden dimension
        deterministic: Enable deterministic sampling (for eval/inference)
        sample_strategy: Sampling strategy ("random", "start", "end", "voiced")
        fixed_duration_sec: Fixed duration when deterministic=True
        frame_rate: (Ignored) Frame rate, stored in config if needed
        audio_offset: (Ignored) Audio offset in codebook layout
        text_padding_token_id: (Ignored) Text padding token ID

    Returns:
        Configured AudioPromptModule

    Examples:
        # Method 1: Using config object (recommended)
        config = AudioPromptConfig(
            enable=True,
            mode="audio_text",
            deterministic=True,
            sample_strategy="start",
        )
        module = create_audio_prompt_module(config=config)

        # Method 2: Using individual parameters
        module = create_audio_prompt_module(
            enable=True,
            mode="audio_text",
            min_duration_sec=10.0,
            max_duration_sec=15.0,
            deterministic=False,
            sample_strategy="random",
        )

        # For evaluation (deterministic sampling)
        module = create_audio_prompt_module(
            enable=True,
            mode="audio_text",
            deterministic=True,
            sample_strategy="start",
            fixed_duration_sec=10.0,
        )
    """
    # Use provided config or create one from individual parameters
    if config is not None:
        final_config = config
    else:
        final_config = AudioPromptConfig(
            enable=enable,
            mode=mode,
            min_duration_sec=min_duration_sec,
            max_duration_sec=max_duration_sec,
            sample_strategy=sample_strategy,
            deterministic=deterministic,
            fixed_duration_sec=fixed_duration_sec,
        )

    return AudioPromptModule(final_config, hidden_dim=hidden_dim)


def get_default_audio_prompt_config() -> AudioPromptConfig:
    """Get default audio prompt configuration.

    Returns default configuration with audio prompting disabled
    for backward compatibility.
    """
    return AudioPromptConfig(
        enable=False,
        mode="speaker_embedding",  # Use global speaker embedding only
    )


def get_eval_audio_prompt_config(
    fixed_duration_sec: float = 10.0,
    sample_strategy: str = "start",
) -> AudioPromptConfig:
    """Get audio prompt configuration optimized for evaluation/inference.

    This configuration ensures deterministic behavior:
    - No random sampling
    - Fixed duration from file start
    - Reproducible across runs

    Args:
        fixed_duration_sec: Fixed reference duration (default: 10 seconds)
        sample_strategy: Position strategy (default: "start" for reproducibility)

    Returns:
        AudioPromptConfig optimized for evaluation

    Example:
        # For evaluation
        config = get_eval_audio_prompt_config(fixed_duration_sec=10.0)
        module = AudioPromptModule(config)

        # Sample will always return first 10 seconds
        sample = module.sampler.sample_single(codes, deterministic=True)
    """
    return AudioPromptConfig(
        enable=True,
        mode="audio_text",
        sample_strategy=sample_strategy,
        deterministic=True,
        fixed_duration_sec=fixed_duration_sec,
        avoid_overlap=False,  # Not needed for evaluation
    )
