# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Word-level alignment using NeMo CTC models with ctc-segmentation.

This module provides forced alignment for Korean speech using pre-trained
CTC-based acoustic models. It processes stereo audio (SPEAKER_MAIN on left,
SPEAKER_USER on right) and produces word-level timestamps.

Key Features:
- Segment-based processing (max 20s per segment as required by Korean model)
- Both SPEAKER_MAIN and SPEAKER_USER alignment
- Korean CTC model support (SungBeom/stt_kr_conformer_ctc_medium)
- 8-machine distributed processing support

Installation:
    pip install nemo_toolkit[asr] ctc-segmentation

Reference:
    - NeMo NFA: https://github.com/NVIDIA/NeMo/tree/main/tools/nemo_forced_aligner
    - Korean CTC: https://huggingface.co/SungBeom/stt_kr_conformer_ctc_medium
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any, Sequence
import gc
import json
import logging
import re
import tempfile

import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import nemo.collections.asr as nemo_asr
    from nemo.collections.asr.models import EncDecCTCModel
    HAS_NEMO = True
except ImportError:
    HAS_NEMO = False

try:
    import ctc_segmentation
    from ctc_segmentation import prepare_text, prepare_token_list
    HAS_CTC_SEG = True
except ImportError:
    HAS_CTC_SEG = False
    prepare_text = None
    prepare_token_list = None

try:
    import soundfile as sf
    HAS_SOUNDFILE = True
except ImportError:
    HAS_SOUNDFILE = False

from ..config import NFAConfig
from .whisper_timestamped import Word, WordAlignment, AlignmentResult

logger = logging.getLogger(__name__)


@dataclass
class AlignmentDiagnostics:
    """Diagnostics and statistics for alignment failures and issues.

    Tracks various failure modes and provides summary statistics for
    debugging and monitoring alignment quality.
    """
    total_segments: int = 0
    successful_segments: int = 0
    failed_segments: int = 0
    fallback_segments: int = 0  # Used uniform alignment as fallback

    # Failure reasons
    empty_audio_count: int = 0
    empty_text_count: int = 0
    ctc_failures: int = 0
    dimension_mismatches: int = 0
    too_long_segments: int = 0
    vad_split_count: int = 0
    chunked_processing_count: int = 0  # Segments using chunked log probs

    # Quality metrics
    total_words_aligned: int = 0
    low_confidence_words: int = 0  # < 0.5 confidence
    very_low_confidence_words: int = 0  # < 0.3 confidence

    # Timing issues
    zero_duration_words: int = 0
    negative_duration_words: int = 0
    excessively_long_words: int = 0  # > max_word_duration

    # Details for debugging
    failure_details: List[Dict[str, Any]] = field(default_factory=list)

    def record_failure(
        self,
        reason: str,
        segment_id: str = "",
        duration: float = 0.0,
        text_preview: str = "",
        error_msg: str = "",
    ) -> None:
        """Record a failure with details for debugging."""
        self.failed_segments += 1
        self.failure_details.append({
            "reason": reason,
            "segment_id": segment_id,
            "duration": round(duration, 2),
            "text_preview": text_preview[:50] if text_preview else "",
            "error": error_msg,
        })

        # Update specific counters
        if reason == "empty_audio":
            self.empty_audio_count += 1
        elif reason == "empty_text":
            self.empty_text_count += 1
        elif reason == "ctc_error":
            self.ctc_failures += 1
        elif reason == "dimension_mismatch":
            self.dimension_mismatches += 1
        elif reason == "too_long":
            self.too_long_segments += 1

    def record_word(self, word: "WordTimestamp") -> None:
        """Record word-level metrics."""
        self.total_words_aligned += 1

        duration = word.end - word.start
        if duration <= 0:
            self.zero_duration_words += 1
        if duration < 0:
            self.negative_duration_words += 1

        if word.confidence < 0.3:
            self.very_low_confidence_words += 1
        elif word.confidence < 0.5:
            self.low_confidence_words += 1

    @property
    def success_rate(self) -> float:
        """Calculate segment success rate."""
        if self.total_segments == 0:
            return 0.0
        return self.successful_segments / self.total_segments

    @property
    def fallback_rate(self) -> float:
        """Calculate fallback rate (segments that needed uniform alignment)."""
        if self.total_segments == 0:
            return 0.0
        return self.fallback_segments / self.total_segments

    @property
    def word_quality_score(self) -> float:
        """Calculate overall word quality score (0-1)."""
        if self.total_words_aligned == 0:
            return 0.0

        # Penalize low confidence and timing issues
        good_words = (
            self.total_words_aligned
            - self.low_confidence_words
            - self.very_low_confidence_words
            - self.zero_duration_words
            - self.negative_duration_words
        )
        return max(0, good_words / self.total_words_aligned)

    def get_summary(self) -> Dict[str, Any]:
        """Get diagnostic summary."""
        return {
            "segments": {
                "total": self.total_segments,
                "successful": self.successful_segments,
                "failed": self.failed_segments,
                "fallback": self.fallback_segments,
                "success_rate": round(self.success_rate, 3),
                "fallback_rate": round(self.fallback_rate, 3),
            },
            "failure_breakdown": {
                "empty_audio": self.empty_audio_count,
                "empty_text": self.empty_text_count,
                "ctc_errors": self.ctc_failures,
                "dimension_mismatches": self.dimension_mismatches,
                "too_long_segments": self.too_long_segments,
            },
            "words": {
                "total_aligned": self.total_words_aligned,
                "low_confidence": self.low_confidence_words,
                "very_low_confidence": self.very_low_confidence_words,
                "timing_issues": self.zero_duration_words + self.negative_duration_words,
                "quality_score": round(self.word_quality_score, 3),
            },
            "processing": {
                "vad_splits": self.vad_split_count,
                "chunked_processing": self.chunked_processing_count,
            },
        }

    def log_summary(self, level: int = logging.INFO) -> None:
        """Log diagnostic summary."""
        summary = self.get_summary()

        logger.log(level, "=" * 50)
        logger.log(level, "ALIGNMENT DIAGNOSTICS SUMMARY")
        logger.log(level, "=" * 50)

        # Segments
        seg = summary["segments"]
        logger.log(level, f"Segments: {seg['successful']}/{seg['total']} successful "
                         f"({seg['success_rate']*100:.1f}%)")
        if seg["fallback"] > 0:
            logger.log(level, f"  Fallback alignments: {seg['fallback']} "
                             f"({seg['fallback_rate']*100:.1f}%)")

        # Failures
        fails = summary["failure_breakdown"]
        if any(v > 0 for v in fails.values()):
            logger.log(level, "Failure breakdown:")
            for reason, count in fails.items():
                if count > 0:
                    logger.log(level, f"  - {reason}: {count}")

        # Words
        words = summary["words"]
        logger.log(level, f"Words aligned: {words['total_aligned']} "
                         f"(quality score: {words['quality_score']*100:.1f}%)")
        if words["low_confidence"] > 0 or words["very_low_confidence"] > 0:
            logger.log(level, f"  Low confidence: {words['low_confidence']}, "
                             f"Very low: {words['very_low_confidence']}")

        # Processing info
        proc = summary["processing"]
        if proc["vad_splits"] > 0 or proc["chunked_processing"] > 0:
            logger.log(level, "Processing:")
            if proc["vad_splits"] > 0:
                logger.log(level, f"  VAD splits: {proc['vad_splits']}")
            if proc["chunked_processing"] > 0:
                logger.log(level, f"  Chunked processing: {proc['chunked_processing']} segments")

        logger.log(level, "=" * 50)

    def get_failure_report(self, max_failures: int = 10) -> str:
        """Get detailed failure report for debugging."""
        lines = ["FAILURE DETAILS (first {max_failures}):"]
        for detail in self.failure_details[:max_failures]:
            lines.append(f"  - [{detail['reason']}] {detail['segment_id']}: "
                        f"dur={detail['duration']}s, text='{detail['text_preview']}...' "
                        f"error={detail['error']}")
        if len(self.failure_details) > max_failures:
            lines.append(f"  ... and {len(self.failure_details) - max_failures} more failures")
        return "\n".join(lines)

# Korean CTC model requirements
# NOTE: The HuggingFace model card states 20s limit, but this was for training.
# For inference/alignment, we disable VAD splitting to maintain text-audio correspondence.
# For very long segments (>60s), we use chunked log probs approach (NeMo-style).
MAX_SEGMENT_DURATION = 60.0  # seconds (VAD disabled for < 60s segments)
CHUNKED_THRESHOLD = 60.0  # seconds (use chunked processing above this)
SAMPLE_RATE = 16000  # Hz (model requirement)

# Chunked processing parameters (NeMo-style buffered chunked streaming)
# Reference: NeMo NFA uses chunk_len_in_secs=1.6, total_buffer_in_secs=4.0
# We use larger chunks since we're doing offline processing (not streaming)
CHUNK_LEN_SEC = 20.0  # seconds per chunk (safe for Korean CTC model)
CHUNK_OVERLAP_SEC = 2.0  # seconds overlap between chunks (handles border distortions)

# VAD parameters for splitting long segments
VAD_FRAME_MS = 30  # Frame size for VAD in milliseconds
VAD_MIN_SILENCE_MS = 300  # Minimum silence duration for split point
VAD_ENERGY_THRESHOLD_RATIO = 0.1  # Energy threshold as ratio of max energy

# GPU memory management parameters
GPU_MEMORY_CLEANUP_INTERVAL = 10  # Clean GPU memory every N batches
GPU_MEMORY_CLEANUP_THRESHOLD = 0.8  # Clean if memory usage > 80%


def cleanup_gpu_memory(force: bool = False) -> None:
    """Clean up GPU memory to prevent OOM during long processing runs.

    This function should be called periodically during batch processing
    to prevent memory fragmentation and cache accumulation.

    Args:
        force: If True, always clean. If False, only clean if memory pressure is high.
    """
    if not HAS_TORCH or not torch.cuda.is_available():
        return

    try:
        if force:
            # Force garbage collection and empty CUDA cache
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            logger.debug("GPU memory cleaned (forced)")
        else:
            # Check memory pressure before cleaning
            allocated = torch.cuda.memory_allocated()
            reserved = torch.cuda.memory_reserved()
            if reserved > 0:
                usage_ratio = allocated / reserved
                if usage_ratio < GPU_MEMORY_CLEANUP_THRESHOLD:
                    # Memory is fragmented (lots of reserved but not allocated)
                    gc.collect()
                    torch.cuda.empty_cache()
                    logger.debug(f"GPU memory cleaned (fragmentation detected: {usage_ratio:.2%} usage)")
    except Exception as e:
        logger.debug(f"GPU memory cleanup failed: {e}")


def get_gpu_memory_info() -> Dict[str, float]:
    """Get current GPU memory usage information.

    Returns:
        Dictionary with memory info in GB
    """
    if not HAS_TORCH or not torch.cuda.is_available():
        return {}

    try:
        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        max_allocated = torch.cuda.max_memory_allocated() / (1024**3)
        return {
            "allocated_gb": round(allocated, 2),
            "reserved_gb": round(reserved, 2),
            "max_allocated_gb": round(max_allocated, 2),
        }
    except Exception:
        return {}


def find_silence_points(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> List[float]:
    """Find silence points in audio using energy-based VAD.

    Returns timestamps (in seconds) of silence centers, suitable for splitting.

    Args:
        audio: Audio array (mono)
        sample_rate: Sample rate in Hz

    Returns:
        List of timestamps (seconds) where silence occurs
    """
    frame_samples = int(sample_rate * VAD_FRAME_MS / 1000)
    min_silence_frames = int(VAD_MIN_SILENCE_MS / VAD_FRAME_MS)

    # Calculate frame energies
    num_frames = len(audio) // frame_samples
    if num_frames < 2:
        return []

    energies = []
    for i in range(num_frames):
        start = i * frame_samples
        end = start + frame_samples
        frame = audio[start:end]
        energy = np.sum(frame ** 2)
        energies.append(energy)

    energies = np.array(energies)
    if len(energies) == 0:
        return []

    # Calculate threshold
    max_energy = np.max(energies)
    if max_energy == 0:
        return []

    threshold = max_energy * VAD_ENERGY_THRESHOLD_RATIO

    # Find silence regions
    is_silence = energies < threshold
    silence_points = []

    # Find consecutive silence frames
    i = 0
    while i < len(is_silence):
        if is_silence[i]:
            start_idx = i
            while i < len(is_silence) and is_silence[i]:
                i += 1
            end_idx = i

            # If silence is long enough, mark the center as a split point
            if end_idx - start_idx >= min_silence_frames:
                center_idx = (start_idx + end_idx) // 2
                center_time = center_idx * VAD_FRAME_MS / 1000
                silence_points.append(center_time)
        else:
            i += 1

    return silence_points


def split_audio_at_silences(
    audio: np.ndarray,
    text: str,
    max_duration: float = MAX_SEGMENT_DURATION,
    sample_rate: int = SAMPLE_RATE,
) -> List[Tuple[np.ndarray, str, float]]:
    """Split long audio at silence points.

    When audio exceeds max_duration, find silence points and split there.
    Text is distributed proportionally across segments.

    Args:
        audio: Audio array (mono)
        text: Transcript text
        max_duration: Maximum segment duration in seconds
        sample_rate: Sample rate in Hz

    Returns:
        List of (audio_segment, text_segment, offset) tuples
    """
    audio_duration = len(audio) / sample_rate

    # If audio is short enough, return as-is
    if audio_duration <= max_duration:
        return [(audio, text, 0.0)]

    # Find silence points
    silence_points = find_silence_points(audio, sample_rate)

    if not silence_points:
        # No silence found, use uniform splitting
        return _uniform_split(audio, text, max_duration, sample_rate)

    # Filter silence points to get good split points
    split_times = [0.0]
    target_duration = max_duration * 0.8  # Aim for 80% of max to leave margin

    current_time = 0.0
    for silence_time in silence_points:
        if silence_time - current_time >= target_duration:
            split_times.append(silence_time)
            current_time = silence_time

    # Add end time
    split_times.append(audio_duration)

    # Handle case where we still have segments > max_duration
    final_split_times = [0.0]
    for i in range(1, len(split_times)):
        if split_times[i] - final_split_times[-1] > max_duration:
            # Force split at max_duration intervals
            num_subsplits = int(np.ceil((split_times[i] - final_split_times[-1]) / max_duration))
            subsplit_duration = (split_times[i] - final_split_times[-1]) / num_subsplits
            for j in range(1, num_subsplits):
                final_split_times.append(final_split_times[-1] + subsplit_duration)
        final_split_times.append(split_times[i])

    # Remove duplicates and sort
    final_split_times = sorted(set(final_split_times))

    # Split audio and distribute text
    words = text.split()
    total_words = len(words)
    segments = []

    for i in range(len(final_split_times) - 1):
        start_time = final_split_times[i]
        end_time = final_split_times[i + 1]

        start_sample = int(start_time * sample_rate)
        end_sample = int(end_time * sample_rate)

        audio_segment = audio[start_sample:end_sample]

        # Distribute words proportionally
        segment_ratio = (end_time - start_time) / audio_duration
        segment_words = max(1, int(total_words * segment_ratio))

        word_start = int(i * total_words / (len(final_split_times) - 1))
        word_end = min(word_start + segment_words, total_words)

        text_segment = " ".join(words[word_start:word_end])

        if len(audio_segment) > 0 and text_segment:
            segments.append((audio_segment, text_segment, start_time))

    return segments if segments else [(audio, text, 0.0)]


def _uniform_split(
    audio: np.ndarray,
    text: str,
    max_duration: float,
    sample_rate: int,
) -> List[Tuple[np.ndarray, str, float]]:
    """Uniform splitting when no silence points available."""
    audio_duration = len(audio) / sample_rate
    num_segments = int(np.ceil(audio_duration / max_duration))
    segment_duration = audio_duration / num_segments

    words = text.split()
    words_per_segment = max(1, len(words) // num_segments)

    segments = []
    for i in range(num_segments):
        start_time = i * segment_duration
        end_time = min((i + 1) * segment_duration, audio_duration)

        start_sample = int(start_time * sample_rate)
        end_sample = int(end_time * sample_rate)

        audio_segment = audio[start_sample:end_sample]

        word_start = i * words_per_segment
        word_end = min((i + 1) * words_per_segment, len(words))
        if i == num_segments - 1:
            word_end = len(words)

        text_segment = " ".join(words[word_start:word_end])

        if len(audio_segment) > 0 and text_segment:
            segments.append((audio_segment, text_segment, start_time))

    return segments


@dataclass
class SegmentInfo:
    """Information about a single segment from Phase 1 metadata.

    Enhanced to include original speaker ID for multi-speaker tracking.
    This allows word-level alignments to preserve original speaker identity
    even when multiple speakers are merged into SPEAKER_USER channel.
    """
    text: str
    start: float
    end: float
    speaker: str  # Role: SPEAKER_MAIN or SPEAKER_USER
    original_speaker_id: str = ""  # Original speaker ID from source data
    channel: int = 0  # 0=left (MAIN), 1=right (USER)

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class WordTimestamp:
    """A single word with timestamp and speaker information.

    Enhanced to support:
    - Basic Moshi format (backward compatible)
    - Extended format with original speaker ID for multi-speaker tracking
    - Channel information for stereo audio processing
    """
    text: str
    start: float
    end: float
    confidence: float = 1.0
    speaker: str = "SPEAKER_MAIN"  # Role: SPEAKER_MAIN or SPEAKER_USER
    original_speaker_id: str = ""  # Original speaker ID from source data
    channel: int = 0  # 0=left (MAIN), 1=right (USER)
    segment_index: int = -1  # Index of source segment (-1 if unknown)

    def to_moshi_format(self) -> list:
        """Convert to Moshi alignment format [word, [start, end], speaker].

        This is the basic format required for Moshi finetuning.
        """
        return [self.text, [round(self.start, 3), round(self.end, 3)], self.speaker]

    def to_extended_format(self) -> dict:
        """Convert to extended format with full speaker metadata.

        This format preserves original speaker identity for:
        - Multi-speaker diarization analysis
        - Speaker-specific fine-tuning
        - Quality analysis per speaker
        """
        return {
            "word": self.text,
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "confidence": round(self.confidence, 3),
            "speaker_role": self.speaker,
            "original_speaker_id": self.original_speaker_id,
            "channel": self.channel,
            "segment_index": self.segment_index,
        }

    def to_word(self) -> Word:
        """Convert to standard Word format."""
        return Word(
            text=self.text,
            start=self.start,
            end=self.end,
            confidence=self.confidence,
        )


class NFAAligner:
    """Word-level forced alignment using NeMo CTC models.

    Uses CTC-segmentation algorithm with pre-trained Korean CTC models
    to produce accurate word-level timestamps for pre-transcribed audio.

    Architecture:
        1. Load Korean CTC model (SungBeom/stt_kr_conformer_ctc_medium)
        2. For each segment from Phase 1 metadata:
           a. Extract audio segment (respecting 20s limit)
           b. Get CTC log probabilities from model
           c. Apply CTC-segmentation for word alignment
        3. Merge all word timestamps with global offsets

    Example usage:
        aligner = NFAAligner(config)

        result = aligner.align_conversation(
            audio_path=Path("audio/conv_001.flac"),
            metadata_path=Path("metadata/conv_001.json"),
            conversation_id="conv_001",
        )

        if result.is_valid:
            result.save_moshi_format(Path("audio/conv_001.json"))
    """

    def __init__(self, config: Optional[NFAConfig] = None):
        """Initialize the NFA aligner.

        Args:
            config: NFA configuration
        """
        self.config = config or NFAConfig()
        self.model = None
        self.vocabulary = None
        self.diagnostics = AlignmentDiagnostics()

        self._check_dependencies()

    def reset_diagnostics(self) -> None:
        """Reset diagnostics for a new alignment session."""
        self.diagnostics = AlignmentDiagnostics()

    def get_diagnostics(self) -> AlignmentDiagnostics:
        """Get current diagnostics."""
        return self.diagnostics

    def validate_audio(
        self,
        audio: np.ndarray,
        expected_channels: int = 2,
    ) -> Tuple[bool, str]:
        """Validate audio array for alignment.

        Args:
            audio: Audio array to validate
            expected_channels: Expected number of channels (2 for stereo)

        Returns:
            Tuple of (is_valid, error_message)
        """
        if audio is None:
            return False, "Audio is None"

        if len(audio) == 0:
            return False, "Audio is empty"

        if audio.ndim == 1:
            if expected_channels == 2:
                return False, "Expected stereo audio, got mono"
        elif audio.ndim == 2:
            if audio.shape[1] != expected_channels:
                return False, f"Expected {expected_channels} channels, got {audio.shape[1]}"
        else:
            return False, f"Invalid audio dimensions: {audio.ndim}"

        # Check for all zeros (silence)
        if np.abs(audio).max() < 1e-10:
            return False, "Audio appears to be silence (all zeros)"

        # Check duration
        if audio.ndim == 1:
            duration = len(audio) / SAMPLE_RATE
        else:
            duration = audio.shape[0] / SAMPLE_RATE

        if duration < 0.1:
            return False, f"Audio too short: {duration:.2f}s (min 0.1s)"

        return True, ""

    def validate_segment(
        self,
        segment: SegmentInfo,
        segment_id: str = "",
    ) -> Tuple[bool, str]:
        """Validate a segment before alignment.

        Args:
            segment: Segment information to validate
            segment_id: Identifier for logging

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check text
        if not segment.text or not segment.text.strip():
            self.diagnostics.record_failure(
                "empty_text", segment_id, segment.duration, "", "No text in segment"
            )
            return False, "Empty text"

        # Check timing
        if segment.duration <= 0:
            self.diagnostics.record_failure(
                "invalid_timing", segment_id, segment.duration,
                segment.text, "Non-positive duration"
            )
            return False, f"Invalid duration: {segment.duration}"

        if segment.start < 0:
            return False, f"Negative start time: {segment.start}"

        if segment.end <= segment.start:
            return False, f"End time ({segment.end}) <= start time ({segment.start})"

        # Check duration limit (warning only, will be split)
        if segment.duration > MAX_SEGMENT_DURATION:
            self.diagnostics.too_long_segments += 1
            logger.debug(
                f"Segment {segment_id} exceeds {MAX_SEGMENT_DURATION}s "
                f"({segment.duration:.1f}s), will be split"
            )

        return True, ""

    def _check_dependencies(self):
        """Check required dependencies are available."""
        missing = []
        if not HAS_TORCH:
            missing.append("torch")
        if not HAS_NEMO:
            missing.append("nemo_toolkit[asr]")
        if not HAS_CTC_SEG:
            missing.append("ctc-segmentation")
        if not HAS_SOUNDFILE:
            missing.append("soundfile")

        if missing:
            logger.warning(
                f"Missing dependencies: {', '.join(missing)}. "
                f"Install with: pip install {' '.join(missing)}"
            )

    def load_model(self) -> None:
        """Load the CTC acoustic model."""
        if self.model is not None:
            return

        if not HAS_NEMO:
            raise ImportError(
                "NeMo is required for NFA alignment. Install with: "
                "pip install nemo_toolkit[asr]"
            )

        if not HAS_CTC_SEG:
            raise ImportError(
                "ctc-segmentation is required. Install with: "
                "pip install ctc-segmentation"
            )

        logger.info(f"Loading CTC model: {self.config.acoustic_model}")

        try:
            model_path = Path(self.config.acoustic_model)

            # Determine loading method based on path type
            if model_path.exists():
                if model_path.is_file() and model_path.suffix == ".nemo":
                    # Local .nemo checkpoint file
                    logger.info(f"Loading from .nemo file: {model_path}")
                    self.model = EncDecCTCModel.restore_from(str(model_path))
                elif model_path.is_dir():
                    # Local directory (HuggingFace format or NeMo extracted)
                    # Check for .nemo file inside directory
                    nemo_files = list(model_path.glob("*.nemo"))
                    if nemo_files:
                        logger.info(f"Loading from .nemo file in directory: {nemo_files[0]}")
                        self.model = EncDecCTCModel.restore_from(str(nemo_files[0]))
                    else:
                        # Try as HuggingFace local path
                        logger.info(f"Loading from local HuggingFace directory: {model_path}")
                        self.model = EncDecCTCModel.from_pretrained(str(model_path))
                else:
                    raise ValueError(f"Unknown model path type: {model_path}")
            elif self.config.acoustic_model.startswith("/"):
                # Absolute path that doesn't exist
                raise FileNotFoundError(f"Model path does not exist: {model_path}")
            else:
                # HuggingFace model name (remote)
                logger.info(f"Loading from HuggingFace: {self.config.acoustic_model}")
                self.model = EncDecCTCModel.from_pretrained(self.config.acoustic_model)

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

        # Move to GPU if available
        if self.config.use_gpu and torch.cuda.is_available():
            self.model = self.model.cuda()
            logger.info("Model loaded on GPU")
        else:
            logger.info("Model loaded on CPU")

        self.model.eval()

        # Get vocabulary for CTC segmentation
        self.vocabulary = self._get_vocabulary()
        logger.info(f"Vocabulary size: {len(self.vocabulary)}")
        logger.info("CTC model loaded successfully")

    def _get_vocabulary(self) -> List[str]:
        """Get vocabulary list from the model."""
        vocab = None

        if hasattr(self.model, 'decoder') and hasattr(self.model.decoder, 'vocabulary'):
            vocab = list(self.model.decoder.vocabulary)
        elif hasattr(self.model, 'cfg') and hasattr(self.model.cfg, 'labels'):
            vocab = list(self.model.cfg.labels)
        elif hasattr(self.model, 'tokenizer'):
            # For models with tokenizer
            vocab = []
            for i in range(self.model.tokenizer.vocab_size):
                token = self.model.tokenizer.ids_to_tokens([i])[0]
                vocab.append(token)

        if vocab is None:
            raise ValueError("Cannot extract vocabulary from model")

        # Detect blank token index
        self.blank_idx = self._detect_blank_index(vocab)
        logger.info(f"Detected blank token at index {self.blank_idx}: '{vocab[self.blank_idx]}'")

        return vocab

    def _detect_blank_index(self, vocab: List[str]) -> int:
        """Detect the blank token index in the vocabulary.

        Common blank token representations:
        - "_" (CTC default)
        - "<blank>" (NeMo style)
        - "<ctc>" (some models)
        - Index 0 (common convention)
        """
        blank_candidates = ["<blank>", "_", "<ctc>", "<pad>", "▁"]  # ▁ is SentencePiece space

        for i, token in enumerate(vocab):
            if token in blank_candidates:
                return i
            # Check for common blank patterns
            if token.lower() in ["blank", "<blank>", "[blank]"]:
                return i

        # Default to index 0 if no explicit blank found
        logger.warning(f"Could not detect blank token, defaulting to index 0 (token: '{vocab[0]}')")
        return 0

    def _load_audio(self, audio_path: Path) -> Tuple[np.ndarray, int]:
        """Load audio file and return stereo array with sample rate."""
        audio, sr = sf.read(audio_path, dtype="float32")

        # Resample if needed
        if sr != SAMPLE_RATE:
            try:
                import librosa
                if audio.ndim == 1:
                    audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
                else:
                    audio = np.stack([
                        librosa.resample(audio[:, i], orig_sr=sr, target_sr=SAMPLE_RATE)
                        for i in range(audio.shape[1])
                    ], axis=1)
            except ImportError:
                from scipy import signal
                ratio = SAMPLE_RATE / sr
                if audio.ndim == 1:
                    audio = signal.resample(audio, int(len(audio) * ratio))
                else:
                    audio = np.stack([
                        signal.resample(audio[:, i], int(len(audio) * ratio))
                        for i in range(audio.shape[1])
                    ], axis=1)
            sr = SAMPLE_RATE

        return audio, sr

    def _extract_channel(
        self,
        audio: np.ndarray,
        channel_idx: int,
    ) -> np.ndarray:
        """Extract single channel from stereo audio."""
        if audio.ndim == 1:
            return audio
        return audio[:, channel_idx]

    def _extract_segment(
        self,
        audio: np.ndarray,
        start: float,
        end: float,
        sample_rate: int,
    ) -> np.ndarray:
        """Extract audio segment by timestamps."""
        start_sample = int(start * sample_rate)
        end_sample = int(end * sample_rate)

        # Clamp to valid range
        start_sample = max(0, start_sample)
        end_sample = min(len(audio), end_sample)

        return audio[start_sample:end_sample]

    def _get_ctc_log_probs(self, audio: np.ndarray) -> np.ndarray:
        """Get CTC log probabilities from the model (single audio)."""
        with torch.no_grad():
            # Prepare audio tensor
            audio_tensor = torch.from_numpy(audio).float()
            if audio_tensor.dim() == 1:
                audio_tensor = audio_tensor.unsqueeze(0)

            if self.config.use_gpu and torch.cuda.is_available():
                audio_tensor = audio_tensor.cuda()

            # Get audio length
            audio_len = torch.tensor([audio_tensor.shape[1]], dtype=torch.long)
            if self.config.use_gpu and torch.cuda.is_available():
                audio_len = audio_len.cuda()

            # Get log probabilities
            log_probs, encoded_len, _ = self.model.forward(
                input_signal=audio_tensor,
                input_signal_length=audio_len,
            )

            # Convert to numpy
            log_probs = log_probs.cpu().numpy()[0]

        return log_probs

    def _get_ctc_log_probs_batched(
        self,
        audio_list: List[np.ndarray],
        batch_size: int = 32,
    ) -> List[np.ndarray]:
        """Get CTC log probabilities for a batch of audio segments.

        This method processes multiple audio segments in parallel on GPU
        for significantly better hardware utilization.

        Args:
            audio_list: List of audio arrays to process
            batch_size: Number of segments to process at once (default: 32)

        Returns:
            List of log probability arrays (one per input audio)
        """
        if not audio_list:
            return []

        all_log_probs = []
        num_batches = (len(audio_list) + batch_size - 1) // batch_size

        # Process in batches with periodic GPU memory cleanup
        for batch_idx, batch_start in enumerate(range(0, len(audio_list), batch_size)):
            batch_end = min(batch_start + batch_size, len(audio_list))
            batch_audio = audio_list[batch_start:batch_end]

            # Find max length in this batch for padding
            max_len = max(len(audio) for audio in batch_audio)

            # Pad all audio to same length
            padded_batch = []
            audio_lengths = []
            for audio in batch_audio:
                pad_len = max_len - len(audio)
                if pad_len > 0:
                    padded = np.pad(audio, (0, pad_len), mode='constant', constant_values=0)
                else:
                    padded = audio
                padded_batch.append(padded)
                audio_lengths.append(len(audio))

            with torch.no_grad():
                # Create batched tensor [B, T]
                batch_tensor = torch.from_numpy(np.stack(padded_batch)).float()

                if self.config.use_gpu and torch.cuda.is_available():
                    batch_tensor = batch_tensor.cuda()

                # Create length tensor
                len_tensor = torch.tensor(audio_lengths, dtype=torch.long)
                if self.config.use_gpu and torch.cuda.is_available():
                    len_tensor = len_tensor.cuda()

                # Forward pass on entire batch
                log_probs_batch, encoded_lens, _ = self.model.forward(
                    input_signal=batch_tensor,
                    input_signal_length=len_tensor,
                )

                # Convert to numpy and extract valid portions
                log_probs_batch = log_probs_batch.cpu().numpy()
                encoded_lens = encoded_lens.cpu().numpy()

                for i, enc_len in enumerate(encoded_lens):
                    # Extract only valid frames (not padding)
                    all_log_probs.append(log_probs_batch[i, :enc_len, :])

            # Explicit cleanup of GPU tensors
            del batch_tensor, len_tensor, log_probs_batch
            if 'encoded_lens' in dir():
                del encoded_lens

            # Periodic GPU memory cleanup to prevent fragmentation
            if (batch_idx + 1) % GPU_MEMORY_CLEANUP_INTERVAL == 0:
                cleanup_gpu_memory(force=True)
                if logger.isEnabledFor(logging.DEBUG):
                    mem_info = get_gpu_memory_info()
                    logger.debug(
                        f"Batch {batch_idx + 1}/{num_batches} - "
                        f"GPU: {mem_info.get('allocated_gb', 0):.1f}GB allocated, "
                        f"{mem_info.get('reserved_gb', 0):.1f}GB reserved"
                    )
            else:
                # Light cleanup between batches
                cleanup_gpu_memory(force=False)

        # Final cleanup after all batches
        cleanup_gpu_memory(force=True)

        return all_log_probs

    def _get_ctc_log_probs_chunked(
        self,
        audio: np.ndarray,
        chunk_len_sec: float = CHUNK_LEN_SEC,
        overlap_sec: float = CHUNK_OVERLAP_SEC,
    ) -> np.ndarray:
        """Get CTC log probabilities for long audio using chunked processing.

        This method handles audio longer than CHUNKED_THRESHOLD (60s) by:
        1. Splitting audio into overlapping chunks
        2. Getting log probs for each chunk
        3. Stitching log probs together (removing overlap regions)
        4. Returning concatenated log probs for full-sequence alignment

        This is based on NeMo's "buffered chunked streaming" approach which
        handles "CTC activation distortion at partition borders" by using
        overlapping partitions.

        Reference:
            - NeMo NFA: use_buffered_chunked_streaming=True
            - ctc_segmentation.get_partitions() for overlap handling

        Args:
            audio: Audio array for the full segment (may be > 60s)
            chunk_len_sec: Length of each chunk in seconds
            overlap_sec: Overlap between chunks in seconds

        Returns:
            Concatenated log probabilities array [T_total, V]
        """
        audio_duration = len(audio) / SAMPLE_RATE

        # If audio is short enough, use regular processing
        if audio_duration <= CHUNKED_THRESHOLD:
            return self._get_ctc_log_probs(audio)

        logger.debug(
            f"Using chunked log probs for {audio_duration:.1f}s audio "
            f"(chunk={chunk_len_sec}s, overlap={overlap_sec}s)"
        )

        # Calculate chunk boundaries
        chunk_samples = int(chunk_len_sec * SAMPLE_RATE)
        overlap_samples = int(overlap_sec * SAMPLE_RATE)
        stride_samples = chunk_samples - overlap_samples

        # Generate chunk boundaries
        chunks = []
        chunk_start = 0
        while chunk_start < len(audio):
            chunk_end = min(chunk_start + chunk_samples, len(audio))
            chunks.append((chunk_start, chunk_end))

            # Move to next chunk (with overlap)
            chunk_start += stride_samples

            # Stop if remaining audio is too short
            if chunk_end >= len(audio):
                break

        logger.debug(f"Split into {len(chunks)} chunks")

        # Get log probs for each chunk
        chunk_log_probs = []
        chunk_frame_counts = []

        for i, (start_sample, end_sample) in enumerate(chunks):
            chunk_audio = audio[start_sample:end_sample]

            # Get log probs for this chunk
            log_probs = self._get_ctc_log_probs(chunk_audio)
            chunk_log_probs.append(log_probs)
            chunk_frame_counts.append(log_probs.shape[0])

            logger.debug(
                f"Chunk {i}: samples [{start_sample}:{end_sample}], "
                f"frames={log_probs.shape[0]}"
            )

            # Cleanup GPU memory after each chunk for very long audio
            if (i + 1) % 5 == 0:  # Every 5 chunks
                cleanup_gpu_memory(force=False)

        # Calculate overlap frames to remove
        # Estimate frame rate from first chunk
        if len(chunks) > 0 and chunk_frame_counts[0] > 0:
            first_chunk_duration = (chunks[0][1] - chunks[0][0]) / SAMPLE_RATE
            frame_rate = chunk_frame_counts[0] / first_chunk_duration
            overlap_frames = int(overlap_sec * frame_rate)
        else:
            # Fallback: assume typical Conformer downsampling (~50 Hz)
            overlap_frames = int(overlap_sec * 50)

        # Stitch log probs together
        stitched_log_probs = []

        for i, log_probs in enumerate(chunk_log_probs):
            num_frames = log_probs.shape[0]

            if len(chunks) == 1:
                # Only one chunk - use all frames
                stitched_log_probs.append(log_probs)
            elif i == 0:
                # First chunk: remove overlap from end
                # Keep frames [0 : num_frames - overlap_frames/2]
                end_cut = min(overlap_frames // 2, num_frames // 4)
                stitched_log_probs.append(log_probs[:-end_cut] if end_cut > 0 else log_probs)
            elif i == len(chunks) - 1:
                # Last chunk: remove overlap from start
                # Keep frames [overlap_frames/2 : end]
                start_cut = min(overlap_frames // 2, num_frames // 4)
                stitched_log_probs.append(log_probs[start_cut:])
            else:
                # Middle chunks: remove overlap from both ends
                # Keep frames [overlap_frames/2 : num_frames - overlap_frames/2]
                start_cut = min(overlap_frames // 2, num_frames // 4)
                end_cut = min(overlap_frames // 2, num_frames // 4)
                stitched_log_probs.append(log_probs[start_cut:-end_cut] if end_cut > 0 else log_probs[start_cut:])

        # Concatenate all chunks
        if stitched_log_probs:
            result = np.concatenate(stitched_log_probs, axis=0)
            logger.debug(
                f"Stitched log probs: {result.shape[0]} frames "
                f"(from {sum(chunk_frame_counts)} original frames)"
            )
            return result
        else:
            logger.warning("No log probs generated from chunked processing")
            return np.array([])

    def _prepare_text_for_alignment(self, text: str) -> str:
        """Prepare text for CTC alignment."""
        # Remove extra whitespace
        text = " ".join(text.split())

        # For Korean, we might need to handle spacing differently
        # Keep the text as-is for now
        return text.strip()

    def _align_segment_ctc(
        self,
        audio: np.ndarray,
        text: str,
        segment_offset: float,
        speaker: str,
        original_speaker_id: str = "",
        channel: int = 0,
        segment_index: int = -1,
    ) -> List[WordTimestamp]:
        """Align a single segment using CTC-segmentation.

        Args:
            audio: Audio array for this segment
            text: Transcript text for this segment
            segment_offset: Global time offset for this segment
            speaker: Speaker role (SPEAKER_MAIN or SPEAKER_USER)
            original_speaker_id: Original speaker ID from source data
            channel: Audio channel (0=left, 1=right)
            segment_index: Index of source segment

        Returns:
            List of WordTimestamp objects with full speaker metadata
        """
        segment_id = f"{speaker}_seg{segment_index}"
        self.diagnostics.total_segments += 1

        # Validate audio
        if len(audio) < 100:  # Too short
            self.diagnostics.record_failure(
                "empty_audio", segment_id, len(audio) / SAMPLE_RATE,
                text, "Audio too short (<100 samples)"
            )
            return []

        text = self._prepare_text_for_alignment(text)
        if not text:
            self.diagnostics.record_failure(
                "empty_text", segment_id, len(audio) / SAMPLE_RATE,
                "", "Empty text after preprocessing"
            )
            return []

        try:
            # Get CTC log probabilities
            # For long audio (> CHUNKED_THRESHOLD), use chunked processing
            # This maintains text-audio correspondence without VAD splitting
            audio_duration = len(audio) / SAMPLE_RATE
            if audio_duration > CHUNKED_THRESHOLD:
                logger.info(
                    f"[{segment_id}] Using chunked log probs for long segment "
                    f"({audio_duration:.1f}s > {CHUNKED_THRESHOLD}s threshold)"
                )
                log_probs = self._get_ctc_log_probs_chunked(audio)
                self.diagnostics.chunked_processing_count += 1
            else:
                log_probs = self._get_ctc_log_probs(audio)

            # Get vocabulary
            vocab = self.vocabulary

            # Split text into words (keep for later word extraction)
            words = text.split()
            if not words:
                self.diagnostics.record_failure(
                    "empty_text", segment_id, len(audio) / SAMPLE_RATE,
                    text, "No words after split"
                )
                return []

            # Configure CTC segmentation
            config = ctc_segmentation.CtcSegmentationParameters()
            config.char_list = vocab
            config.blank = getattr(self, 'blank_idx', 0)  # Use detected blank index

            # Calculate frame duration (audio_duration already calculated above)
            num_frames = log_probs.shape[0]
            if num_frames == 0:
                logger.warning(f"[{segment_id}] Empty log_probs returned from model")
                self.diagnostics.record_failure(
                    "ctc_error", segment_id, audio_duration,
                    text, "Empty log_probs from model"
                )
                return self._fallback_uniform_alignment(
                    text, audio, segment_offset, speaker,
                    original_speaker_id, channel, segment_index
                )
            config.index_duration = audio_duration / num_frames

            # Use ctc-segmentation's prepare_text which handles vocabulary filtering
            # This filters out characters not in vocabulary instead of raising errors
            if prepare_text is not None:
                ground_truth_mat, utt_begin_indices = prepare_text(config, text, vocab)
            else:
                # Fallback to custom prepare if library function not available
                ground_truth_mat, utt_begin_indices = self._prepare_ground_truth(
                    words, vocab
                )

            # Validate dimensions before calling CTC segmentation
            if ground_truth_mat.shape[1] != log_probs.shape[1]:
                logger.warning(
                    f"[{segment_id}] Vocabulary dimension mismatch: "
                    f"ground_truth={ground_truth_mat.shape[1]}, log_probs={log_probs.shape[1]}. "
                    f"Falling back to uniform alignment."
                )
                self.diagnostics.record_failure(
                    "dimension_mismatch", segment_id, audio_duration,
                    text, f"ground_truth={ground_truth_mat.shape[1]} vs log_probs={log_probs.shape[1]}"
                )
                return self._fallback_uniform_alignment(
                    text, audio, segment_offset, speaker,
                    original_speaker_id, channel, segment_index
                )

            # Run CTC segmentation
            timings, char_probs, state_list = ctc_segmentation.ctc_segmentation(
                config, log_probs, ground_truth_mat
            )

            # Extract word-level timestamps with full speaker metadata
            word_timestamps = []

            # When using prepare_text, utt_begin_indices contains indices for each word
            # We need to map back to our word list
            num_segments = len(utt_begin_indices) - 1

            for i in range(min(num_segments, len(words))):
                start_idx = utt_begin_indices[i]
                end_idx = utt_begin_indices[i + 1] - 1 if i + 1 < len(utt_begin_indices) else len(timings) - 1

                if start_idx < len(timings) and end_idx < len(timings) and end_idx >= start_idx:
                    start_time = timings[start_idx] + segment_offset
                    end_time = timings[end_idx] + segment_offset

                    # Get confidence from character probabilities
                    if start_idx < len(char_probs):
                        confidence = float(np.exp(char_probs[start_idx]))
                    else:
                        confidence = 0.5  # Default confidence

                    # Ensure valid timing
                    if end_time <= start_time:
                        end_time = start_time + 0.01  # Minimum duration

                    duration = end_time - start_time

                    # Filter by confidence and duration
                    if (confidence >= self.config.min_confidence and
                        duration <= self.config.max_word_duration and
                        duration > 0):
                        word = WordTimestamp(
                            text=words[i] if i < len(words) else "",
                            start=start_time,
                            end=end_time,
                            confidence=confidence,
                            speaker=speaker,
                            original_speaker_id=original_speaker_id,
                            channel=channel,
                            segment_index=segment_index,
                        )
                        word_timestamps.append(word)
                        self.diagnostics.record_word(word)
                    else:
                        # Track filtered words
                        if duration > self.config.max_word_duration:
                            self.diagnostics.excessively_long_words += 1

            # Mark as successful
            self.diagnostics.successful_segments += 1
            return word_timestamps

        except Exception as e:
            error_msg = str(e)
            logger.warning(f"[{segment_id}] CTC segmentation failed: {error_msg}")
            self.diagnostics.record_failure(
                "ctc_error", segment_id, len(audio) / SAMPLE_RATE,
                text, error_msg
            )
            return self._fallback_uniform_alignment(
                text, audio, segment_offset, speaker,
                original_speaker_id, channel, segment_index
            )

    def _prepare_ground_truth(
        self,
        words: List[str],
        vocab: List[str],
    ) -> Tuple[np.ndarray, List[int]]:
        """Prepare ground truth matrix for CTC segmentation."""
        # Create character to index mapping
        char2idx = {c: i for i, c in enumerate(vocab)}
        blank_idx = 0

        # Build character sequence with spaces between words
        char_sequence = []
        utt_begin_indices = [0]

        for word in words:
            for char in word:
                if char in char2idx:
                    char_sequence.append(char2idx[char])
                else:
                    # Handle unknown characters
                    char_sequence.append(blank_idx)

            # Add space between words
            if ' ' in char2idx:
                char_sequence.append(char2idx[' '])

            utt_begin_indices.append(len(char_sequence))

        # Create ground truth matrix (T x V)
        num_chars = len(char_sequence)
        num_vocab = len(vocab)
        ground_truth = np.full((num_chars, num_vocab), -np.inf)

        for i, idx in enumerate(char_sequence):
            ground_truth[i, idx] = 0.0

        return ground_truth, utt_begin_indices

    def _fallback_uniform_alignment(
        self,
        text: str,
        audio: np.ndarray,
        segment_offset: float,
        speaker: str,
        original_speaker_id: str = "",
        channel: int = 0,
        segment_index: int = -1,
    ) -> List[WordTimestamp]:
        """Fallback to uniform word distribution when CTC fails.

        Records fallback usage in diagnostics for monitoring.
        """
        words = text.split()
        if not words:
            return []

        # Track fallback usage
        self.diagnostics.fallback_segments += 1

        audio_duration = len(audio) / SAMPLE_RATE
        word_duration = audio_duration / len(words)

        timestamps = []
        for i, word in enumerate(words):
            start_time = segment_offset + i * word_duration
            end_time = segment_offset + (i + 1) * word_duration

            word_ts = WordTimestamp(
                text=word,
                start=start_time,
                end=end_time,
                confidence=0.5,  # Lower confidence for fallback
                speaker=speaker,
                original_speaker_id=original_speaker_id,
                channel=channel,
                segment_index=segment_index,
            )
            timestamps.append(word_ts)
            self.diagnostics.record_word(word_ts)

        logger.debug(
            f"Fallback alignment for {speaker}_seg{segment_index}: "
            f"{len(words)} words, {audio_duration:.2f}s"
        )

        return timestamps

    def _load_phase1_metadata(self, metadata_path: Path) -> Dict[str, List[SegmentInfo]]:
        """Load Phase 1 metadata and extract segment info by speaker.

        Preserves original speaker IDs from Phase 1 for word-level tracking.
        """
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        segments_by_speaker = {
            "SPEAKER_MAIN": [],
            "SPEAKER_USER": [],
        }

        segments = metadata.get("segments", {})

        # Get main speaker ID as fallback
        speakers_info = metadata.get("speakers", {})
        main_speaker_id = speakers_info.get("main", {}).get("id", "unknown_main")

        # Load SPEAKER_MAIN segments
        for idx, seg in enumerate(segments.get("main", [])):
            segments_by_speaker["SPEAKER_MAIN"].append(SegmentInfo(
                text=seg.get("text", ""),
                start=seg.get("start", 0.0),
                end=seg.get("end", 0.0),
                speaker="SPEAKER_MAIN",
                original_speaker_id=seg.get("original_speaker_id", main_speaker_id),
                channel=0,  # Left channel
            ))

        # Load SPEAKER_USER segments
        # User channel may have multiple speakers merged
        for idx, seg in enumerate(segments.get("user", [])):
            segments_by_speaker["SPEAKER_USER"].append(SegmentInfo(
                text=seg.get("text", ""),
                start=seg.get("start", 0.0),
                end=seg.get("end", 0.0),
                speaker="SPEAKER_USER",
                original_speaker_id=seg.get("original_speaker_id", "unknown_user"),
                channel=1,  # Right channel
            ))

        return segments_by_speaker

    def _process_speaker_channel(
        self,
        audio: np.ndarray,
        segments: List[SegmentInfo],
        speaker: str,
    ) -> List[WordTimestamp]:
        """Process all segments for a single speaker channel.

        Preserves original speaker ID for each word, allowing tracking of
        individual speakers even when multiple are merged into one channel.
        """
        # Use batched processing if available and batch_size > 1
        batch_size = getattr(self.config, 'batch_size', 1)
        if batch_size > 1 and HAS_TORCH:
            return self._process_speaker_channel_batched(audio, segments, speaker, batch_size)

        all_words = []

        for seg_idx, segment in enumerate(segments):
            if not segment.text or not segment.text.strip():
                continue

            # Extract segment audio - _align_segment_ctc will handle long segments
            # using chunked log probs (NeMo-style) to maintain text-audio correspondence
            segment_audio = self._extract_segment(
                audio, segment.start, segment.end, SAMPLE_RATE
            )

            words = self._align_segment_ctc(
                segment_audio,
                segment.text,
                segment.start,
                speaker,
                original_speaker_id=segment.original_speaker_id,
                channel=segment.channel,
                segment_index=seg_idx,
            )
            all_words.extend(words)

        return all_words

    def _process_speaker_channel_batched(
        self,
        audio: np.ndarray,
        segments: List[SegmentInfo],
        speaker: str,
        batch_size: int = 32,
    ) -> List[WordTimestamp]:
        """Process all segments for a speaker channel using batched GPU inference.

        This method collects multiple segments and processes them together
        for significantly better GPU utilization (10-50x speedup).

        Args:
            audio: Full audio array for this channel
            segments: List of segment info from Phase 1
            speaker: Speaker role (SPEAKER_MAIN or SPEAKER_USER)
            batch_size: Number of segments to process in parallel

        Returns:
            List of WordTimestamp objects with full speaker metadata
        """
        if not segments:
            return []

        all_words = []

        # First pass: separate long segments from short segments
        # Long segments (> CHUNKED_THRESHOLD) use chunked processing via _align_segment_ctc
        # Short segments can be batched together for better GPU utilization
        short_segment_data = []  # [(audio_segment, text, offset, segment_info, seg_idx), ...]
        long_segments = []  # [(segment, seg_idx), ...]

        for seg_idx, segment in enumerate(segments):
            if not segment.text or not segment.text.strip():
                continue

            text = self._prepare_text_for_alignment(segment.text)
            if not text:
                continue

            # Separate long segments for chunked processing
            if segment.duration > CHUNKED_THRESHOLD:
                long_segments.append((segment, seg_idx))
            else:
                segment_audio = self._extract_segment(
                    audio, segment.start, segment.end, SAMPLE_RATE
                )
                if len(segment_audio) > 100:
                    short_segment_data.append((
                        segment_audio, text, segment.start, segment, seg_idx
                    ))

        # Process long segments individually using chunked log probs
        # This maintains text-audio correspondence without VAD splitting
        if long_segments:
            logger.debug(
                f"Processing {len(long_segments)} long segments using chunked log probs"
            )
            for segment, seg_idx in long_segments:
                segment_audio = self._extract_segment(
                    audio, segment.start, segment.end, SAMPLE_RATE
                )
                words = self._align_segment_ctc(
                    segment_audio,
                    segment.text,
                    segment.start,
                    speaker,
                    original_speaker_id=segment.original_speaker_id,
                    channel=segment.channel,
                    segment_index=seg_idx,
                )
                all_words.extend(words)

        # Process short segments in batches
        if not short_segment_data:
            return all_words

        segment_data = short_segment_data

        # Extract just the audio arrays for batched CTC processing
        audio_list = [sd[0] for sd in segment_data]

        # Get CTC log probs for all segments in batches
        logger.debug(f"Processing {len(audio_list)} segments in batches of {batch_size}")
        all_log_probs = self._get_ctc_log_probs_batched(audio_list, batch_size)

        # Second pass: run CTC segmentation on each segment
        all_words = []
        vocab = self.vocabulary

        for i, (seg_audio, text, offset, segment, seg_idx) in enumerate(segment_data):
            try:
                log_probs = all_log_probs[i]
                words = text.split()

                if not words:
                    continue

                # Build character-level ground truth
                ground_truth_mat, utt_begin_indices = self._prepare_ground_truth(
                    words, vocab
                )

                # Configure CTC segmentation
                config = ctc_segmentation.CtcSegmentationParameters()
                config.char_list = vocab
                config.blank = 0

                # Calculate frame duration
                audio_duration = len(seg_audio) / SAMPLE_RATE
                num_frames = log_probs.shape[0]
                config.index_duration = audio_duration / num_frames if num_frames > 0 else 0.01

                # Run CTC segmentation
                timings, char_probs, state_list = ctc_segmentation.ctc_segmentation(
                    config, log_probs, ground_truth_mat
                )

                # Extract word-level timestamps
                for j, word in enumerate(words):
                    if j < len(utt_begin_indices) - 1:
                        start_idx = utt_begin_indices[j]
                        end_idx = utt_begin_indices[j + 1] - 1

                        if start_idx < len(timings) and end_idx < len(timings):
                            start_time = timings[start_idx] + offset
                            end_time = timings[end_idx] + offset

                            # Get confidence
                            if start_idx < len(char_probs):
                                confidence = float(np.exp(char_probs[start_idx]))
                            else:
                                confidence = 1.0

                            # Filter by confidence and duration
                            duration = end_time - start_time
                            if (confidence >= self.config.min_confidence and
                                duration <= self.config.max_word_duration and
                                duration > 0):
                                all_words.append(WordTimestamp(
                                    text=word,
                                    start=start_time,
                                    end=end_time,
                                    confidence=confidence,
                                    speaker=speaker,
                                    original_speaker_id=segment.original_speaker_id,
                                    channel=segment.channel,
                                    segment_index=seg_idx,
                                ))

            except Exception as e:
                logger.debug(f"CTC segmentation failed for segment {i}: {e}")
                # Fallback to uniform alignment for this segment
                fallback_words = self._fallback_uniform_alignment(
                    text, seg_audio, offset, speaker,
                    segment.original_speaker_id, segment.channel, seg_idx
                )
                all_words.extend(fallback_words)

        return all_words

    def _split_long_segment(
        self,
        segment: SegmentInfo,
        audio: np.ndarray,
    ) -> List[Tuple[np.ndarray, str, float]]:
        """Split a long segment into sub-segments for batched processing.

        Uses VAD-based splitting to find natural break points at silences.
        Falls back to uniform splitting if no silences are found.
        Tracks VAD splits in diagnostics.

        Returns:
            List of (audio_array, text, time_offset) tuples
        """
        if not segment.text or not segment.text.strip():
            return []

        # Extract the full segment audio
        segment_audio = self._extract_segment(
            audio, segment.start, segment.end, SAMPLE_RATE
        )

        if len(segment_audio) < 100:
            return []

        # Use VAD-based splitting for natural break points
        sub_segments = split_audio_at_silences(
            segment_audio,
            segment.text,
            max_duration=MAX_SEGMENT_DURATION,
            sample_rate=SAMPLE_RATE,
        )

        # Track VAD splits in diagnostics
        if len(sub_segments) > 1:
            self.diagnostics.vad_split_count += len(sub_segments) - 1
            logger.debug(
                f"VAD split segment ({segment.duration:.1f}s) into "
                f"{len(sub_segments)} sub-segments"
            )

        # Adjust offsets to be relative to the original audio
        adjusted_segments = []
        for audio_seg, text_seg, offset in sub_segments:
            global_offset = segment.start + offset
            adjusted_segments.append((audio_seg, text_seg, global_offset))

        return adjusted_segments

    def _process_long_segment(
        self,
        audio: np.ndarray,
        segment: SegmentInfo,
        speaker: str,
        segment_index: int = -1,
    ) -> List[WordTimestamp]:
        """Process a segment longer than MAX_SEGMENT_DURATION by splitting.

        Uses VAD-based splitting to find natural break points at silences.
        Preserves original speaker ID across all sub-segments.
        """
        if not segment.text or not segment.text.strip():
            return []

        # Use the VAD-based splitting method
        sub_segments = self._split_long_segment(segment, audio)

        if not sub_segments:
            return []

        all_words = []

        for subseg_audio, subseg_text, global_offset in sub_segments:
            if len(subseg_audio) < 100 or not subseg_text:
                continue

            # Align with full speaker metadata
            word_timestamps = self._align_segment_ctc(
                subseg_audio,
                subseg_text,
                global_offset,
                speaker,
                original_speaker_id=segment.original_speaker_id,
                channel=segment.channel,
                segment_index=segment_index,
            )
            all_words.extend(word_timestamps)

        return all_words

    def align_conversation(
        self,
        audio_path: Path,
        metadata_path: Path,
        conversation_id: str,
        log_diagnostics: bool = True,
    ) -> "ConversationAlignment":
        """Align a full conversation using Phase 1 metadata.

        This is the main entry point for Phase 2 processing.

        Args:
            audio_path: Path to stereo FLAC file
            metadata_path: Path to Phase 1 metadata JSON
            conversation_id: Conversation identifier
            log_diagnostics: Whether to log diagnostics summary after alignment

        Returns:
            ConversationAlignment with word timestamps for both speakers
        """
        # Reset diagnostics for this conversation
        self.reset_diagnostics()

        try:
            self.load_model()

            # Load audio
            audio, sr = self._load_audio(audio_path)

            # Validate audio
            is_valid, error_msg = self.validate_audio(audio, expected_channels=2)
            if not is_valid:
                logger.warning(f"[{conversation_id}] Audio validation failed: {error_msg}")
                return ConversationAlignment(
                    conversation_id=conversation_id,
                    is_valid=False,
                    error=error_msg,
                )

            if audio.ndim == 1:
                logger.warning(f"[{conversation_id}] Mono audio")
                return ConversationAlignment(
                    conversation_id=conversation_id,
                    is_valid=False,
                    error="Audio is mono, expected stereo",
                )

            # Load Phase 1 metadata
            segments_by_speaker = self._load_phase1_metadata(metadata_path)

            # Validate segments
            main_segments = segments_by_speaker["SPEAKER_MAIN"]
            user_segments = segments_by_speaker["SPEAKER_USER"]

            logger.debug(
                f"[{conversation_id}] Processing: "
                f"{len(main_segments)} MAIN segments, {len(user_segments)} USER segments"
            )

            # Process SPEAKER_MAIN (left channel)
            logger.debug(f"[{conversation_id}] Processing SPEAKER_MAIN")
            main_audio = self._extract_channel(audio, 0)
            main_words = self._process_speaker_channel(
                main_audio,
                main_segments,
                "SPEAKER_MAIN",
            )

            # Process SPEAKER_USER (right channel)
            logger.debug(f"[{conversation_id}] Processing SPEAKER_USER")
            user_audio = self._extract_channel(audio, 1)
            user_words = self._process_speaker_channel(
                user_audio,
                user_segments,
                "SPEAKER_USER",
            )

            # Log diagnostics if enabled
            if log_diagnostics and logger.isEnabledFor(logging.DEBUG):
                self.diagnostics.log_summary(logging.DEBUG)

            # Create alignment result with diagnostics
            result = ConversationAlignment(
                conversation_id=conversation_id,
                main_words=main_words,
                user_words=user_words,
                is_valid=True,
                diagnostics=self.diagnostics,
            )

            # Log summary for significant issues
            diag = self.diagnostics
            if diag.failed_segments > 0 or diag.fallback_segments > diag.total_segments * 0.1:
                logger.info(
                    f"[{conversation_id}] Alignment completed with issues: "
                    f"{diag.successful_segments}/{diag.total_segments} successful, "
                    f"{diag.fallback_segments} fallbacks, {diag.failed_segments} failed"
                )

            return result

        except Exception as e:
            logger.error(f"[{conversation_id}] Error aligning: {e}")

            # Log diagnostics on error
            if self.diagnostics.total_segments > 0:
                logger.error(self.diagnostics.get_failure_report(5))

            return ConversationAlignment(
                conversation_id=conversation_id,
                is_valid=False,
                error=str(e),
            )

    def align_stereo(
        self,
        audio_path: Path,
        conversation_id: str,
        main_transcript: Optional[str] = None,
        user_transcript: Optional[str] = None,
        metadata_path: Optional[Path] = None,
    ) -> AlignmentResult:
        """Align stereo audio file (compatibility method).

        This method provides compatibility with the Phase 2 orchestrator.
        """
        if metadata_path and metadata_path.exists():
            result = self.align_conversation(
                audio_path, metadata_path, conversation_id
            )

            # Convert to AlignmentResult format
            main_alignment = WordAlignment(
                speaker="SPEAKER_MAIN",
                words=[w.to_word() for w in result.main_words],
                total_duration=sum(w.end - w.start for w in result.main_words),
                confidence=np.mean([w.confidence for w in result.main_words]) if result.main_words else 0.0,
            )

            user_alignment = WordAlignment(
                speaker="SPEAKER_USER",
                words=[w.to_word() for w in result.user_words],
                total_duration=sum(w.end - w.start for w in result.user_words),
                confidence=np.mean([w.confidence for w in result.user_words]) if result.user_words else 0.0,
            )

            return AlignmentResult(
                conversation_id=conversation_id,
                main_alignment=main_alignment,
                user_alignment=user_alignment,
                is_valid=result.is_valid,
                error=result.error,
                quality_score=result.quality_score,
            )
        else:
            # Use provided transcripts
            return self._align_with_transcripts(
                audio_path, conversation_id,
                main_transcript or "", user_transcript or ""
            )

    def _align_with_transcripts(
        self,
        audio_path: Path,
        conversation_id: str,
        main_transcript: str,
        user_transcript: str,
    ) -> AlignmentResult:
        """Align using directly provided transcripts."""
        try:
            self.load_model()
            audio, sr = self._load_audio(audio_path)

            if audio.ndim == 1:
                # Mono - only process main
                main_audio = audio
                user_audio = np.zeros_like(audio)
            else:
                main_audio = self._extract_channel(audio, 0)
                user_audio = self._extract_channel(audio, 1)

            # Process main speaker
            main_words = []
            if main_transcript.strip():
                main_words = self._align_segment_ctc(
                    main_audio, main_transcript, 0.0, "SPEAKER_MAIN"
                )

            # Process user speaker
            user_words = []
            if user_transcript.strip():
                user_words = self._align_segment_ctc(
                    user_audio, user_transcript, 0.0, "SPEAKER_USER"
                )

            main_alignment = WordAlignment(
                speaker="SPEAKER_MAIN",
                words=[w.to_word() for w in main_words],
            )
            user_alignment = WordAlignment(
                speaker="SPEAKER_USER",
                words=[w.to_word() for w in user_words],
            )

            return AlignmentResult(
                conversation_id=conversation_id,
                main_alignment=main_alignment,
                user_alignment=user_alignment,
                is_valid=True,
            )

        except Exception as e:
            return AlignmentResult(
                conversation_id=conversation_id,
                is_valid=False,
                error=str(e),
            )


@dataclass
class ConversationAlignment:
    """Complete alignment for a conversation (both speakers)."""
    conversation_id: str
    main_words: List[WordTimestamp] = field(default_factory=list)
    user_words: List[WordTimestamp] = field(default_factory=list)
    is_valid: bool = True
    error: Optional[str] = None
    diagnostics: Optional[AlignmentDiagnostics] = None

    @property
    def all_words(self) -> List[WordTimestamp]:
        """Get all words sorted by timestamp."""
        all_w = self.main_words + self.user_words
        return sorted(all_w, key=lambda w: w.start)

    @property
    def quality_score(self) -> float:
        """Calculate quality score."""
        if not self.is_valid:
            return 0.0

        scores = []

        # Word count score
        total_words = len(self.main_words) + len(self.user_words)
        if total_words > 0:
            scores.append(min(total_words / 100, 1.0))

        # Confidence scores
        if self.main_words:
            scores.append(np.mean([w.confidence for w in self.main_words]))
        if self.user_words:
            scores.append(np.mean([w.confidence for w in self.user_words]))

        # Coverage score
        if self.main_words and self.user_words:
            scores.append(1.0)
        elif self.main_words or self.user_words:
            scores.append(0.5)
        else:
            scores.append(0.0)

        return sum(scores) / len(scores) if scores else 0.0

    def to_moshi_format(self) -> Dict[str, Any]:
        """Convert to Moshi alignment JSON format."""
        all_words = self.all_words
        return {
            "alignments": [w.to_moshi_format() for w in all_words]
        }

    def save_moshi_format(self, output_path: Path) -> bool:
        """Save alignment in Moshi format."""
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(self.to_moshi_format(), f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            logger.error(f"Error saving alignment: {e}")
            return False

    def save_detailed_format(self, output_dir: Path) -> bool:
        """Save detailed alignments (separate files per speaker)."""
        try:
            output_dir.mkdir(parents=True, exist_ok=True)

            # Save main speaker
            main_path = output_dir / "alignment_speaker01" / f"{self.conversation_id}.json"
            main_path.parent.mkdir(parents=True, exist_ok=True)
            with open(main_path, "w", encoding="utf-8") as f:
                json.dump({
                    "speaker": "SPEAKER_MAIN",
                    "words": [w.to_moshi_format() for w in self.main_words],
                    "word_count": len(self.main_words),
                }, f, ensure_ascii=False, indent=2)

            # Save user speaker
            user_path = output_dir / "alignment_speaker02" / f"{self.conversation_id}.json"
            user_path.parent.mkdir(parents=True, exist_ok=True)
            with open(user_path, "w", encoding="utf-8") as f:
                json.dump({
                    "speaker": "SPEAKER_USER",
                    "words": [w.to_moshi_format() for w in self.user_words],
                    "word_count": len(self.user_words),
                }, f, ensure_ascii=False, indent=2)

            return True
        except Exception as e:
            logger.error(f"Error saving detailed alignment: {e}")
            return False

    def to_extended_format(self) -> Dict[str, Any]:
        """Convert to extended format with full speaker metadata.

        This format includes rich speaker information for each word:
        - speaker_role: SPEAKER_MAIN or SPEAKER_USER
        - original_speaker_id: Original speaker ID from source diarization
        - channel: Audio channel (0=left/MAIN, 1=right/USER)
        - segment_index: Index of source segment
        - confidence: Alignment confidence score

        Returns:
            Dictionary with full alignment metadata
        """
        all_words = self.all_words

        # Collect unique original speaker IDs
        unique_speakers = set()
        for w in all_words:
            if w.original_speaker_id:
                unique_speakers.add(w.original_speaker_id)

        # Collect statistics per original speaker
        speaker_stats = {}
        for w in all_words:
            sid = w.original_speaker_id or w.speaker
            if sid not in speaker_stats:
                speaker_stats[sid] = {
                    "word_count": 0,
                    "total_duration": 0.0,
                    "role": w.speaker,
                    "channel": w.channel,
                }
            speaker_stats[sid]["word_count"] += 1
            speaker_stats[sid]["total_duration"] += (w.end - w.start)

        result = {
            "conversation_id": self.conversation_id,
            "format_version": "2.0",  # Extended format version
            "metadata": {
                "total_words": len(all_words),
                "main_speaker_words": len(self.main_words),
                "user_speaker_words": len(self.user_words),
                "unique_original_speakers": list(unique_speakers),
                "speaker_statistics": speaker_stats,
                "quality_score": round(self.quality_score, 3),
            },
            "alignments": [w.to_extended_format() for w in all_words],
            # Also include basic Moshi format for backward compatibility
            "moshi_format": [w.to_moshi_format() for w in all_words],
        }

        # Include diagnostics if available
        if self.diagnostics is not None:
            result["diagnostics"] = self.diagnostics.get_summary()

        return result

    def save_extended_format(self, output_path: Path) -> bool:
        """Save alignment in extended format with full speaker metadata.

        Args:
            output_path: Path to save the extended format JSON

        Returns:
            True if successful, False otherwise
        """
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(self.to_extended_format(), f, ensure_ascii=False, indent=2)
            logger.debug(f"Saved extended alignment: {output_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving extended alignment: {e}")
            return False

    def save_detailed_extended_format(self, output_dir: Path) -> bool:
        """Save detailed alignments per speaker with extended metadata.

        Creates separate files for each speaker role with full metadata.

        Args:
            output_dir: Base output directory

        Returns:
            True if successful, False otherwise
        """
        try:
            output_dir.mkdir(parents=True, exist_ok=True)

            # Collect original speaker IDs per role
            main_speakers = set(
                w.original_speaker_id for w in self.main_words if w.original_speaker_id
            )
            user_speakers = set(
                w.original_speaker_id for w in self.user_words if w.original_speaker_id
            )

            # Save main speaker with extended metadata
            main_path = output_dir / "alignment_speaker01" / f"{self.conversation_id}.json"
            main_path.parent.mkdir(parents=True, exist_ok=True)
            with open(main_path, "w", encoding="utf-8") as f:
                json.dump({
                    "speaker_role": "SPEAKER_MAIN",
                    "channel": 0,
                    "original_speaker_ids": list(main_speakers),
                    "word_count": len(self.main_words),
                    "words": [w.to_extended_format() for w in self.main_words],
                    # Basic format for Moshi compatibility
                    "moshi_format": [w.to_moshi_format() for w in self.main_words],
                }, f, ensure_ascii=False, indent=2)

            # Save user speaker with extended metadata
            user_path = output_dir / "alignment_speaker02" / f"{self.conversation_id}.json"
            user_path.parent.mkdir(parents=True, exist_ok=True)
            with open(user_path, "w", encoding="utf-8") as f:
                json.dump({
                    "speaker_role": "SPEAKER_USER",
                    "channel": 1,
                    "original_speaker_ids": list(user_speakers),
                    "word_count": len(self.user_words),
                    "words": [w.to_extended_format() for w in self.user_words],
                    # Basic format for Moshi compatibility
                    "moshi_format": [w.to_moshi_format() for w in self.user_words],
                }, f, ensure_ascii=False, indent=2)

            return True
        except Exception as e:
            logger.error(f"Error saving detailed extended alignment: {e}")
            return False
