"""
Dialogue Quality Monitor for K-Moshi Training (Full-Duplex Mode).

Provides dialogue quality evaluation including:
- Turn-taking naturalness
- Response latency
- Overlap handling
- Backchannel detection

This module is specifically designed for Full-Duplex (V3) mode where
both Moshi and User audio streams are available during training.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import numpy as np

logger = logging.getLogger("dialogue_monitor")


@dataclass
class DialogueQualityResult:
    """Result of dialogue quality evaluation."""
    # Turn-taking metrics
    turn_taking_score: float = 0.0     # Overall naturalness (0-1)
    turn_count_moshi: int = 0          # Number of Moshi turns
    turn_count_user: int = 0           # Number of User turns
    avg_turn_length_moshi: float = 0.0  # Average Moshi turn length (frames)
    avg_turn_length_user: float = 0.0   # Average User turn length (frames)

    # Overlap metrics
    overlap_ratio: float = 0.0          # Ratio of overlapping frames
    overlap_count: int = 0              # Number of overlap events
    avg_overlap_duration: float = 0.0   # Average overlap duration (frames)

    # Latency metrics
    avg_response_latency_frames: float = 0.0  # Frames between user end and moshi start
    avg_response_latency_ms: float = 0.0      # Milliseconds

    # Silence metrics
    silence_ratio: float = 0.0          # Ratio of silent frames
    avg_silence_duration: float = 0.0   # Average silence duration (frames)

    # Backchannel detection
    backchannels_detected: int = 0      # Short acknowledgments
    interruption_count: int = 0         # Moshi interrupting user

    # Statistics
    sample_count: int = 0
    total_frames: int = 0


class DialogueQualityMonitor:
    """
    Dialogue quality evaluation monitor for Full-Duplex mode.

    Evaluates Moshi's real-time conversation abilities:

    1. Turn-Taking:
       - Natural alternation between speakers
       - Appropriate response timing
       - Balanced turn lengths

    2. Overlap Handling:
       - Detection of simultaneous speech
       - Quality of overlap resolution

    3. Response Latency:
       - Time between user finishing and Moshi responding
       - Natural conversation rhythm

    4. Backchannels:
       - Detection of short acknowledgments ("네", "응", etc.)
       - Appropriate backchannel timing

    Frame Rate:
        Moshi operates at 12.5Hz (80ms per frame)
        Natural response latency: 200-600ms (2.5-7.5 frames)

    Note:
        This monitor is only meaningful in Full-Duplex mode (V3)
        where user audio is available as context.

    Usage:
        monitor = DialogueQualityMonitor(frame_rate=12.5)
        result = monitor.evaluate_dialogue(moshi_audio, user_audio, zero_token_id)
        metrics = monitor.get_summary()
    """

    def __init__(
        self,
        frame_rate: float = 12.5,
        enabled: bool = True,
        overlap_threshold_frames: int = 3,
        silence_threshold_frames: int = 25,
        backchannel_max_frames: int = 10,
        natural_latency_range: Tuple[int, int] = (2, 8),
    ):
        """
        Initialize dialogue quality monitor.

        Args:
            frame_rate: Audio frame rate in Hz (Moshi default: 12.5)
            enabled: Whether to enable this monitor
            overlap_threshold_frames: Minimum frames to count as overlap
            silence_threshold_frames: Minimum frames to count as silence gap
            backchannel_max_frames: Maximum frames for backchannel detection
            natural_latency_range: (min, max) frames for natural response latency
        """
        self.frame_rate = frame_rate
        self.frame_duration_ms = 1000.0 / frame_rate  # 80ms
        self.enabled = enabled
        self.overlap_threshold = overlap_threshold_frames
        self.silence_threshold = silence_threshold_frames
        self.backchannel_max = backchannel_max_frames
        self.natural_latency_range = natural_latency_range

        # Accumulated statistics
        self.total_turn_taking_scores: List[float] = []
        self.total_overlap_ratios: List[float] = []
        self.total_response_latencies: List[float] = []
        self.total_silence_ratios: List[float] = []
        self.num_samples = 0

    def reset(self):
        """Reset accumulated statistics."""
        self.total_turn_taking_scores = []
        self.total_overlap_ratios = []
        self.total_response_latencies = []
        self.total_silence_ratios = []
        self.num_samples = 0

    def evaluate_dialogue(
        self,
        moshi_audio_codes: torch.Tensor,
        user_audio_codes: torch.Tensor,
        zero_token_id: int,
        moshi_text_codes: Optional[torch.Tensor] = None,
    ) -> DialogueQualityResult:
        """
        Evaluate dialogue quality for a batch.

        Args:
            moshi_audio_codes: Moshi audio codes [B, 8, T] or [B, T]
            user_audio_codes: User audio codes [B, 8, T] or [B, T]
            zero_token_id: Token ID indicating silence/no audio
            moshi_text_codes: Optional Moshi text codes for backchannel detection

        Returns:
            DialogueQualityResult with dialogue quality metrics
        """
        if not self.enabled:
            return DialogueQualityResult()

        # Handle tensor dimensions - use first codebook for speech detection
        if moshi_audio_codes.dim() == 3:
            moshi_audio = moshi_audio_codes[:, 0, :]  # [B, T]
        else:
            moshi_audio = moshi_audio_codes

        if user_audio_codes.dim() == 3:
            user_audio = user_audio_codes[:, 0, :]  # [B, T]
        else:
            user_audio = user_audio_codes

        batch_size, seq_len = moshi_audio.shape

        # Aggregate metrics
        batch_results = []

        for b in range(batch_size):
            moshi = moshi_audio[b].cpu()
            user = user_audio[b].cpu()

            # Detect speech segments
            moshi_segments = self._detect_speech_segments(moshi, zero_token_id)
            user_segments = self._detect_speech_segments(user, zero_token_id)

            # Compute metrics
            result = self._analyze_dialogue(
                moshi_segments, user_segments, seq_len
            )
            batch_results.append(result)
            self.num_samples += 1

        # Aggregate batch results
        if not batch_results:
            return DialogueQualityResult()

        # Average across batch
        avg_result = DialogueQualityResult(
            sample_count=batch_size,
            total_frames=batch_size * seq_len,
        )

        # Turn-taking
        avg_result.turn_taking_score = np.mean([r.turn_taking_score for r in batch_results])
        avg_result.turn_count_moshi = sum(r.turn_count_moshi for r in batch_results)
        avg_result.turn_count_user = sum(r.turn_count_user for r in batch_results)
        avg_result.avg_turn_length_moshi = np.mean([r.avg_turn_length_moshi for r in batch_results if r.avg_turn_length_moshi > 0])
        avg_result.avg_turn_length_user = np.mean([r.avg_turn_length_user for r in batch_results if r.avg_turn_length_user > 0])

        # Overlap
        avg_result.overlap_ratio = np.mean([r.overlap_ratio for r in batch_results])
        avg_result.overlap_count = sum(r.overlap_count for r in batch_results)
        avg_result.avg_overlap_duration = np.mean([r.avg_overlap_duration for r in batch_results if r.avg_overlap_duration > 0])

        # Latency
        latencies = [r.avg_response_latency_frames for r in batch_results if r.avg_response_latency_frames > 0]
        if latencies:
            avg_result.avg_response_latency_frames = np.mean(latencies)
            avg_result.avg_response_latency_ms = avg_result.avg_response_latency_frames * self.frame_duration_ms

        # Silence
        avg_result.silence_ratio = np.mean([r.silence_ratio for r in batch_results])
        avg_result.avg_silence_duration = np.mean([r.avg_silence_duration for r in batch_results if r.avg_silence_duration > 0])

        # Backchannels
        avg_result.backchannels_detected = sum(r.backchannels_detected for r in batch_results)
        avg_result.interruption_count = sum(r.interruption_count for r in batch_results)

        # Accumulate for summary
        self.total_turn_taking_scores.append(avg_result.turn_taking_score)
        self.total_overlap_ratios.append(avg_result.overlap_ratio)
        if avg_result.avg_response_latency_frames > 0:
            self.total_response_latencies.append(avg_result.avg_response_latency_frames)
        self.total_silence_ratios.append(avg_result.silence_ratio)

        return avg_result

    def _detect_speech_segments(
        self,
        audio_codes: torch.Tensor,
        zero_token_id: int,
    ) -> List[Tuple[int, int]]:
        """
        Detect speech segments from audio codes.

        Args:
            audio_codes: Audio token IDs [T]
            zero_token_id: Token ID indicating silence

        Returns:
            List of (start_frame, end_frame) tuples
        """
        segments = []
        in_speech = False
        start_frame = 0

        codes_list = audio_codes.tolist()

        for i, code in enumerate(codes_list):
            is_speech = code != zero_token_id and code >= 0

            if is_speech and not in_speech:
                start_frame = i
                in_speech = True
            elif not is_speech and in_speech:
                segments.append((start_frame, i))
                in_speech = False

        # Handle speech at end
        if in_speech:
            segments.append((start_frame, len(codes_list)))

        return segments

    def _analyze_dialogue(
        self,
        moshi_segments: List[Tuple[int, int]],
        user_segments: List[Tuple[int, int]],
        total_frames: int,
    ) -> DialogueQualityResult:
        """
        Analyze dialogue patterns from speech segments.

        Args:
            moshi_segments: Moshi speech segments
            user_segments: User speech segments
            total_frames: Total number of frames

        Returns:
            DialogueQualityResult with analysis
        """
        result = DialogueQualityResult(total_frames=total_frames)

        # Turn counts
        result.turn_count_moshi = len(moshi_segments)
        result.turn_count_user = len(user_segments)

        # Average turn lengths
        if moshi_segments:
            result.avg_turn_length_moshi = np.mean([e - s for s, e in moshi_segments])
        if user_segments:
            result.avg_turn_length_user = np.mean([e - s for s, e in user_segments])

        # Compute overlap
        overlap_info = self._compute_overlap(moshi_segments, user_segments, total_frames)
        result.overlap_ratio = overlap_info["ratio"]
        result.overlap_count = overlap_info["count"]
        result.avg_overlap_duration = overlap_info["avg_duration"]

        # Compute silence
        silence_info = self._compute_silence(moshi_segments, user_segments, total_frames)
        result.silence_ratio = silence_info["ratio"]
        result.avg_silence_duration = silence_info["avg_duration"]

        # Compute response latency
        latencies = self._compute_response_latency(moshi_segments, user_segments)
        if latencies:
            result.avg_response_latency_frames = np.mean(latencies)
            result.avg_response_latency_ms = result.avg_response_latency_frames * self.frame_duration_ms

        # Detect backchannels (short Moshi responses)
        result.backchannels_detected = sum(
            1 for s, e in moshi_segments
            if (e - s) <= self.backchannel_max
        )

        # Detect interruptions (Moshi starting while user speaking)
        result.interruption_count = self._detect_interruptions(moshi_segments, user_segments)

        # Compute turn-taking score
        result.turn_taking_score = self._compute_turn_taking_score(result)

        return result

    def _compute_overlap(
        self,
        moshi_segments: List[Tuple[int, int]],
        user_segments: List[Tuple[int, int]],
        total_frames: int,
    ) -> Dict[str, float]:
        """Compute overlap metrics between speakers."""
        if not moshi_segments or not user_segments:
            return {"ratio": 0.0, "count": 0, "avg_duration": 0.0}

        # Create frame-level masks
        moshi_mask = np.zeros(total_frames, dtype=bool)
        user_mask = np.zeros(total_frames, dtype=bool)

        for s, e in moshi_segments:
            moshi_mask[s:min(e, total_frames)] = True
        for s, e in user_segments:
            user_mask[s:min(e, total_frames)] = True

        # Compute overlap
        overlap_mask = moshi_mask & user_mask
        overlap_frames = np.sum(overlap_mask)
        overlap_ratio = overlap_frames / total_frames if total_frames > 0 else 0.0

        # Count overlap events
        overlap_segments = self._mask_to_segments(overlap_mask)
        overlap_segments = [seg for seg in overlap_segments if seg[1] - seg[0] >= self.overlap_threshold]
        overlap_count = len(overlap_segments)

        # Average overlap duration
        avg_duration = np.mean([e - s for s, e in overlap_segments]) if overlap_segments else 0.0

        return {
            "ratio": float(overlap_ratio),
            "count": overlap_count,
            "avg_duration": float(avg_duration),
        }

    def _compute_silence(
        self,
        moshi_segments: List[Tuple[int, int]],
        user_segments: List[Tuple[int, int]],
        total_frames: int,
    ) -> Dict[str, float]:
        """Compute silence metrics (neither speaker active)."""
        if total_frames == 0:
            return {"ratio": 0.0, "avg_duration": 0.0}

        # Create frame-level masks
        moshi_mask = np.zeros(total_frames, dtype=bool)
        user_mask = np.zeros(total_frames, dtype=bool)

        for s, e in moshi_segments:
            moshi_mask[s:min(e, total_frames)] = True
        for s, e in user_segments:
            user_mask[s:min(e, total_frames)] = True

        # Silence = neither speaking
        silence_mask = ~(moshi_mask | user_mask)
        silence_ratio = np.sum(silence_mask) / total_frames

        # Find silence segments
        silence_segments = self._mask_to_segments(silence_mask)
        silence_segments = [seg for seg in silence_segments if seg[1] - seg[0] >= self.silence_threshold]

        avg_duration = np.mean([e - s for s, e in silence_segments]) if silence_segments else 0.0

        return {
            "ratio": float(silence_ratio),
            "avg_duration": float(avg_duration),
        }

    def _compute_response_latency(
        self,
        moshi_segments: List[Tuple[int, int]],
        user_segments: List[Tuple[int, int]],
    ) -> List[float]:
        """
        Compute response latency (time from user end to Moshi start).

        Returns:
            List of latencies in frames
        """
        if not moshi_segments or not user_segments:
            return []

        latencies = []

        for user_start, user_end in user_segments:
            # Find next Moshi segment after this user segment
            next_moshi = None
            for moshi_start, moshi_end in moshi_segments:
                if moshi_start > user_end:
                    next_moshi = (moshi_start, moshi_end)
                    break

            if next_moshi:
                latency = next_moshi[0] - user_end
                if latency > 0:  # Positive latency only
                    latencies.append(latency)

        return latencies

    def _detect_interruptions(
        self,
        moshi_segments: List[Tuple[int, int]],
        user_segments: List[Tuple[int, int]],
    ) -> int:
        """
        Detect interruptions (Moshi starting while user is speaking).

        Returns:
            Number of interruption events
        """
        interruptions = 0

        for moshi_start, moshi_end in moshi_segments:
            for user_start, user_end in user_segments:
                # Moshi starts during user speech
                if user_start < moshi_start < user_end:
                    interruptions += 1
                    break

        return interruptions

    def _mask_to_segments(self, mask: np.ndarray) -> List[Tuple[int, int]]:
        """Convert boolean mask to list of (start, end) segments."""
        segments = []
        in_segment = False
        start = 0

        for i, val in enumerate(mask):
            if val and not in_segment:
                start = i
                in_segment = True
            elif not val and in_segment:
                segments.append((start, i))
                in_segment = False

        if in_segment:
            segments.append((start, len(mask)))

        return segments

    def _compute_turn_taking_score(self, result: DialogueQualityResult) -> float:
        """
        Compute overall turn-taking naturalness score.

        Factors:
        - Low overlap is good (natural turn-taking)
        - Moderate silence is good (not too much, not too little)
        - Natural response latency is good
        - Both speakers participate
        - Low interruption rate
        """
        scores = []

        # Overlap score (lower is better, but some overlap is natural)
        # Optimal overlap ratio: 5-15%
        overlap_score = 1.0 - min(abs(result.overlap_ratio - 0.1), 0.3) / 0.3
        scores.append(max(0, overlap_score))

        # Silence score (moderate is good)
        # Optimal silence ratio: 20-40%
        if result.silence_ratio < 0.2:
            silence_score = result.silence_ratio / 0.2
        elif result.silence_ratio <= 0.4:
            silence_score = 1.0
        else:
            silence_score = max(0, 1.0 - (result.silence_ratio - 0.4) / 0.4)
        scores.append(silence_score)

        # Latency score (natural latency range)
        if result.avg_response_latency_frames > 0:
            min_lat, max_lat = self.natural_latency_range
            if min_lat <= result.avg_response_latency_frames <= max_lat:
                latency_score = 1.0
            elif result.avg_response_latency_frames < min_lat:
                latency_score = result.avg_response_latency_frames / min_lat
            else:
                latency_score = max(0, 1.0 - (result.avg_response_latency_frames - max_lat) / max_lat)
            scores.append(latency_score)

        # Participation balance
        total_turns = result.turn_count_moshi + result.turn_count_user
        if total_turns > 0:
            moshi_ratio = result.turn_count_moshi / total_turns
            balance_score = 1.0 - abs(moshi_ratio - 0.5) * 2  # Best at 50/50
            scores.append(max(0, balance_score))

        # Interruption penalty
        if result.turn_count_moshi > 0:
            interruption_rate = result.interruption_count / result.turn_count_moshi
            interruption_score = 1.0 - min(interruption_rate, 0.5) * 2
            scores.append(max(0, interruption_score))

        return np.mean(scores) if scores else 0.0

    def get_summary(self) -> Dict[str, float]:
        """Get summary statistics for logging."""
        summary = {
            "num_samples": self.num_samples,
        }

        if self.total_turn_taking_scores:
            summary["turn_taking_score"] = np.mean(self.total_turn_taking_scores)

        if self.total_overlap_ratios:
            summary["overlap_ratio"] = np.mean(self.total_overlap_ratios)

        if self.total_response_latencies:
            summary["avg_response_latency_frames"] = np.mean(self.total_response_latencies)
            summary["avg_response_latency_ms"] = summary["avg_response_latency_frames"] * self.frame_duration_ms

        if self.total_silence_ratios:
            summary["silence_ratio"] = np.mean(self.total_silence_ratios)

        return summary

    def format_log_message(self) -> str:
        """Format a summary log message."""
        summary = self.get_summary()
        parts = ["[DIALOGUE]"]

        if "turn_taking_score" in summary:
            parts.append(f"TurnTaking={summary['turn_taking_score']:.3f}")
        if "overlap_ratio" in summary:
            parts.append(f"Overlap={summary['overlap_ratio']:.1%}")
        if "avg_response_latency_ms" in summary:
            parts.append(f"Latency={summary['avg_response_latency_ms']:.0f}ms")

        return " ".join(parts)
