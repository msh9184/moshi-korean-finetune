"""
Alignment Quality Monitor for K-Moshi Training.

Provides text-audio alignment quality evaluation including:
- Timing accuracy: How well text tokens align with audio frames
- Boundary precision/recall: Word boundary detection accuracy
- Synchronization score: Overall text-audio sync quality

This module validates the quality of:
1. Interleaver's text-audio alignment
2. Training data alignment JSON files
3. Model's temporal consistency during training
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import numpy as np

logger = logging.getLogger("alignment_monitor")


@dataclass
class AlignmentQualityResult:
    """Result of text-audio alignment quality evaluation."""
    # Timing metrics
    timing_accuracy: float = 0.0       # Percentage of correctly timed tokens (0-1)
    avg_timing_error_frames: float = 0.0  # Average timing error in frames
    avg_timing_error_ms: float = 0.0   # Average timing error in milliseconds

    # Boundary metrics
    boundary_precision: float = 0.0    # Precision of word boundary detection
    boundary_recall: float = 0.0       # Recall of word boundary detection
    boundary_f1: float = 0.0           # F1 score of boundary detection

    # Synchronization
    sync_score: float = 0.0            # Overall synchronization score (0-1)

    # Coverage metrics
    text_coverage: float = 0.0         # Percentage of audio frames with text
    silence_ratio: float = 0.0         # Percentage of audio frames without text

    # Statistics
    sample_count: int = 0
    total_frames: int = 0
    total_text_tokens: int = 0
    total_padding_tokens: int = 0


class AlignmentQualityMonitor:
    """
    Text-Audio alignment quality evaluation monitor.

    Evaluates how well Inner Monologue text is synchronized with audio:

    1. Timing Accuracy:
       - Measures if text tokens appear at the correct time frames
       - Compares predicted text positions with alignment JSON

    2. Boundary Detection:
       - Precision: Of detected boundaries, how many are correct
       - Recall: Of actual boundaries, how many are detected

    3. Sync Score:
       - Overall quality of text-audio synchronization
       - Considers timing errors, boundary accuracy, and coverage

    Frame Rate:
        Moshi operates at 12.5Hz (80ms per frame)
        1 second = 12.5 frames

    Usage:
        monitor = AlignmentQualityMonitor(frame_rate=12.5)
        result = monitor.evaluate_alignment(text_codes, padding_id, alignments)
        metrics = monitor.get_summary()
    """

    def __init__(
        self,
        frame_rate: float = 12.5,
        tolerance_frames: int = 2,
        enabled: bool = True,
    ):
        """
        Initialize alignment quality monitor.

        Args:
            frame_rate: Audio frame rate in Hz (Moshi default: 12.5)
            tolerance_frames: Allowed timing error in frames for "correct"
            enabled: Whether to enable this monitor
        """
        self.frame_rate = frame_rate
        self.frame_duration_ms = 1000.0 / frame_rate  # 80ms for 12.5Hz
        self.tolerance_frames = tolerance_frames
        self.enabled = enabled

        # Accumulated statistics
        self.total_timing_errors: List[float] = []
        self.total_boundary_precision: List[float] = []
        self.total_boundary_recall: List[float] = []
        self.total_sync_scores: List[float] = []
        self.num_samples = 0

    def reset(self):
        """Reset accumulated statistics."""
        self.total_timing_errors = []
        self.total_boundary_precision = []
        self.total_boundary_recall = []
        self.total_sync_scores = []
        self.num_samples = 0

    def evaluate_alignment(
        self,
        text_codes: torch.Tensor,
        text_padding_id: int,
        end_of_text_padding_id: int,
        alignments: Optional[List[List[tuple]]] = None,
        main_speaker_label: str = "SPEAKER_MAIN",
    ) -> AlignmentQualityResult:
        """
        Evaluate alignment quality for a batch.

        Args:
            text_codes: Text token tensor [B, 1, T] or [B, T]
            text_padding_id: Padding token ID
            end_of_text_padding_id: End-of-text padding token ID
            alignments: List of alignment data per sample
                Each alignment: [("word", (start_sec, end_sec), "speaker"), ...]
            main_speaker_label: Speaker label to consider (default: SPEAKER_MAIN)

        Returns:
            AlignmentQualityResult with alignment quality metrics
        """
        if not self.enabled:
            return AlignmentQualityResult()

        # Handle tensor dimensions
        if text_codes.dim() == 3:
            text_codes = text_codes.squeeze(1)  # [B, T]

        batch_size, seq_len = text_codes.shape
        device = text_codes.device

        # Aggregate metrics
        batch_timing_errors = []
        batch_boundary_precisions = []
        batch_boundary_recalls = []
        batch_sync_scores = []

        total_frames = 0
        total_text_tokens = 0
        total_padding_tokens = 0

        for b in range(batch_size):
            sample_codes = text_codes[b].cpu()
            sample_alignments = alignments[b] if alignments and b < len(alignments) else None

            # Extract text token positions (non-padding)
            padding_ids = {text_padding_id, end_of_text_padding_id}
            text_mask = torch.tensor([
                int(tok.item()) not in padding_ids and int(tok.item()) >= 0
                for tok in sample_codes
            ])

            text_positions = torch.where(text_mask)[0].tolist()
            padding_positions = torch.where(~text_mask)[0].tolist()

            total_frames += seq_len
            total_text_tokens += len(text_positions)
            total_padding_tokens += len(padding_positions)

            if sample_alignments is None:
                # No alignment data - can only compute coverage metrics
                continue

            # Convert alignments to expected frame positions
            expected_boundaries = self._alignments_to_frame_boundaries(
                sample_alignments, main_speaker_label
            )

            if not expected_boundaries:
                continue

            # Extract detected boundaries from text codes
            detected_boundaries = self._extract_text_boundaries(text_mask)

            # Compute timing accuracy
            timing_error = self._compute_timing_error(
                detected_boundaries, expected_boundaries
            )
            batch_timing_errors.append(timing_error)

            # Compute boundary precision/recall
            precision, recall = self._compute_boundary_metrics(
                detected_boundaries, expected_boundaries
            )
            batch_boundary_precisions.append(precision)
            batch_boundary_recalls.append(recall)

            # Compute sync score
            sync_score = self._compute_sync_score(
                timing_error, precision, recall, len(text_positions), seq_len
            )
            batch_sync_scores.append(sync_score)

            self.num_samples += 1

        # Compute result
        result = AlignmentQualityResult(
            sample_count=batch_size,
            total_frames=total_frames,
            total_text_tokens=total_text_tokens,
            total_padding_tokens=total_padding_tokens,
        )

        # Text coverage
        if total_frames > 0:
            result.text_coverage = total_text_tokens / total_frames
            result.silence_ratio = total_padding_tokens / total_frames

        # Timing metrics
        if batch_timing_errors:
            avg_error = np.mean(batch_timing_errors)
            result.avg_timing_error_frames = avg_error
            result.avg_timing_error_ms = avg_error * self.frame_duration_ms
            result.timing_accuracy = np.mean([
                1.0 if e <= self.tolerance_frames else 0.0
                for e in batch_timing_errors
            ])
            self.total_timing_errors.extend(batch_timing_errors)

        # Boundary metrics
        if batch_boundary_precisions:
            result.boundary_precision = np.mean(batch_boundary_precisions)
            result.boundary_recall = np.mean(batch_boundary_recalls)
            if result.boundary_precision + result.boundary_recall > 0:
                result.boundary_f1 = (
                    2 * result.boundary_precision * result.boundary_recall /
                    (result.boundary_precision + result.boundary_recall)
                )
            self.total_boundary_precision.extend(batch_boundary_precisions)
            self.total_boundary_recall.extend(batch_boundary_recalls)

        # Sync score
        if batch_sync_scores:
            result.sync_score = np.mean(batch_sync_scores)
            self.total_sync_scores.extend(batch_sync_scores)

        return result

    def _alignments_to_frame_boundaries(
        self,
        alignments: List[tuple],
        speaker_label: str,
    ) -> List[Tuple[int, int]]:
        """
        Convert alignment data to frame-level boundaries.

        Args:
            alignments: [("word", (start_sec, end_sec), "speaker"), ...]
            speaker_label: Speaker to filter for

        Returns:
            List of (start_frame, end_frame) tuples
        """
        boundaries = []
        for item in alignments:
            if len(item) < 3:
                continue
            word, timing, speaker = item[0], item[1], item[2]

            if speaker != speaker_label:
                continue

            if not isinstance(timing, (list, tuple)) or len(timing) < 2:
                continue

            start_sec, end_sec = timing[0], timing[1]
            start_frame = int(start_sec * self.frame_rate)
            end_frame = int(end_sec * self.frame_rate)

            if end_frame > start_frame:
                boundaries.append((start_frame, end_frame))

        return boundaries

    def _extract_text_boundaries(
        self,
        text_mask: torch.Tensor,
    ) -> List[Tuple[int, int]]:
        """
        Extract word boundaries from text token mask.

        Finds contiguous regions of text tokens (non-padding).

        Args:
            text_mask: Boolean mask where True = text token

        Returns:
            List of (start_frame, end_frame) tuples
        """
        boundaries = []
        in_word = False
        start_frame = 0

        for i, is_text in enumerate(text_mask.tolist()):
            if is_text and not in_word:
                # Start of word
                start_frame = i
                in_word = True
            elif not is_text and in_word:
                # End of word
                boundaries.append((start_frame, i))
                in_word = False

        # Handle word at end of sequence
        if in_word:
            boundaries.append((start_frame, len(text_mask)))

        return boundaries

    def _compute_timing_error(
        self,
        detected: List[Tuple[int, int]],
        expected: List[Tuple[int, int]],
    ) -> float:
        """
        Compute average timing error between detected and expected boundaries.

        Uses Hungarian matching to find optimal pairing.

        Args:
            detected: Detected (start, end) boundaries
            expected: Expected (start, end) boundaries

        Returns:
            Average timing error in frames
        """
        if not detected or not expected:
            return float('inf') if (detected or expected) else 0.0

        # Simple greedy matching for efficiency
        # For each detected, find closest expected
        errors = []
        used_expected = set()

        for d_start, d_end in detected:
            d_center = (d_start + d_end) / 2

            best_error = float('inf')
            best_idx = -1

            for i, (e_start, e_end) in enumerate(expected):
                if i in used_expected:
                    continue
                e_center = (e_start + e_end) / 2
                error = abs(d_center - e_center)
                if error < best_error:
                    best_error = error
                    best_idx = i

            if best_idx >= 0:
                used_expected.add(best_idx)
                errors.append(best_error)

        return np.mean(errors) if errors else float('inf')

    def _compute_boundary_metrics(
        self,
        detected: List[Tuple[int, int]],
        expected: List[Tuple[int, int]],
    ) -> Tuple[float, float]:
        """
        Compute boundary detection precision and recall.

        A detected boundary is considered correct if it overlaps with
        an expected boundary within tolerance.

        Args:
            detected: Detected boundaries
            expected: Expected boundaries

        Returns:
            (precision, recall) tuple
        """
        if not detected and not expected:
            return 1.0, 1.0
        if not detected:
            return 0.0, 0.0
        if not expected:
            return 0.0, 1.0  # No expected = nothing to recall

        tolerance = self.tolerance_frames

        # Count true positives for precision
        true_positives_det = 0
        for d_start, d_end in detected:
            for e_start, e_end in expected:
                # Check if boundaries overlap within tolerance
                if (abs(d_start - e_start) <= tolerance or
                    abs(d_end - e_end) <= tolerance or
                    (d_start >= e_start - tolerance and d_end <= e_end + tolerance)):
                    true_positives_det += 1
                    break

        # Count true positives for recall
        true_positives_exp = 0
        for e_start, e_end in expected:
            for d_start, d_end in detected:
                if (abs(d_start - e_start) <= tolerance or
                    abs(d_end - e_end) <= tolerance or
                    (d_start >= e_start - tolerance and d_end <= e_end + tolerance)):
                    true_positives_exp += 1
                    break

        precision = true_positives_det / len(detected) if detected else 0.0
        recall = true_positives_exp / len(expected) if expected else 0.0

        return precision, recall

    def _compute_sync_score(
        self,
        timing_error: float,
        precision: float,
        recall: float,
        text_tokens: int,
        total_frames: int,
    ) -> float:
        """
        Compute overall synchronization score.

        Combines timing accuracy, boundary quality, and coverage.

        Args:
            timing_error: Average timing error in frames
            precision: Boundary precision
            recall: Boundary recall
            text_tokens: Number of text tokens
            total_frames: Total number of frames

        Returns:
            Sync score (0-1)
        """
        # Timing component (inverse of error, normalized)
        max_error = 10.0  # 10 frames = 800ms max expected error
        timing_score = max(0, 1.0 - timing_error / max_error)

        # Boundary F1
        if precision + recall > 0:
            boundary_f1 = 2 * precision * recall / (precision + recall)
        else:
            boundary_f1 = 0.0

        # Coverage component
        coverage = text_tokens / max(total_frames, 1)
        # Optimal coverage is around 30-50% for natural speech
        coverage_score = 1.0 - abs(coverage - 0.4) * 2  # Peak at 40%
        coverage_score = max(0, min(1, coverage_score))

        # Weighted combination
        sync_score = (
            0.4 * timing_score +
            0.4 * boundary_f1 +
            0.2 * coverage_score
        )

        return sync_score

    def get_summary(self) -> Dict[str, float]:
        """
        Get summary statistics for logging.

        Returns:
            Dictionary with alignment quality metrics
        """
        summary = {
            "num_samples": self.num_samples,
        }

        if self.total_timing_errors:
            summary["avg_timing_error_frames"] = np.mean(self.total_timing_errors)
            summary["avg_timing_error_ms"] = np.mean(self.total_timing_errors) * self.frame_duration_ms
            summary["timing_accuracy"] = np.mean([
                1.0 if e <= self.tolerance_frames else 0.0
                for e in self.total_timing_errors
            ])

        if self.total_boundary_precision:
            summary["boundary_precision"] = np.mean(self.total_boundary_precision)
            summary["boundary_recall"] = np.mean(self.total_boundary_recall)
            p = summary["boundary_precision"]
            r = summary["boundary_recall"]
            if p + r > 0:
                summary["boundary_f1"] = 2 * p * r / (p + r)

        if self.total_sync_scores:
            summary["sync_score"] = np.mean(self.total_sync_scores)

        return summary

    def format_log_message(self) -> str:
        """Format a summary log message."""
        summary = self.get_summary()
        parts = ["[ALIGNMENT]"]

        if "timing_accuracy" in summary:
            parts.append(f"TimingAcc={summary['timing_accuracy']:.1%}")
        if "boundary_f1" in summary:
            parts.append(f"BoundF1={summary['boundary_f1']:.3f}")
        if "sync_score" in summary:
            parts.append(f"Sync={summary['sync_score']:.3f}")

        return " ".join(parts)
