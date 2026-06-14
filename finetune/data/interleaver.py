import json
import logging
import math
import os
import random
from collections import deque
from dataclasses import dataclass, field
from functools import reduce
from pathlib import Path
from typing import Iterator, Optional, NamedTuple

import numpy as np
import sentencepiece
import torch
from moshi.conditioners import ConditionAttributes

logger = logging.getLogger("interleaver")


# Use NamedTuple for structured alignment data with named fields
class Alignment(NamedTuple):
    """Alignment data with text, time span, and speaker label."""
    text: str
    time_span: tuple[float, float]
    speaker: str


class TokenizedAlignment(NamedTuple):
    """Tokenized alignment data with token IDs, time span, and speaker label."""
    tokens: list[int]
    time_span: tuple[float, float]
    speaker: str


# Character-level alignment: (character, start_time, end_time)
CharacterAlignment = tuple[str, float, float]


# =============================================================================
# Data Augmentation: Segment Statistics
# =============================================================================

@dataclass
class SegmentStatistics:
    """
    Statistics for a segment's speaker distribution.
    Used for quality filtering and monitoring.
    """
    # Moshi (main speaker) statistics
    moshi_word_count: int = 0
    moshi_duration_sec: float = 0.0
    moshi_alignment_count: int = 0

    # User (other speaker) statistics
    user_word_count: int = 0
    user_duration_sec: float = 0.0
    user_alignment_count: int = 0

    # Segment metadata
    segment_duration_sec: float = 0.0
    segment_path: str = ""
    segment_start_sec: float = 0.0

    # Computed ratios
    @property
    def total_speech_duration(self) -> float:
        return self.moshi_duration_sec + self.user_duration_sec

    @property
    def moshi_ratio(self) -> float:
        """Ratio of Moshi speech to total speech duration."""
        if self.total_speech_duration == 0:
            return 0.0
        return self.moshi_duration_sec / self.total_speech_duration

    @property
    def user_ratio(self) -> float:
        """Ratio of User speech to total speech duration."""
        if self.total_speech_duration == 0:
            return 0.0
        return self.user_duration_sec / self.total_speech_duration

    @property
    def has_moshi_speech(self) -> bool:
        return self.moshi_word_count > 0 and self.moshi_duration_sec > 0

    @property
    def has_user_speech(self) -> bool:
        return self.user_word_count > 0 and self.user_duration_sec > 0

    @property
    def has_both_speakers(self) -> bool:
        return self.has_moshi_speech and self.has_user_speech

    def to_dict(self) -> dict:
        return {
            "moshi_words": self.moshi_word_count,
            "moshi_duration_sec": round(self.moshi_duration_sec, 2),
            "moshi_ratio": round(self.moshi_ratio, 3),
            "user_words": self.user_word_count,
            "user_duration_sec": round(self.user_duration_sec, 2),
            "user_ratio": round(self.user_ratio, 3),
            "segment_duration_sec": round(self.segment_duration_sec, 2),
            "has_both_speakers": self.has_both_speakers,
        }


def compute_segment_statistics(
    alignments: list[Alignment],
    main_speaker_label: str,
    segment_duration_sec: float,
    segment_path: str = "",
    segment_start_sec: float = 0.0,
) -> SegmentStatistics:
    """
    Compute statistics for a segment's speaker distribution.

    Args:
        alignments: List of (word, (start, end), speaker) tuples
        main_speaker_label: Label for main speaker (Moshi)
        segment_duration_sec: Total segment duration
        segment_path: Path to audio file (for logging)
        segment_start_sec: Start time in original audio (for logging)

    Returns:
        SegmentStatistics with computed values
    """
    stats = SegmentStatistics(
        segment_duration_sec=segment_duration_sec,
        segment_path=segment_path,
        segment_start_sec=segment_start_sec,
    )

    for word, (start, end), speaker in alignments:
        duration = max(0, end - start)
        word_count = len(word.split()) if word.strip() else 0

        if speaker == main_speaker_label:
            stats.moshi_word_count += word_count if word_count > 0 else 1
            stats.moshi_duration_sec += duration
            stats.moshi_alignment_count += 1
        else:
            stats.user_word_count += word_count if word_count > 0 else 1
            stats.user_duration_sec += duration
            stats.user_alignment_count += 1

    return stats


# =============================================================================
# UNIFIED FILTERING STATISTICS (Layer 5: Logging & Debugging)
# =============================================================================
# Redesigned to match the new 5-layer segment_filtering structure.
# Provides comprehensive statistics for each filtering layer.
# =============================================================================

@dataclass
class FilteringStatistics:
    """
    Unified filtering statistics tracker for the 5-layer segment_filtering system.

    Layer Structure:
        Layer 1: Case Control (case_*) - Structural validity checks
        Layer 2: Quality (quality_*) - Quality requirement checks
        Layer 3: Preferences (pref_*) - Probabilistic preference checks
        Layer 4: Role Swapping (swap_*) - Data augmentation statistics
        Layer 5: Meta (total_*, passed_*) - Overall statistics
    """
    # =========================================================================
    # Layer 5 Meta Statistics
    # =========================================================================
    total_segments: int = 0           # Total segments processed
    passed_segments: int = 0          # Segments passing ALL filters
    total_samples_yielded: int = 0    # Total samples yielded (including swapped)

    # =========================================================================
    # Layer 1: Case Control Statistics
    # =========================================================================
    case_checks: int = 0              # Total case control checks performed
    case_passed: int = 0              # Segments passing case control

    # Case detection counts
    case1_detected: int = 0           # Full dialogue detected
    case2_detected: int = 0           # Moshi monologue detected
    case3_detected: int = 0           # User audio only detected
    case4_detected: int = 0           # Missing Moshi audio detected
    case5_detected: int = 0           # Missing Moshi text detected

    # Case filtered counts (segment was valid but case not allowed)
    case1_filtered: int = 0
    case2_filtered: int = 0
    case3_filtered: int = 0
    case4_filtered: int = 0
    case5_filtered: int = 0

    # =========================================================================
    # Layer 2: Quality Statistics
    # =========================================================================
    quality_checks: int = 0           # Total quality checks performed
    quality_passed: int = 0           # Segments passing quality checks

    # Quality filter skip reasons
    quality_no_moshi: int = 0         # No Moshi speech at all
    quality_low_moshi_words: int = 0  # Below min_moshi_words
    quality_low_moshi_duration: int = 0  # Below min_moshi_duration_sec
    quality_low_moshi_ratio: int = 0  # Below min_moshi_ratio
    quality_low_user_words: int = 0   # Below min_user_words
    quality_short_segment: int = 0    # Below min_segment_duration_sec
    quality_long_segment: int = 0     # Above max_segment_duration_sec

    # =========================================================================
    # Layer 3: Preferences Statistics
    # =========================================================================
    pref_checks: int = 0              # Total preference checks performed
    pref_passed: int = 0              # Segments passing preference checks
    pref_user_first_skipped: int = 0  # Skipped due to prefer_moshi_start
    pref_single_speaker_skipped: int = 0  # Skipped due to prefer_both_speakers

    # =========================================================================
    # Layer 4: Role Swapping Statistics
    # =========================================================================
    swap_original_yielded: int = 0    # Original samples yielded
    swap_swapped_yielded: int = 0     # Role-swapped samples yielded
    swap_post_check_passed: int = 0   # Swapped samples passing recheck
    swap_post_check_failed: int = 0   # Swapped samples failing recheck

    @property
    def skip_rate(self) -> float:
        """Overall skip rate (1 - pass rate)."""
        if self.total_segments == 0:
            return 0.0
        return 1.0 - (self.passed_segments / self.total_segments)

    @property
    def case_pass_rate(self) -> float:
        """Case control pass rate."""
        if self.case_checks == 0:
            return 0.0
        return self.case_passed / self.case_checks

    @property
    def quality_pass_rate(self) -> float:
        """Quality check pass rate."""
        if self.quality_checks == 0:
            return 0.0
        return self.quality_passed / self.quality_checks

    @property
    def pref_pass_rate(self) -> float:
        """Preferences check pass rate."""
        if self.pref_checks == 0:
            return 0.0
        return self.pref_passed / self.pref_checks

    @property
    def data_multiplier(self) -> float:
        """Effective data multiplier from role swapping."""
        if self.swap_original_yielded == 0:
            return 1.0
        return (self.swap_original_yielded + self.swap_swapped_yielded) / self.swap_original_yielded

    def to_dict(self) -> dict:
        """Export statistics as dictionary for JSON serialization."""
        return {
            "meta": {
                "total_segments": self.total_segments,
                "passed_segments": self.passed_segments,
                "skip_rate": round(self.skip_rate * 100, 2),
                "total_samples_yielded": self.total_samples_yielded,
            },
            "layer1_case_control": {
                "checks": self.case_checks,
                "passed": self.case_passed,
                "pass_rate": round(self.case_pass_rate * 100, 2),
                "case1": {"detected": self.case1_detected, "filtered": self.case1_filtered},
                "case2": {"detected": self.case2_detected, "filtered": self.case2_filtered},
                "case3": {"detected": self.case3_detected, "filtered": self.case3_filtered},
                "case4": {"detected": self.case4_detected, "filtered": self.case4_filtered},
                "case5": {"detected": self.case5_detected, "filtered": self.case5_filtered},
            },
            "layer2_quality": {
                "checks": self.quality_checks,
                "passed": self.quality_passed,
                "pass_rate": round(self.quality_pass_rate * 100, 2),
                "skipped": {
                    "no_moshi": self.quality_no_moshi,
                    "low_moshi_words": self.quality_low_moshi_words,
                    "low_moshi_duration": self.quality_low_moshi_duration,
                    "low_moshi_ratio": self.quality_low_moshi_ratio,
                    "low_user_words": self.quality_low_user_words,
                    "short_segment": self.quality_short_segment,
                    "long_segment": self.quality_long_segment,
                },
            },
            "layer3_preferences": {
                "checks": self.pref_checks,
                "passed": self.pref_passed,
                "pass_rate": round(self.pref_pass_rate * 100, 2),
                "skipped": {
                    "user_first": self.pref_user_first_skipped,
                    "single_speaker": self.pref_single_speaker_skipped,
                },
            },
            "layer4_role_swapping": {
                "original_yielded": self.swap_original_yielded,
                "swapped_yielded": self.swap_swapped_yielded,
                "post_check_passed": self.swap_post_check_passed,
                "post_check_failed": self.swap_post_check_failed,
                "data_multiplier": round(self.data_multiplier, 2),
            },
        }

    def log_summary(self, log_fn=None, verbosity: int = 1):
        """
        Log filtering statistics summary.

        Args:
            log_fn: Logging function (default: logger.info)
            verbosity: 0=minimal, 1=summary, 2=detailed, 3=debug
        """
        if log_fn is None:
            log_fn = logger.info

        # =====================================================================
        # Header
        # =====================================================================
        log_fn("╔══════════════════════════════════════════════════════════════════╗")
        log_fn("║               SEGMENT FILTERING STATISTICS                        ║")
        log_fn("╚══════════════════════════════════════════════════════════════════╝")

        # =====================================================================
        # Meta Statistics (always shown)
        # =====================================================================
        log_fn(f"  [Stats] Total Segments:  {self.total_segments:,}")
        log_fn(f"  [OK] Passed Segments:    {self.passed_segments:,} ({100*(1-self.skip_rate):.1f}%)")
        log_fn(f"  [X] Skip Rate:           {100*self.skip_rate:.1f}%")
        log_fn(f"  >>> Samples Yielded:     {self.total_samples_yielded:,}")

        if verbosity >= 1:
            # =================================================================
            # Layer 1: Case Control
            # =================================================================
            log_fn("  ─────────────────────────────────────────────────────────────────")
            log_fn("  Layer 1: CASE CONTROL (Structural Validity)")
            log_fn(f"    Checks: {self.case_checks:,}  →  Passed: {self.case_passed:,} ({self.case_pass_rate*100:.1f}%)")

            if verbosity >= 2:
                log_fn("    ┌────────┬──────────────────────────────┬──────────┬──────────┐")
                log_fn("    │  Case  │  Description                 │ Detected │ Filtered │")
                log_fn("    ├────────┼──────────────────────────────┼──────────┼──────────┤")
                log_fn(f"    │  1     │  Full dialogue               │  {self.case1_detected:>6,} │  {self.case1_filtered:>6,} │")
                log_fn(f"    │  2     │  Moshi monologue             │  {self.case2_detected:>6,} │  {self.case2_filtered:>6,} │")
                log_fn(f"    │  3     │  User audio only             │  {self.case3_detected:>6,} │  {self.case3_filtered:>6,} │")
                log_fn(f"    │  4     │  Missing Moshi audio         │  {self.case4_detected:>6,} │  {self.case4_filtered:>6,} │")
                log_fn(f"    │  5     │  Missing Moshi text          │  {self.case5_detected:>6,} │  {self.case5_filtered:>6,} │")
                log_fn("    └────────┴──────────────────────────────┴──────────┴──────────┘")

            # =================================================================
            # Layer 2: Quality
            # =================================================================
            log_fn("  ─────────────────────────────────────────────────────────────────")
            log_fn("  Layer 2: QUALITY (Hard Minimums)")
            log_fn(f"    Checks: {self.quality_checks:,}  →  Passed: {self.quality_passed:,} ({self.quality_pass_rate*100:.1f}%)")

            if verbosity >= 2:
                skips = [
                    ("No Moshi speech", self.quality_no_moshi),
                    ("Low Moshi words", self.quality_low_moshi_words),
                    ("Low Moshi duration", self.quality_low_moshi_duration),
                    ("Low Moshi ratio", self.quality_low_moshi_ratio),
                    ("Low User words", self.quality_low_user_words),
                    ("Short segment", self.quality_short_segment),
                    ("Long segment", self.quality_long_segment),
                ]
                for reason, count in skips:
                    if count > 0:
                        log_fn(f"    - {reason}: {count:,}")

            # =================================================================
            # Layer 3: Preferences
            # =================================================================
            if self.pref_checks > 0:
                log_fn("  ─────────────────────────────────────────────────────────────────")
                log_fn("  Layer 3: PREFERENCES (Probabilistic)")
                log_fn(f"    Checks: {self.pref_checks:,}  →  Passed: {self.pref_passed:,} ({self.pref_pass_rate*100:.1f}%)")

                if verbosity >= 2:
                    if self.pref_user_first_skipped > 0:
                        log_fn(f"    - User speaks first: {self.pref_user_first_skipped:,}")
                    if self.pref_single_speaker_skipped > 0:
                        log_fn(f"    - Single speaker: {self.pref_single_speaker_skipped:,}")

            # =================================================================
            # Layer 4: Role Swapping
            # =================================================================
            if self.swap_original_yielded > 0 or self.swap_swapped_yielded > 0:
                log_fn("  ─────────────────────────────────────────────────────────────────")
                log_fn("  Layer 4: ROLE SWAPPING (Data Augmentation)")
                log_fn(f"    Original samples: {self.swap_original_yielded:,}")
                log_fn(f"    Swapped samples:  {self.swap_swapped_yielded:,}")
                log_fn(f"    Data multiplier:  {self.data_multiplier:.2f}x")

                if self.swap_post_check_failed > 0:
                    log_fn(f"    Post-swap recheck: passed={self.swap_post_check_passed}, failed={self.swap_post_check_failed}")

        log_fn("  ═══════════════════════════════════════════════════════════════════")


# Global filtering statistics tracker
_filtering_stats = FilteringStatistics()


def get_filtering_statistics() -> FilteringStatistics:
    """Get the global filtering statistics tracker."""
    return _filtering_stats


def reset_filtering_statistics():
    """Reset the global filtering statistics tracker."""
    global _filtering_stats
    _filtering_stats = FilteringStatistics()


# =============================================================================
# FILTERING LOGGER - Advanced Logging System (Layer 5)
# =============================================================================

class FilteringLogger:
    """
    Advanced logging system for the segment filtering pipeline.

    Features:
        - Verbosity control (0=minimal, 1=summary, 2=detailed, 3=debug)
        - First-filter/first-pass logging for quick debugging
        - Optional file output for persistent logging
        - Per-layer logging control for precise debugging

    Usage:
        logger = FilteringLogger(config.logging, run_dir="/path/to/run")
        logger.log_case_detection(case_num, is_allowed, segment_info)
        logger.log_quality_check(stats, skip_reason)
        logger.log_epoch_end(stats)
    """

    def __init__(
        self,
        config,  # FilteringLoggerArgs
        run_dir: str | Path | None = None,
    ):
        """
        Initialize the filtering logger.

        Args:
            config: FilteringLoggerArgs configuration
            run_dir: Directory for log file output (if save_to_file=True)
        """
        self.config = config
        self.enabled = config.enabled
        self.verbosity = config.verbosity

        # Track first events for log_first_* options
        self._first_filter_logged = False
        self._first_pass_logged = False
        self._first_case_logged = {1: False, 2: False, 3: False, 4: False, 5: False}

        # File logging setup
        self._file_handler = None
        if config.save_to_file and run_dir is not None:
            log_dir = Path(run_dir) / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            log_path = log_dir / config.log_filename

            # Create file handler
            self._file_handler = logging.FileHandler(log_path, mode='a', encoding='utf-8')
            self._file_handler.setFormatter(
                logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            )
            logger.addHandler(self._file_handler)
            logger.info(f"[FILTER LOG] Logging to: {log_path}")

    def close(self):
        """Close file handler if open."""
        if self._file_handler is not None:
            self._file_handler.close()
            logger.removeHandler(self._file_handler)
            self._file_handler = None

    def _should_log(self, layer_flag: bool, min_verbosity: int = 1) -> bool:
        """Check if logging should occur based on config and verbosity."""
        if not self.enabled:
            return False
        if self.verbosity < min_verbosity:
            return False
        return layer_flag

    def log_init(self, segment_filtering):
        """Log initialization of segment filtering system."""
        if not self.enabled:
            return

        logger.info("┌─────────────────────────────────────────────────────────────────┐")
        logger.info("│          SEGMENT FILTERING SYSTEM INITIALIZED                   │")
        logger.info("└─────────────────────────────────────────────────────────────────┘")
        logger.info(f"  Layer 1 (Case Control):   {'[ON]' if segment_filtering.case_control.enabled else '[OFF]'}")
        logger.info(f"  Layer 2 (Quality):        {'[ON]' if segment_filtering.quality.enabled else '[OFF]'}")
        logger.info(f"  Layer 3 (Preferences):    {'[ON]' if segment_filtering.preferences.enabled else '[OFF]'}")
        logger.info(f"  Layer 4 (Role Swapping):  {'[ON]' if segment_filtering.role_swapping.enabled else '[OFF]'}")
        logger.info(f"  Layer 5 (Logging):        verbosity={self.verbosity}")

        if segment_filtering.case_control.enabled:
            cc = segment_filtering.case_control
            logger.info(f"    Case control: allow=[1:{cc.allow_case1}, 2:{cc.allow_case2}, 3:{cc.allow_case3}, 4:{cc.allow_case4}, 5:{cc.allow_case5}]")

        if segment_filtering.quality.enabled:
            q = segment_filtering.quality
            logger.info(f"    Quality: min_moshi_words={q.min_moshi_words}, min_user_words={q.min_user_words}, min_duration={q.min_segment_duration_sec}s")

        if segment_filtering.role_swapping.enabled:
            rs = segment_filtering.role_swapping
            logger.info(f"    Role swapping: yield_both={rs.yield_both}, recheck_after_swap={rs.recheck_after_swap}")

    def log_case_detection(
        self,
        case_num: int,
        is_allowed: bool,
        segment_path: str = "",
        details: dict | None = None,
    ):
        """Log case detection result."""
        if not self._should_log(self.config.log_case_detection, min_verbosity=2):
            # Check for first filter/pass
            if self.config.log_first_filter and not is_allowed and not self._first_filter_logged:
                self._first_filter_logged = True
                logger.warning(f"[FIRST FILTER] Case {case_num} not allowed: {segment_path}")
                if details:
                    logger.warning(f"  Details: {details}")
            elif self.config.log_first_pass and is_allowed and not self._first_case_logged.get(case_num, False):
                self._first_case_logged[case_num] = True
                logger.info(f"[FIRST PASS] Case {case_num} detected and allowed: {segment_path}")
            return

        status = "[ALLOWED]" if is_allowed else "[FILTERED]"
        logger.debug(f"[CASE {case_num}] {status}: {segment_path}")
        if details and self.verbosity >= 3:
            logger.debug(f"  {details}")

    def log_quality_check(
        self,
        stats,  # SegmentStatistics
        skip_reason: str | None,
        is_swapped: bool = False,
    ):
        """Log quality check result."""
        if not self._should_log(self.config.log_quality_checks, min_verbosity=2):
            return

        swap_label = "[SWAPPED] " if is_swapped else ""
        if skip_reason is None:
            logger.debug(
                f"[QUALITY] {swap_label}PASSED: "
                f"moshi_words={stats.moshi_word_count}, user_words={stats.user_word_count}, "
                f"moshi_dur={stats.moshi_duration_sec:.1f}s"
            )
        else:
            logger.debug(f"[QUALITY] {swap_label}SKIPPED: {skip_reason}")

    def log_preferences_check(
        self,
        skip_reason: str | None,
        first_speaker: str = "",
        has_both: bool = True,
    ):
        """Log preferences check result."""
        if not self._should_log(self.config.log_quality_checks, min_verbosity=2):
            return

        if skip_reason is None:
            logger.debug(f"[PREFS] PASSED: first_speaker={first_speaker}, has_both={has_both}")
        else:
            logger.debug(f"[PREFS] SKIPPED: {skip_reason}")

    def log_role_swapping(
        self,
        action: str,  # "swap", "yield_original", "yield_swapped", "recheck_pass", "recheck_fail"
        segment_path: str = "",
        details: str = "",
    ):
        """Log role swapping operation."""
        if not self._should_log(self.config.log_role_swapping, min_verbosity=2):
            return

        symbols = {
            "swap": "<->",
            "yield_original": ">>>",
            "yield_swapped": "<~>",
            "recheck_pass": "[OK]",
            "recheck_fail": "[X]",
        }
        symbol = symbols.get(action, "•")
        logger.debug(f"[SWAP] {symbol} {action}: {segment_path} {details}")

    def log_first_filter(self, skip_reason: str, segment_path: str):
        """Log first filtered segment (for quick debugging)."""
        if not self.config.log_first_filter or self._first_filter_logged:
            return

        self._first_filter_logged = True
        logger.warning("┌─────────────────────────────────────────────────────────────────┐")
        logger.warning("│                    FIRST FILTERED SEGMENT                        │")
        logger.warning("└─────────────────────────────────────────────────────────────────┘")
        logger.warning(f"  Path:   {segment_path}")
        logger.warning(f"  Reason: {skip_reason}")

    def log_first_pass(self, segment_path: str, stats):
        """Log first passed segment (for quick debugging)."""
        if not self.config.log_first_pass or self._first_pass_logged:
            return

        self._first_pass_logged = True
        logger.info("┌─────────────────────────────────────────────────────────────────┐")
        logger.info("│                     FIRST PASSED SEGMENT                         │")
        logger.info("└─────────────────────────────────────────────────────────────────┘")
        logger.info(f"  Path:        {segment_path}")
        logger.info(f"  Moshi words: {stats.moshi_word_count}")
        logger.info(f"  User words:  {stats.user_word_count}")
        logger.info(f"  Duration:    {stats.segment_duration_sec:.1f}s")

    def log_epoch_end(self, stats: FilteringStatistics):
        """Log epoch end summary."""
        if not self._should_log(self.config.log_epoch_summary, min_verbosity=1):
            return

        stats.log_summary(log_fn=logger.info, verbosity=self.verbosity)

    def save_stats_json(self, stats: FilteringStatistics, run_dir: str | Path):
        """Save statistics as JSON file."""
        if not self.config.save_to_file:
            return

        import json
        stats_path = Path(run_dir) / "logs" / "filtering_stats.json"
        stats_path.parent.mkdir(parents=True, exist_ok=True)

        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats.to_dict(), f, indent=2, ensure_ascii=False)

        logger.info(f"[FILTER LOG] Statistics saved to: {stats_path}")


@dataclass
class Sample:
    """
    A single training sample with codes and metadata.

    For data augmentation, includes:
    - statistics: Segment quality statistics (for filtering decisions)
    - is_role_swapped: Whether this sample has swapped Moshi/User roles
    - skip_reason: Reason for skipping (if filtered out)

    For speaker conditioning, includes:
    - speaker_reference_audio: Reference audio for speaker embedding extraction
    - speaker_reference_text: Reference text (if available)
    """
    codes: torch.Tensor
    condition_attributes: ConditionAttributes | None = None
    user_text_alignments: list | None = None  # User's text alignments for reference
    moshi_text_raw: str | None = None  # Original Moshi text (no truncation) for logging/WER

    # Data augmentation metadata
    statistics: SegmentStatistics | None = None  # Segment quality statistics
    is_role_swapped: bool = False  # True if Moshi/User roles were swapped
    skip_reason: str | None = None  # Reason for skipping (None if not skipped)

    # Source tracking for dialogue saving
    audio_path: str | None = None  # Original audio file path for complete dialogue saving

    # Speaker conditioning reference (Phase 2)
    speaker_reference_audio: torch.Tensor | None = None  # [T] at 16kHz for speaker encoder
    speaker_reference_text: str | None = None  # Optional reference text
    speaker_reference_start_sec: float = 0.0  # Reference segment start time (in source audio)
    speaker_reference_end_sec: float = 0.0  # Reference segment end time (in source audio)


@dataclass
class Batch:
    codes: torch.Tensor
    condition_attributes: list[ConditionAttributes] | None = None
    user_text_alignments: list[list] | None = None  # Per-sample User text alignments
    moshi_text_raw_list: list[str] | None = None  # Per-sample original Moshi text
    audio_paths: list[str] | None = None  # Per-sample original audio file paths

    # Speaker conditioning (Phase 2)
    speaker_reference_audios: list[torch.Tensor] | None = None  # Per-sample reference audio
    speaker_reference_texts: list[str] | None = None  # Per-sample reference text
    speaker_reference_start_secs: list[float] | None = None  # Per-sample reference start time
    speaker_reference_end_secs: list[float] | None = None  # Per-sample reference end time

    @classmethod
    def collate(cls, batch: list[Sample]) -> "Batch":
        codes = torch.cat([b.codes for b in batch])
        condition_attrs = None
        user_alignments = None
        moshi_texts = None
        audio_paths = None
        speaker_ref_audios = None
        speaker_ref_texts = None
        speaker_ref_start_secs = None
        speaker_ref_end_secs = None

        if batch[0].condition_attributes is not None:
            condition_attrs = [b.condition_attributes for b in batch]

        # Collect user_text_alignments from each sample
        if any(b.user_text_alignments is not None for b in batch):
            user_alignments = [b.user_text_alignments for b in batch]

        # Collect moshi_text_raw from each sample
        if any(b.moshi_text_raw is not None for b in batch):
            moshi_texts = [b.moshi_text_raw for b in batch]

        # Collect audio_paths from each sample
        if any(b.audio_path is not None for b in batch):
            audio_paths = [b.audio_path for b in batch]

        # Collect speaker reference audios, texts, and timing (Phase 2)
        if any(b.speaker_reference_audio is not None for b in batch):
            speaker_ref_audios = [b.speaker_reference_audio for b in batch]
        if any(b.speaker_reference_text is not None for b in batch):
            speaker_ref_texts = [b.speaker_reference_text for b in batch]
        if any(b.speaker_reference_start_sec > 0 for b in batch):
            speaker_ref_start_secs = [b.speaker_reference_start_sec for b in batch]
            speaker_ref_end_secs = [b.speaker_reference_end_sec for b in batch]

        return Batch(
            codes, condition_attrs, user_alignments, moshi_texts, audio_paths,
            speaker_ref_audios, speaker_ref_texts, speaker_ref_start_secs, speaker_ref_end_secs
        )


def tokenize(
    tokenizer: sentencepiece.SentencePieceProcessor,
    text: str,
    bos: bool = True,
    alpha: float | None = None,
):
    """Tokenize the given string, accounting for new lines, potentially adding a BOS token."""
    nl_piece = tokenizer.encode("\n")[-1]
    if alpha is not None:
        tokens = tokenizer.encode(
            text.split("\n"), enable_sampling=True, alpha=alpha, nbest_size=-1
        )
    else:
        tokens = tokenizer.encode(text.split("\n"))
    tokens = reduce(lambda a, b: [*a, nl_piece, *b], tokens)
    if bos:
        tokens = [tokenizer.bos_id(), *tokens]
    return tokens


class Interleaver:
    """
    Text-Audio Interleaver for precise alignment of text tokens with audio frames.

    This class handles the critical task of aligning text tokens (from SentencePiece
    tokenization) with audio frames (12.5Hz = 80ms per frame). The alignment quality
    directly impacts the Inner Monologue learning.

    Key Features:
        - Character-level interpolation (J-Moshi style): Distributes word timestamps
          to character level for more precise token placement
        - Adaptive token distribution: Prevents token loss when text exceeds frames
        - Overflow detection and logging: Tracks alignment quality metrics

    Args:
        tokenizer: SentencePiece text tokenizer used by the model.
        audio_frame_rate (float): Frame rate of the audio tokenizer (typically 12.5Hz).
        text_padding (int): Special token used for text padding (no text at this frame).
        end_of_text_padding (int): Special token used to mark the frame before text.
        zero_padding (int): Special token for no-input positions (silence).
        in_word_padding (int | None): Padding used within a word segment.
            Defaults to text_padding if None.
        keep_main_only (bool): If True, only keep alignments from the main speaker.
            This is essential for Inner Monologue training where only Moshi's
            text should be in the text stream. Default: False.
        main_speaker_label (str): Label identifying the main speaker in alignments.
            Default: "SPEAKER_MAIN".
        use_bos_eos (bool): If True, inserts BOS/EOS tokens at speaker turns.
            Default: False.
        keep_and_shift (bool): Token queue behavior when words overlap.
            If True: Extend queue (keeps all tokens, may delay placement).
            If False: Replace queue (original Moshi/J-Moshi behavior).
            Default: False (recommended for timing accuracy).
        audio_delay (float): Delay between text and audio in seconds.
            Positive value means text is ahead of audio. Default: 0.0.
        proba (float): Probability of keeping the text. Default: 1.0.
        device: Device location for output tensors. Default: "cuda".
        adaptive_distribute (bool): If True, distribute tokens evenly when overflow
            is detected. This prevents token loss for fast speech. Default: False.
        warn_on_overflow (bool): If True, log warnings when token overflow occurs.
            Useful for monitoring alignment quality. Default: True.
        character_level_interpolation (bool): If True, apply J-Moshi style
            character-level timestamp interpolation. This converts word-level
            timestamps to character-level for more precise token placement.
            Recommended for Korean/Japanese. Default: False.
    """

    def __init__(
        self,
        tokenizer: sentencepiece.SentencePieceProcessor,
        audio_frame_rate: float,
        text_padding: int,
        end_of_text_padding: int,
        zero_padding: int,
        in_word_padding: int | None = None,
        keep_main_only: bool = False,
        main_speaker_label: str = "SPEAKER_MAIN",
        use_bos_eos: bool = False,
        keep_and_shift: bool = False,
        audio_delay: float = 0.0,
        proba: float = 1.0,
        device: str | torch.device = "cuda",
        adaptive_distribute: bool = False,
        warn_on_overflow: bool = True,
        character_level_interpolation: bool = False,
    ):
        self.tokenizer = tokenizer
        self.audio_frame_rate = audio_frame_rate
        self.text_padding = text_padding
        self.end_of_text_padding = end_of_text_padding
        self.zero_padding = zero_padding
        self.in_word_padding = (
            self.text_padding if in_word_padding is None else in_word_padding
        )
        self.keep_main_only = keep_main_only
        self.main_speaker_label = main_speaker_label
        self.use_bos_eos = use_bos_eos
        self.keep_and_shift = keep_and_shift
        self.audio_delay = audio_delay
        self.proba = proba
        self.device = device
        self.adaptive_distribute = adaptive_distribute
        self.warn_on_overflow = warn_on_overflow
        self.character_level_interpolation = character_level_interpolation

        # Overflow statistics tracking
        self._overflow_count = 0
        self._total_segments = 0
        self._total_tokens_lost = 0

        # Log configuration on first use
        if character_level_interpolation:
            logger.info(
                "[INTERLEAVER] Character-level interpolation enabled - "
                "word timestamps will be distributed to character level"
            )

    @property
    def special_tokens(self) -> set[int]:
        """Return the set of special tokens used by this interleaver."""
        return {
            self.text_padding,
            self.end_of_text_padding,
            self.tokenizer.bos_id(),
            self.tokenizer.eos_id(),
            self.zero_padding,
            self.in_word_padding,
        }

    def get_overflow_stats(self) -> dict:
        """Get statistics about token overflow events.

        Returns:
            dict with keys:
                - total_segments: Total number of segments processed
                - overflow_count: Number of segments with token overflow
                - overflow_rate: Percentage of segments with overflow
                - total_tokens_lost: Total number of tokens lost due to overflow
        """
        overflow_rate = (
            self._overflow_count / self._total_segments * 100
            if self._total_segments > 0
            else 0.0
        )
        return {
            "total_segments": self._total_segments,
            "overflow_count": self._overflow_count,
            "overflow_rate": overflow_rate,
            "total_tokens_lost": self._total_tokens_lost,
        }

    def _word_to_character_alignments(
        self, alignments: list[Alignment]
    ) -> list[CharacterAlignment]:
        """
        Convert word-level alignments to character-level alignments (J-Moshi style).

        This method distributes word timestamps evenly across characters,
        enabling more precise token placement. For Korean and Japanese,
        where subword tokens may span multiple characters, this approach
        provides better timing than word-level alignment.

        Algorithm:
            For each word with start_time and end_time:
            1. Count the number of characters
            2. Calculate per-character duration = (end - start) / num_chars
            3. Assign each character its interpolated time range

        Example:
            Word "안녕" at (0.0, 0.5s) with 2 characters:
            - "안" → (0.0, 0.25)
            - "녕" → (0.25, 0.5)

        Args:
            alignments: Word-level alignments [(word, (start, end), speaker), ...]

        Returns:
            Character-level alignments [(char, start, end), ...]
        """
        char_alignments: list[CharacterAlignment] = []

        for word, (start, end), speaker in alignments:
            word_stripped = word.strip()
            if not word_stripped:
                continue

            num_chars = len(word_stripped)
            if num_chars == 0:
                continue

            # Calculate per-character duration
            duration = end - start
            char_duration = duration / num_chars

            # Create character-level alignments
            for i, char in enumerate(word_stripped):
                char_start = start + i * char_duration
                char_end = start + (i + 1) * char_duration
                char_alignments.append((char, char_start, char_end))

        return char_alignments

    def _tokenize_with_character_interpolation(
        self, alignments: list[Alignment]
    ) -> list[TokenizedAlignment]:
        """
        Tokenize with character-level timestamp interpolation.

        This method:
        1. Converts word-level timestamps to character-level
        2. Tokenizes the full text to get token-character mapping
        3. Assigns each token a timestamp based on its constituent characters

        The key insight is that SentencePiece tokens may span multiple
        characters (e.g., "안녕" → ["▁안", "녕"]). By having character-level
        timestamps, we can assign more accurate timing to each token.

        Algorithm:
            1. Convert words to character-level alignments
            2. Build full text and track character positions
            3. Tokenize full text (no BOS for internal tokens)
            4. Map each token back to character timestamps
            5. Assign token timestamp as the start time of its first character

        Args:
            alignments: Word-level alignments [(word, (start, end), speaker), ...]

        Returns:
            Token-level alignments [(token_ids, (start, end), speaker), ...]
        """
        if not alignments:
            return []

        # Step 1: Convert to character-level alignments
        char_alignments = self._word_to_character_alignments(alignments)
        if not char_alignments:
            return []

        # Step 2: Build full text and character-to-time mapping
        full_text = "".join(char for char, _, _ in char_alignments)
        char_start_times = [start for _, start, _ in char_alignments]
        char_end_times = [end for _, _, end in char_alignments]

        # Step 3: Tokenize the full text
        # We tokenize without BOS here; BOS handling is done in _insert_bos_eos
        tokens = tokenize(self.tokenizer, full_text, bos=False)
        if not tokens:
            return []

        # Step 4: Map tokens back to characters using SentencePiece decode
        # Get the string representation of each token to find its character span
        tokenized_alignments: list[TokenizedAlignment] = []
        char_idx = 0

        for token_id in tokens:
            # Decode single token to get its string representation
            token_str = self.tokenizer.decode([token_id])
            # Remove leading space marker (▁) for length calculation
            token_chars = token_str.lstrip("▁ ")

            if not token_chars:
                # Handle special tokens or space-only tokens
                token_chars = " "

            token_len = len(token_chars)

            # Find the character range for this token
            if char_idx < len(char_start_times):
                start_time = char_start_times[char_idx]
                # End index is the last character of this token
                end_char_idx = min(char_idx + token_len - 1, len(char_end_times) - 1)
                end_time = char_end_times[end_char_idx]

                # Create token alignment with single token
                # Note: Each token gets its own alignment entry for frame-level placement
                tokenized_alignments.append(
                    ([token_id], (start_time, end_time), alignments[0][2])
                )

                char_idx += token_len
            else:
                # Fallback: use the last available timestamp
                if char_end_times:
                    start_time = char_end_times[-1]
                    end_time = char_end_times[-1] + 0.08  # One frame duration
                    tokenized_alignments.append(
                        ([token_id], (start_time, end_time), alignments[0][2])
                    )

        return tokenized_alignments

    def _count_total_tokens(
        self, alignments: list[TokenizedAlignment] | None
    ) -> int:
        """Count total tokens in alignments."""
        if alignments is None or len(alignments) == 0:
            return 0
        return sum(len(a[0]) for a in alignments)

    def _build_adaptive_token_stream(
        self,
        alignments: list[TokenizedAlignment],
        segment_duration: float,
    ) -> torch.Tensor:
        """Build token stream with adaptive distribution to prevent overflow.

        This method distributes tokens more evenly across the available frames
        when the total token count exceeds the frame count, while still
        respecting the general timing order of words.

        Args:
            alignments: Tokenized alignments with timing info
            segment_duration: Duration of the segment in seconds

        Returns:
            Token stream tensor [1, 1, T]
        """
        T = math.ceil(segment_duration * self.audio_frame_rate)
        total_tokens = self._count_total_tokens(alignments)

        # If no overflow, fall back to standard method
        if total_tokens <= T:
            return self._build_standard_token_stream(alignments, segment_duration)

        # Adaptive distribution: spread tokens evenly while preserving order
        all_tokens = []
        for toks, ts, speaker in alignments:
            all_tokens.extend(toks)

        # Calculate distribution step (how many frames per token)
        # Use floor division to ensure we fit all tokens
        step = T / total_tokens

        text_tokens = [self.text_padding] * T
        for i, token in enumerate(all_tokens):
            frame_idx = min(int(i * step), T - 1)
            # If this frame is already occupied, find next available
            while frame_idx < T and text_tokens[frame_idx] != self.text_padding:
                frame_idx += 1
            if frame_idx < T:
                # Set end_of_text_padding before the token if applicable
                if frame_idx > 0 and text_tokens[frame_idx - 1] == self.text_padding:
                    text_tokens[frame_idx - 1] = self.end_of_text_padding
                text_tokens[frame_idx] = token

        # Log the adaptive distribution
        if self.warn_on_overflow:
            logger.warning(
                f"[INTERLEAVER] Adaptive distribution applied: "
                f"{total_tokens} tokens → {T} frames (step={step:.2f})"
            )

        return torch.tensor(text_tokens, device=self.device).view(1, 1, -1)

    def _build_standard_token_stream(
        self,
        alignments: list[TokenizedAlignment] | None,
        segment_duration: float,
    ) -> torch.Tensor:
        """Standard token stream building (original logic).

        This is the original build_token_stream logic, extracted for clarity.
        """
        T = math.ceil(segment_duration * self.audio_frame_rate)
        if alignments is None or len(alignments) == 0:
            text_tokens = [self.zero_padding] * T
        else:
            text_tokens = [self.text_padding] * T
            i = 0
            to_append_stack: deque = deque()
            last_word_end = -1
            for t in range(T):
                while (
                    i < len(alignments)
                    and alignments[i][1][0] * self.audio_frame_rate < t + 1
                ):
                    tokenized = alignments[i][0]
                    last_word_end = int(alignments[i][1][1] * self.audio_frame_rate)
                    if self.keep_and_shift:
                        to_append_stack.extend(tokenized)
                    else:
                        to_append_stack = deque(tokenized)
                    i += 1
                if to_append_stack:
                    if t > 0 and text_tokens[t - 1] in [
                        self.text_padding,
                        self.in_word_padding,
                    ]:
                        text_tokens[t - 1] = self.end_of_text_padding
                    next_token = to_append_stack.popleft()
                    text_tokens[t] = next_token
                elif t <= last_word_end:
                    text_tokens[t] = self.in_word_padding

            # Check for overflow (tokens remaining in queue)
            if len(to_append_stack) > 0:
                self._overflow_count += 1
                self._total_tokens_lost += len(to_append_stack)
                if self.warn_on_overflow:
                    logger.warning(
                        f"[INTERLEAVER] Token overflow detected: "
                        f"{len(to_append_stack)} tokens lost at segment end"
                    )

        if self.audio_delay < 0:
            prefix_length = int(self.audio_frame_rate * -self.audio_delay)
            text_tokens[:prefix_length] = [self.zero_padding] * prefix_length

        return torch.tensor(text_tokens, device=self.device).view(1, 1, -1)

    def _tokenize(self, alignments: list[Alignment]) -> list[TokenizedAlignment]:
        """
        Tokenize word-level alignments into token-level alignments.

        If character_level_interpolation is enabled, uses J-Moshi style
        character-level timestamp distribution for more precise token placement.

        Args:
            alignments: Word-level alignments [(word, (start, end), speaker), ...]

        Returns:
            Token-level alignments [(token_ids, (start, end), speaker), ...]
        """
        if self.character_level_interpolation:
            return self._tokenize_with_character_interpolation(alignments)

        # Original word-level tokenization
        out = []
        for word, ts, speaker in alignments:
            toks = tokenize(self.tokenizer, word.strip(), bos=False)
            out.append((toks, ts, speaker))
        return out

    def _keep_main_only(
        self, alignments: list[TokenizedAlignment], main_speaker: str
    ) -> list[TokenizedAlignment]:
        return [a for a in alignments if a[2] == main_speaker]

    def _keep_those_with_duration(
        self, alignments: list[TokenizedAlignment]
    ) -> list[TokenizedAlignment]:
        # Removes all words with negative or 0 durations.
        return [a for a in alignments if a[1][0] < a[1][1]]

    def _add_delay(
        self, alignments: list[TokenizedAlignment]
    ) -> list[TokenizedAlignment]:
        # Delay the audio with respect to the text, e.g. positive values mean the audio is late on the text.
        return [
            (a[0], (a[1][0] - self.audio_delay, a[1][1] - self.audio_delay), a[2])
            for a in alignments
            if a[1][1] > self.audio_delay
        ]

    def _insert_bos_eos(
        self, alignments: list[TokenizedAlignment], main_speaker: str
    ) -> list[TokenizedAlignment]:
        # EOS and BOS is different from what it was in the old Interleaver, it is now symmetrical:
        # if the main speaker talks after another speaker (or is the first to talk), BOS is prepended to the first word.
        # Similary, if any other speaker speaks either first, or after the main speaker, a EOS is prepended.
        # This is in contrast with the legacy Interleaver, where the EOS would be inserted immediately
        # at the end of the turn of the main speaker.
        out: list[TokenizedAlignment] = []
        last_speaker = None
        for toks, ts, speaker in alignments:
            toks = list(toks)
            if speaker == last_speaker:
                pass
            elif speaker == main_speaker:
                toks.insert(0, self.tokenizer.bos_id())
            elif last_speaker == main_speaker:
                assert out
                toks.insert(0, self.tokenizer.eos_id())
            last_speaker = speaker
            out.append((toks, ts, speaker))
        return out

    def build_token_stream(
        self,
        alignments: list[TokenizedAlignment] | None,
        segment_duration: float,
    ) -> torch.Tensor:
        """Builds the token stream from the tokenized alignments.

        This method now supports two modes:
        1. Standard mode (default): Original timing-based placement with overflow detection
        2. Adaptive mode (adaptive_distribute=True): Distributes tokens evenly when
           overflow is detected to prevent token loss

        The method tracks statistics about overflow events which can be retrieved
        via get_overflow_stats().
        """
        # Track total segments processed
        self._total_segments += 1

        # Check if adaptive distribution should be used
        if self.adaptive_distribute and alignments is not None and len(alignments) > 0:
            T = math.ceil(segment_duration * self.audio_frame_rate)
            total_tokens = self._count_total_tokens(alignments)

            # Use adaptive distribution only when overflow would occur
            if total_tokens > T:
                return self._build_adaptive_token_stream(alignments, segment_duration)

        # Use standard token stream building
        return self._build_standard_token_stream(alignments, segment_duration)

    def prepare_item(
        self,
        alignments: list[Alignment] | None,
        segment_duration: float,
        main_speaker: str | None = None,
    ) -> torch.Tensor:
        """Responsible with processing the alignments and calling `build_token_stream`."""
        if alignments is None:
            tokenized = None
        else:
            tokenized = self._tokenize(sorted(alignments, key=lambda x: x[1][0]))
            if self.keep_main_only:
                main_speaker = main_speaker or self.main_speaker_label
                tokenized = self._keep_main_only(tokenized, main_speaker)
            elif self.use_bos_eos:
                main_speaker = main_speaker or self.main_speaker_label
                tokenized = self._insert_bos_eos(tokenized, main_speaker)
            tokenized = self._keep_those_with_duration(tokenized)
            if self.audio_delay != 0:
                tokenized = self._add_delay(tokenized)
        return self.build_token_stream(tokenized, segment_duration)


def dicho(alignment, val, i=0, j=None):
    if j is None:
        j = len(alignment)
    if i == j:
        return i
    k = (i + j) // 2
    if alignment[k][1][0] < val:
        return dicho(alignment, val, k + 1, j)
    else:
        return dicho(alignment, val, i, k)


class InterleavedTokenizer:
    """
    Tokenizer that interleaves text and audio tokens.

    Args:
        mimi: Mimi audio codec model
        interleaver: Interleaver instance
        duration_sec: Chunk duration in seconds
        jsonl_base_dir: Parent directory of JSONL file (used for path resolution)
        speaker_conditioning_config: Optional speaker conditioning configuration
    """

    def __init__(
        self,
        mimi,
        interleaver,
        duration_sec: float,
        jsonl_base_dir: str | Path | None = None,
        speaker_conditioning_config: Optional[dict] = None,
    ):
        self.mimi = mimi
        self.interleaver = interleaver
        self.duration_sec = duration_sec
        self.num_audio_frames = math.ceil(duration_sec * mimi.frame_rate)
        # Store parent directory of JSONL file (used for relative path resolution)
        self.jsonl_base_dir = Path(jsonl_base_dir) if jsonl_base_dir else None
        self._path_resolution_logged = False

        # Speaker conditioning configuration (Phase 2)
        self.speaker_conditioning_enabled = False
        self.speaker_ref_min_duration_sec = 3.0
        self.speaker_ref_max_duration_sec = 10.0
        self.speaker_ref_target_sample_rate = 16000  # Speaker encoder expects 16kHz

        if speaker_conditioning_config:
            self.speaker_conditioning_enabled = speaker_conditioning_config.get('enabled', False)
            ref_sampler_cfg = speaker_conditioning_config.get('reference_sampler', {})
            self.speaker_ref_min_duration_sec = ref_sampler_cfg.get('min_duration_sec', 3.0)
            self.speaker_ref_max_duration_sec = ref_sampler_cfg.get('max_duration_sec', 10.0)
            self.speaker_ref_target_sample_rate = ref_sampler_cfg.get('target_sample_rate', 16000)

            if self.speaker_conditioning_enabled:
                logger.info(
                    f"[InterleavedTokenizer] Speaker conditioning enabled: "
                    f"ref_duration={self.speaker_ref_min_duration_sec}-{self.speaker_ref_max_duration_sec}s, "
                    f"target_sr={self.speaker_ref_target_sample_rate}Hz"
                )

    def _resolve_json_path(self, wav_path: str) -> str:
        """
        Resolve JSON alignment file path from WAV path.

        The path returned by sphn.dataset_jsonl() can be:
        1. Absolute path (if sphn resolved it relative to JSONL)
        2. Relative path (as stored in JSONL)

        This function tries multiple possible paths to find the JSON file,
        including the data_preparation output format where alignments are
        in a separate 'alignments/' directory.

        Search order:
        1. Co-located JSON (same name as audio, e.g., audio/conv.wav -> audio/conv.json)
        2. alignments/ directory (Moshi format from data_preparation)
        3. alignment_speaker01/ directory (per-speaker format)
        """
        # Get base name without extension
        wav_basename = os.path.splitext(os.path.basename(wav_path))[0]
        json_suffix = os.path.splitext(wav_path)[0] + ".json"

        # Build list of candidate paths to try
        candidates = []

        # 1. Try as-is (absolute path or already resolved by sphn)
        candidates.append(json_suffix)

        # 2. If WAV is absolute path (resolved by sphn)
        if os.path.isabs(wav_path):
            candidates.append(os.path.splitext(wav_path)[0] + ".json")

            # Also try alignments/ directory relative to audio file
            wav_dir = Path(wav_path).parent
            base_dir = wav_dir.parent  # Go up from audio/ to parent dir

            # Try alignments/ directory (Moshi format from data_preparation)
            alignments_path = base_dir / "alignments" / f"{wav_basename}.json"
            candidates.append(str(alignments_path))

            # Try alignment_speaker01/ directory (per-speaker format)
            speaker01_path = base_dir / "alignment_speaker01" / f"{wav_basename}.json"
            candidates.append(str(speaker01_path))

        # 3. Resolve relative to JSONL base dir (most important!)
        if self.jsonl_base_dir:
            # Resolve relative path based on JSONL file's directory
            resolved = self.jsonl_base_dir / json_suffix
            candidates.append(str(resolved))

            # Try alignments/ directory in JSONL base dir
            alignments_path = self.jsonl_base_dir / "alignments" / f"{wav_basename}.json"
            candidates.append(str(alignments_path))

            # Try alignment_speaker01/ directory in JSONL base dir
            speaker01_path = self.jsonl_base_dir / "alignment_speaker01" / f"{wav_basename}.json"
            candidates.append(str(speaker01_path))

            # Also handle case where wav_path already contains path within JSONL base dir
            if not os.path.isabs(wav_path):
                # Example: wav_path = "audio/0.wav"
                # jsonl_base_dir = "data/train"
                # → "data/train/audio/0.json"
                candidates.append(str(self.jsonl_base_dir / json_suffix))

        # 4. Try relative to current working directory
        candidates.append(os.path.join(os.getcwd(), json_suffix))

        # Remove duplicates and find existing file
        seen = set()
        for candidate in candidates:
            if candidate in seen:
                continue
            seen.add(candidate)

            if os.path.exists(candidate):
                if not self._path_resolution_logged:
                    logger.info(f"JSON path resolved: {wav_path} -> {candidate}")
                    self._path_resolution_logged = True
                return candidate

        # 찾지 못한 경우 에러 메시지와 함께 예외 발생
        error_msg = (
            f"JSON alignment file not found for: {wav_path}\n"
            f"Tried paths:\n" + "\n".join(f"  - {c}" for c in seen)
        )
        if self.jsonl_base_dir:
            error_msg += f"\nJSONL base dir: {self.jsonl_base_dir}"

        raise FileNotFoundError(error_msg)

    def _sample_speaker_reference(
        self,
        full_audio: np.ndarray,
        alignments: list[Alignment],
        target_start_sec: float,
        target_end_sec: float,
        main_speaker_label: str = "SPEAKER_MAIN",
    ) -> tuple[torch.Tensor | None, str | None, float, float]:
        """
        Sample speaker reference audio from MOSHI channel, avoiding target segment.

        This method samples a random segment from the MOSHI (SPEAKER_MAIN) channel
        that doesn't overlap with the current training target segment.

        Args:
            full_audio: Full audio array (mono or stereo)
            alignments: Word-level alignments [(word, (start, end), speaker), ...]
            target_start_sec: Start of target segment to exclude
            target_end_sec: End of target segment to exclude
            main_speaker_label: Label for main speaker (default: "SPEAKER_MAIN")

        Returns:
            Tuple of (reference_audio_16khz, reference_text, ref_start_sec, ref_end_sec)
            or (None, None, 0.0, 0.0) if no valid region
        """
        import torchaudio.functional as F

        if not self.speaker_conditioning_enabled:
            return None, None, 0.0, 0.0

        sr = self.mimi.sample_rate  # 24000
        min_samples = int(self.speaker_ref_min_duration_sec * sr)
        max_samples = int(self.speaker_ref_max_duration_sec * sr)

        # Extract Moshi audio channel
        if full_audio.ndim == 2 and full_audio.shape[0] == 2:
            moshi_audio = full_audio[0]  # Left channel = Moshi
        elif full_audio.ndim == 2 and full_audio.shape[0] == 1:
            moshi_audio = full_audio[0]
        else:
            moshi_audio = full_audio

        total_samples = len(moshi_audio)
        total_duration_sec = total_samples / sr

        # Find valid MOSHI speech regions (not overlapping with target)
        buffer_sec = 0.5  # Avoid regions too close to target
        exclude_start_sec = max(0, target_start_sec - buffer_sec)
        exclude_end_sec = min(total_duration_sec, target_end_sec + buffer_sec)

        # Collect MOSHI speaker regions from alignments
        moshi_regions = []
        current_region_start = None
        current_region_end = None
        current_region_text = []

        for word, (start, end), speaker in alignments:
            if speaker == main_speaker_label:
                # Skip if overlaps with target region
                if start < exclude_end_sec and end > exclude_start_sec:
                    continue

                if current_region_start is None:
                    current_region_start = start
                    current_region_end = end
                    current_region_text = [word]
                elif start - current_region_end < 1.0:  # Merge if gap < 1 second
                    current_region_end = end
                    current_region_text.append(word)
                else:
                    # Save current region and start new
                    duration = current_region_end - current_region_start
                    if duration >= self.speaker_ref_min_duration_sec:
                        moshi_regions.append((
                            current_region_start,
                            current_region_end,
                            " ".join(current_region_text)
                        ))
                    current_region_start = start
                    current_region_end = end
                    current_region_text = [word]

        # Don't forget last region
        if current_region_start is not None:
            duration = current_region_end - current_region_start
            if duration >= self.speaker_ref_min_duration_sec:
                moshi_regions.append((
                    current_region_start,
                    current_region_end,
                    " ".join(current_region_text)
                ))

        if not moshi_regions:
            logger.debug("[Speaker Ref] No valid MOSHI regions found for reference sampling")
            return None, None, 0.0, 0.0

        # Random selection
        orig_start_sec, orig_end_sec, orig_ref_text = random.choice(moshi_regions)
        start_sec, end_sec = orig_start_sec, orig_end_sec

        # Clip to max duration
        duration = end_sec - start_sec
        if duration > self.speaker_ref_max_duration_sec:
            max_start = end_sec - self.speaker_ref_max_duration_sec
            start_sec = random.uniform(start_sec, max_start)
            end_sec = start_sec + self.speaker_ref_max_duration_sec

        # =================================================================
        # CRITICAL FIX: Re-extract text only for the clipped time range
        # =================================================================
        # When audio is clipped to max_duration, we must also extract only
        # the text that corresponds to the clipped audio segment.
        # Otherwise, reference_text contains text from the entire region
        # (e.g., 885 chars for a 3-second audio clip).
        # =================================================================
        ref_text_words = []
        for word, (word_start, word_end), speaker in alignments:
            if speaker != main_speaker_label:
                continue
            # Only include words that overlap with the clipped segment
            if word_end > start_sec and word_start < end_sec:
                ref_text_words.append(word)

        ref_text = " ".join(ref_text_words) if ref_text_words else ""

        # Extract audio segment
        start_sample = int(start_sec * sr)
        end_sample = min(int(end_sec * sr), total_samples)
        ref_audio = moshi_audio[start_sample:end_sample]

        # Convert to tensor and resample to 16kHz for speaker encoder
        ref_audio_tensor = torch.from_numpy(ref_audio.copy()).float()
        if sr != self.speaker_ref_target_sample_rate:
            ref_audio_tensor = F.resample(
                ref_audio_tensor.unsqueeze(0),
                sr,
                self.speaker_ref_target_sample_rate,
            ).squeeze(0)

        logger.debug(
            f"[Speaker Ref] Sampled reference: {start_sec:.2f}s-{end_sec:.2f}s "
            f"(duration={end_sec-start_sec:.2f}s, samples={ref_audio_tensor.shape[-1]}, "
            f"text_len={len(ref_text)} chars)"
        )

        return ref_audio_tensor, ref_text, start_sec, end_sec

    def __call__(self, wav: np.ndarray, start_sec: float, path: str) -> Sample:
        with torch.no_grad():
            # Moshiko (7B model) expects MONO audio:
            # - 9 codebooks total: text(1) + audio(8)
            # - For stereo input, we use the LEFT channel (Moshi's voice) only
            #   This matches K-Moshi data format: LEFT=Moshi/SPEAKER_MAIN, RIGHT=User

            # Initialize flag on first call
            if not hasattr(self, '_first_sample_logged'):
                self._first_sample_logged = False

            # CRITICAL DEBUG: Log first sample
            if not self._first_sample_logged:
                logger.warning(f"[INTERLEAVER DEBUG] ===== FIRST SAMPLE =====")
                logger.warning(
                    f"[INTERLEAVER DEBUG] wav input: shape={wav.shape}, ndim={wav.ndim}, "
                    f"dtype={wav.dtype}, min={wav.min():.4f}, max={wav.max():.4f}"
                )
                logger.warning(f"[INTERLEAVER DEBUG] path={path}, start_sec={start_sec}")
                logger.warning(
                    f"[INTERLEAVER DEBUG] mimi.sample_rate={self.mimi.sample_rate}, "
                    f"mimi.frame_rate={self.mimi.frame_rate}"
                )

            # Convert stereo to mono if needed (Moshiko only supports mono)
            if wav.ndim == 2 and wav.shape[0] == 2:
                # Stereo: use LEFT channel (Moshi's voice) for training
                # K-Moshi data format: LEFT=Moshi/SPEAKER_MAIN, RIGHT=User
                audio = wav[0]  # Left channel = Moshi's voice (SPEAKER_MAIN)
                if not self._first_sample_logged:
                    logger.warning(
                        f"[INTERLEAVER DEBUG] Stereo detected, using LEFT channel (Moshi voice)"
                    )
            elif wav.ndim == 2 and wav.shape[0] == 1:
                audio = wav.squeeze(0)
                if not self._first_sample_logged:
                    logger.warning(f"[INTERLEAVER DEBUG] Mono (1 channel) detected")
            else:
                audio = wav
                if not self._first_sample_logged:
                    logger.warning(f"[INTERLEAVER DEBUG] 1D audio detected")

            if not self._first_sample_logged:
                logger.warning(
                    f"[INTERLEAVER DEBUG] audio after conversion: shape={audio.shape}"
                )

            try:
                # Encode audio with Mimi → 8 codebooks
                audio_tensor = torch.Tensor(audio).cuda()

                if not self._first_sample_logged:
                    logger.warning(
                        f"[INTERLEAVER DEBUG] audio_tensor: shape={audio_tensor.shape}, "
                        f"dtype={audio_tensor.dtype}"
                    )

                # Mimi expects [B, C, T] = [batch, channels, samples]
                # Original code used [:, None] which creates [samples, 1] - WRONG!
                # Correct format: [None, None, :] creates [1, 1, samples] = [B, C, T]
                mimi_input = audio_tensor[None, None, :]

                if not self._first_sample_logged:
                    logger.warning(
                        f"[INTERLEAVER DEBUG] mimi_input shape: {mimi_input.shape} (should be [1, 1, T])"
                    )

                audio_tokens = self.mimi.encode(mimi_input)

                if not self._first_sample_logged:
                    logger.warning(
                        f"[INTERLEAVER DEBUG] audio_tokens after encode: shape={audio_tokens.shape}"
                    )

            except Exception as e:
                logger.error(f"[INTERLEAVER ERROR] Mimi encoding failed! {type(e).__name__}: {e}")
                logger.error(f"[INTERLEAVER ERROR] audio shape: {audio.shape}")
                raise

            audio_tokens = audio_tokens[..., : self.num_audio_frames]
            this_num_audio_frames = audio_tokens.shape[-1]
            audio_tokens = torch.nn.functional.pad(
                audio_tokens[..., : self.num_audio_frames],
                (0, self.num_audio_frames - this_num_audio_frames),
                value=self.interleaver.zero_padding,
            )
            audio_tokens = audio_tokens.view(1, -1, self.num_audio_frames)

            # Improved path resolution: Find JSON file
            info_file = self._resolve_json_path(path)
            with open(info_file) as f:
                data = json.load(f)
                alignments = data["alignments"]

            start_alignment = dicho(alignments, start_sec)
            end_alignment = dicho(alignments, start_sec + self.duration_sec)
            alignments = [
                (a[0], (a[1][0] - start_sec, a[1][1] - start_sec), a[2])
                for a in alignments[start_alignment:end_alignment]
            ]

            # Debug: Check for empty alignments (changed to debug level to reduce log noise)
            # These segments are filtered out by quality check anyway
            if len(alignments) == 0:
                logger.debug(
                    f"Empty alignments for {path} at {start_sec:.2f}s - "
                    f"will be filtered by quality check"
                )

            text_tokens = self.interleaver.prepare_item(
                alignments, this_num_audio_frames
            )
            text_tokens = torch.nn.functional.pad(
                text_tokens,
                (0, self.num_audio_frames - text_tokens.shape[-1]),
                value=self.interleaver.zero_padding,
            )

            codes = torch.cat([text_tokens, audio_tokens], dim=1)

            # Phase 2: Sample speaker reference audio if enabled
            speaker_reference_audio = None
            speaker_reference_text = None
            speaker_reference_start_sec = 0.0
            speaker_reference_end_sec = 0.0
            if self.speaker_conditioning_enabled:
                # Get the full alignments (before trimming to chunk)
                with open(info_file) as f:
                    full_data = json.load(f)
                    full_alignments = full_data["alignments"]

                # Parse alignments into Alignment objects
                parsed_alignments = []
                for a in full_alignments:
                    if len(a) >= 3:
                        parsed_alignments.append(
                            Alignment(text=a[0], time_span=(a[1][0], a[1][1]), speaker=a[2])
                        )

                # Sample speaker reference avoiding the target segment
                (
                    speaker_reference_audio,
                    speaker_reference_text,
                    speaker_reference_start_sec,
                    speaker_reference_end_sec,
                ) = self._sample_speaker_reference(
                    full_audio=wav,
                    alignments=parsed_alignments,
                    target_start_sec=start_sec,
                    target_end_sec=start_sec + self.duration_sec,
                    main_speaker_label="SPEAKER_MAIN",
                )

                if not self._first_sample_logged and speaker_reference_audio is not None:
                    logger.warning(
                        f"[SPEAKER COND] Sampled reference audio: "
                        f"shape={speaker_reference_audio.shape}, "
                        f"duration={speaker_reference_audio.shape[-1] / self.speaker_ref_target_sample_rate:.2f}s"
                    )

            # Log final codes shape on first call and SET FLAG TO TRUE
            if not self._first_sample_logged:
                # Data produces 9 codebooks: text(1) + audio(8)
                # Model expects 17 codebooks (n_q=16), padding happens in train.py
                expected_codebooks = 9
                logger.warning(
                    f"[INTERLEAVER DEBUG] Final codes: shape={codes.shape}, "
                    f"expected=[1, {expected_codebooks}, {self.num_audio_frames}]"
                )
                logger.warning(
                    f"[INTERLEAVER DEBUG] text_tokens.shape={text_tokens.shape}, "
                    f"audio_tokens.shape={audio_tokens.shape}"
                )
                logger.warning(
                    f"[INTERLEAVER DEBUG] Codes will be padded to 17 codebooks in train.py"
                )
                # Verify data codebook count (before padding)
                if codes.shape[1] != expected_codebooks:
                    logger.error(
                        f"[INTERLEAVER ERROR] Codebook mismatch! Got {codes.shape[1]}, expected {expected_codebooks}. "
                        f"text={text_tokens.shape[1]}, audio={audio_tokens.shape[1]}"
                    )

                # Validate final codes tensor
                codes_min = codes.min().item()
                codes_max = codes.max().item()
                logger.warning(
                    f"[INTERLEAVER DEBUG] codes value range: min={codes_min}, max={codes_max}"
                )

                # Check for NaN/Inf
                if torch.isnan(codes).any() or torch.isinf(codes).any():
                    logger.error(
                        f"[INTERLEAVER ERROR] Codes contain NaN or Inf! This will cause NaN loss."
                    )

                logger.warning("[INTERLEAVER DEBUG] ===== END FIRST SAMPLE =====")
                # CRITICAL: Set flag to True to prevent logging every sample
                self._first_sample_logged = True

            return Sample(
                codes=codes,
                condition_attributes=data.get("text_conditions", None),
                audio_path=path,  # Track source for dialogue saving
                speaker_reference_audio=speaker_reference_audio,  # Phase 2: Speaker conditioning
                speaker_reference_text=speaker_reference_text,     # Phase 2: Speaker conditioning
                speaker_reference_start_sec=speaker_reference_start_sec,  # Reference segment timing
                speaker_reference_end_sec=speaker_reference_end_sec,      # Reference segment timing
            )


class StereoInterleavedTokenizer:
    """
    Tokenizer for stereo audio that produces 17 codebooks for user stream training.

    This tokenizer is designed for full finetuning with dep_q=16, where both
    Moshi's and User's audio streams are modeled.

    Output codebook structure (17 total):
        - Stream 0: Text tokens (Moshi's inner monologue from SPEAKER_MAIN)
        - Streams 1-8: Moshi's audio (8 codebooks from LEFT channel)
        - Streams 9-16: User's audio (8 codebooks from RIGHT channel)

    Note:
        - LEFT channel (wav[0]) = Moshi/AI voice (SPEAKER_MAIN)
        - RIGHT channel (wav[1]) = User voice
        - This matches K-Moshi data format (see docs/)

    Unified Filtering Pipeline (5 Layers):
        Layer 1: Case Control - Structural validity (allow_case1..5)
        Layer 2: Quality - Hard minimums (min_moshi_words, etc.)
        Layer 3: Preferences - Probabilistic preferences (optional)
        Layer 4: Role Swapping - Data augmentation (2x data)
        Layer 5: Logging - Debug and statistics

    Args:
        mimi: Mimi audio codec model
        interleaver: Interleaver instance for text tokenization
        duration_sec: Chunk duration in seconds
        jsonl_base_dir: Parent directory of JSONL file (used for path resolution)
        segment_filtering: SegmentFilteringArgs for unified filtering
        run_dir: Optional run directory for log file output
    """

    def __init__(
        self,
        mimi,
        interleaver,
        duration_sec: float,
        jsonl_base_dir: str | Path | None = None,
        segment_filtering=None,  # SegmentFilteringArgs
        run_dir: str | Path | None = None,
        speaker_conditioning_config: Optional[dict] = None,  # Phase 2: Speaker conditioning
    ):
        """
        Initialize the stereo tokenizer with unified segment filtering.

        Args:
            mimi: Mimi audio codec model
            interleaver: Interleaver instance for text tokenization
            duration_sec: Chunk duration in seconds
            jsonl_base_dir: Parent directory of JSONL file (used for path resolution)
            segment_filtering: SegmentFilteringArgs for unified filtering pipeline
            run_dir: Run directory for log file output
            speaker_conditioning_config: Optional dict with speaker conditioning settings
                - enabled: bool, whether to sample speaker reference
                - min_duration_sec: float, minimum reference duration (default 3.0)
                - max_duration_sec: float, maximum reference duration (default 10.0)
                - target_sample_rate: int, target sample rate for speaker encoder (default 16000)
        """
        self.mimi = mimi
        self.interleaver = interleaver
        self.duration_sec = duration_sec
        self.num_audio_frames = math.ceil(duration_sec * mimi.frame_rate)
        self.jsonl_base_dir = Path(jsonl_base_dir) if jsonl_base_dir else None
        self._path_resolution_logged = False
        self._first_sample_logged = False

        # =====================================================================
        # Unified Segment Filtering Configuration (5-Layer System)
        # =====================================================================
        self.segment_filtering = segment_filtering
        self.filter_logger: FilteringLogger | None = None

        # Initialize FilteringLogger if segment_filtering is provided
        if segment_filtering is not None:
            self.filter_logger = FilteringLogger(
                config=segment_filtering.logging,
                run_dir=run_dir,
            )
            self.filter_logger.log_init(segment_filtering)

        # =====================================================================
        # Phase 2: Speaker Conditioning Configuration
        # =====================================================================
        self.speaker_conditioning_enabled = False
        self.speaker_ref_min_duration_sec = 3.0
        self.speaker_ref_max_duration_sec = 10.0
        self.speaker_ref_target_sample_rate = 16000

        if speaker_conditioning_config is not None:
            self.speaker_conditioning_enabled = speaker_conditioning_config.get("enabled", False)
            self.speaker_ref_min_duration_sec = speaker_conditioning_config.get(
                "min_duration_sec", 3.0
            )
            self.speaker_ref_max_duration_sec = speaker_conditioning_config.get(
                "max_duration_sec", 10.0
            )
            self.speaker_ref_target_sample_rate = speaker_conditioning_config.get(
                "target_sample_rate", 16000
            )
            if self.speaker_conditioning_enabled:
                logger.info(
                    f"[STEREO TOKENIZER] Speaker conditioning enabled: "
                    f"ref_duration=[{self.speaker_ref_min_duration_sec}, "
                    f"{self.speaker_ref_max_duration_sec}]s, "
                    f"target_sr={self.speaker_ref_target_sample_rate}Hz"
                )

    def _resolve_json_path(self, wav_path: str) -> str:
        """Resolve JSON alignment file path from WAV path.

        Supports data_preparation output format with alignments/ directory.
        """
        wav_basename = os.path.splitext(os.path.basename(wav_path))[0]
        json_suffix = os.path.splitext(wav_path)[0] + ".json"
        candidates = []

        candidates.append(json_suffix)

        if os.path.isabs(wav_path):
            candidates.append(os.path.splitext(wav_path)[0] + ".json")

            # Try alignments/ directory relative to audio file
            wav_dir = Path(wav_path).parent
            base_dir = wav_dir.parent
            alignments_path = base_dir / "alignments" / f"{wav_basename}.json"
            candidates.append(str(alignments_path))
            speaker01_path = base_dir / "alignment_speaker01" / f"{wav_basename}.json"
            candidates.append(str(speaker01_path))

        if self.jsonl_base_dir:
            resolved = self.jsonl_base_dir / json_suffix
            candidates.append(str(resolved))

            alignments_path = self.jsonl_base_dir / "alignments" / f"{wav_basename}.json"
            candidates.append(str(alignments_path))
            speaker01_path = self.jsonl_base_dir / "alignment_speaker01" / f"{wav_basename}.json"
            candidates.append(str(speaker01_path))

            if not os.path.isabs(wav_path):
                candidates.append(str(self.jsonl_base_dir / json_suffix))

        candidates.append(os.path.join(os.getcwd(), json_suffix))

        seen = set()
        for candidate in candidates:
            if candidate in seen:
                continue
            seen.add(candidate)

            if os.path.exists(candidate):
                if not self._path_resolution_logged:
                    logger.info(f"JSON path resolved: {wav_path} -> {candidate}")
                    self._path_resolution_logged = True
                return candidate

        error_msg = (
            f"JSON alignment file not found for: {wav_path}\n"
            f"Tried paths:\n" + "\n".join(f"  - {c}" for c in seen)
        )
        if self.jsonl_base_dir:
            error_msg += f"\nJSONL base dir: {self.jsonl_base_dir}"

        raise FileNotFoundError(error_msg)

    def _encode_audio_channel(self, audio: np.ndarray) -> torch.Tensor:
        """
        Encode a single audio channel with Mimi.

        Args:
            audio: 1D audio array (samples,)

        Returns:
            Audio tokens tensor [1, 8, T]
        """
        audio_tensor = torch.Tensor(audio).cuda()
        # Mimi expects [B, C, T] = [batch, channels, samples]
        mimi_input = audio_tensor[None, None, :]
        audio_tokens = self.mimi.encode(mimi_input)  # [1, 8, T]
        return audio_tokens

    # =========================================================================
    # Data Filtering Methods (Pre-Augmentation)
    # =========================================================================

    def _compute_audio_energy(self, audio: np.ndarray) -> float:
        """
        Compute RMS energy of an audio signal.

        Args:
            audio: 1D audio array (samples,)

        Returns:
            RMS energy value (higher = louder audio)
        """
        if len(audio) == 0:
            return 0.0
        return float(np.sqrt(np.mean(audio ** 2)))

    def _compute_audio_presence_ratio(
        self,
        audio: np.ndarray,
        energy_threshold: float,
        frame_size: int = 1600,  # 100ms at 16kHz or ~66ms at 24kHz
    ) -> float:
        """
        Compute the ratio of frames with audio above energy threshold.

        Args:
            audio: 1D audio array (samples,)
            energy_threshold: Minimum RMS energy to consider as audio
            frame_size: Number of samples per frame for analysis

        Returns:
            Ratio of frames with audio (0.0 to 1.0)
        """
        if len(audio) < frame_size:
            # For very short audio, just check overall energy
            return 1.0 if self._compute_audio_energy(audio) > energy_threshold else 0.0

        num_frames = len(audio) // frame_size
        active_frames = 0

        for i in range(num_frames):
            frame = audio[i * frame_size : (i + 1) * frame_size]
            frame_energy = self._compute_audio_energy(frame)
            if frame_energy > energy_threshold:
                active_frames += 1

        return active_frames / num_frames if num_frames > 0 else 0.0

    def _sample_speaker_reference(
        self,
        full_audio: np.ndarray,
        alignments: list[Alignment],
        target_start_sec: float,
        target_end_sec: float,
        main_speaker_label: str = "SPEAKER_MAIN",
    ) -> tuple[torch.Tensor | None, str | None, float, float]:
        """
        Sample speaker reference audio from MOSHI channel, avoiding target segment.

        This method extracts a reference audio segment for speaker conditioning
        from regions where SPEAKER_MAIN is speaking, outside the target training segment.

        Args:
            full_audio: Full stereo audio array [2, samples] at Mimi sample rate (24kHz)
            alignments: List of Alignment objects for the full audio
            target_start_sec: Start time of target segment to avoid
            target_end_sec: End time of target segment to avoid
            main_speaker_label: Label for MOSHI speaker (default "SPEAKER_MAIN")

        Returns:
            Tuple of (reference_audio, reference_text, start_sec, end_sec):
            - reference_audio: Tensor [T] at target_sample_rate (16kHz) or None
            - reference_text: Concatenated text from reference region or None
            - start_sec: Start time of reference segment in source audio
            - end_sec: End time of reference segment in source audio
        """
        if not self.speaker_conditioning_enabled:
            return None, None, 0.0, 0.0

        # Get Mimi sample rate (should be 24000)
        mimi_sample_rate = self.mimi.sample_rate

        # Find MOSHI speech regions outside target segment
        moshi_regions = []
        for align in alignments:
            if align.speaker != main_speaker_label:
                continue

            seg_start, seg_end = align.time_span

            # Skip if overlaps with target segment
            if seg_end <= target_start_sec or seg_start >= target_end_sec:
                moshi_regions.append({
                    "start": seg_start,
                    "end": seg_end,
                    "text": align.text,
                })

        if not moshi_regions:
            return None, None, 0.0, 0.0

        # Sort by start time and merge adjacent regions
        moshi_regions.sort(key=lambda x: x["start"])

        # Try to find a contiguous region of sufficient length
        min_samples = int(self.speaker_ref_min_duration_sec * mimi_sample_rate)
        max_samples = int(self.speaker_ref_max_duration_sec * mimi_sample_rate)

        # Merge adjacent regions (within 0.5s gap)
        merged_regions = []
        current_region = None
        gap_threshold = 0.5  # seconds

        for region in moshi_regions:
            if current_region is None:
                current_region = {
                    "start": region["start"],
                    "end": region["end"],
                    "texts": [region["text"]],
                }
            elif region["start"] - current_region["end"] <= gap_threshold:
                # Merge with current region
                current_region["end"] = region["end"]
                current_region["texts"].append(region["text"])
            else:
                # Save current and start new
                merged_regions.append(current_region)
                current_region = {
                    "start": region["start"],
                    "end": region["end"],
                    "texts": [region["text"]],
                }

        if current_region is not None:
            merged_regions.append(current_region)

        # Find regions that meet minimum duration
        valid_regions = []
        for region in merged_regions:
            duration = region["end"] - region["start"]
            if duration >= self.speaker_ref_min_duration_sec:
                valid_regions.append(region)

        if not valid_regions:
            # Fall back to longest region even if below minimum
            if merged_regions:
                valid_regions = [max(merged_regions, key=lambda x: x["end"] - x["start"])]
            else:
                return None, None, 0.0, 0.0

        # Select a random valid region
        import random
        selected = random.choice(valid_regions)

        # Extract audio from MOSHI channel (channel 0 = left = Moshi)
        start_sample = int(selected["start"] * mimi_sample_rate)
        end_sample = int(selected["end"] * mimi_sample_rate)

        # Limit to max duration
        clipped = False
        if end_sample - start_sample > max_samples:
            # Random start within valid range
            max_start = end_sample - max_samples
            start_sample = random.randint(start_sample, max_start)
            end_sample = start_sample + max_samples
            clipped = True

        # Ensure within bounds
        if full_audio.ndim == 2:
            moshi_channel = full_audio[0]  # Left channel = Moshi
        else:
            moshi_channel = full_audio

        start_sample = max(0, min(start_sample, len(moshi_channel) - min_samples))
        end_sample = min(len(moshi_channel), end_sample)

        if end_sample <= start_sample:
            return None, None, 0.0, 0.0

        ref_audio = moshi_channel[start_sample:end_sample]

        # Resample from Mimi sample rate (24kHz) to target (16kHz)
        ref_audio_tensor = torch.from_numpy(ref_audio.copy()).float()

        if mimi_sample_rate != self.speaker_ref_target_sample_rate:
            import torchaudio.functional as F
            ref_audio_tensor = F.resample(
                ref_audio_tensor,
                orig_freq=mimi_sample_rate,
                new_freq=self.speaker_ref_target_sample_rate,
            )

        # =================================================================
        # CRITICAL FIX: Re-extract text only for the clipped time range
        # =================================================================
        # When audio is clipped to max_duration, we must extract only the
        # text that corresponds to the clipped audio segment.
        # =================================================================
        if clipped:
            # Convert samples back to seconds for text extraction
            clip_start_sec = start_sample / mimi_sample_rate
            clip_end_sec = end_sample / mimi_sample_rate

            # Re-extract text for the clipped segment
            ref_text_parts = []
            for align in alignments:
                if align.speaker != main_speaker_label:
                    continue
                seg_start, seg_end = align.time_span
                # Only include text that overlaps with clipped segment
                if seg_end > clip_start_sec and seg_start < clip_end_sec:
                    ref_text_parts.append(align.text)

            ref_text = " ".join(ref_text_parts) if ref_text_parts else ""
            # Use clipped segment timing
            final_start_sec = clip_start_sec
            final_end_sec = clip_end_sec
        else:
            # No clipping - use original concatenated texts
            ref_text = " ".join(selected["texts"])
            # Use original selected region timing
            final_start_sec = selected["start"]
            final_end_sec = selected["end"]

        return ref_audio_tensor, ref_text, final_start_sec, final_end_sec

    def _detect_segment_case(
        self,
        has_moshi_audio: bool,
        has_moshi_text: bool,
        has_user_audio: bool,
    ) -> int:
        """
        Detect which case (1-5) a segment belongs to based on stream presence.

        ┌──────┬────────────────────────────────────────┬─────────────────────┐
        │ Case │ Configuration                          │ Description         │
        ├──────┼────────────────────────────────────────┼─────────────────────┤
        │ 1    │ moshi_audio + moshi_text + user_audio  │ Full dialogue       │
        │ 2    │ moshi_audio + moshi_text               │ Moshi monologue     │
        │ 3    │ user_audio only                        │ No Moshi content    │
        │ 4    │ moshi_text + user_audio                │ Missing Moshi audio │
        │ 5    │ moshi_audio + user_audio               │ Missing Moshi text  │
        │ 0    │ Other / Empty                          │ No valid content    │
        └──────┴────────────────────────────────────────┴─────────────────────┘

        Returns:
            Case number (1-5) or 0 for unclassified
        """
        if has_moshi_audio and has_moshi_text and has_user_audio:
            return 1  # Full dialogue
        elif has_moshi_audio and has_moshi_text and not has_user_audio:
            return 2  # Moshi monologue
        elif not has_moshi_audio and not has_moshi_text and has_user_audio:
            return 3  # User audio only
        elif not has_moshi_audio and has_moshi_text and has_user_audio:
            return 4  # Moshi text + user audio (no moshi audio)
        elif has_moshi_audio and not has_moshi_text and has_user_audio:
            return 5  # Moshi audio + user audio (no moshi text)
        elif has_moshi_audio and not has_moshi_text and not has_user_audio:
            return 5  # Moshi audio only (no text, no user) - treat as Case 5
        else:
            return 0  # Unclassified (e.g., no content at all)

    def _check_case_control(
        self,
        wav: np.ndarray,
        stats: SegmentStatistics,
        segment_path: str = "",
        is_post_swap: bool = False,
    ) -> str | None:
        """
        Layer 1: Case Control - Check structural validity.

        Uses EXPLICIT case control (allow_case1 through allow_case5) for intuitive
        configuration in YAML.

        +------+----------------------------------------+---------+---------------------+
        | Case | Configuration                          | Default | Description         |
        +------+----------------------------------------+---------+---------------------+
        | 1    | moshi_audio + moshi_text + user_audio  | [ALLOW] | Full dialogue       |
        | 2    | moshi_audio + moshi_text               | [ALLOW] | Moshi monologue     |
        | 3    | user_audio only                        | [SKIP]  | No Moshi content    |
        | 4    | moshi_text + user_audio                | [SKIP]  | Missing Moshi audio |
        | 5    | moshi_audio + user_audio               | [SKIP]  | Missing Moshi text  |
        +------+----------------------------------------+---------+---------------------+

        Args:
            wav: Stereo audio array [2, samples]
            stats: Pre-computed segment statistics
            segment_path: Path to segment file (for logging)
            is_post_swap: If True, this check is for a post-swap sample

        Returns:
            None if valid (case is allowed), otherwise skip reason string
        """
        # Get case control config from segment_filtering
        if self.segment_filtering is None:
            return None
        cfg = self.segment_filtering.case_control
        if not cfg.enabled:
            return None

        filter_stats = get_filtering_statistics()

        # Track check count
        if is_post_swap:
            # Post-swap tracking handled separately
            pass
        else:
            filter_stats.case_checks += 1

        # =====================================================================
        # Step 1: Check Moshi audio presence (LEFT channel = wav[0])
        # =====================================================================
        moshi_audio = wav[0]
        moshi_audio_energy = self._compute_audio_energy(moshi_audio)
        moshi_audio_presence = self._compute_audio_presence_ratio(
            moshi_audio, cfg.min_audio_energy
        )
        has_moshi_audio = (
            moshi_audio_energy > cfg.min_audio_energy
            and moshi_audio_presence >= cfg.min_audio_presence_ratio
        )

        # =====================================================================
        # Step 2: Check User audio presence (RIGHT channel = wav[1])
        # =====================================================================
        user_audio = wav[1]
        user_audio_energy = self._compute_audio_energy(user_audio)
        user_audio_presence = self._compute_audio_presence_ratio(
            user_audio, cfg.min_audio_energy
        )
        has_user_audio = (
            user_audio_energy > cfg.min_audio_energy
            and user_audio_presence >= cfg.min_audio_presence_ratio
        )

        # =====================================================================
        # Step 3: Check Moshi text presence (from alignments)
        # =====================================================================
        # Use min_moshi_text_words from legacy config or default to 1
        min_text_words = getattr(cfg, 'min_moshi_text_words', 1)
        has_moshi_text = stats.moshi_word_count >= min_text_words

        # =====================================================================
        # Step 4: Detect segment case (1-5)
        # =====================================================================
        case_num = self._detect_segment_case(has_moshi_audio, has_moshi_text, has_user_audio)

        # Update case detection statistics (only for original samples, not post-swap)
        if not is_post_swap:
            if case_num == 1:
                filter_stats.case1_detected += 1
            elif case_num == 2:
                filter_stats.case2_detected += 1
            elif case_num == 3:
                filter_stats.case3_detected += 1
            elif case_num == 4:
                filter_stats.case4_detected += 1
            elif case_num == 5:
                filter_stats.case5_detected += 1

        # =====================================================================
        # Step 5: Check if this case is allowed (explicit case control)
        # =====================================================================
        case_allowed = {
            1: cfg.allow_case1,
            2: cfg.allow_case2,
            3: cfg.allow_case3,
            4: cfg.allow_case4,
            5: cfg.allow_case5,
            0: False,  # Unclassified cases are always filtered
        }

        case_descriptions = {
            1: "full_dialogue",
            2: "moshi_monologue",
            3: "user_audio_only",
            4: "missing_moshi_audio",
            5: "missing_moshi_text",
            0: "no_valid_content",
        }

        is_allowed = case_allowed.get(case_num, False)

        # Log via FilteringLogger
        if self.filter_logger is not None:
            details = {
                "moshi_audio_energy": f"{moshi_audio_energy:.6f}",
                "user_audio_energy": f"{user_audio_energy:.6f}",
                "moshi_words": stats.moshi_word_count,
            }
            self.filter_logger.log_case_detection(
                case_num, is_allowed, segment_path, details
            )

        if is_allowed:
            # Segment passes case control
            if is_post_swap:
                filter_stats.swap_post_check_passed += 1
            else:
                filter_stats.case_passed += 1
            return None  # VALID

        # =====================================================================
        # Step 6: Case not allowed - build skip reason and update statistics
        # =====================================================================
        swap_label = "post_swap_" if is_post_swap else ""
        skip_reason = (
            f"{swap_label}case{case_num}:{case_descriptions.get(case_num, 'unknown')}"
        )

        # Update filtering statistics
        if is_post_swap:
            filter_stats.swap_post_check_failed += 1
        else:
            # Update case-specific filtered counts
            if case_num == 1:
                filter_stats.case1_filtered += 1
            elif case_num == 2:
                filter_stats.case2_filtered += 1
            elif case_num == 3:
                filter_stats.case3_filtered += 1
            elif case_num == 4:
                filter_stats.case4_filtered += 1
            elif case_num == 5:
                filter_stats.case5_filtered += 1

        # Log first filter
        if self.filter_logger is not None:
            self.filter_logger.log_first_filter(skip_reason, segment_path)

        return skip_reason

    # =========================================================================
    # Layer 2: Quality Requirements
    # =========================================================================

    def _check_quality(
        self,
        stats: SegmentStatistics,
        is_swapped_version: bool = False,
    ) -> str | None:
        """
        Layer 2: Quality Requirements - Check hard minimums.

        Args:
            stats: Segment statistics computed from alignments
            is_swapped_version: If True, skip statistics tracking (already counted)

        Returns:
            None if segment passes, otherwise skip reason string
        """
        # Get quality config from segment_filtering
        if self.segment_filtering is None:
            return None
        cfg = self.segment_filtering.quality
        if not cfg.enabled:
            return None

        filter_stats = get_filtering_statistics()

        # Only track for original samples (not swapped versions)
        if not is_swapped_version:
            filter_stats.total_segments += 1
            filter_stats.quality_checks += 1

        # Check: No Moshi speech at all
        if not stats.has_moshi_speech:
            if not is_swapped_version:
                filter_stats.quality_no_moshi += 1
            return "no_moshi_speech"

        # Check: Minimum Moshi word count
        if stats.moshi_word_count < cfg.min_moshi_words:
            if not is_swapped_version:
                filter_stats.quality_low_moshi_words += 1
            return f"low_moshi_words ({stats.moshi_word_count} < {cfg.min_moshi_words})"

        # Check: Minimum Moshi duration
        if stats.moshi_duration_sec < cfg.min_moshi_duration_sec:
            if not is_swapped_version:
                filter_stats.quality_low_moshi_duration += 1
            return f"low_moshi_duration ({stats.moshi_duration_sec:.1f}s < {cfg.min_moshi_duration_sec}s)"

        # Check: Minimum Moshi ratio (only if there's any speech)
        if cfg.min_moshi_ratio > 0 and stats.total_speech_duration > 0:
            if stats.moshi_ratio < cfg.min_moshi_ratio:
                if not is_swapped_version:
                    filter_stats.quality_low_moshi_ratio += 1
                return f"low_moshi_ratio ({stats.moshi_ratio:.2f} < {cfg.min_moshi_ratio})"

        # Check: Minimum User word count (for dialogue training)
        if cfg.min_user_words > 0:
            if stats.user_word_count < cfg.min_user_words:
                if not is_swapped_version:
                    filter_stats.quality_low_user_words += 1
                return f"low_user_words ({stats.user_word_count} < {cfg.min_user_words})"

        # Check: Minimum segment duration
        if stats.segment_duration_sec < cfg.min_segment_duration_sec:
            if not is_swapped_version:
                filter_stats.quality_short_segment += 1
            return f"short_segment ({stats.segment_duration_sec:.1f}s < {cfg.min_segment_duration_sec}s)"

        # Check: Maximum segment duration (if configured)
        if cfg.max_segment_duration_sec is not None:
            if stats.segment_duration_sec > cfg.max_segment_duration_sec:
                if not is_swapped_version:
                    filter_stats.quality_long_segment += 1
                return f"long_segment ({stats.segment_duration_sec:.1f}s > {cfg.max_segment_duration_sec}s)"

        # Passed quality check
        if not is_swapped_version:
            filter_stats.quality_passed += 1

        # Log quality check
        if self.filter_logger is not None:
            self.filter_logger.log_quality_check(stats, None, is_swapped_version)

        return None

    def _swap_audio_channels(
        self,
        moshi_audio: np.ndarray,
        user_audio: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Swap Moshi and User audio channels.

        After swapping:
        - Original User audio becomes new Moshi audio
        - Original Moshi audio becomes new User audio

        Returns:
            (new_moshi_audio, new_user_audio)
        """
        return user_audio.copy(), moshi_audio.copy()

    def _swap_alignments(
        self,
        alignments: list[Alignment],
        main_speaker_label: str,
    ) -> list[Alignment]:
        """
        Swap speaker labels in alignments.

        SPEAKER_MAIN becomes other speaker(s), other speaker(s) become SPEAKER_MAIN.

        Args:
            alignments: List of (word, (start, end), speaker) tuples
            main_speaker_label: Label for main speaker (e.g., "SPEAKER_MAIN")

        Returns:
            New list of alignments with swapped speaker labels
        """
        swapped = []
        for word, times, speaker in alignments:
            if speaker == main_speaker_label:
                # Moshi → User (use generic "SPEAKER_USER" label)
                new_speaker = "SPEAKER_USER"
            else:
                # User → Moshi
                new_speaker = main_speaker_label
            swapped.append((word, times, new_speaker))
        return swapped

    # =========================================================================
    # Layer 3: Preferences (Probabilistic)
    # =========================================================================

    def _check_preferences(
        self,
        stats: SegmentStatistics,
        alignments: list[Alignment],
        is_swapped_version: bool = False,
    ) -> str | None:
        """
        Layer 3: Preferences - Check probabilistic preferences.

        This method implements PROBABILISTIC FILTERING:
        - All controls are probabilistic (skip with probability P)
        - Works as a DATA FILTER (segments not meeting criteria are skipped)

        Args:
            stats: Segment statistics computed from alignments
            alignments: List of (word, (start, end), speaker) tuples
            is_swapped_version: If True, skip statistics tracking (already counted)

        Returns:
            None if segment passes, otherwise skip reason string
        """
        # Get preferences config from segment_filtering
        if self.segment_filtering is None:
            return None
        cfg = self.segment_filtering.preferences
        if not cfg.enabled:
            return None

        filter_stats = get_filtering_statistics()
        main_speaker_label = self.interleaver.main_speaker_label

        # Only track for original samples
        if not is_swapped_version:
            filter_stats.pref_checks += 1

        first_speaker = alignments[0][2] if alignments else "none"
        has_both = stats.has_moshi_speech and stats.has_user_speech

        # =====================================================================
        # Check 1: prefer_both_speakers (probabilistic)
        # =====================================================================
        prefer_both = cfg.prefer_both_speakers
        prefer_both_prob = cfg.prefer_both_speakers_prob

        if prefer_both:
            if not has_both:
                # Apply probability: skip with prefer_both_speakers_prob
                if random.random() < prefer_both_prob:
                    if not is_swapped_version:
                        filter_stats.pref_single_speaker_skipped += 1
                    missing = "moshi" if not stats.has_moshi_speech else "user"
                    return f"single_speaker ({missing} missing, prob={prefer_both_prob})"
                # Else: allow through despite single speaker

        # =====================================================================
        # Check 2: prefer_moshi_start (probabilistic)
        # =====================================================================
        if cfg.prefer_moshi_start and len(alignments) > 0:
            if first_speaker != main_speaker_label:
                # User speaks first - apply probabilistic filter
                if random.random() < cfg.prefer_moshi_start_prob:
                    if not is_swapped_version:
                        filter_stats.pref_user_first_skipped += 1
                    return f"user_speaks_first (prob={cfg.prefer_moshi_start_prob})"
                # Else: allow through despite user speaking first

        # Passed preferences check
        if not is_swapped_version:
            filter_stats.pref_passed += 1

        # Log preferences check
        if self.filter_logger is not None:
            self.filter_logger.log_preferences_check(None, first_speaker, has_both)

        return None

    def __call__(
        self,
        wav: np.ndarray,
        start_sec: float,
        path: str,
        is_swapped_version: bool = False,
    ) -> Sample | list[Sample]:
        """
        Process stereo audio and produce 17 codebooks with data augmentation.

        When role_swapping.yield_both=True, returns a list of [original, swapped] samples.
        Otherwise returns a single Sample.

        Args:
            wav: Stereo audio array [2, samples] or [samples, 2]
            start_sec: Start time in seconds
            path: Path to audio file
            is_swapped_version: Internal flag for recursive swapped sample creation

        Returns:
            Sample or list[Sample] with codes tensor [1, 17, T]
        """
        with torch.no_grad():
            # Debug logging for first sample
            if not self._first_sample_logged:
                logger.warning(f"[STEREO TOKENIZER] ===== FIRST SAMPLE =====")
                logger.warning(
                    f"[STEREO TOKENIZER] wav input: shape={wav.shape}, ndim={wav.ndim}, "
                    f"dtype={wav.dtype}"
                )
                logger.warning(f"[STEREO TOKENIZER] path={path}, start_sec={start_sec}")
                if self.segment_filtering is not None:
                    logger.warning(
                        f"[STEREO TOKENIZER] Segment filtering: "
                        f"case_control={self.segment_filtering.case_control.enabled}, "
                        f"quality={self.segment_filtering.quality.enabled}, "
                        f"role_swapping={self.segment_filtering.role_swapping.enabled}"
                    )

            # Validate stereo input
            if wav.ndim != 2 or wav.shape[0] != 2:
                raise ValueError(
                    f"StereoInterleavedTokenizer requires stereo audio [2, samples], "
                    f"got shape={wav.shape}"
                )

            # =====================================================================
            # Phase 1: Load alignments FIRST (before audio encoding for efficiency)
            # =====================================================================
            info_file = self._resolve_json_path(path)
            with open(info_file) as f:
                data = json.load(f)
                raw_alignments = data["alignments"]

            start_alignment = dicho(raw_alignments, start_sec)
            end_alignment = dicho(raw_alignments, start_sec + self.duration_sec)
            alignments = [
                (a[0], (a[1][0] - start_sec, a[1][1] - start_sec), a[2])
                for a in raw_alignments[start_alignment:end_alignment]
            ]

            if len(alignments) == 0:
                logger.debug(f"Empty alignments for {path} at {start_sec:.2f}s")

            # =====================================================================
            # Phase 2: Compute segment statistics for filtering
            # =====================================================================
            main_speaker_label = self.interleaver.main_speaker_label
            stats = compute_segment_statistics(
                alignments=alignments,
                main_speaker_label=main_speaker_label,
                segment_duration_sec=self.duration_sec,
                segment_path=path,
                segment_start_sec=start_sec,
            )

            # =====================================================================
            # Phase 2.5: Case Control (Layer 1) - Pre-Augmentation Filter
            # =====================================================================
            # This runs BEFORE data augmentation to filter fundamental data issues:
            # - Case 1: moshi_audio + moshi_text + user_audio -> VALID (dialogue)
            # - Case 2: moshi_audio + moshi_text              -> VALID (monologue)
            # - Case 3: user_audio only                       -> INVALID
            # - Case 4: moshi_text + user_audio               -> INVALID
            # - Case 5: moshi_audio + user_audio              -> INVALID
            skip_reason = self._check_case_control(wav, stats, segment_path=path)
            if skip_reason is not None:
                if not self._first_sample_logged:
                    logger.warning(
                        f"[STEREO TOKENIZER] Case control failed: {skip_reason}"
                    )
                return Sample(
                    codes=torch.zeros(1, 17, self.num_audio_frames, dtype=torch.long),
                    condition_attributes=None,
                    user_text_alignments=None,
                    moshi_text_raw=None,
                    statistics=stats,
                    is_role_swapped=False,
                    skip_reason=f"case_control:{skip_reason}",
                    audio_path=path,
                )

            # =====================================================================
            # Phase 3: Quality Requirements (Layer 2)
            # =====================================================================
            # Pass is_swapped_version to avoid double-counting statistics
            skip_reason = self._check_quality(stats, is_swapped_version=is_swapped_version)
            if skip_reason is not None:
                # Return a "skipped" Sample with empty codes
                # The data loader should handle this and skip it
                if not self._first_sample_logged:
                    logger.warning(
                        f"[STEREO TOKENIZER] Sample skipped: {skip_reason} "
                        f"(moshi_words={stats.moshi_word_count}, "
                        f"moshi_dur={stats.moshi_duration_sec:.1f}s, "
                        f"ratio={stats.moshi_ratio:.2f})"
                    )
                return Sample(
                    codes=torch.zeros(1, 17, self.num_audio_frames, dtype=torch.long),
                    condition_attributes=None,
                    user_text_alignments=None,
                    moshi_text_raw=None,
                    statistics=stats,
                    is_role_swapped=False,
                    skip_reason=skip_reason,
                    audio_path=path,
                )

            # =====================================================================
            # Phase 3.5: Preferences Filtering (Layer 3)
            # =====================================================================
            skip_reason = self._check_preferences(
                stats, alignments, is_swapped_version=is_swapped_version
            )
            if skip_reason is not None:
                if not self._first_sample_logged:
                    logger.warning(
                        f"[STEREO TOKENIZER] Sample skipped by preferences: {skip_reason} "
                        f"(moshi_words={stats.moshi_word_count}, "
                        f"user_words={stats.user_word_count})"
                    )
                return Sample(
                    codes=torch.zeros(1, 17, self.num_audio_frames, dtype=torch.long),
                    condition_attributes=None,
                    user_text_alignments=None,
                    moshi_text_raw=None,
                    statistics=stats,
                    is_role_swapped=False,
                    skip_reason=skip_reason,
                    audio_path=path,
                )

            # =====================================================================
            # Phase 3.9: Update passed_segments count (AFTER all filters pass)
            # =====================================================================
            # Only count for original samples (not swapped versions)
            if not is_swapped_version and self.segment_filtering is not None:
                filter_stats = get_filtering_statistics()
                filter_stats.passed_segments += 1

            # =====================================================================
            # Phase 4: Role Swapping Decision (Layer 4)
            # =====================================================================
            # For yield_both mode: generate original first, then swapped
            # For probability mode: decide whether to swap this sample
            should_swap = False
            yield_both_samples = False

            # Check role swapping config
            if (
                self.segment_filtering is not None
                and self.segment_filtering.role_swapping.enabled
                and not is_swapped_version
            ):
                cfg = self.segment_filtering.role_swapping
                if cfg.yield_both:
                    yield_both_samples = True
                elif random.random() < cfg.probability:
                    should_swap = True

            # =====================================================================
            # Phase 5: Extract and potentially swap audio channels
            # =====================================================================
            # Convention: LEFT (wav[0]) = Moshi, RIGHT (wav[1]) = User
            moshi_audio = wav[0]  # Moshi's voice (LEFT channel, SPEAKER_MAIN)
            user_audio = wav[1]   # User's voice (RIGHT channel)

            if should_swap or is_swapped_version:
                # Swap audio channels: User becomes new Moshi, Moshi becomes new User
                moshi_audio, user_audio = self._swap_audio_channels(moshi_audio, user_audio)
                # Swap alignments: SPEAKER_MAIN ↔ SPEAKER_USER
                alignments = self._swap_alignments(alignments, main_speaker_label)

                # Log role swapping event
                if self.filter_logger is not None:
                    self.filter_logger.log_role_swapping("swap", path)

                if not self._first_sample_logged:
                    logger.warning(
                        "[STEREO TOKENIZER] Role swapping applied: "
                        "User→Moshi audio, Moshi→User audio, speaker labels swapped"
                    )

                # =============================================================
                # Phase 5.5: Post-Swap Integrity Check (Layer 1 after swap)
                # =============================================================
                # After role swapping, check if the swapped sample still passes
                # case control. E.g., Original Case 1 → After swap might
                # become Case 5 if the new Moshi (original User) has no text.
                should_recheck = False

                # Check if post-swap recheck is enabled
                if (
                    self.segment_filtering is not None
                    and self.segment_filtering.case_control.enabled
                    and self.segment_filtering.role_swapping.recheck_after_swap
                ):
                    should_recheck = True

                if should_recheck:
                    # Create swapped wav array for integrity check
                    swapped_wav = np.stack([moshi_audio, user_audio], axis=0)

                    # Recompute stats for swapped alignments
                    swapped_stats = compute_segment_statistics(
                        alignments=alignments,  # Already swapped
                        main_speaker_label=main_speaker_label,
                        segment_duration_sec=self.duration_sec,
                        segment_path=path,
                        segment_start_sec=start_sec,
                    )

                    # Run case control check on swapped sample
                    post_swap_skip_reason = self._check_case_control(
                        swapped_wav, swapped_stats, path, is_post_swap=True
                    )

                    if post_swap_skip_reason is not None:
                        if not self._first_sample_logged:
                            logger.warning(
                                f"[STEREO TOKENIZER] Post-swap integrity failed: {post_swap_skip_reason}"
                            )

                        # Log role swapping failure
                        if self.filter_logger is not None:
                            self.filter_logger.log_role_swapping(
                                "recheck_fail", path, post_swap_skip_reason
                            )

                        # Return skipped Sample for post-swap failure
                        return Sample(
                            codes=torch.zeros(1, 17, self.num_audio_frames, dtype=torch.long),
                            condition_attributes=None,
                            user_text_alignments=None,
                            moshi_text_raw=None,
                            statistics=swapped_stats,
                            is_role_swapped=True,
                            skip_reason=post_swap_skip_reason,
                            audio_path=path,
                        )

                    # Log successful recheck
                    if self.filter_logger is not None:
                        self.filter_logger.log_role_swapping("recheck_pass", path)

                    # Update stats to use swapped stats for the rest of processing
                    stats = swapped_stats

            if not self._first_sample_logged:
                logger.warning(
                    f"[STEREO TOKENIZER] moshi_audio: shape={moshi_audio.shape}, "
                    f"user_audio: shape={user_audio.shape}"
                )

            # =====================================================================
            # Phase 6: Encode audio with Mimi
            # =====================================================================
            try:
                moshi_tokens = self._encode_audio_channel(moshi_audio)  # [1, 8, T]
                user_tokens = self._encode_audio_channel(user_audio)    # [1, 8, T]

                if not self._first_sample_logged:
                    logger.warning(
                        f"[STEREO TOKENIZER] moshi_tokens: {moshi_tokens.shape}, "
                        f"user_tokens: {user_tokens.shape}"
                    )

            except Exception as e:
                logger.error(f"[STEREO TOKENIZER] Mimi encoding failed: {e}")
                raise

            # =====================================================================
            # Phase 7: Truncate/pad to target frame count
            # =====================================================================
            this_num_frames = min(moshi_tokens.shape[-1], user_tokens.shape[-1])
            this_num_frames = min(this_num_frames, self.num_audio_frames)

            # Truncate and pad Moshi audio tokens
            moshi_tokens = moshi_tokens[..., :this_num_frames]
            moshi_tokens = torch.nn.functional.pad(
                moshi_tokens,
                (0, self.num_audio_frames - this_num_frames),
                value=self.interleaver.zero_padding,
            )
            moshi_tokens = moshi_tokens.view(1, -1, self.num_audio_frames)  # [1, 8, T]

            # Truncate and pad User audio tokens
            user_tokens = user_tokens[..., :this_num_frames]
            user_tokens = torch.nn.functional.pad(
                user_tokens,
                (0, self.num_audio_frames - this_num_frames),
                value=self.interleaver.zero_padding,
            )
            user_tokens = user_tokens.view(1, -1, self.num_audio_frames)  # [1, 8, T]

            # =====================================================================
            # Phase 8: Build text tokens and extract metadata
            # =====================================================================
            # Extract User alignments for reference
            user_alignments = [
                a for a in alignments
                if a[2] != main_speaker_label
            ]

            # Extract Moshi (main speaker) original text - NO truncation
            moshi_alignments = [
                a for a in alignments
                if a[2] == main_speaker_label
            ]
            moshi_text_raw = " ".join([a[0] for a in moshi_alignments]) if moshi_alignments else ""

            # Build text tokens using interleaver (will filter to SPEAKER_MAIN only)
            text_tokens = self.interleaver.prepare_item(alignments, this_num_frames)
            text_tokens = torch.nn.functional.pad(
                text_tokens,
                (0, self.num_audio_frames - text_tokens.shape[-1]),
                value=self.interleaver.zero_padding,
            )  # [1, 1, T]

            # =====================================================================
            # Phase 9: Combine all streams → 17 codebooks
            # =====================================================================
            codes = torch.cat([text_tokens, moshi_tokens, user_tokens], dim=1)

            # =====================================================================
            # Phase 9.5: Sample speaker reference audio (if enabled)
            # =====================================================================
            speaker_reference_audio = None
            speaker_reference_text = None
            speaker_reference_start_sec = 0.0
            speaker_reference_end_sec = 0.0
            if self.speaker_conditioning_enabled and not is_swapped_version:
                # Only sample for original samples (swapped samples inherit from original)
                # Get full alignments for reference sampling
                with open(info_file) as f:
                    full_data = json.load(f)
                    full_alignments = full_data["alignments"]

                # Parse alignments into Alignment objects
                parsed_full_alignments = []
                for a in full_alignments:
                    if len(a) >= 3:
                        parsed_full_alignments.append(
                            Alignment(text=a[0], time_span=(a[1][0], a[1][1]), speaker=a[2])
                        )

                # Sample speaker reference avoiding the target segment
                (
                    speaker_reference_audio,
                    speaker_reference_text,
                    speaker_reference_start_sec,
                    speaker_reference_end_sec,
                ) = self._sample_speaker_reference(
                    full_audio=wav,
                    alignments=parsed_full_alignments,
                    target_start_sec=start_sec,
                    target_end_sec=start_sec + self.duration_sec,
                    main_speaker_label=main_speaker_label,
                )

                if not self._first_sample_logged and speaker_reference_audio is not None:
                    logger.warning(
                        f"[STEREO TOKENIZER] Sampled speaker reference: "
                        f"shape={speaker_reference_audio.shape}, "
                        f"duration={speaker_reference_audio.shape[-1] / self.speaker_ref_target_sample_rate:.2f}s"
                    )

            # Debug logging for first sample
            if not self._first_sample_logged:
                logger.warning(
                    f"[STEREO TOKENIZER] Final codes: shape={codes.shape}, "
                    f"expected=[1, 17, {self.num_audio_frames}]"
                )
                logger.warning(
                    f"[STEREO TOKENIZER] text={text_tokens.shape}, "
                    f"moshi_audio={moshi_tokens.shape}, user_audio={user_tokens.shape}"
                )

                if codes.shape[1] != 17:
                    logger.error(
                        f"[STEREO TOKENIZER] Codebook mismatch! Got {codes.shape[1]}, expected 17"
                    )

                # Validate codes
                if torch.isnan(codes).any() or torch.isinf(codes).any():
                    logger.error("[STEREO TOKENIZER] Codes contain NaN or Inf!")

                # Log statistics summary
                if stats:
                    logger.warning(
                        f"[STEREO TOKENIZER] Segment stats: "
                        f"moshi_words={stats.moshi_word_count}, "
                        f"user_words={stats.user_word_count}, "
                        f"moshi_dur={stats.moshi_duration_sec:.1f}s, "
                        f"user_dur={stats.user_duration_sec:.1f}s, "
                        f"ratio={stats.moshi_ratio:.2f}"
                    )

                logger.warning("[STEREO TOKENIZER] ===== END FIRST SAMPLE =====")
                self._first_sample_logged = True

            # =====================================================================
            # Phase 10: Create Sample with metadata
            # =====================================================================
            is_swapped = should_swap or is_swapped_version
            sample = Sample(
                codes=codes,
                condition_attributes=data.get("text_conditions", None),
                user_text_alignments=user_alignments if user_alignments else None,
                moshi_text_raw=moshi_text_raw if moshi_text_raw else None,
                statistics=stats,
                is_role_swapped=is_swapped,
                skip_reason=None,
                audio_path=path,  # Track source for dialogue saving
                speaker_reference_audio=speaker_reference_audio,  # Phase 2: Speaker conditioning
                speaker_reference_text=speaker_reference_text,     # Phase 2: Speaker conditioning
                speaker_reference_start_sec=speaker_reference_start_sec,  # Reference segment timing
                speaker_reference_end_sec=speaker_reference_end_sec,      # Reference segment timing
            )

            # Track yield statistics
            filter_stats = get_filtering_statistics()
            if is_swapped:
                filter_stats.swap_swapped_yielded += 1
                if self.filter_logger is not None:
                    self.filter_logger.log_role_swapping("yield_swapped", path)
            else:
                filter_stats.swap_original_yielded += 1
                if self.filter_logger is not None:
                    self.filter_logger.log_role_swapping("yield_original", path)
            filter_stats.total_samples_yielded += 1

            # Log first pass
            if self.filter_logger is not None and not is_swapped:
                self.filter_logger.log_first_pass(path, stats)

            # =====================================================================
            # Phase 11: Handle yield_both mode (return both original and swapped)
            # =====================================================================
            if yield_both_samples:
                # Generate swapped version recursively
                # Note: We pass the original wav (before swapping), and the recursive call
                # will apply the swap because is_swapped_version=True
                swapped_sample = self.__call__(
                    wav=wav,
                    start_sec=start_sec,
                    path=path,
                    is_swapped_version=True,
                )

                # Return list of [original, swapped]
                # The data loader needs to flatten this list
                return [sample, swapped_sample]

            return sample


def get_interleaved_tokenizer(
    mimi,
    spm: sentencepiece.SentencePieceProcessor,
    duration_sec: float,
    text_padding_token_id: int,
    end_of_text_padding_id: int,
    zero_token_id: int,
    jsonl_base_dir: str | Path | None = None,
    enable_user_stream: bool = False,
    full_duplex_input: bool = True,  # Original Moshi / J-Moshi default
    # Interleaver configuration options (from InterleaverArgs)
    keep_main_only: bool = True,
    keep_and_shift: bool = False,
    adaptive_distribute: bool = True,
    warn_on_overflow: bool = True,
    character_level_interpolation: bool = True,
    main_speaker_label: str = "SPEAKER_MAIN",
    # Unified segment filtering configuration (from SegmentFilteringArgs)
    segment_filtering=None,
    # Run directory for log file output
    run_dir: str | Path | None = None,
    # Phase 2: Speaker conditioning configuration
    speaker_conditioning_config: Optional[dict] = None,
):
    """
    Factory function to get the appropriate interleaved tokenizer.

    This function creates the Interleaver internally and returns the appropriate
    tokenizer based on the enable_user_stream flag.

    Args:
        mimi: Mimi audio codec model
        spm: SentencePiece tokenizer
        duration_sec: Chunk duration in seconds
        text_padding_token_id: Token ID for text padding
        end_of_text_padding_id: Token ID for end of text padding
        zero_token_id: Token ID for zero/no-input positions
        jsonl_base_dir: Parent directory of JSONL file
        enable_user_stream: If True, use StereoInterleavedTokenizer for 17 codebooks

        Interleaver configuration (from InterleaverArgs YAML):
        keep_main_only: If True, only use SPEAKER_MAIN text for Inner Monologue.
            Default: True (recommended for training).
        keep_and_shift: Token queue behavior when words overlap.
            If False: Replace queue (original Moshi/J-Moshi behavior, recommended).
            If True: Extend queue (keeps all tokens, may delay placement).
            Default: False.
        adaptive_distribute: If True, distribute tokens evenly when overflow is detected.
            This helps prevent token loss when text tokens exceed audio frames.
            Default: True (recommended for Korean/Japanese).
        warn_on_overflow: If True, log warnings when token overflow occurs.
            Default: True.
        character_level_interpolation: If True, use J-Moshi style character-level
            timestamp interpolation for more precise token placement.
            Default: True (recommended for Korean/Japanese).
        main_speaker_label: Label identifying the main speaker in alignments.
            Default: "SPEAKER_MAIN".

        segment_filtering: SegmentFilteringArgs for unified 5-layer filtering.
            - Layer 1: Case control (allow_case1..5)
            - Layer 2: Quality requirements (min_moshi_words, etc.)
            - Layer 3: Preferences (prefer_moshi_start, etc.)
            - Layer 4: Role swapping (data augmentation)
            - Layer 5: Logging configuration
            Only used for StereoInterleavedTokenizer (user stream mode).

        run_dir: Optional run directory for log file output.

        speaker_conditioning_config: Optional dict with speaker conditioning settings
            (Phase 2). Includes: enabled, min_duration_sec, max_duration_sec,
            target_sample_rate.

    Returns:
        InterleavedTokenizer or StereoInterleavedTokenizer
    """
    # Log interleaver configuration
    logger.info(
        f"[INTERLEAVER CONFIG] keep_main_only={keep_main_only}, "
        f"keep_and_shift={keep_and_shift}, adaptive_distribute={adaptive_distribute}, "
        f"character_level_interpolation={character_level_interpolation}, "
        f"main_speaker_label='{main_speaker_label}'"
    )

    # Create the Interleaver instance with all configuration options
    interleaver = Interleaver(
        tokenizer=spm,
        audio_frame_rate=mimi.frame_rate,
        text_padding=text_padding_token_id,
        end_of_text_padding=end_of_text_padding_id,
        zero_padding=zero_token_id,
        keep_main_only=keep_main_only,
        main_speaker_label=main_speaker_label,
        keep_and_shift=keep_and_shift,
        adaptive_distribute=adaptive_distribute,
        warn_on_overflow=warn_on_overflow,
        character_level_interpolation=character_level_interpolation,
    )

    # Determine if stereo data is needed
    # - enable_user_stream: USER-STREAM mode (17 codebooks, dep_q=16)
    # - full_duplex_input: FULL-DUPLEX mode (17 codebooks, dep_q=8)
    use_stereo = enable_user_stream or full_duplex_input

    if use_stereo:
        if enable_user_stream:
            logger.info("Using StereoInterleavedTokenizer for USER-STREAM (17 codebooks, dep_q=16)")
        else:
            logger.info("Using StereoInterleavedTokenizer for FULL-DUPLEX (17 codebooks, dep_q=8)")

        # Log segment filtering configuration
        if segment_filtering is not None:
            logger.info("  Segment filtering: UNIFIED 5-LAYER SYSTEM")
            logger.info(f"    Layer 1 (Case Control): {'[ON]' if segment_filtering.case_control.enabled else '[OFF]'}")
            logger.info(f"    Layer 2 (Quality): {'[ON]' if segment_filtering.quality.enabled else '[OFF]'}")
            logger.info(f"    Layer 3 (Preferences): {'[ON]' if segment_filtering.preferences.enabled else '[OFF]'}")
            logger.info(f"    Layer 4 (Role Swapping): {'[ON]' if segment_filtering.role_swapping.enabled else '[OFF]'}")

        return StereoInterleavedTokenizer(
            mimi=mimi,
            interleaver=interleaver,
            duration_sec=duration_sec,
            jsonl_base_dir=jsonl_base_dir,
            segment_filtering=segment_filtering,
            run_dir=run_dir,
            speaker_conditioning_config=speaker_conditioning_config,  # Phase 2
        )
    else:
        logger.info("Using InterleavedTokenizer for MONOLOGUE (9 codebooks, dep_q=8)")
        return InterleavedTokenizer(
            mimi=mimi,
            interleaver=interleaver,
            duration_sec=duration_sec,
            jsonl_base_dir=jsonl_base_dir,
            speaker_conditioning_config=speaker_conditioning_config,  # Phase 2
        )
