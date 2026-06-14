# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Speaker selection for SPEAKER_MAIN assignment.

Uses hybrid scoring based on duration and turn count to select
the primary speaker (SPEAKER_MAIN) for Moshi training format.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional
import logging

from ..readers.lhotse_shar import Conversation, Speaker
from ..config import SpeakerSelectionConfig

logger = logging.getLogger(__name__)


class SpeakerRole(Enum):
    """Speaker roles in Moshi format."""
    SPEAKER_MAIN = "SPEAKER_MAIN"  # Left channel
    SPEAKER_USER = "SPEAKER_USER"  # Right channel


@dataclass
class SpeakerInfo:
    """Detailed speaker information for diarization metadata.

    Preserves individual speaker statistics for future N-speaker
    diarization support while maintaining Moshi 2-stream compatibility.
    """
    speaker_id: str
    role: SpeakerRole  # Assigned Moshi role (MAIN or USER)
    total_duration: float  # Total speech duration in seconds
    turn_count: int  # Number of speaking turns
    hybrid_score: float  # Computed selection score
    rank: int  # Rank by score (1 = highest)
    segments: list[dict] = None  # [{"start": 0.0, "end": 1.5, "text": "..."}]

    def __post_init__(self):
        if self.segments is None:
            self.segments = []

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "speaker_id": self.speaker_id,
            "role": self.role.value,
            "total_duration": round(self.total_duration, 3),
            "turn_count": self.turn_count,
            "hybrid_score": round(self.hybrid_score, 4),
            "rank": self.rank,
            "num_segments": len(self.segments),
            "segments": self.segments,
        }


@dataclass
class SpeakerAssignment:
    """Result of speaker role assignment."""
    conversation_id: str
    main_speaker: Speaker
    main_score: float
    user_speakers: list[Speaker]
    user_score: float
    total_speakers: int
    is_valid: bool
    skip_reason: Optional[str] = None


class SpeakerSelector:
    """Selects SPEAKER_MAIN using hybrid scoring algorithm.

    Hybrid Score = duration_weight × duration_ratio + turn_weight × turn_ratio

    Where:
    - duration_ratio = speaker_duration / total_speech_duration
    - turn_ratio = speaker_turns / total_turns

    Example usage:
        selector = SpeakerSelector(config)
        assignment = selector.assign_roles(conversation)

        if assignment.is_valid:
            main = assignment.main_speaker
            print(f"SPEAKER_MAIN: {main.id} (score: {assignment.main_score:.3f})")
    """

    def __init__(self, config: Optional[SpeakerSelectionConfig] = None):
        """Initialize the selector.

        Args:
            config: Speaker selection configuration
        """
        self.config = config or SpeakerSelectionConfig()

    def compute_score(
        self,
        speaker: Speaker,
        total_duration: float,
        total_turns: int,
    ) -> float:
        """Compute hybrid score for a speaker.

        Args:
            speaker: Speaker to score
            total_duration: Total speech duration across all speakers
            total_turns: Total turn count across all speakers

        Returns:
            Hybrid score between 0 and 1
        """
        if total_duration <= 0 or total_turns <= 0:
            return 0.0

        duration_ratio = speaker.total_duration / total_duration
        turn_ratio = speaker.turn_count / total_turns

        score = (
            self.config.duration_weight * duration_ratio +
            self.config.turn_count_weight * turn_ratio
        )

        return score

    def validate_conversation(self, conv: Conversation) -> tuple[bool, Optional[str]]:
        """Validate a conversation for processing.

        Args:
            conv: Conversation to validate

        Returns:
            Tuple of (is_valid, skip_reason)
        """
        # Check duration bounds
        if conv.duration < self.config.min_duration_sec:
            return False, f"Too short: {conv.duration:.1f}s < {self.config.min_duration_sec}s"

        if conv.duration > self.config.max_duration_sec:
            return False, f"Too long: {conv.duration:.1f}s > {self.config.max_duration_sec}s"

        # Check minimum turns
        total_turns = sum(s.turn_count for s in conv.speakers.values())
        if total_turns < self.config.min_turns:
            return False, f"Too few turns: {total_turns} < {self.config.min_turns}"

        # Check speaker count
        if conv.num_speakers < 1:
            return False, "No speakers found"

        return True, None

    def assign_roles(self, conv: Conversation) -> SpeakerAssignment:
        """Assign speaker roles (SPEAKER_MAIN, SPEAKER_USER) to a conversation.

        Args:
            conv: Conversation to process

        Returns:
            SpeakerAssignment with role assignments
        """
        # Validate conversation
        is_valid, skip_reason = self.validate_conversation(conv)

        if not is_valid:
            # Return invalid assignment
            return SpeakerAssignment(
                conversation_id=conv.id,
                main_speaker=None,
                main_score=0.0,
                user_speakers=[],
                user_score=0.0,
                total_speakers=conv.num_speakers,
                is_valid=False,
                skip_reason=skip_reason,
            )

        # Compute total statistics
        total_duration = sum(s.total_duration for s in conv.speakers.values())
        total_turns = sum(s.turn_count for s in conv.speakers.values())

        # Score all speakers
        scored_speakers = []
        for speaker in conv.speakers.values():
            score = self.compute_score(speaker, total_duration, total_turns)
            scored_speakers.append((speaker, score))

        # Sort by score (descending)
        scored_speakers.sort(key=lambda x: x[1], reverse=True)

        # Assign SPEAKER_MAIN (highest score)
        main_speaker, main_score = scored_speakers[0]

        # Assign SPEAKER_USER (remaining speakers)
        user_speakers = []
        user_score = 0.0

        if len(scored_speakers) > 1:
            for speaker, score in scored_speakers[1:]:
                user_speakers.append(speaker)
                user_score += score

        return SpeakerAssignment(
            conversation_id=conv.id,
            main_speaker=main_speaker,
            main_score=main_score,
            user_speakers=user_speakers,
            user_score=user_score,
            total_speakers=conv.num_speakers,
            is_valid=True,
        )

    def get_utterances_by_role(
        self,
        conv: Conversation,
        assignment: SpeakerAssignment,
    ) -> dict[SpeakerRole, list]:
        """Get utterances grouped by speaker role.

        Args:
            conv: Original conversation
            assignment: Speaker role assignment

        Returns:
            Dictionary mapping SpeakerRole to list of utterances
        """
        if not assignment.is_valid:
            return {SpeakerRole.SPEAKER_MAIN: [], SpeakerRole.SPEAKER_USER: []}

        main_id = assignment.main_speaker.id
        user_ids = {s.id for s in assignment.user_speakers}

        result = {
            SpeakerRole.SPEAKER_MAIN: [],
            SpeakerRole.SPEAKER_USER: [],
        }

        for utt in conv.utterances:
            if utt.speaker_id == main_id:
                result[SpeakerRole.SPEAKER_MAIN].append(utt)
            elif utt.speaker_id in user_ids:
                result[SpeakerRole.SPEAKER_USER].append(utt)
            else:
                # Unknown speaker - merge into SPEAKER_USER
                if self.config.merge_minor_speakers:
                    result[SpeakerRole.SPEAKER_USER].append(utt)
                else:
                    logger.warning(
                        f"Unknown speaker {utt.speaker_id} in {conv.id}"
                    )

        return result

    def get_all_speakers_info(
        self,
        conv: Conversation,
        assignment: SpeakerAssignment,
    ) -> list[SpeakerInfo]:
        """Get detailed information for ALL speakers in the conversation.

        This method preserves individual speaker statistics for future
        N-speaker diarization support, while maintaining compatibility
        with the current Moshi 2-stream (MAIN/USER) format.

        Args:
            conv: Original conversation
            assignment: Speaker role assignment

        Returns:
            List of SpeakerInfo objects sorted by rank (score descending)
        """
        if not assignment.is_valid:
            return []

        # Compute totals for scoring
        total_duration = sum(s.total_duration for s in conv.speakers.values())
        total_turns = sum(s.turn_count for s in conv.speakers.values())

        # Get main speaker ID
        main_id = assignment.main_speaker.id

        # Group utterances by speaker
        utterances_by_speaker: dict[str, list] = {}
        for utt in conv.utterances:
            if utt.speaker_id not in utterances_by_speaker:
                utterances_by_speaker[utt.speaker_id] = []
            utterances_by_speaker[utt.speaker_id].append(utt)

        # Build speaker info list with scores
        speaker_infos = []
        for speaker in conv.speakers.values():
            score = self.compute_score(speaker, total_duration, total_turns)

            # Determine role
            role = (
                SpeakerRole.SPEAKER_MAIN
                if speaker.id == main_id
                else SpeakerRole.SPEAKER_USER
            )

            # Build segments list
            segments = []
            for utt in utterances_by_speaker.get(speaker.id, []):
                segments.append({
                    "start": round(utt.start, 3),
                    "end": round(utt.end, 3),
                    "text": utt.text,
                })

            # Sort segments by start time
            segments.sort(key=lambda x: x["start"])

            speaker_infos.append(SpeakerInfo(
                speaker_id=speaker.id,
                role=role,
                total_duration=speaker.total_duration,
                turn_count=speaker.turn_count,
                hybrid_score=score,
                rank=0,  # Will be set after sorting
                segments=segments,
            ))

        # Sort by score descending and assign ranks
        speaker_infos.sort(key=lambda x: x.hybrid_score, reverse=True)
        for i, info in enumerate(speaker_infos):
            info.rank = i + 1

        return speaker_infos

    def get_diarization_metadata(
        self,
        conv: Conversation,
        assignment: SpeakerAssignment,
    ) -> dict:
        """Generate comprehensive diarization metadata.

        Creates a metadata structure that supports:
        1. Current Moshi 2-stream format (main/user)
        2. Future N-speaker diarization
        3. Speaker overlap analysis
        4. Turn-taking patterns

        Args:
            conv: Original conversation
            assignment: Speaker role assignment

        Returns:
            Dictionary with diarization metadata
        """
        if not assignment.is_valid:
            return {}

        speakers_info = self.get_all_speakers_info(conv, assignment)

        # Compute speaker overlap regions
        overlap_regions = self._compute_overlap_regions(conv)

        # Compute turn-taking statistics
        turn_stats = self._compute_turn_statistics(conv)

        return {
            "num_speakers": len(speakers_info),
            "speakers": [info.to_dict() for info in speakers_info],
            "main_speaker_id": assignment.main_speaker.id,
            "user_speaker_ids": [s.id for s in assignment.user_speakers],
            "overlap_analysis": {
                "total_overlap_duration": round(overlap_regions["total_duration"], 3),
                "overlap_ratio": round(
                    overlap_regions["total_duration"] / max(conv.duration, 0.001), 4
                ),
                "num_overlap_regions": overlap_regions["count"],
            },
            "turn_taking": turn_stats,
            "scoring": {
                "duration_weight": self.config.duration_weight,
                "turn_count_weight": self.config.turn_count_weight,
            },
        }

    def _compute_overlap_regions(self, conv: Conversation) -> dict:
        """Compute regions where speakers overlap.

        Returns:
            Dict with overlap statistics
        """
        if not conv.utterances:
            return {"total_duration": 0.0, "count": 0, "regions": []}

        # Sort utterances by start time
        sorted_utts = sorted(conv.utterances, key=lambda u: u.start)

        overlap_regions = []
        total_overlap = 0.0

        for i, utt1 in enumerate(sorted_utts):
            for utt2 in sorted_utts[i + 1:]:
                if utt2.start >= utt1.end:
                    break  # No more overlaps possible

                if utt1.speaker_id != utt2.speaker_id:
                    # Found overlap between different speakers
                    overlap_start = utt2.start
                    overlap_end = min(utt1.end, utt2.end)
                    overlap_duration = overlap_end - overlap_start

                    if overlap_duration > 0:
                        overlap_regions.append({
                            "start": round(overlap_start, 3),
                            "end": round(overlap_end, 3),
                            "speakers": [utt1.speaker_id, utt2.speaker_id],
                        })
                        total_overlap += overlap_duration

        return {
            "total_duration": total_overlap,
            "count": len(overlap_regions),
            "regions": overlap_regions[:10],  # Limit to first 10 for brevity
        }

    def _compute_turn_statistics(self, conv: Conversation) -> dict:
        """Compute turn-taking statistics.

        Returns:
            Dict with turn-taking analysis
        """
        if not conv.utterances:
            return {
                "total_turns": 0,
                "avg_turn_duration": 0.0,
                "speaker_transitions": 0,
            }

        sorted_utts = sorted(conv.utterances, key=lambda u: u.start)

        total_duration = sum(u.duration for u in sorted_utts)
        transitions = 0
        prev_speaker = None

        for utt in sorted_utts:
            if prev_speaker is not None and utt.speaker_id != prev_speaker:
                transitions += 1
            prev_speaker = utt.speaker_id

        return {
            "total_turns": len(sorted_utts),
            "avg_turn_duration": round(
                total_duration / max(len(sorted_utts), 1), 3
            ),
            "speaker_transitions": transitions,
            "transitions_per_minute": round(
                transitions / max(conv.duration / 60, 0.001), 2
            ),
        }
