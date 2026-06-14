# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Segment-level alignment generation for Phase 1.

Creates intermediate alignment files containing segment-level timestamps,
which are later refined to word-level in Phase 2.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import json
import logging

from ..readers.lhotse_shar import Conversation, Utterance
from .speaker_selector import SpeakerRole, SpeakerAssignment

logger = logging.getLogger(__name__)


@dataclass
class Segment:
    """A single aligned segment."""
    speaker: str  # SPEAKER_MAIN or SPEAKER_USER
    start: float
    end: float
    text: str
    original_speaker_id: str = ""

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization.

        Includes original_speaker_id for multi-speaker tracking.
        """
        result = {
            "speaker": self.speaker,
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "text": self.text,
        }
        # Include original_speaker_id for extensibility
        if self.original_speaker_id:
            result["original_speaker_id"] = self.original_speaker_id
        return result


@dataclass
class SpeakerInfo:
    """Information about a speaker's role assignment."""
    role: str  # SPEAKER_MAIN or SPEAKER_USER
    original_id: str
    score: float
    total_duration: float
    turn_count: int

    def to_dict(self) -> dict:
        return {
            "original_id": self.original_id,
            "score": round(self.score, 4),
            "total_duration": round(self.total_duration, 3),
            "turn_count": self.turn_count,
        }


@dataclass
class SegmentAlignment:
    """Complete segment-level alignment for a conversation."""
    conversation_id: str
    duration: float
    speakers: dict[str, SpeakerInfo] = field(default_factory=dict)
    segments: list[Segment] = field(default_factory=list)
    is_valid: bool = True
    error: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "conversation_id": self.conversation_id,
            "duration": round(self.duration, 3),
            "speakers": {
                role: info.to_dict()
                for role, info in self.speakers.items()
            },
            "segments": [seg.to_dict() for seg in self.segments],
        }

    def save(self, path: Path) -> bool:
        """Save alignment to JSON file."""
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            logger.error(f"Error saving alignment to {path}: {e}")
            return False

    @classmethod
    def load(cls, path: Path) -> "SegmentAlignment":
        """Load alignment from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        speakers = {}
        for role, info in data.get("speakers", {}).items():
            speakers[role] = SpeakerInfo(
                role=role,
                original_id=info["original_id"],
                score=info["score"],
                total_duration=info["total_duration"],
                turn_count=info["turn_count"],
            )

        segments = []
        for seg in data.get("segments", []):
            segments.append(Segment(
                speaker=seg["speaker"],
                start=seg["start"],
                end=seg["end"],
                text=seg["text"],
                original_speaker_id=seg.get("original_speaker_id", ""),
            ))

        return cls(
            conversation_id=data["conversation_id"],
            duration=data["duration"],
            speakers=speakers,
            segments=segments,
        )


class SegmentAligner:
    """Generates segment-level alignments from conversation data.

    This is the output of Phase 1 processing, containing:
    - Speaker role assignments (SPEAKER_MAIN, SPEAKER_USER)
    - Segment-level timestamps and text
    - Speaker statistics

    Example usage:
        aligner = SegmentAligner()
        alignment = aligner.create_alignment(conversation, assignment)
        alignment.save(output_path)
    """

    def __init__(self):
        pass

    def create_alignment(
        self,
        conversation: Conversation,
        assignment: SpeakerAssignment,
        utterances_by_role: dict[SpeakerRole, list[Utterance]],
    ) -> SegmentAlignment:
        """Create segment alignment from conversation and role assignment.

        Args:
            conversation: Source conversation
            assignment: Speaker role assignment
            utterances_by_role: Utterances grouped by role

        Returns:
            SegmentAlignment object
        """
        if not assignment.is_valid:
            return SegmentAlignment(
                conversation_id=conversation.id,
                duration=conversation.duration,
                is_valid=False,
                error=assignment.skip_reason,
            )

        # Build speaker info
        speakers = {}

        # SPEAKER_MAIN
        main_speaker = assignment.main_speaker
        main_utts = utterances_by_role[SpeakerRole.SPEAKER_MAIN]
        speakers["SPEAKER_MAIN"] = SpeakerInfo(
            role="SPEAKER_MAIN",
            original_id=main_speaker.id,
            score=assignment.main_score,
            total_duration=sum(u.duration for u in main_utts),
            turn_count=len(main_utts),
        )

        # SPEAKER_USER (may combine multiple speakers)
        user_utts = utterances_by_role[SpeakerRole.SPEAKER_USER]
        if assignment.user_speakers:
            original_ids = [s.id for s in assignment.user_speakers]
            speakers["SPEAKER_USER"] = SpeakerInfo(
                role="SPEAKER_USER",
                original_id=",".join(original_ids) if len(original_ids) > 1 else original_ids[0] if original_ids else "",
                score=assignment.user_score,
                total_duration=sum(u.duration for u in user_utts),
                turn_count=len(user_utts),
            )
        else:
            speakers["SPEAKER_USER"] = SpeakerInfo(
                role="SPEAKER_USER",
                original_id="",
                score=0.0,
                total_duration=0.0,
                turn_count=0,
            )

        # Build segments (sorted by start time)
        segments = []

        for utt in main_utts:
            segments.append(Segment(
                speaker="SPEAKER_MAIN",
                start=utt.start,
                end=utt.end,
                text=utt.text,
                original_speaker_id=utt.speaker_id,
            ))

        for utt in user_utts:
            segments.append(Segment(
                speaker="SPEAKER_USER",
                start=utt.start,
                end=utt.end,
                text=utt.text,
                original_speaker_id=utt.speaker_id,
            ))

        # Sort by start time
        segments.sort(key=lambda s: s.start)

        return SegmentAlignment(
            conversation_id=conversation.id,
            duration=conversation.duration,
            speakers=speakers,
            segments=segments,
            is_valid=True,
        )

    def merge_adjacent_segments(
        self,
        alignment: SegmentAlignment,
        max_gap: float = 0.5,
    ) -> SegmentAlignment:
        """Merge adjacent segments from the same speaker.

        Args:
            alignment: Original alignment
            max_gap: Maximum gap (seconds) to merge

        Returns:
            New alignment with merged segments
        """
        if not alignment.segments:
            return alignment

        merged = []
        current = alignment.segments[0]

        for seg in alignment.segments[1:]:
            if seg.speaker == current.speaker and seg.start - current.end <= max_gap:
                # Merge segments
                current = Segment(
                    speaker=current.speaker,
                    start=current.start,
                    end=seg.end,
                    text=current.text + " " + seg.text,
                    original_speaker_id=current.original_speaker_id,
                )
            else:
                merged.append(current)
                current = seg

        merged.append(current)

        return SegmentAlignment(
            conversation_id=alignment.conversation_id,
            duration=alignment.duration,
            speakers=alignment.speakers,
            segments=merged,
            is_valid=alignment.is_valid,
        )
