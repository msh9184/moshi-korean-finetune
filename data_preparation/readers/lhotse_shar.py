# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Lhotse Shar format reader for Korean broadcast data.

Lhotse Shar format consists of:
- cuts.XXXXXX.jsonl.gz: Cut manifests with supervision information
- recording.XXXXXX.tar: Audio files corresponding to cuts

File naming convention:
- Cut ID: "A220001-0" (used for audio file names in tar)
- Recording ID: "A220001" (reference ID in cuts metadata)
- Audio files in tar: "{cut_id}.wav" or "{cut_id}.flac"

This reader groups cuts by conversation_id and provides unified access
to audio and text data for the data preparation pipeline.
"""

import gzip
import json
import tarfile
import tempfile
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional
import io
import logging

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Speaker:
    """Speaker information extracted from supervisions."""
    id: str
    total_duration: float = 0.0
    turn_count: int = 0
    utterances: list = field(default_factory=list)

    @property
    def avg_turn_duration(self) -> float:
        """Average duration per turn."""
        if self.turn_count == 0:
            return 0.0
        return self.total_duration / self.turn_count


@dataclass
class Utterance:
    """Single utterance/segment within a conversation."""
    id: str
    speaker_id: str
    text: str
    start: float
    end: float
    language: str = "ko"

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class Conversation:
    """A complete conversation with multiple speakers and utterances."""
    id: str
    recording_id: str
    cut_id: str  # Added: cut ID for audio file matching
    utterances: list[Utterance] = field(default_factory=list)
    speakers: dict[str, Speaker] = field(default_factory=dict)
    audio_path: Optional[Path] = None
    duration: float = 0.0
    sample_rate: int = 16000

    def __post_init__(self):
        """Build speaker statistics from utterances."""
        self._build_speaker_stats()

    def _build_speaker_stats(self):
        """Calculate speaker statistics from utterances."""
        speaker_data = defaultdict(lambda: {"duration": 0.0, "count": 0, "utterances": []})

        for utt in self.utterances:
            speaker_data[utt.speaker_id]["duration"] += utt.duration
            speaker_data[utt.speaker_id]["count"] += 1
            speaker_data[utt.speaker_id]["utterances"].append(utt)

        self.speakers = {
            spk_id: Speaker(
                id=spk_id,
                total_duration=data["duration"],
                turn_count=data["count"],
                utterances=data["utterances"],
            )
            for spk_id, data in speaker_data.items()
        }

    @property
    def num_speakers(self) -> int:
        return len(self.speakers)

    @property
    def total_speech_duration(self) -> float:
        """Total duration of all speech segments."""
        return sum(utt.duration for utt in self.utterances)

    def get_sorted_speakers(self, by: str = "duration") -> list[Speaker]:
        """Get speakers sorted by specified criterion.

        Args:
            by: Sort criterion - "duration" or "turn_count"

        Returns:
            List of speakers sorted in descending order
        """
        if by == "duration":
            return sorted(
                self.speakers.values(),
                key=lambda s: s.total_duration,
                reverse=True
            )
        elif by == "turn_count":
            return sorted(
                self.speakers.values(),
                key=lambda s: s.turn_count,
                reverse=True
            )
        else:
            raise ValueError(f"Unknown sort criterion: {by}")


class LhotseSharReader:
    """Reader for Lhotse Shar format Korean broadcast data.

    Reads cuts.XXXXXX.jsonl.gz and recording.XXXXXX.tar pairs,
    grouping cuts by conversation_id for downstream processing.

    Important: Audio files in tar are named by cut["id"], not recording["id"].
    - Cut ID example: "A220001-0" → audio file: "A220001-0.wav"
    - Recording ID example: "A220001" (not used for file naming)

    Example usage:
        reader = LhotseSharReader(
            source_dir="/path/to/lhotse/data",
            machine_id=0,
            total_machines=20
        )

        for shard_idx, conversations in reader.iter_shards():
            for conv in conversations:
                print(f"Conversation {conv.id}: {conv.num_speakers} speakers")
    """

    def __init__(
        self,
        source_dir: Path,
        machine_id: int = 0,
        total_machines: int = 1,
    ):
        """Initialize the reader.

        Args:
            source_dir: Path to Lhotse Shar data directory
            machine_id: ID of current machine (0-indexed)
            total_machines: Total number of machines for distributed processing
        """
        self.source_dir = Path(source_dir)
        self.machine_id = machine_id
        self.total_machines = total_machines

        # Discover shard files
        self.shard_indices = self._discover_shards()
        logger.info(f"Found {len(self.shard_indices)} total shards in {source_dir}")

        # Filter shards for this machine
        self.my_shards = [
            idx for i, idx in enumerate(self.shard_indices)
            if i % total_machines == machine_id
        ]
        logger.info(
            f"Machine {machine_id}/{total_machines}: "
            f"Processing {len(self.my_shards)} shards"
        )

    def _discover_shards(self) -> list[int]:
        """Discover available shard indices from cuts files."""
        cuts_files = sorted(self.source_dir.glob("cuts.*.jsonl.gz"))
        indices = []

        for f in cuts_files:
            # Extract shard index from filename: cuts.000123.jsonl.gz -> 123
            try:
                idx_str = f.name.split(".")[1]
                indices.append(int(idx_str))
            except (IndexError, ValueError):
                logger.warning(f"Could not parse shard index from {f.name}")

        return sorted(indices)

    def _get_cuts_path(self, shard_idx: int) -> Path:
        """Get path to cuts file for a shard."""
        return self.source_dir / f"cuts.{shard_idx:06d}.jsonl.gz"

    def _get_recording_path(self, shard_idx: int) -> Path:
        """Get path to recording tar file for a shard."""
        return self.source_dir / f"recording.{shard_idx:06d}.tar"

    def _read_cuts(self, shard_idx: int) -> list[dict]:
        """Read all cuts from a shard's cuts file."""
        cuts_path = self._get_cuts_path(shard_idx)
        cuts = []

        with gzip.open(cuts_path, "rt", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    cuts.append(json.loads(line))

        return cuts

    def _extract_audio_from_tar(
        self,
        shard_idx: int,
        cut_ids: set[str],
        temp_dir: Path,
    ) -> dict[str, Path]:
        """Extract audio files from recording tar.

        Args:
            shard_idx: Shard index
            cut_ids: Set of cut IDs to extract (e.g., "A220001-0")
            temp_dir: Directory to extract audio files to

        Returns:
            Mapping from cut_id to extracted audio path
        """
        tar_path = self._get_recording_path(shard_idx)
        audio_paths = {}

        if not tar_path.exists():
            logger.warning(f"Recording tar not found: {tar_path}")
            return audio_paths

        with tarfile.open(tar_path, "r") as tar:
            for member in tar.getmembers():
                # Skip non-audio files (e.g., .json files)
                if not member.name.endswith(('.wav', '.flac', '.mp3')):
                    continue

                # Audio files are named: {cut_id}.wav (e.g., A220001-0.wav)
                cut_id = Path(member.name).stem

                if cut_id in cut_ids:
                    tar.extract(member, temp_dir)
                    audio_paths[cut_id] = temp_dir / member.name

        return audio_paths

    def _group_cuts_by_conversation(
        self,
        cuts: list[dict],
    ) -> dict[str, list[dict]]:
        """Group cuts by conversation_id.

        For this dataset, each cut represents one conversation.
        The cut["id"] is used as the conversation ID.
        """
        conversations = defaultdict(list)

        for cut in cuts:
            # Use cut["id"] as the conversation ID
            # This matches the audio file naming in the tar
            conv_id = cut.get("id", "unknown")
            conversations[conv_id].append(cut)

        return dict(conversations)

    def _build_conversation(
        self,
        conv_id: str,
        cuts: list[dict],
        audio_paths: dict[str, Path],
    ) -> Optional[Conversation]:
        """Build a Conversation object from grouped cuts.

        Args:
            conv_id: Conversation ID (same as cut ID, e.g., "A220001-0")
            cuts: List of cuts belonging to this conversation
            audio_paths: Mapping from cut_id to audio path

        Returns:
            Conversation object or None if invalid
        """
        utterances = []
        recording_id = None
        cut_id = conv_id  # In this format, conv_id == cut_id
        duration = 0.0
        sample_rate = 16000

        for cut in cuts:
            # Get recording info
            if "recording" in cut:
                rec = cut["recording"]
                recording_id = rec.get("id", recording_id)
                duration = max(duration, rec.get("duration", 0.0))
                sample_rate = rec.get("sampling_rate", sample_rate)
            else:
                recording_id = recording_id or cut.get("id")
                duration = max(duration, cut.get("duration", 0.0))

            # Process supervisions (utterances)
            for sup in cut.get("supervisions", []):
                speaker_id = sup.get("speaker", "unknown")
                text = sup.get("text", "")
                start = sup.get("start", 0.0)
                dur = sup.get("duration", 0.0)
                language = sup.get("language", "ko")

                utterances.append(Utterance(
                    id=sup.get("id", f"{conv_id}_{len(utterances)}"),
                    speaker_id=speaker_id,
                    text=text,
                    start=start,
                    end=start + dur,
                    language=language,
                ))

        if not utterances:
            logger.debug(f"Skipping conversation {conv_id}: no utterances")
            return None

        # Sort utterances by start time
        utterances.sort(key=lambda u: u.start)

        # Get audio path using cut_id (not recording_id!)
        audio_path = audio_paths.get(cut_id)

        conv = Conversation(
            id=conv_id,
            recording_id=recording_id or conv_id,
            cut_id=cut_id,
            utterances=utterances,
            audio_path=audio_path,
            duration=duration,
            sample_rate=sample_rate,
        )

        return conv

    def read_shard(
        self,
        shard_idx: int,
        temp_dir: Optional[Path] = None,
        extract_audio: bool = True,
    ) -> list[Conversation]:
        """Read all conversations from a single shard.

        Args:
            shard_idx: Shard index to read
            temp_dir: Directory for extracted audio (created if None)
            extract_audio: Whether to extract audio files

        Returns:
            List of Conversation objects
        """
        # Read cuts
        cuts = self._read_cuts(shard_idx)
        logger.debug(f"Shard {shard_idx}: Read {len(cuts)} cuts")

        # Group by conversation (cut_id)
        conv_groups = self._group_cuts_by_conversation(cuts)
        logger.debug(f"Shard {shard_idx}: {len(conv_groups)} conversations")

        # Extract audio if needed
        audio_paths = {}
        if extract_audio:
            # Collect cut_ids (not recording_ids!)
            cut_ids = set(conv_groups.keys())

            if temp_dir is None:
                temp_dir = Path(tempfile.mkdtemp(prefix=f"shard_{shard_idx}_"))

            audio_paths = self._extract_audio_from_tar(
                shard_idx, cut_ids, temp_dir
            )
            logger.debug(f"Shard {shard_idx}: Extracted {len(audio_paths)} audio files")

        # Build conversation objects
        conversations = []
        for conv_id, cuts_list in conv_groups.items():
            conv = self._build_conversation(conv_id, cuts_list, audio_paths)
            if conv is not None:
                conversations.append(conv)

        return conversations

    def iter_shards(
        self,
        extract_audio: bool = True,
    ) -> Iterator[tuple[int, list[Conversation]]]:
        """Iterate over all shards assigned to this machine.

        Yields:
            Tuple of (shard_index, list of conversations)
        """
        for shard_idx in self.my_shards:
            try:
                conversations = self.read_shard(shard_idx, extract_audio=extract_audio)
                yield shard_idx, conversations
            except Exception as e:
                logger.error(f"Error reading shard {shard_idx}: {e}")
                continue

    def iter_conversations(
        self,
        extract_audio: bool = True,
    ) -> Iterator[Conversation]:
        """Iterate over all conversations assigned to this machine.

        Yields:
            Conversation objects
        """
        for _, conversations in self.iter_shards(extract_audio=extract_audio):
            yield from conversations

    def get_statistics(self) -> dict:
        """Compute statistics over all shards (without loading audio)."""
        stats = {
            "total_shards": len(self.shard_indices),
            "my_shards": len(self.my_shards),
            "total_conversations": 0,
            "total_utterances": 0,
            "total_duration_hours": 0.0,
            "speakers_per_conversation": [],
        }

        for shard_idx in self.my_shards:
            try:
                conversations = self.read_shard(shard_idx, extract_audio=False)
                stats["total_conversations"] += len(conversations)

                for conv in conversations:
                    stats["total_utterances"] += len(conv.utterances)
                    stats["total_duration_hours"] += conv.duration / 3600
                    stats["speakers_per_conversation"].append(conv.num_speakers)

            except Exception as e:
                logger.error(f"Error computing stats for shard {shard_idx}: {e}")

        # Compute averages
        if stats["speakers_per_conversation"]:
            stats["avg_speakers"] = np.mean(stats["speakers_per_conversation"])
        else:
            stats["avg_speakers"] = 0.0

        return stats
