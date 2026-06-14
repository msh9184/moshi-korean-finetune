# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Moshi format output writer.

Writes data in the format expected by Moshi finetuning:
- Stereo WAV audio files (24kHz, 16-bit)
- Word-level alignment JSON files per speaker
- JSONL manifest file
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import json
import logging

from ..config import DatasetConfig
from ..processors.stereo_converter import StereoAudio
from ..processors.segment_aligner import SegmentAlignment
from ..aligners.whisper_timestamped import WordAlignment, AlignmentResult
from ..aligners.nfa_aligner import ConversationAlignment

logger = logging.getLogger(__name__)


@dataclass
class ManifestEntry:
    """A single entry in the manifest.jsonl file."""
    audio: str  # Relative path to audio file
    alignment_speaker01: str  # Relative path to SPEAKER_MAIN alignment
    alignment_speaker02: str  # Relative path to SPEAKER_USER alignment
    duration: float
    speakers: int = 2
    conversation_id: str = ""
    alignment: str = ""  # Relative path to Moshi combined alignment
    alignment_extended: str = ""  # Relative path to extended alignment with speaker metadata

    def to_dict(self) -> dict:
        """Return basic manifest dict for Moshi training compatibility."""
        return {
            "audio": self.audio,
            "alignment_speaker01": self.alignment_speaker01,
            "alignment_speaker02": self.alignment_speaker02,
            "duration": round(self.duration, 3),
            "speakers": self.speakers,
        }

    def to_extended_dict(self) -> dict:
        """Return extended manifest dict with all alignment paths."""
        result = self.to_dict()
        result["conversation_id"] = self.conversation_id
        if self.alignment:
            result["alignment"] = self.alignment
        if self.alignment_extended:
            result["alignment_extended"] = self.alignment_extended
        return result

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    def to_extended_json(self) -> str:
        return json.dumps(self.to_extended_dict(), ensure_ascii=False)


class MoshiFormatWriter:
    """Writes data in Moshi finetuning format.

    Output structure:
        {output_dir}/
        ├── audio/
        │   ├── conv_000001.wav
        │   └── ...
        ├── alignment_speaker01/
        │   ├── conv_000001.json
        │   └── ...
        ├── alignment_speaker02/
        │   ├── conv_000001.json
        │   └── ...
        ├── alignments/              # Moshi format (combined)
        │   ├── conv_000001.json
        │   └── ...
        └── manifest.jsonl

    Example usage:
        writer = MoshiFormatWriter(dataset_config)

        # Write audio
        writer.write_audio(stereo_audio, "conv_001")

        # Write alignments (Phase 2)
        writer.write_alignments(alignment_result, "conv_001")

        # Add to manifest
        writer.add_manifest_entry(entry)

        # Finalize
        writer.finalize()
    """

    def __init__(self, config: DatasetConfig, machine_id: Optional[int] = None):
        """Initialize the writer.

        Args:
            config: Dataset configuration
            machine_id: Machine ID for distributed processing (None for single machine)
        """
        self.config = config
        self.machine_id = machine_id
        self.manifest_entries: list[ManifestEntry] = []

        # Create output directories
        self._create_directories()

    def _create_directories(self) -> None:
        """Create output directory structure."""
        self.config.audio_output_dir.mkdir(parents=True, exist_ok=True)
        self.config.alignment_speaker01_dir.mkdir(parents=True, exist_ok=True)
        self.config.alignment_speaker02_dir.mkdir(parents=True, exist_ok=True)
        self.config.segment_alignment_dir.mkdir(parents=True, exist_ok=True)
        self.config.alignments_dir.mkdir(parents=True, exist_ok=True)
        self.config.extended_alignments_dir.mkdir(parents=True, exist_ok=True)

    def get_audio_path(self, conversation_id: str) -> Path:
        """Get output path for audio file."""
        return self.config.audio_output_dir / f"{conversation_id}.wav"

    def get_alignment_path(self, conversation_id: str, speaker: int) -> Path:
        """Get output path for alignment file.

        Args:
            conversation_id: Conversation ID
            speaker: 1 for SPEAKER_MAIN, 2 for SPEAKER_USER
        """
        if speaker == 1:
            return self.config.alignment_speaker01_dir / f"{conversation_id}.json"
        else:
            return self.config.alignment_speaker02_dir / f"{conversation_id}.json"

    def get_segment_alignment_path(self, conversation_id: str) -> Path:
        """Get output path for segment-level alignment (Phase 1)."""
        return self.config.segment_alignment_dir / f"{conversation_id}.json"

    def get_extended_alignment_path(self, conversation_id: str) -> Path:
        """Get output path for extended alignment with speaker metadata."""
        return self.config.extended_alignments_dir / f"{conversation_id}.json"

    def get_moshi_alignment_path(self, conversation_id: str) -> Path:
        """Get output path for Moshi format combined alignment."""
        return self.config.alignments_dir / f"{conversation_id}.json"

    def write_audio(
        self,
        stereo_audio: StereoAudio,
        conversation_id: str,
    ) -> Optional[Path]:
        """Write stereo audio to file.

        Args:
            stereo_audio: StereoAudio object
            conversation_id: Conversation ID

        Returns:
            Path to written file, or None on error
        """
        import soundfile as sf

        output_path = self.get_audio_path(conversation_id)

        try:
            # soundfile expects (samples, channels)
            sf.write(
                output_path,
                stereo_audio.audio.T,
                stereo_audio.sample_rate,
                subtype="PCM_16",
            )
            logger.debug(f"Wrote audio: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Error writing audio {conversation_id}: {e}")
            return None

    def write_segment_alignment(
        self,
        segment_alignment: SegmentAlignment,
        conversation_id: str,
    ) -> Optional[Path]:
        """Write segment-level alignment (Phase 1 output).

        Args:
            segment_alignment: SegmentAlignment object
            conversation_id: Conversation ID

        Returns:
            Path to written file, or None on error
        """
        output_path = self.get_segment_alignment_path(conversation_id)

        if segment_alignment.save(output_path):
            logger.debug(f"Wrote segment alignment: {output_path}")
            return output_path
        return None

    def write_word_alignments(
        self,
        alignment_result: AlignmentResult,
        conversation_id: str,
    ) -> tuple[Optional[Path], Optional[Path]]:
        """Write word-level alignments (Phase 2 output).

        Args:
            alignment_result: AlignmentResult object
            conversation_id: Conversation ID

        Returns:
            Tuple of (speaker01_path, speaker02_path), None on error
        """
        main_path = None
        user_path = None

        if alignment_result.main_alignment:
            main_output = self.get_alignment_path(conversation_id, 1)
            if alignment_result.main_alignment.save(main_output):
                main_path = main_output
                logger.debug(f"Wrote SPEAKER_MAIN alignment: {main_output}")

        if alignment_result.user_alignment:
            user_output = self.get_alignment_path(conversation_id, 2)
            if alignment_result.user_alignment.save(user_output):
                user_path = user_output
                logger.debug(f"Wrote SPEAKER_USER alignment: {user_output}")

        return main_path, user_path

    def write_conversation_alignment(
        self,
        alignment: ConversationAlignment,
        save_extended: bool = True,
        save_moshi: bool = True,
    ) -> tuple[Optional[Path], Optional[Path]]:
        """Write conversation alignment in both Moshi and extended formats.

        Args:
            alignment: ConversationAlignment object with word-level alignments
            save_extended: Whether to save extended format with speaker metadata
            save_moshi: Whether to save basic Moshi format

        Returns:
            Tuple of (moshi_path, extended_path), None for skipped/failed outputs
        """
        moshi_path = None
        extended_path = None

        if not alignment.is_valid:
            logger.warning(f"Invalid alignment for {alignment.conversation_id}: {alignment.error}")
            return moshi_path, extended_path

        # Save Moshi format (basic)
        if save_moshi:
            output_path = self.get_moshi_alignment_path(alignment.conversation_id)
            if alignment.save_moshi_format(output_path):
                moshi_path = output_path
                logger.debug(f"Wrote Moshi alignment: {output_path}")

        # Save extended format (with full speaker metadata)
        if save_extended:
            output_path = self.get_extended_alignment_path(alignment.conversation_id)
            if alignment.save_extended_format(output_path):
                extended_path = output_path
                logger.debug(f"Wrote extended alignment: {output_path}")

        return moshi_path, extended_path

    def create_manifest_entry(
        self,
        conversation_id: str,
        duration: float,
        speakers: int = 2,
        include_extended: bool = True,
    ) -> ManifestEntry:
        """Create a manifest entry.

        Args:
            conversation_id: Conversation ID
            duration: Audio duration in seconds
            speakers: Number of speakers
            include_extended: Whether to include extended alignment paths

        Returns:
            ManifestEntry object
        """
        entry = ManifestEntry(
            audio=f"audio/{conversation_id}.wav",
            alignment_speaker01=f"alignment_speaker01/{conversation_id}.json",
            alignment_speaker02=f"alignment_speaker02/{conversation_id}.json",
            duration=duration,
            speakers=speakers,
            conversation_id=conversation_id,
            alignment=f"alignments/{conversation_id}.json",
        )
        if include_extended:
            entry.alignment_extended = f"alignments_extended/{conversation_id}.json"
        return entry

    def add_manifest_entry(self, entry: ManifestEntry) -> None:
        """Add an entry to the manifest."""
        self.manifest_entries.append(entry)

    def write_manifest(self, append: bool = False) -> Path:
        """Write manifest.jsonl file.

        Args:
            append: If True, append to existing manifest

        Returns:
            Path to manifest file
        """
        mode = "a" if append else "w"
        manifest_path = self.config.manifest_path

        with open(manifest_path, mode, encoding="utf-8") as f:
            for entry in self.manifest_entries:
                f.write(entry.to_json() + "\n")

        logger.info(f"Wrote manifest with {len(self.manifest_entries)} entries: {manifest_path}")
        return manifest_path

    def write_phase1_manifest(self) -> Path:
        """Write Phase 1 intermediate manifest.

        This manifest is used as input for Phase 2 processing.
        """
        manifest_path = self.config.phase1_manifest_path

        with open(manifest_path, "w", encoding="utf-8") as f:
            for entry in self.manifest_entries:
                # Phase 1 manifest includes segment alignment path
                data = {
                    "conversation_id": entry.conversation_id,
                    "audio": entry.audio,
                    "segment_alignment": f"segment_alignment/{entry.conversation_id}.json",
                    "duration": entry.duration,
                    "speakers": entry.speakers,
                }
                f.write(json.dumps(data, ensure_ascii=False) + "\n")

        logger.info(f"Wrote Phase 1 manifest: {manifest_path}")
        return manifest_path

    def finalize(self, phase: int = 2, machine_id: Optional[int] = None) -> None:
        """Finalize output and write manifest.

        Args:
            phase: 1 for Phase 1 (segment-level), 2 for Phase 2 (word-level)
            machine_id: Machine ID for distributed processing
        """
        machine_id = machine_id if machine_id is not None else self.machine_id

        if phase == 1:
            if machine_id is not None:
                self.write_machine_manifest(machine_id, phase=1)
            else:
                self.write_phase1_manifest()
        else:
            if machine_id is not None:
                self.write_machine_manifest(machine_id, phase=2)
            else:
                self.write_manifest()

        # Clear entries for next batch
        self.manifest_entries = []

    def write_machine_manifest(self, machine_id: int, phase: int = 2) -> Path:
        """Write machine-specific manifest for later merging.

        Args:
            machine_id: Machine ID
            phase: 1 for Phase 1, 2 for Phase 2

        Returns:
            Path to manifest file
        """
        checkpoint_dir = self.config.checkpoint_dir
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        if phase == 1:
            manifest_path = checkpoint_dir / f"manifest_phase1_{machine_id:03d}.jsonl"
        else:
            manifest_path = checkpoint_dir / f"manifest_phase2_{machine_id:03d}.jsonl"

        with open(manifest_path, "w", encoding="utf-8") as f:
            for entry in self.manifest_entries:
                if phase == 1:
                    # Phase 1 manifest format
                    data = {
                        "conversation_id": entry.conversation_id,
                        "audio": entry.audio,
                        "segment_alignment": f"segment_alignment/{entry.conversation_id}.json",
                        "duration": entry.duration,
                        "speakers": entry.speakers,
                    }
                else:
                    # Phase 2 manifest format - use extended dict with all paths
                    data = entry.to_extended_dict()
                    # Ensure alignment paths are present
                    if not data.get("alignment"):
                        data["alignment"] = f"alignments/{entry.conversation_id}.json"
                    if not data.get("alignment_extended"):
                        data["alignment_extended"] = f"alignments_extended/{entry.conversation_id}.json"

                f.write(json.dumps(data, ensure_ascii=False) + "\n")

        logger.info(f"Wrote machine {machine_id} manifest with {len(self.manifest_entries)} entries: {manifest_path}")
        return manifest_path

    def get_statistics(self) -> dict:
        """Get statistics about written data."""
        audio_files = list(self.config.audio_output_dir.glob("*.wav"))
        align01_files = list(self.config.alignment_speaker01_dir.glob("*.json"))
        align02_files = list(self.config.alignment_speaker02_dir.glob("*.json"))
        moshi_align_files = list(self.config.alignments_dir.glob("*.json"))
        extended_align_files = list(self.config.extended_alignments_dir.glob("*.json"))

        # Count manifest entries
        manifest_count = 0
        if self.config.manifest_path.exists():
            with open(self.config.manifest_path) as f:
                manifest_count = sum(1 for _ in f)

        return {
            "audio_files": len(audio_files),
            "alignment_speaker01_files": len(align01_files),
            "alignment_speaker02_files": len(align02_files),
            "moshi_alignment_files": len(moshi_align_files),
            "extended_alignment_files": len(extended_align_files),
            "manifest_entries": manifest_count,
        }
