# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Phase 1 CPU-based orchestrator - Enhanced version.

Handles:
1. Reading Lhotse Shar format data
2. Speaker selection (SPEAKER_MAIN assignment)
3. Mono to stereo conversion with FLAC compression
4. Rich metadata generation for Phase 2
5. Writing intermediate format for Phase 2

Features:
- Flexible distributed processing (N machines)
- Checkpoint/resume support
- FLAC output for disk savings
- Rich metadata preservation
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import json
import gzip
import logging
import shutil
import tempfile
import time
import os

from ..config import PipelineConfig, DatasetConfig
from ..readers.lhotse_shar import LhotseSharReader, Conversation
from ..processors.speaker_selector import SpeakerSelector, SpeakerRole
from ..processors.stereo_converter import StereoConverter

logger = logging.getLogger(__name__)


@dataclass
class ConversationMetadata:
    """Rich metadata for a processed conversation (for Phase 2).

    Includes comprehensive speaker diarization information for:
    1. Moshi 2-stream training (main/user format)
    2. Future N-speaker diarization support
    3. Speaker overlap and turn-taking analysis
    """
    conversation_id: str
    audio_path: str  # Relative path to audio file
    duration: float
    sample_rate: int

    # Speaker info (Moshi 2-stream format)
    main_speaker_id: str
    user_speaker_ids: list[str]
    main_score: float
    num_speakers: int

    # Segment info (Moshi 2-stream format)
    main_segments: list[dict]  # [{"start": 0.0, "end": 1.5, "text": "..."}]
    user_segments: list[dict]
    main_total_duration: float
    user_total_duration: float

    # Original source info
    original_recording_id: str
    source_shard: int

    # Processing info
    processed_at: str
    machine_id: int

    # Extended diarization info (for future N-speaker support)
    diarization_info: dict = None  # Comprehensive speaker diarization data

    def __post_init__(self):
        if self.diarization_info is None:
            self.diarization_info = {}

    def to_dict(self) -> dict:
        result = {
            "conversation_id": self.conversation_id,
            "audio_path": self.audio_path,
            "duration": round(self.duration, 3),
            "sample_rate": self.sample_rate,
            # Moshi 2-stream format (backward compatible)
            "speakers": {
                "main": {
                    "id": self.main_speaker_id,
                    "score": round(self.main_score, 4),
                    "total_duration": round(self.main_total_duration, 3),
                    "num_segments": len(self.main_segments),
                },
                "user": {
                    "ids": self.user_speaker_ids,
                    "total_duration": round(self.user_total_duration, 3),
                    "num_segments": len(self.user_segments),
                },
                "total": self.num_speakers,
            },
            # Moshi 2-stream segments
            "segments": {
                "main": self.main_segments,
                "user": self.user_segments,
            },
            "source": {
                "recording_id": self.original_recording_id,
                "shard": self.source_shard,
            },
            "processing": {
                "timestamp": self.processed_at,
                "machine_id": self.machine_id,
            },
        }

        # Add extended diarization info if available
        if self.diarization_info:
            result["diarization"] = self.diarization_info

        return result


@dataclass
class Phase1Stats:
    """Statistics for Phase 1 processing."""
    total_conversations: int = 0
    processed_conversations: int = 0
    skipped_conversations: int = 0
    failed_conversations: int = 0
    total_duration_hours: float = 0.0
    total_main_duration_hours: float = 0.0
    total_user_duration_hours: float = 0.0
    processing_time_sec: float = 0.0
    total_audio_bytes: int = 0
    total_metadata_bytes: int = 0

    def to_dict(self) -> dict:
        return {
            "total_conversations": self.total_conversations,
            "processed_conversations": self.processed_conversations,
            "skipped_conversations": self.skipped_conversations,
            "failed_conversations": self.failed_conversations,
            "total_duration_hours": round(self.total_duration_hours, 2),
            "total_main_duration_hours": round(self.total_main_duration_hours, 2),
            "total_user_duration_hours": round(self.total_user_duration_hours, 2),
            "processing_time_sec": round(self.processing_time_sec, 1),
            "speed_ratio": round(
                self.total_duration_hours * 3600 / max(self.processing_time_sec, 1), 2
            ),
            "total_audio_mb": round(self.total_audio_bytes / (1024 * 1024), 2),
            "total_metadata_mb": round(self.total_metadata_bytes / (1024 * 1024), 2),
        }

    def merge(self, other: "Phase1Stats") -> "Phase1Stats":
        """Merge two stats objects."""
        return Phase1Stats(
            total_conversations=self.total_conversations + other.total_conversations,
            processed_conversations=self.processed_conversations + other.processed_conversations,
            skipped_conversations=self.skipped_conversations + other.skipped_conversations,
            failed_conversations=self.failed_conversations + other.failed_conversations,
            total_duration_hours=self.total_duration_hours + other.total_duration_hours,
            total_main_duration_hours=self.total_main_duration_hours + other.total_main_duration_hours,
            total_user_duration_hours=self.total_user_duration_hours + other.total_user_duration_hours,
            processing_time_sec=max(self.processing_time_sec, other.processing_time_sec),
            total_audio_bytes=self.total_audio_bytes + other.total_audio_bytes,
            total_metadata_bytes=self.total_metadata_bytes + other.total_metadata_bytes,
        )


@dataclass
class CheckpointData:
    """Checkpoint data for resume support."""
    machine_id: int
    total_machines: int
    processed_ids: set[str] = field(default_factory=set)
    stats: Phase1Stats = field(default_factory=Phase1Stats)
    last_shard: int = -1
    timestamp: str = ""

    def to_dict(self) -> dict:
        return {
            "machine_id": self.machine_id,
            "total_machines": self.total_machines,
            "processed_ids": list(self.processed_ids),
            "stats": self.stats.to_dict(),
            "last_shard": self.last_shard,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CheckpointData":
        stats = Phase1Stats(
            total_conversations=data["stats"]["total_conversations"],
            processed_conversations=data["stats"]["processed_conversations"],
            skipped_conversations=data["stats"]["skipped_conversations"],
            failed_conversations=data["stats"]["failed_conversations"],
            total_duration_hours=data["stats"]["total_duration_hours"],
            processing_time_sec=data["stats"]["processing_time_sec"],
        )
        return cls(
            machine_id=data["machine_id"],
            total_machines=data["total_machines"],
            processed_ids=set(data.get("processed_ids", [])),
            stats=stats,
            last_shard=data.get("last_shard", -1),
            timestamp=data.get("timestamp", ""),
        )


class Phase1Orchestrator:
    """Orchestrates Phase 1 CPU-based processing with enhanced features.

    Features:
    - FLAC audio output (50-60% disk savings)
    - Rich metadata for Phase 2
    - Checkpoint/resume support
    - Flexible N-machine distribution

    Example usage:
        config = PipelineConfig.from_yaml("config.yaml")
        orchestrator = Phase1Orchestrator(config)

        for dataset in config.datasets:
            stats = orchestrator.process_dataset(dataset)
            print(f"Processed {stats.processed_conversations} conversations")
    """

    def __init__(self, config: PipelineConfig):
        """Initialize the orchestrator.

        Args:
            config: Pipeline configuration
        """
        self.config = config
        self.selector = SpeakerSelector(config.speaker_selection)
        self.converter = StereoConverter(config.audio)

    def process_conversation(
        self,
        conversation: Conversation,
        dataset_config: DatasetConfig,
        shard_idx: int,
    ) -> tuple[Optional[ConversationMetadata], Optional[str]]:
        """Process a single conversation.

        Args:
            conversation: Conversation to process
            dataset_config: Dataset configuration
            shard_idx: Source shard index

        Returns:
            Tuple of (metadata, error_message)
        """
        try:
            # Step 1: Assign speaker roles
            assignment = self.selector.assign_roles(conversation)
            if not assignment.is_valid:
                return None, assignment.skip_reason

            # Step 2: Get utterances by role
            utterances_by_role = self.selector.get_utterances_by_role(
                conversation, assignment
            )

            # Step 3: Load source audio
            if conversation.audio_path is None or not conversation.audio_path.exists():
                return None, "Audio file not found"

            source_audio, source_sr = self.converter.load_audio(conversation.audio_path)

            # Step 4: Convert to stereo
            stereo_audio = self.converter.convert(
                conversation=conversation,
                assignment=assignment,
                source_audio=source_audio,
                source_sr=source_sr,
                utterances_by_role=utterances_by_role,
            )

            if not stereo_audio.is_valid:
                return None, stereo_audio.error

            # Step 5: Save audio (FLAC format)
            audio_filename = f"{conversation.id}.{self.config.audio.extension}"
            audio_path = dataset_config.audio_output_dir / audio_filename

            if not self.converter.save(stereo_audio, audio_path):
                return None, "Failed to save audio"

            # Step 6: Create rich metadata
            main_utts = utterances_by_role[SpeakerRole.SPEAKER_MAIN]
            user_utts = utterances_by_role[SpeakerRole.SPEAKER_USER]

            # Include original speaker_id for each segment for extensibility
            # This allows tracking individual speakers within MAIN/USER channels
            main_segments = [
                {
                    "start": round(u.start, 3),
                    "end": round(u.end, 3),
                    "text": u.text,
                    "original_speaker_id": u.speaker_id,  # Preserve original speaker ID
                }
                for u in main_utts
            ]
            user_segments = [
                {
                    "start": round(u.start, 3),
                    "end": round(u.end, 3),
                    "text": u.text,
                    "original_speaker_id": u.speaker_id,  # Preserve original speaker ID
                }
                for u in user_utts
            ]

            # Step 7: Generate extended diarization metadata for future N-speaker support
            diarization_info = self.selector.get_diarization_metadata(
                conversation, assignment
            )

            metadata = ConversationMetadata(
                conversation_id=conversation.id,
                audio_path=f"audio/{audio_filename}",
                duration=stereo_audio.duration,
                sample_rate=stereo_audio.sample_rate,
                main_speaker_id=assignment.main_speaker.id,
                user_speaker_ids=[s.id for s in assignment.user_speakers],
                main_score=assignment.main_score,
                num_speakers=assignment.total_speakers,
                main_segments=main_segments,
                user_segments=user_segments,
                main_total_duration=stereo_audio.main_total_duration,
                user_total_duration=stereo_audio.user_total_duration,
                original_recording_id=conversation.recording_id,
                source_shard=shard_idx,
                processed_at=time.strftime("%Y-%m-%d %H:%M:%S"),
                machine_id=self.config.phase1.machine_id,
                diarization_info=diarization_info,
            )

            return metadata, None

        except Exception as e:
            logger.error(f"Error processing {conversation.id}: {e}")
            return None, str(e)

    def save_metadata(
        self,
        metadata: ConversationMetadata,
        dataset_config: DatasetConfig,
    ) -> int:
        """Save metadata to file.

        Returns:
            Size of metadata file in bytes
        """
        metadata_path = dataset_config.metadata_dir / f"{metadata.conversation_id}.json"
        metadata_path.parent.mkdir(parents=True, exist_ok=True)

        content = json.dumps(metadata.to_dict(), ensure_ascii=False, indent=2)
        metadata_path.write_text(content, encoding="utf-8")

        return len(content.encode("utf-8"))

    def save_checkpoint(
        self,
        checkpoint: CheckpointData,
        dataset_config: DatasetConfig,
    ) -> Path:
        """Save checkpoint for resume support."""
        checkpoint.timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        checkpoint_path = dataset_config.get_machine_checkpoint_path(checkpoint.machine_id)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        with open(checkpoint_path, "w") as f:
            json.dump(checkpoint.to_dict(), f, indent=2)

        return checkpoint_path

    def load_checkpoint(
        self,
        dataset_config: DatasetConfig,
    ) -> Optional[CheckpointData]:
        """Load checkpoint if exists."""
        checkpoint_path = dataset_config.get_machine_checkpoint_path(
            self.config.phase1.machine_id
        )

        if not checkpoint_path.exists():
            return None

        try:
            with open(checkpoint_path) as f:
                data = json.load(f)
            return CheckpointData.from_dict(data)
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}")
            return None

    def process_dataset(
        self,
        dataset_config: DatasetConfig,
        progress_callback: Optional[callable] = None,
    ) -> Phase1Stats:
        """Process a dataset with checkpoint support.

        Args:
            dataset_config: Dataset to process
            progress_callback: Optional callback(processed, total)

        Returns:
            Processing statistics
        """
        start_time = time.time()

        # Create output directories
        dataset_config.audio_output_dir.mkdir(parents=True, exist_ok=True)
        dataset_config.metadata_dir.mkdir(parents=True, exist_ok=True)

        # Load checkpoint if resuming
        checkpoint = None
        if self.config.phase1.resume_from_checkpoint:
            checkpoint = self.load_checkpoint(dataset_config)
            if checkpoint:
                logger.info(
                    f"Resuming from checkpoint: {checkpoint.stats.processed_conversations} "
                    f"already processed"
                )

        if checkpoint is None:
            checkpoint = CheckpointData(
                machine_id=self.config.phase1.machine_id,
                total_machines=self.config.phase1.total_machines,
            )

        stats = checkpoint.stats
        processed_ids = checkpoint.processed_ids

        # Initialize reader
        reader = LhotseSharReader(
            source_dir=dataset_config.source_path,
            machine_id=self.config.phase1.machine_id,
            total_machines=self.config.phase1.total_machines,
        )

        # Open manifest file for appending
        manifest_path = dataset_config.get_machine_manifest_path(
            self.config.phase1.machine_id
        )
        manifest_path.parent.mkdir(parents=True, exist_ok=True)

        with open(manifest_path, "a", encoding="utf-8") as manifest_file:
            # Process shards
            for shard_idx, conversations in reader.iter_shards():
                if shard_idx <= checkpoint.last_shard:
                    logger.debug(f"Skipping already processed shard {shard_idx}")
                    continue

                logger.info(
                    f"Processing shard {shard_idx} ({len(conversations)} conversations)"
                )

                for conv in conversations:
                    stats.total_conversations += 1

                    # Skip if already processed
                    if conv.id in processed_ids:
                        continue

                    # Process conversation
                    metadata, error = self.process_conversation(
                        conv, dataset_config, shard_idx
                    )

                    if metadata:
                        stats.processed_conversations += 1
                        stats.total_duration_hours += metadata.duration / 3600
                        stats.total_main_duration_hours += metadata.main_total_duration / 3600
                        stats.total_user_duration_hours += metadata.user_total_duration / 3600

                        # Save metadata
                        meta_bytes = self.save_metadata(metadata, dataset_config)
                        stats.total_metadata_bytes += meta_bytes

                        # Get audio file size
                        audio_path = dataset_config.audio_output_dir / f"{conv.id}.{self.config.audio.extension}"
                        if audio_path.exists():
                            stats.total_audio_bytes += audio_path.stat().st_size

                        # Write to manifest
                        manifest_entry = {
                            "conversation_id": metadata.conversation_id,
                            "audio_path": metadata.audio_path,
                            "metadata_path": f"metadata/{metadata.conversation_id}.json",
                            "duration": round(metadata.duration, 3),
                            "num_speakers": metadata.num_speakers,
                        }
                        manifest_file.write(json.dumps(manifest_entry, ensure_ascii=False) + "\n")
                        manifest_file.flush()

                        processed_ids.add(conv.id)

                    elif error:
                        if "Too short" in error or "Too long" in error or "Too few" in error:
                            stats.skipped_conversations += 1
                        else:
                            stats.failed_conversations += 1
                            logger.warning(f"Failed {conv.id}: {error}")

                    # Progress callback
                    if progress_callback and stats.total_conversations % self.config.phase1.log_interval == 0:
                        progress_callback(stats.processed_conversations, stats.total_conversations)

                    # Checkpoint
                    if stats.processed_conversations % self.config.phase1.checkpoint_interval == 0:
                        checkpoint.processed_ids = processed_ids
                        checkpoint.stats = stats
                        checkpoint.last_shard = shard_idx
                        self.save_checkpoint(checkpoint, dataset_config)

                # Update shard progress
                checkpoint.last_shard = shard_idx

        # Final checkpoint
        checkpoint.processed_ids = processed_ids
        checkpoint.stats = stats
        self.save_checkpoint(checkpoint, dataset_config)

        stats.processing_time_sec = time.time() - start_time

        # Save final stats
        self.save_stats(stats, dataset_config)

        return stats

    def save_stats(self, stats: Phase1Stats, dataset_config: DatasetConfig) -> None:
        """Save final statistics."""
        stats_path = dataset_config.stats_path
        stats_path.parent.mkdir(parents=True, exist_ok=True)

        with open(stats_path, "w") as f:
            json.dump(stats.to_dict(), f, indent=2)

        logger.info(f"Saved stats to {stats_path}")

    def process_dataset_parallel(
        self,
        dataset_config: DatasetConfig,
        num_workers: int = 8,
        progress_callback: Optional[callable] = None,
    ) -> Phase1Stats:
        """Process a dataset with parallel workers.

        Uses ThreadPoolExecutor for I/O-bound audio processing.

        Args:
            dataset_config: Dataset to process
            num_workers: Number of parallel workers
            progress_callback: Optional callback(processed, total)

        Returns:
            Processing statistics
        """
        start_time = time.time()

        # Create output directories
        dataset_config.audio_output_dir.mkdir(parents=True, exist_ok=True)
        dataset_config.metadata_dir.mkdir(parents=True, exist_ok=True)

        # Load checkpoint if resuming
        checkpoint = None
        if self.config.phase1.resume_from_checkpoint:
            checkpoint = self.load_checkpoint(dataset_config)
            if checkpoint:
                logger.info(
                    f"Resuming from checkpoint: {checkpoint.stats.processed_conversations} "
                    f"already processed"
                )

        if checkpoint is None:
            checkpoint = CheckpointData(
                machine_id=self.config.phase1.machine_id,
                total_machines=self.config.phase1.total_machines,
            )

        stats = checkpoint.stats
        processed_ids = checkpoint.processed_ids

        # Initialize reader
        reader = LhotseSharReader(
            source_dir=dataset_config.source_path,
            machine_id=self.config.phase1.machine_id,
            total_machines=self.config.phase1.total_machines,
        )

        # Open manifest file for appending
        manifest_path = dataset_config.get_machine_manifest_path(
            self.config.phase1.machine_id
        )
        manifest_path.parent.mkdir(parents=True, exist_ok=True)

        def process_single(conv_shard: tuple) -> tuple:
            """Process a single conversation (for parallel execution)."""
            conv, shard_idx = conv_shard
            if conv.id in processed_ids:
                return None, None, "already_processed"
            metadata, error = self.process_conversation(conv, dataset_config, shard_idx)
            return conv.id, metadata, error

        with open(manifest_path, "a", encoding="utf-8") as manifest_file:
            # Process shards
            for shard_idx, conversations in reader.iter_shards():
                if shard_idx <= checkpoint.last_shard:
                    logger.debug(f"Skipping already processed shard {shard_idx}")
                    continue

                logger.info(
                    f"Processing shard {shard_idx} ({len(conversations)} conversations) "
                    f"with {num_workers} workers"
                )

                # Prepare work items
                work_items = [(conv, shard_idx) for conv in conversations]

                # Process in parallel
                with ThreadPoolExecutor(max_workers=num_workers) as executor:
                    futures = {
                        executor.submit(process_single, item): item
                        for item in work_items
                    }

                    for future in as_completed(futures):
                        stats.total_conversations += 1
                        conv_id, metadata, error = future.result()

                        if error == "already_processed":
                            continue

                        if metadata:
                            stats.processed_conversations += 1
                            stats.total_duration_hours += metadata.duration / 3600
                            stats.total_main_duration_hours += metadata.main_total_duration / 3600
                            stats.total_user_duration_hours += metadata.user_total_duration / 3600

                            # Save metadata
                            meta_bytes = self.save_metadata(metadata, dataset_config)
                            stats.total_metadata_bytes += meta_bytes

                            # Get audio file size
                            audio_path = dataset_config.audio_output_dir / f"{conv_id}.{self.config.audio.extension}"
                            if audio_path.exists():
                                stats.total_audio_bytes += audio_path.stat().st_size

                            # Write to manifest
                            manifest_entry = {
                                "conversation_id": metadata.conversation_id,
                                "audio_path": metadata.audio_path,
                                "metadata_path": f"metadata/{metadata.conversation_id}.json",
                                "duration": round(metadata.duration, 3),
                                "num_speakers": metadata.num_speakers,
                            }
                            manifest_file.write(json.dumps(manifest_entry, ensure_ascii=False) + "\n")
                            manifest_file.flush()

                            processed_ids.add(conv_id)

                        elif error:
                            if "Too short" in error or "Too long" in error or "Too few" in error:
                                stats.skipped_conversations += 1
                            else:
                                stats.failed_conversations += 1
                                logger.warning(f"Failed {conv_id}: {error}")

                        # Progress callback
                        if progress_callback and stats.total_conversations % self.config.phase1.log_interval == 0:
                            progress_callback(stats.processed_conversations, stats.total_conversations)

                # Checkpoint after each shard
                checkpoint.processed_ids = processed_ids
                checkpoint.stats = stats
                checkpoint.last_shard = shard_idx
                self.save_checkpoint(checkpoint, dataset_config)
                logger.info(f"Shard {shard_idx} complete: {stats.processed_conversations} processed so far")

        # Final checkpoint
        checkpoint.processed_ids = processed_ids
        checkpoint.stats = stats
        self.save_checkpoint(checkpoint, dataset_config)

        stats.processing_time_sec = time.time() - start_time

        # Save final stats
        self.save_stats(stats, dataset_config)

        return stats

    @staticmethod
    def merge_manifests(dataset_config: DatasetConfig, total_machines: int) -> Path:
        """Merge manifests from all machines into final manifest.

        Args:
            dataset_config: Dataset configuration
            total_machines: Total number of machines

        Returns:
            Path to merged manifest
        """
        merged_path = dataset_config.manifest_path
        merged_path.parent.mkdir(parents=True, exist_ok=True)

        with open(merged_path, "w", encoding="utf-8") as out_file:
            for machine_id in range(total_machines):
                machine_manifest = dataset_config.get_machine_manifest_path(machine_id)
                if machine_manifest.exists():
                    with open(machine_manifest, "r", encoding="utf-8") as in_file:
                        for line in in_file:
                            out_file.write(line)
                    logger.info(f"Merged manifest from machine {machine_id}")

        logger.info(f"Created merged manifest: {merged_path}")
        return merged_path

    @staticmethod
    def merge_stats(dataset_config: DatasetConfig, total_machines: int) -> Phase1Stats:
        """Merge stats from all machines.

        Args:
            dataset_config: Dataset configuration
            total_machines: Total number of machines

        Returns:
            Merged statistics
        """
        merged_stats = Phase1Stats()

        for machine_id in range(total_machines):
            checkpoint_path = dataset_config.get_machine_checkpoint_path(machine_id)
            if checkpoint_path.exists():
                try:
                    with open(checkpoint_path) as f:
                        data = json.load(f)

                    stats_data = data.get("stats", {})
                    machine_stats = Phase1Stats(
                        total_conversations=stats_data.get("total_conversations", 0),
                        processed_conversations=stats_data.get("processed_conversations", 0),
                        skipped_conversations=stats_data.get("skipped_conversations", 0),
                        failed_conversations=stats_data.get("failed_conversations", 0),
                        total_duration_hours=stats_data.get("total_duration_hours", 0.0),
                        total_main_duration_hours=stats_data.get("total_main_duration_hours", 0.0),
                        total_user_duration_hours=stats_data.get("total_user_duration_hours", 0.0),
                        # Handle both formats: total_audio_mb (legacy) and total_audio_bytes
                        total_audio_bytes=int(stats_data.get("total_audio_mb", 0) * 1024 * 1024),
                        total_metadata_bytes=int(stats_data.get("total_metadata_mb", 0) * 1024 * 1024),
                    )
                    merged_stats = merged_stats.merge(machine_stats)
                    logger.debug(f"Merged stats from machine {machine_id}")
                except Exception as e:
                    logger.warning(f"Failed to load stats from machine {machine_id}: {e}")

        return merged_stats

    @staticmethod
    def validate_merged_manifest(
        dataset_config: DatasetConfig,
        total_machines: int,
    ) -> dict:
        """Validate merged manifest integrity.

        Checks for duplicates, missing fields, and data consistency.

        Args:
            dataset_config: Dataset configuration
            total_machines: Total number of machines

        Returns:
            Validation report dict
        """
        report = {
            "is_valid": True,
            "total_entries": 0,
            "duplicates": [],
            "missing_audio": [],
            "missing_metadata": [],
            "format_errors": [],
            "machines_coverage": {i: 0 for i in range(total_machines)},
        }

        seen_ids = set()
        manifest_path = dataset_config.manifest_path

        if not manifest_path.exists():
            report["is_valid"] = False
            report["error"] = f"Manifest not found: {manifest_path}"
            return report

        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        entry = json.loads(line.strip())
                        report["total_entries"] += 1

                        conv_id = entry.get("conversation_id")
                        if not conv_id:
                            report["format_errors"].append(f"Line {line_num}: missing conversation_id")
                            continue

                        # Check for duplicates
                        if conv_id in seen_ids:
                            report["duplicates"].append(conv_id)
                        seen_ids.add(conv_id)

                        # Check audio file exists
                        audio_path = dataset_config.output_path / dataset_config.split / entry.get("audio_path", "")
                        if not audio_path.exists():
                            report["missing_audio"].append(conv_id)

                        # Check metadata file exists
                        metadata_path = dataset_config.output_path / dataset_config.split / entry.get("metadata_path", "")
                        if not metadata_path.exists():
                            report["missing_metadata"].append(conv_id)

                    except json.JSONDecodeError as e:
                        report["format_errors"].append(f"Line {line_num}: invalid JSON - {e}")

        except Exception as e:
            report["is_valid"] = False
            report["error"] = str(e)
            return report

        # Determine validity
        if report["duplicates"]:
            report["is_valid"] = False
        if len(report["format_errors"]) > 0:
            report["is_valid"] = False

        # Log summary
        logger.info(f"Manifest validation: {report['total_entries']} entries")
        if report["duplicates"]:
            logger.warning(f"  Duplicates: {len(report['duplicates'])}")
        if report["missing_audio"]:
            logger.warning(f"  Missing audio: {len(report['missing_audio'])}")
        if report["missing_metadata"]:
            logger.warning(f"  Missing metadata: {len(report['missing_metadata'])}")

        return report

    @staticmethod
    def merge_manifests_with_validation(
        dataset_config: DatasetConfig,
        total_machines: int,
        deduplicate: bool = True,
    ) -> tuple[Path, dict]:
        """Merge manifests with duplicate detection and validation.

        Args:
            dataset_config: Dataset configuration
            total_machines: Total number of machines
            deduplicate: Remove duplicate entries

        Returns:
            Tuple of (merged manifest path, validation report)
        """
        merged_path = dataset_config.manifest_path
        merged_path.parent.mkdir(parents=True, exist_ok=True)

        seen_ids = set()
        duplicate_count = 0
        total_entries = 0
        entries_per_machine = {i: 0 for i in range(total_machines)}

        with open(merged_path, "w", encoding="utf-8") as out_file:
            for machine_id in range(total_machines):
                machine_manifest = dataset_config.get_machine_manifest_path(machine_id)
                if not machine_manifest.exists():
                    logger.warning(f"Machine {machine_id} manifest not found: {machine_manifest}")
                    continue

                with open(machine_manifest, "r", encoding="utf-8") as in_file:
                    for line in in_file:
                        try:
                            entry = json.loads(line.strip())
                            conv_id = entry.get("conversation_id", "")

                            if conv_id in seen_ids:
                                if deduplicate:
                                    duplicate_count += 1
                                    continue
                            seen_ids.add(conv_id)

                            out_file.write(line)
                            total_entries += 1
                            entries_per_machine[machine_id] += 1

                        except json.JSONDecodeError:
                            logger.warning(f"Invalid JSON in machine {machine_id} manifest")
                            continue

                logger.info(f"Merged {entries_per_machine[machine_id]} entries from machine {machine_id}")

        report = {
            "total_entries": total_entries,
            "duplicate_count": duplicate_count,
            "entries_per_machine": entries_per_machine,
            "merged_path": str(merged_path),
        }

        logger.info(f"Created merged manifest: {merged_path}")
        logger.info(f"Total entries: {total_entries}, Duplicates removed: {duplicate_count}")

        return merged_path, report
