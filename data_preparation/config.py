# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Configuration for Korean Moshi dataset preparation pipeline.

Enhanced for Phase 1 focus with FLAC support and flexible distributed processing.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Literal
import yaml


@dataclass
class DatasetConfig:
    """Configuration for a single dataset source."""
    name: str
    source_path: Path
    output_path: Path
    split: str = "train"  # train or valid

    def get_audio_extension(self, audio_format: str = "flac") -> str:
        """Get file extension for audio format."""
        return audio_format.lower()

    @property
    def audio_output_dir(self) -> Path:
        return self.output_path / self.split / "audio"

    @property
    def metadata_dir(self) -> Path:
        """Rich metadata for Phase 2."""
        return self.output_path / self.split / "metadata"

    @property
    def alignment_speaker01_dir(self) -> Path:
        """Word alignments for SPEAKER_MAIN."""
        return self.output_path / self.split / "alignment_speaker01"

    @property
    def alignment_speaker02_dir(self) -> Path:
        """Word alignments for SPEAKER_USER."""
        return self.output_path / self.split / "alignment_speaker02"

    @property
    def segment_alignment_dir(self) -> Path:
        """Segment-level alignments from Phase 1."""
        return self.output_path / self.split / "segment_alignment"

    @property
    def alignments_dir(self) -> Path:
        """Moshi format combined alignments."""
        return self.output_path / self.split / "alignments"

    @property
    def extended_alignments_dir(self) -> Path:
        """Extended alignments with full speaker metadata."""
        return self.output_path / self.split / "alignments_extended"

    @property
    def manifest_path(self) -> Path:
        return self.output_path / self.split / "manifest.jsonl"

    @property
    def phase1_manifest_path(self) -> Path:
        """Phase 1 intermediate manifest for Phase 2 input."""
        return self.output_path / self.split / "manifest_phase1.jsonl"

    @property
    def stats_path(self) -> Path:
        return self.output_path / self.split / "stats.json"

    @property
    def checkpoint_dir(self) -> Path:
        return self.output_path / self.split / ".checkpoints"

    def get_machine_checkpoint_path(self, machine_id: int) -> Path:
        """Get checkpoint path for specific machine."""
        return self.checkpoint_dir / f"machine_{machine_id:03d}.json"

    def get_machine_manifest_path(self, machine_id: int) -> Path:
        """Get manifest path for specific machine (before merge)."""
        return self.checkpoint_dir / f"manifest_{machine_id:03d}.jsonl"


@dataclass
class SpeakerSelectionConfig:
    """Configuration for SPEAKER_MAIN selection algorithm."""
    # Hybrid score weights
    duration_weight: float = 0.6
    turn_count_weight: float = 0.4

    # Minimum requirements
    min_duration_sec: float = 10.0
    max_duration_sec: float = 7200.0  # 2 hours
    min_turns: int = 2

    # Multi-speaker handling
    max_speakers: int = 2  # SPEAKER_MAIN + SPEAKER_USER
    merge_minor_speakers: bool = True  # Merge 3rd+ speakers into SPEAKER_USER


@dataclass
class AudioConfig:
    """Audio processing configuration with FLAC support."""
    sample_rate: int = 16000  # Match source data sample rate (avoid upsampling)
    channels: int = 2  # Stereo output
    bit_depth: int = 16

    # Format: "flac" for disk savings (~50-60% smaller), "wav" for compatibility
    format: Literal["flac", "wav"] = "flac"

    # FLAC compression level (0-8, higher = smaller file but slower)
    flac_compression: int = 5

    # Channel assignment
    main_channel: int = 0  # Left channel for SPEAKER_MAIN
    user_channel: int = 1  # Right channel for SPEAKER_USER

    @property
    def extension(self) -> str:
        return self.format.lower()


@dataclass
class Phase1Config:
    """Phase 1 (CPU) processing configuration with flexible distribution."""
    # Worker configuration
    num_workers: int = 8  # Per machine parallel workers

    # Distributed processing - FLEXIBLE machine count
    machine_id: int = 0  # Current machine (0-indexed)
    total_machines: int = 1  # Total machines (set this to your desired count)

    # Shard assignment mode
    shard_assignment: Literal["round_robin", "range"] = "round_robin"

    # Progress tracking
    checkpoint_interval: int = 50  # Save checkpoint every N conversations
    log_interval: int = 10

    # Resume support
    resume_from_checkpoint: bool = True

    # Error handling
    skip_on_error: bool = True
    max_retries: int = 3

    # Memory optimization
    clear_audio_cache: bool = True  # Clear extracted audio after processing
    batch_size: int = 100  # Conversations per batch for memory management


@dataclass
class MetadataConfig:
    """Rich metadata configuration for Phase 2 compatibility."""
    # What to preserve
    save_speaker_info: bool = True
    save_segment_timestamps: bool = True
    save_utterance_texts: bool = True
    save_original_paths: bool = True
    save_processing_info: bool = True

    # Metadata format
    format: Literal["json", "jsonl"] = "json"
    compress_metadata: bool = False  # gzip compression


@dataclass
class Phase2Config:
    """Phase 2 (GPU) processing configuration."""
    # GPU configuration
    gpu_ids: list[int] = field(default_factory=lambda: [0])
    num_gpus: int = 1
    gpu_id: int = 0  # Single GPU mode - which GPU to use

    # Distributed processing (8-machine mode)
    machine_id: int = 0  # Current machine (0-indexed)
    total_machines: int = 1  # Total machines for distributed processing

    # Aligner type: "nfa" (NeMo Forced Aligner) or "whisper" (whisper-timestamped)
    aligner_type: Literal["nfa", "whisper"] = "nfa"

    # Processing
    batch_size: int = 32  # Batch size for NFA processing
    num_workers: int = 4  # DataLoader workers

    # Progress tracking
    log_interval: int = 100
    checkpoint_interval: int = 500

    # Task sorting (longest first for better GPU utilization)
    sort_by_duration: bool = True

    # Resume support
    resume_from_checkpoint: bool = True


@dataclass
class NFAConfig:
    """NeMo Forced Aligner (NFA) configuration.

    NFA uses CTC-based acoustic models for forced alignment.
    For Korean, use a pre-trained Korean CTC model.
    """
    # Acoustic model - path to pre-trained Korean CTC model (.nemo file)
    # Examples:
    #   - Local path: "/path/to/korean_ctc_model.nemo"
    #   - HuggingFace: "nvidia/stt_ko_conformer_ctc_large"
    acoustic_model: str = "SungBeom/stt_kr_conformer_ctc_medium"

    # Language
    language: str = "ko"

    # Alignment parameters
    window_size: int = 8000  # Window size for alignment (samples at 16kHz = 0.5s)
    shift_size: int = 4000   # Shift size between windows

    # Output format
    output_format: Literal["json", "ctm", "textgrid"] = "json"

    # Quality filtering
    min_confidence: float = 0.5  # Minimum confidence score for word alignments
    max_word_duration: float = 3.0  # Maximum duration for a single word (seconds)

    # Processing options
    use_gpu: bool = True
    batch_size: int = 32
    compute_type: str = "float16"  # float16 or float32

    # Channel processing
    process_both_channels: bool = True  # Process both L (main) and R (user) channels


@dataclass
class WhisperAlignmentConfig:
    """Whisper-timestamped alignment configuration (alternative to NFA)."""
    # Model selection
    whisper_model: str = "large-v3"

    # Language
    language: str = "ko"

    # Accuracy settings
    accurate: bool = True  # Use more accurate but slower alignment
    vad: bool = True  # Use voice activity detection

    # Quality thresholds
    min_word_confidence: float = 0.5

    # Processing
    compute_type: str = "float16"


@dataclass
class AlignmentConfig:
    """Generic alignment configuration used by whisper_timestamped aligner.

    This provides a unified interface for the WhisperTimestampedAligner class.
    """
    # Model selection
    whisper_model: str = "large-v3"

    # Device
    device: str = "cuda"  # "cuda" or "cpu"

    # Language
    language: str = "ko"

    # VAD (Voice Activity Detection)
    use_vad: bool = True

    # Quality thresholds
    min_word_confidence: float = 0.5

    # Processing
    compute_type: str = "float16"


@dataclass
class PipelineConfig:
    """Complete pipeline configuration for Phase 1 and Phase 2."""
    # Dataset configurations
    datasets: list[DatasetConfig] = field(default_factory=list)

    # Phase 1 (CPU) configs
    speaker_selection: SpeakerSelectionConfig = field(
        default_factory=SpeakerSelectionConfig
    )
    audio: AudioConfig = field(default_factory=AudioConfig)
    phase1: Phase1Config = field(default_factory=Phase1Config)
    metadata: MetadataConfig = field(default_factory=MetadataConfig)

    # Phase 2 (GPU) configs
    phase2: Phase2Config = field(default_factory=Phase2Config)
    nfa: NFAConfig = field(default_factory=NFAConfig)
    whisper_alignment: WhisperAlignmentConfig = field(default_factory=WhisperAlignmentConfig)

    # Output paths
    output_base: Path = Path("/path/to/data")

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> "PipelineConfig":
        """Load configuration from YAML file."""
        with open(yaml_path) as f:
            data = yaml.safe_load(f)

        datasets = []
        for ds in data.get("datasets", []):
            datasets.append(DatasetConfig(
                name=ds["name"],
                source_path=Path(ds["source_path"]),
                output_path=Path(ds["output_path"]),
                split=ds.get("split", "train"),
            ))

        config = cls(datasets=datasets)

        # Override component configs if specified
        if "speaker_selection" in data:
            for k, v in data["speaker_selection"].items():
                if hasattr(config.speaker_selection, k):
                    setattr(config.speaker_selection, k, v)

        if "audio" in data:
            for k, v in data["audio"].items():
                if hasattr(config.audio, k):
                    setattr(config.audio, k, v)

        if "phase1" in data:
            for k, v in data["phase1"].items():
                if hasattr(config.phase1, k):
                    setattr(config.phase1, k, v)

        if "metadata" in data:
            for k, v in data["metadata"].items():
                if hasattr(config.metadata, k):
                    setattr(config.metadata, k, v)

        if "output_base" in data:
            config.output_base = Path(data["output_base"])

        return config

    def to_yaml(self, yaml_path: Path) -> None:
        """Save configuration to YAML file."""
        data = {
            "output_base": str(self.output_base),
            "datasets": [
                {
                    "name": ds.name,
                    "source_path": str(ds.source_path),
                    "output_path": str(ds.output_path),
                    "split": ds.split,
                }
                for ds in self.datasets
            ],
            "speaker_selection": {
                "duration_weight": self.speaker_selection.duration_weight,
                "turn_count_weight": self.speaker_selection.turn_count_weight,
                "min_duration_sec": self.speaker_selection.min_duration_sec,
                "max_duration_sec": self.speaker_selection.max_duration_sec,
                "min_turns": self.speaker_selection.min_turns,
            },
            "audio": {
                "sample_rate": self.audio.sample_rate,
                "channels": self.audio.channels,
                "format": self.audio.format,
                "flac_compression": self.audio.flac_compression,
            },
            "phase1": {
                "num_workers": self.phase1.num_workers,
                "machine_id": self.phase1.machine_id,
                "total_machines": self.phase1.total_machines,
                "checkpoint_interval": self.phase1.checkpoint_interval,
                "resume_from_checkpoint": self.phase1.resume_from_checkpoint,
            },
            "metadata": {
                "save_speaker_info": self.metadata.save_speaker_info,
                "save_segment_timestamps": self.metadata.save_segment_timestamps,
                "save_utterance_texts": self.metadata.save_utterance_texts,
            },
        }

        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        with open(yaml_path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)

    def create_machine_config(self, machine_id: int, total_machines: int) -> "PipelineConfig":
        """Create a config for a specific machine in distributed setup."""
        import copy
        new_config = copy.deepcopy(self)
        new_config.phase1.machine_id = machine_id
        new_config.phase1.total_machines = total_machines
        return new_config


# Default dataset configurations
DEFAULT_DATASETS = [
    DatasetConfig(
        name="aihub-broadcast-key463-839g-train",
        source_path=Path("/path/to/data"),
        output_path=Path("/path/to/data"),
        split="train",
    ),
    DatasetConfig(
        name="aihub-broadcast-key463-839g-valid",
        source_path=Path("/path/to/data"),
        output_path=Path("/path/to/data"),
        split="valid",
    ),
    DatasetConfig(
        name="aihub-broadcast-key71314-559g-train",
        source_path=Path("/path/to/data"),
        output_path=Path("/path/to/data"),
        split="train",
    ),
    DatasetConfig(
        name="aihub-broadcast-key71314-559g-valid",
        source_path=Path("/path/to/data"),
        output_path=Path("/path/to/data"),
        split="valid",
    ),
]


def get_default_config() -> PipelineConfig:
    """Get default pipeline configuration."""
    return PipelineConfig(datasets=DEFAULT_DATASETS)
