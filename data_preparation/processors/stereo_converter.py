# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Stereo audio conversion for Moshi format with FLAC support.

Converts mono audio with speaker timestamps to stereo format:
- Left channel (0): SPEAKER_MAIN
- Right channel (1): SPEAKER_USER

Audio sample rate is preserved from source by default (16kHz for Korean data).
Resampling only occurs if target sample rate differs from source.
Supports both WAV and FLAC output formats.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import logging

import numpy as np
import soundfile as sf

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False

from ..readers.lhotse_shar import Conversation, Utterance
from ..config import AudioConfig
from .speaker_selector import SpeakerRole, SpeakerAssignment

logger = logging.getLogger(__name__)


@dataclass
class StereoAudio:
    """Result of stereo conversion."""
    conversation_id: str
    audio: np.ndarray  # Shape: (2, num_samples)
    sample_rate: int
    duration: float
    main_segments: list[tuple[float, float]]  # (start, end) pairs
    user_segments: list[tuple[float, float]]
    is_valid: bool
    error: Optional[str] = None

    # Additional metadata for Phase 2
    main_total_duration: float = 0.0
    user_total_duration: float = 0.0
    num_main_segments: int = 0
    num_user_segments: int = 0


class StereoConverter:
    """Converts mono audio to stereo based on speaker timestamps.

    Places SPEAKER_MAIN audio in the left channel and SPEAKER_USER
    audio in the right channel, resampled to the target sample rate.

    Supports FLAC output for 50-60% disk space savings.

    Example usage:
        converter = StereoConverter(config)

        # Load source audio
        audio, sr = converter.load_audio(audio_path)

        # Convert to stereo
        result = converter.convert(
            conversation=conv,
            assignment=assignment,
            source_audio=audio,
            source_sr=sr
        )

        if result.is_valid:
            converter.save(result, output_path)  # Saves as FLAC by default
    """

    def __init__(self, config: Optional[AudioConfig] = None):
        """Initialize the converter.

        Args:
            config: Audio configuration
        """
        self.config = config or AudioConfig()

        if not HAS_LIBROSA:
            logger.warning(
                "librosa not available, resampling will use scipy (slower)"
            )

    def load_audio(self, path: Path) -> tuple[np.ndarray, int]:
        """Load audio from file.

        Args:
            path: Path to audio file (supports WAV, FLAC, MP3, etc.)

        Returns:
            Tuple of (audio array, sample rate)
        """
        audio, sr = sf.read(path, dtype="float32")

        # Ensure mono
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        return audio, sr

    def resample(
        self,
        audio: np.ndarray,
        source_sr: int,
        target_sr: int,
    ) -> np.ndarray:
        """Resample audio to target sample rate.

        Args:
            audio: Audio array (mono)
            source_sr: Source sample rate
            target_sr: Target sample rate

        Returns:
            Resampled audio array
        """
        if source_sr == target_sr:
            return audio

        if HAS_LIBROSA:
            return librosa.resample(
                audio,
                orig_sr=source_sr,
                target_sr=target_sr,
            )
        else:
            # Fallback to scipy
            from scipy import signal
            num_samples = int(len(audio) * target_sr / source_sr)
            return signal.resample(audio, num_samples).astype(np.float32)

    def extract_segments(
        self,
        audio: np.ndarray,
        sample_rate: int,
        utterances: list[Utterance],
    ) -> np.ndarray:
        """Extract and combine audio segments for utterances.

        Creates a single-channel audio containing only the specified
        segments, with silence elsewhere.

        Args:
            audio: Source audio array (mono)
            sample_rate: Sample rate of audio
            utterances: List of utterances to extract

        Returns:
            Audio array with only specified segments
        """
        output = np.zeros_like(audio)

        for utt in utterances:
            start_sample = int(utt.start * sample_rate)
            end_sample = int(utt.end * sample_rate)

            # Clip to valid range
            start_sample = max(0, start_sample)
            end_sample = min(len(audio), end_sample)

            if start_sample < end_sample:
                output[start_sample:end_sample] = audio[start_sample:end_sample]

        return output

    def convert(
        self,
        conversation: Conversation,
        assignment: SpeakerAssignment,
        source_audio: np.ndarray,
        source_sr: int,
        utterances_by_role: Optional[dict[SpeakerRole, list]] = None,
    ) -> StereoAudio:
        """Convert mono audio to stereo format.

        Args:
            conversation: Conversation metadata
            assignment: Speaker role assignment
            source_audio: Source audio array (mono)
            source_sr: Source sample rate
            utterances_by_role: Pre-computed utterances grouped by role

        Returns:
            StereoAudio result
        """
        try:
            if not assignment.is_valid:
                return StereoAudio(
                    conversation_id=conversation.id,
                    audio=np.zeros((2, 0)),
                    sample_rate=self.config.sample_rate,
                    duration=0.0,
                    main_segments=[],
                    user_segments=[],
                    is_valid=False,
                    error=f"Invalid assignment: {assignment.skip_reason}",
                )

            # Resample to target sample rate
            if source_sr != self.config.sample_rate:
                audio = self.resample(
                    source_audio, source_sr, self.config.sample_rate
                )
            else:
                audio = source_audio.copy()

            # Get utterances by role
            if utterances_by_role is None:
                from .speaker_selector import SpeakerSelector
                selector = SpeakerSelector()
                utterances_by_role = selector.get_utterances_by_role(
                    conversation, assignment
                )

            # Extract segments for each channel
            main_utts = utterances_by_role[SpeakerRole.SPEAKER_MAIN]
            user_utts = utterances_by_role[SpeakerRole.SPEAKER_USER]

            left_channel = self.extract_segments(
                audio, self.config.sample_rate, main_utts
            )
            right_channel = self.extract_segments(
                audio, self.config.sample_rate, user_utts
            )

            # Combine into stereo
            stereo_audio = np.stack([left_channel, right_channel], axis=0)

            # Collect segment times
            main_segments = [(u.start, u.end) for u in main_utts]
            user_segments = [(u.start, u.end) for u in user_utts]

            # Calculate durations
            main_total_duration = sum(u.end - u.start for u in main_utts)
            user_total_duration = sum(u.end - u.start for u in user_utts)

            duration = len(audio) / self.config.sample_rate

            return StereoAudio(
                conversation_id=conversation.id,
                audio=stereo_audio,
                sample_rate=self.config.sample_rate,
                duration=duration,
                main_segments=main_segments,
                user_segments=user_segments,
                is_valid=True,
                main_total_duration=main_total_duration,
                user_total_duration=user_total_duration,
                num_main_segments=len(main_segments),
                num_user_segments=len(user_segments),
            )

        except Exception as e:
            logger.error(f"Error converting {conversation.id}: {e}")
            return StereoAudio(
                conversation_id=conversation.id,
                audio=np.zeros((2, 0)),
                sample_rate=self.config.sample_rate,
                duration=0.0,
                main_segments=[],
                user_segments=[],
                is_valid=False,
                error=str(e),
            )

    def save(
        self,
        stereo_audio: StereoAudio,
        output_path: Path,
        format_override: Optional[str] = None,
    ) -> bool:
        """Save stereo audio to file.

        Args:
            stereo_audio: StereoAudio object to save
            output_path: Output file path
            format_override: Override format from config ("flac" or "wav")

        Returns:
            True if successful
        """
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)

            audio_format = format_override or self.config.format

            # Ensure correct extension
            if not output_path.suffix.lower() == f".{audio_format}":
                output_path = output_path.with_suffix(f".{audio_format}")

            # soundfile expects (samples, channels)
            audio_data = stereo_audio.audio.T

            if audio_format == "flac":
                # FLAC: lossless compression, 50-60% smaller than WAV
                sf.write(
                    output_path,
                    audio_data,
                    stereo_audio.sample_rate,
                    format="FLAC",
                    subtype="PCM_16",
                )
            else:
                # WAV: uncompressed
                sf.write(
                    output_path,
                    audio_data,
                    stereo_audio.sample_rate,
                    subtype="PCM_16",
                )

            return True

        except Exception as e:
            logger.error(f"Error saving {output_path}: {e}")
            return False

    def get_output_path(
        self,
        output_dir: Path,
        conversation_id: str,
    ) -> Path:
        """Get output path with correct extension.

        Args:
            output_dir: Output directory
            conversation_id: Conversation ID

        Returns:
            Full output path with correct extension
        """
        return output_dir / f"{conversation_id}.{self.config.extension}"
