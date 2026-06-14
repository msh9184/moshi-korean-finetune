# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Word-level alignment using whisper-timestamped.

whisper-timestamped provides word-level timestamps without requiring
language-specific alignment models, making it ideal for Korean.

Installation:
    pip install whisper-timestamped

Reference:
    https://github.com/linto-ai/whisper-timestamped
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import json
import logging

import numpy as np

try:
    import whisper_timestamped as whisper
    HAS_WHISPER_TIMESTAMPED = True
except ImportError:
    HAS_WHISPER_TIMESTAMPED = False

from ..config import AlignmentConfig

logger = logging.getLogger(__name__)


@dataclass
class Word:
    """A single word with timestamp."""
    text: str
    start: float
    end: float
    confidence: float = 1.0

    def to_list(self, speaker: str) -> list:
        """Convert to Moshi alignment format [word, [start, end], speaker]."""
        return [self.text, [round(self.start, 3), round(self.end, 3)], speaker]


@dataclass
class WordAlignment:
    """Word-level alignment for a single speaker/channel."""
    speaker: str  # SPEAKER_MAIN or SPEAKER_USER
    words: list[Word] = field(default_factory=list)
    total_duration: float = 0.0
    confidence: float = 0.0

    def to_moshi_format(self) -> dict:
        """Convert to Moshi alignment JSON format."""
        return {
            "alignments": [word.to_list(self.speaker) for word in self.words]
        }

    def save(self, path: Path) -> bool:
        """Save alignment to JSON file in Moshi format."""
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.to_moshi_format(), f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            logger.error(f"Error saving alignment to {path}: {e}")
            return False

    @classmethod
    def load(cls, path: Path) -> "WordAlignment":
        """Load alignment from Moshi format JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        words = []
        speaker = ""

        for item in data.get("alignments", []):
            text = item[0]
            start, end = item[1]
            spk = item[2] if len(item) > 2 else ""

            if not speaker:
                speaker = spk

            words.append(Word(text=text, start=start, end=end))

        alignment = cls(speaker=speaker, words=words)
        if words:
            alignment.total_duration = words[-1].end - words[0].start
            alignment.confidence = sum(w.confidence for w in words) / len(words)

        return alignment


@dataclass
class AlignmentResult:
    """Result of word-level alignment for both channels."""
    conversation_id: str
    main_alignment: Optional[WordAlignment] = None
    user_alignment: Optional[WordAlignment] = None
    is_valid: bool = True
    error: Optional[str] = None
    quality_score: float = 0.0


class WhisperTimestampedAligner:
    """Word-level alignment using whisper-timestamped.

    Processes stereo audio and generates word-level alignments
    for both SPEAKER_MAIN (left channel) and SPEAKER_USER (right channel).

    Example usage:
        aligner = WhisperTimestampedAligner(config)

        # Process stereo audio file
        result = aligner.align_stereo(
            audio_path=Path("audio/conv_001.wav"),
            conversation_id="conv_001"
        )

        if result.is_valid:
            result.main_alignment.save(Path("alignment_speaker01/conv_001.json"))
            result.user_alignment.save(Path("alignment_speaker02/conv_001.json"))
    """

    def __init__(self, config: Optional[AlignmentConfig] = None):
        """Initialize the aligner.

        Args:
            config: Alignment configuration
        """
        self.config = config or AlignmentConfig()
        self.model = None

        if not HAS_WHISPER_TIMESTAMPED:
            raise ImportError(
                "whisper-timestamped is required. Install with: "
                "pip install whisper-timestamped"
            )

    def load_model(self) -> None:
        """Load Whisper model."""
        if self.model is not None:
            return

        logger.info(f"Loading Whisper model: {self.config.whisper_model}")
        self.model = whisper.load_model(
            self.config.whisper_model,
            device=self.config.device,
        )
        logger.info("Model loaded successfully")

    def _transcribe_channel(
        self,
        audio: np.ndarray,
        sample_rate: int,
        speaker: str,
    ) -> WordAlignment:
        """Transcribe a single audio channel.

        Args:
            audio: Audio array (mono)
            sample_rate: Sample rate
            speaker: Speaker label

        Returns:
            WordAlignment for this channel
        """
        # Resample to 16kHz if needed (Whisper expects 16kHz)
        if sample_rate != 16000:
            from scipy import signal
            num_samples = int(len(audio) * 16000 / sample_rate)
            audio = signal.resample(audio, num_samples)

        # Check if channel has any audio
        if np.abs(audio).max() < 1e-6:
            logger.debug(f"Empty audio for {speaker}")
            return WordAlignment(speaker=speaker, words=[])

        # Transcribe with word timestamps
        result = whisper.transcribe(
            self.model,
            audio,
            language=self.config.language,
            vad=self.config.use_vad,
        )

        # Extract words
        words = []
        total_confidence = 0.0

        for segment in result.get("segments", []):
            for word_data in segment.get("words", []):
                confidence = word_data.get("confidence", 1.0)

                # Filter by confidence
                if confidence < self.config.min_word_confidence:
                    continue

                word = Word(
                    text=word_data["text"].strip(),
                    start=word_data["start"],
                    end=word_data["end"],
                    confidence=confidence,
                )
                words.append(word)
                total_confidence += confidence

        alignment = WordAlignment(
            speaker=speaker,
            words=words,
            total_duration=words[-1].end - words[0].start if words else 0.0,
            confidence=total_confidence / len(words) if words else 0.0,
        )

        return alignment

    def align_stereo(
        self,
        audio_path: Path,
        conversation_id: str,
    ) -> AlignmentResult:
        """Align stereo audio file.

        Processes both channels:
        - Left channel (0): SPEAKER_MAIN
        - Right channel (1): SPEAKER_USER

        Args:
            audio_path: Path to stereo WAV file
            conversation_id: Conversation identifier

        Returns:
            AlignmentResult with alignments for both channels
        """
        try:
            import soundfile as sf

            # Load model if needed
            self.load_model()

            # Load stereo audio
            audio, sample_rate = sf.read(audio_path, dtype="float32")

            if audio.ndim == 1:
                # Mono file - process as SPEAKER_MAIN only
                logger.warning(f"Mono audio for {conversation_id}, processing as SPEAKER_MAIN only")
                main_alignment = self._transcribe_channel(
                    audio, sample_rate, "SPEAKER_MAIN"
                )
                return AlignmentResult(
                    conversation_id=conversation_id,
                    main_alignment=main_alignment,
                    user_alignment=WordAlignment(speaker="SPEAKER_USER", words=[]),
                    is_valid=True,
                    quality_score=main_alignment.confidence,
                )

            # Process left channel (SPEAKER_MAIN)
            logger.debug(f"Processing SPEAKER_MAIN for {conversation_id}")
            main_alignment = self._transcribe_channel(
                audio[:, 0], sample_rate, "SPEAKER_MAIN"
            )

            # Process right channel (SPEAKER_USER)
            logger.debug(f"Processing SPEAKER_USER for {conversation_id}")
            user_alignment = self._transcribe_channel(
                audio[:, 1], sample_rate, "SPEAKER_USER"
            )

            # Calculate quality score
            quality_score = self._calculate_quality(
                main_alignment, user_alignment
            )

            return AlignmentResult(
                conversation_id=conversation_id,
                main_alignment=main_alignment,
                user_alignment=user_alignment,
                is_valid=True,
                quality_score=quality_score,
            )

        except Exception as e:
            logger.error(f"Error aligning {conversation_id}: {e}")
            return AlignmentResult(
                conversation_id=conversation_id,
                is_valid=False,
                error=str(e),
            )

    def align_from_arrays(
        self,
        left_channel: np.ndarray,
        right_channel: np.ndarray,
        sample_rate: int,
        conversation_id: str,
    ) -> AlignmentResult:
        """Align from audio arrays directly.

        Args:
            left_channel: SPEAKER_MAIN audio
            right_channel: SPEAKER_USER audio
            sample_rate: Sample rate
            conversation_id: Conversation identifier

        Returns:
            AlignmentResult
        """
        try:
            self.load_model()

            main_alignment = self._transcribe_channel(
                left_channel, sample_rate, "SPEAKER_MAIN"
            )
            user_alignment = self._transcribe_channel(
                right_channel, sample_rate, "SPEAKER_USER"
            )

            quality_score = self._calculate_quality(
                main_alignment, user_alignment
            )

            return AlignmentResult(
                conversation_id=conversation_id,
                main_alignment=main_alignment,
                user_alignment=user_alignment,
                is_valid=True,
                quality_score=quality_score,
            )

        except Exception as e:
            logger.error(f"Error aligning {conversation_id}: {e}")
            return AlignmentResult(
                conversation_id=conversation_id,
                is_valid=False,
                error=str(e),
            )

    def _calculate_quality(
        self,
        main_alignment: WordAlignment,
        user_alignment: WordAlignment,
    ) -> float:
        """Calculate alignment quality score.

        Args:
            main_alignment: SPEAKER_MAIN alignment
            user_alignment: SPEAKER_USER alignment

        Returns:
            Quality score between 0 and 1
        """
        scores = []

        # Word count score
        total_words = len(main_alignment.words) + len(user_alignment.words)
        if total_words > 0:
            scores.append(min(total_words / 100, 1.0))  # Expect ~100 words

        # Confidence score
        if main_alignment.words:
            scores.append(main_alignment.confidence)
        if user_alignment.words:
            scores.append(user_alignment.confidence)

        # Coverage score (both channels should have content)
        if main_alignment.words and user_alignment.words:
            scores.append(1.0)
        elif main_alignment.words or user_alignment.words:
            scores.append(0.5)
        else:
            scores.append(0.0)

        return sum(scores) / len(scores) if scores else 0.0

    def validate_alignment(
        self,
        result: AlignmentResult,
        segment_alignment_path: Optional[Path] = None,
    ) -> dict:
        """Validate alignment result against segment-level alignment.

        Args:
            result: Alignment result to validate
            segment_alignment_path: Optional path to segment alignment JSON

        Returns:
            Validation metrics dictionary
        """
        metrics = {
            "is_valid": result.is_valid,
            "quality_score": result.quality_score,
            "main_word_count": len(result.main_alignment.words) if result.main_alignment else 0,
            "user_word_count": len(result.user_alignment.words) if result.user_alignment else 0,
            "main_confidence": result.main_alignment.confidence if result.main_alignment else 0,
            "user_confidence": result.user_alignment.confidence if result.user_alignment else 0,
        }

        # Validate against segment alignment if provided
        if segment_alignment_path and segment_alignment_path.exists():
            from ..processors.segment_aligner import SegmentAlignment
            seg_align = SegmentAlignment.load(segment_alignment_path)

            # Check text coverage
            seg_text_main = " ".join(
                s.text for s in seg_align.segments if s.speaker == "SPEAKER_MAIN"
            )
            word_text_main = " ".join(
                w.text for w in (result.main_alignment.words if result.main_alignment else [])
            )

            # Simple coverage metric (character overlap)
            if seg_text_main:
                common_chars = set(seg_text_main) & set(word_text_main)
                metrics["main_coverage"] = len(common_chars) / len(set(seg_text_main))
            else:
                metrics["main_coverage"] = 1.0

        return metrics
