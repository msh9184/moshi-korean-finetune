# Copyright (c) 2026 Sunghwan Mun. Licensed under the Apache License, Version 2.0.
"""K-Moshi Reference Voice Preparation.

This module provides tools to validate and prepare reference audio
for TTS voice cloning. The reference audio is used by OpenAudio S1 Mini
to clone a consistent voice for K-Moshi.

Requirements:
- Reference audio: 10-30 seconds of clean speech
- Transcript: Exact text of the reference audio
- Sample rate: 24kHz preferred
- SNR: > 30dB recommended

Usage:
    # Validate existing reference
    result = validate_reference_audio("raw_reference.wav")
    if not result["valid"]:
        print(result["issues"])

    # Prepare reference for voice cloning
    prepare_reference_audio("raw.wav", "processed.wav")
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
import logging

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ReferenceAudioConfig:
    """Configuration for reference audio validation.

    Attributes:
        min_duration: Minimum duration in seconds.
        max_duration: Maximum duration in seconds.
        target_sr: Target sample rate (Hz).
        min_snr: Minimum signal-to-noise ratio (dB).
        max_silence_ratio: Maximum ratio of silence.
        target_db: Target loudness (dBFS).
    """
    min_duration: float = 10.0
    max_duration: float = 30.0
    target_sr: int = 24000
    min_snr: float = 30.0
    max_silence_ratio: float = 0.3
    target_db: float = -23.0


# Default K-Moshi reference transcript
MOSHI_REFERENCE_TRANSCRIPT = """안녕하세요, 저는 케이모시입니다.
한국어 음성 대화 AI예요.
무엇이든 편하게 물어봐 주세요.
오늘 하루도 좋은 하루 되세요."""

# Alternative transcripts for variety
MOSHI_REFERENCE_TRANSCRIPTS = [
    """안녕하세요, 케이모시라고 합니다.
한국어로 대화할 수 있는 AI 어시스턴트예요.
궁금한 거 있으면 편하게 물어보세요.""",

    """안녕하세요, 저는 케이모시예요.
여러분의 질문에 답변해 드리는 AI입니다.
뭐든지 물어봐 주세요, 최선을 다해 도와드릴게요.""",

    """케이모시입니다, 반갑습니다.
한국어 음성 대화를 위해 개발되었어요.
무엇을 도와드릴까요?""",
]


def validate_reference_audio(
    audio_path: Union[str, Path],
    config: Optional[ReferenceAudioConfig] = None,
) -> Dict:
    """Validate reference audio for voice cloning.

    Checks:
    - Duration within acceptable range
    - Sample rate (recommends 24kHz)
    - Mono/stereo (recommends mono)
    - SNR estimation
    - Silence ratio
    - Clipping detection

    Args:
        audio_path: Path to audio file.
        config: Validation configuration.

    Returns:
        Dictionary with validation results:
        - valid: Boolean indicating if audio passes all checks
        - duration: Actual duration in seconds
        - sample_rate: Original sample rate
        - estimated_snr: Estimated SNR in dB
        - issues: List of problems found
        - recommendations: List of suggested improvements
    """
    try:
        import soundfile as sf
    except ImportError:
        raise ImportError("soundfile not installed. Run: pip install soundfile")

    if config is None:
        config = ReferenceAudioConfig()

    audio_path = Path(audio_path)
    if not audio_path.exists():
        return {
            "valid": False,
            "issues": [f"File not found: {audio_path}"],
            "recommendations": [],
        }

    # Load audio
    try:
        audio, sr = sf.read(audio_path)
    except Exception as e:
        return {
            "valid": False,
            "issues": [f"Failed to read audio: {e}"],
            "recommendations": [],
        }

    issues = []
    recommendations = []

    # Duration check
    duration = len(audio) / sr
    if duration < config.min_duration:
        issues.append(
            f"Duration too short: {duration:.1f}s < {config.min_duration}s"
        )
        recommendations.append(
            f"Record longer reference ({config.min_duration}-{config.max_duration} seconds)"
        )
    elif duration > config.max_duration:
        recommendations.append(
            f"Consider trimming to {config.max_duration}s for optimal results"
        )

    # Sample rate check
    if sr != config.target_sr:
        recommendations.append(f"Resample from {sr}Hz to {config.target_sr}Hz")

    # Mono check
    if len(audio.shape) > 1 and audio.shape[1] > 1:
        recommendations.append("Convert stereo to mono for voice cloning")
        audio = audio.mean(axis=1)  # For subsequent analysis

    # Ensure 1D array
    if len(audio.shape) > 1:
        audio = audio.squeeze()

    # SNR estimation
    estimated_snr = _estimate_snr(audio)
    if estimated_snr < config.min_snr:
        issues.append(f"Low SNR: {estimated_snr:.1f}dB < {config.min_snr}dB")
        recommendations.append("Re-record in a quieter environment")

    # Silence ratio check
    silence_ratio = _calculate_silence_ratio(audio)
    if silence_ratio > config.max_silence_ratio:
        recommendations.append(
            f"High silence ratio ({silence_ratio:.0%}), consider trimming"
        )

    # Clipping detection
    clipping_ratio = _detect_clipping(audio)
    if clipping_ratio > 0.001:  # > 0.1% clipping
        issues.append(f"Audio clipping detected: {clipping_ratio:.2%}")
        recommendations.append("Re-record with lower input level")

    # Loudness check
    rms_db = 20 * np.log10(np.sqrt(np.mean(audio ** 2)) + 1e-8)
    if rms_db < -40:
        issues.append(f"Audio too quiet: {rms_db:.1f}dBFS")
        recommendations.append("Increase recording level or normalize audio")
    elif rms_db > -10:
        issues.append(f"Audio too loud: {rms_db:.1f}dBFS")
        recommendations.append("Reduce recording level to prevent distortion")

    return {
        "valid": len(issues) == 0,
        "duration": duration,
        "sample_rate": sr,
        "estimated_snr": estimated_snr,
        "silence_ratio": silence_ratio,
        "loudness_db": rms_db,
        "clipping_ratio": clipping_ratio,
        "issues": issues,
        "recommendations": recommendations,
    }


def _estimate_snr(audio: np.ndarray) -> float:
    """Estimate signal-to-noise ratio.

    Uses a simple method comparing RMS of loud portions
    to quiet portions (assumed noise floor).
    """
    rms = np.sqrt(np.mean(audio ** 2))
    noise_floor = np.percentile(np.abs(audio), 5)
    return 20 * np.log10(rms / (noise_floor + 1e-8))


def _calculate_silence_ratio(
    audio: np.ndarray,
    threshold_db: float = -40,
) -> float:
    """Calculate ratio of silent frames."""
    frame_size = 1024
    hop_size = 512

    threshold_amp = 10 ** (threshold_db / 20)
    silent_frames = 0
    total_frames = 0

    for i in range(0, len(audio) - frame_size, hop_size):
        frame = audio[i:i + frame_size]
        rms = np.sqrt(np.mean(frame ** 2))
        if rms < threshold_amp:
            silent_frames += 1
        total_frames += 1

    return silent_frames / max(total_frames, 1)


def _detect_clipping(audio: np.ndarray, threshold: float = 0.99) -> float:
    """Detect clipping (samples at or near maximum amplitude)."""
    clipped = np.sum(np.abs(audio) >= threshold)
    return clipped / len(audio)


def prepare_reference_audio(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    config: Optional[ReferenceAudioConfig] = None,
    trim_silence: bool = True,
    normalize: bool = True,
) -> Path:
    """Prepare reference audio for voice cloning.

    Processing steps:
    1. Load and convert to mono
    2. Resample to target sample rate
    3. Trim leading/trailing silence
    4. Normalize loudness
    5. Save as WAV

    Args:
        input_path: Source audio file.
        output_path: Output path for processed audio.
        config: Processing configuration.
        trim_silence: Remove leading/trailing silence.
        normalize: Apply loudness normalization.

    Returns:
        Path to processed audio file.

    Raises:
        ImportError: If required packages not installed.
        FileNotFoundError: If input file doesn't exist.
    """
    try:
        import librosa
        import soundfile as sf
    except ImportError:
        raise ImportError(
            "librosa and soundfile required. Run: pip install librosa soundfile"
        )

    if config is None:
        config = ReferenceAudioConfig()

    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    logger.info(f"Processing reference audio: {input_path}")

    # Load audio (resampling to target SR, mono)
    audio, sr = librosa.load(
        str(input_path),
        sr=config.target_sr,
        mono=True,
    )

    # Trim silence
    if trim_silence:
        audio, _ = librosa.effects.trim(audio, top_db=30)
        logger.debug(f"Trimmed to {len(audio)/sr:.2f}s")

    # Normalize loudness
    if normalize:
        # Peak normalization to 95%
        peak = np.max(np.abs(audio))
        if peak > 0:
            audio = audio / peak * 0.95

        # RMS normalization to target dB
        current_rms = np.sqrt(np.mean(audio ** 2))
        target_rms = 10 ** (config.target_db / 20)
        if current_rms > 0:
            audio = audio * (target_rms / current_rms)
            # Clip to prevent overflow
            audio = np.clip(audio, -1.0, 1.0)

        logger.debug(f"Normalized to {config.target_db}dBFS")

    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save
    sf.write(output_path, audio, config.target_sr)
    logger.info(f"Saved processed audio: {output_path}")

    return output_path


def generate_reference_transcript(
    variant: int = 0,
) -> str:
    """Get a reference transcript for K-Moshi voice recording.

    Args:
        variant: Transcript variant index (0=default, 1-3=alternatives).

    Returns:
        Reference transcript string.
    """
    if variant == 0:
        return MOSHI_REFERENCE_TRANSCRIPT
    elif 1 <= variant <= len(MOSHI_REFERENCE_TRANSCRIPTS):
        return MOSHI_REFERENCE_TRANSCRIPTS[variant - 1]
    else:
        return MOSHI_REFERENCE_TRANSCRIPT


def create_voice_profile(
    audio_path: Union[str, Path],
    transcript: str,
    voice_id: str = "kmoshi_v1",
    output_dir: Optional[Union[str, Path]] = None,
) -> Dict:
    """Create a complete voice profile for K-Moshi.

    A voice profile includes:
    - Processed reference audio
    - Transcript file
    - Validation results
    - Profile metadata

    Args:
        audio_path: Path to raw reference audio.
        transcript: Exact transcript of the audio.
        voice_id: Unique voice identifier.
        output_dir: Output directory for profile.

    Returns:
        Dictionary containing profile information.
    """
    audio_path = Path(audio_path)
    if output_dir is None:
        output_dir = audio_path.parent / "profiles"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Validate original
    validation = validate_reference_audio(audio_path)

    # Process audio
    processed_path = output_dir / f"{voice_id}_reference.wav"
    prepare_reference_audio(audio_path, processed_path)

    # Save transcript
    transcript_path = output_dir / f"{voice_id}_transcript.txt"
    with open(transcript_path, "w", encoding="utf-8") as f:
        f.write(transcript)

    # Create profile metadata
    profile = {
        "voice_id": voice_id,
        "reference_audio": str(processed_path),
        "transcript": str(transcript_path),
        "validation": validation,
        "created_at": str(Path(audio_path).stat().st_mtime),
    }

    # Save profile
    import json
    profile_path = output_dir / f"{voice_id}_profile.json"
    with open(profile_path, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2, ensure_ascii=False)

    logger.info(f"Created voice profile: {profile_path}")
    return profile


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="K-Moshi Reference Voice Preparation")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate reference audio")
    validate_parser.add_argument("audio", help="Audio file to validate")

    # Prepare command
    prepare_parser = subparsers.add_parser("prepare", help="Prepare reference audio")
    prepare_parser.add_argument("input", help="Input audio file")
    prepare_parser.add_argument("output", help="Output audio file")
    prepare_parser.add_argument(
        "--no-trim", action="store_true", help="Don't trim silence"
    )
    prepare_parser.add_argument(
        "--no-normalize", action="store_true", help="Don't normalize"
    )

    # Profile command
    profile_parser = subparsers.add_parser("profile", help="Create voice profile")
    profile_parser.add_argument("audio", help="Reference audio file")
    profile_parser.add_argument(
        "--voice-id", default="kmoshi_v1", help="Voice ID"
    )
    profile_parser.add_argument("--output-dir", help="Output directory")
    profile_parser.add_argument(
        "--transcript-variant", type=int, default=0, help="Transcript variant"
    )

    # Transcript command
    transcript_parser = subparsers.add_parser(
        "transcript", help="Get reference transcript"
    )
    transcript_parser.add_argument(
        "--variant", type=int, default=0, help="Transcript variant (0-3)"
    )

    args = parser.parse_args()

    if args.command == "validate":
        result = validate_reference_audio(args.audio)
        print(f"\n{'✅ VALID' if result['valid'] else '❌ INVALID'}")
        print(f"\nDuration: {result.get('duration', 0):.2f}s")
        print(f"Sample Rate: {result.get('sample_rate', 0)}Hz")
        print(f"Estimated SNR: {result.get('estimated_snr', 0):.1f}dB")
        print(f"Loudness: {result.get('loudness_db', 0):.1f}dBFS")

        if result["issues"]:
            print("\nIssues:")
            for issue in result["issues"]:
                print(f"  ❌ {issue}")

        if result["recommendations"]:
            print("\nRecommendations:")
            for rec in result["recommendations"]:
                print(f"  💡 {rec}")

    elif args.command == "prepare":
        output = prepare_reference_audio(
            args.input,
            args.output,
            trim_silence=not args.no_trim,
            normalize=not args.no_normalize,
        )
        print(f"Prepared: {output}")

    elif args.command == "profile":
        transcript = generate_reference_transcript(args.transcript_variant)
        profile = create_voice_profile(
            args.audio,
            transcript,
            voice_id=args.voice_id,
            output_dir=args.output_dir,
        )
        print(f"Created profile: {profile['voice_id']}")

    elif args.command == "transcript":
        transcript = generate_reference_transcript(args.variant)
        print("=" * 60)
        print("K-Moshi Reference Transcript")
        print("=" * 60)
        print(transcript)
        print("=" * 60)
        print("\nRecord this text clearly for voice cloning reference.")
